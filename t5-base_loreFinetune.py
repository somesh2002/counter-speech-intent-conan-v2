
# This script is designed to train a T5 model for generating counterspeech
# in response to hate speech. It uses the LoRA technique for parameter-efficient fine-tuning.
# The script includes data loading, preprocessing, model training, and evaluation.
# It also includes a function for generating counterspeech given an intent and hate speech input.
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
from peft import get_peft_model, LoraConfig, TaskType

# Dataset
ds = load_dataset("Aswini123/IntentCONANv2")

# Model Initialization
model_name = "t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Preprocessing 
max_input_length = 256
max_target_length = 256
#   # preprocessing function to tokenize the dataset
# The function takes a batch of data and tokenizes the input and target texts
# It also creates the input prompts for the model
# The input prompts are formatted as "generate {csType}: {hatespeech}"
def preprocess(batch):
    prompts = [f"generate {ct.lower()}: {hs}" for ct, hs in zip(batch["csType"], batch["hatespeech"])]
    
    model_inputs = tokenizer(
        prompts,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["counterspeech"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = ds.map(preprocess, batched=True)

# Apply PEFT with LoRA
# LoRA (Low-Rank Adaptation) is a technique for parameter-efficient fine-tuning of large language models.
# It allows for the training of a smaller number of parameters while maintaining the performance of the full model.
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
peft_model = get_peft_model(model, peft_config)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# Define the compute_metrics function
# This function computes the evaluation metrics for the model's predictions
# It takes the predictions and labels as input and computes the ROUGE, BLEU, and BERTScore metrics.
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {key: value * 100 for key, value in rouge_result.items()}

    bleu_result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    bert_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"]) * 100

    return {
        **rouge_result,
        "bleu": bleu_result["bleu"] * 100,
        "bertscore_f1": bert_f1
    }

# Training Arguments
# The training arguments define the parameters for the training process
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model)

# Trainer Initialization
# The Seq2SeqTrainer is used to train the model
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
trainer.train()
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# Evaluation 
metrics = trainer.evaluate()
print(metrics)

# Inference Function for Counterspeech Generation
def generate_counterspeech(intent: str, hate_text: str):
    input_text = f"generate {intent.lower()}: {hate_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_length=64,
        num_beams=5,
        repetition_penalty=1.2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_counterspeech("denouncing", "Go back to your country!"))
