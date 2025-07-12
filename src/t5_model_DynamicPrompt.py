# Importing necessary libraries
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
from torch.optim import AdamW
import random
import re

# Load Dataset
ds = load_dataset("Aswini123/IntentCONANv2")

# Model Initialization
model_name = "t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Preprocessing
max_input_length = 256
max_target_length = 256

# preprocessing function to tokenize the dataset
# The function takes a batch of data and tokenizes the input and target texts
# It also creates the input prompts for the model
# The input prompts are formatted as "generate {csType}: {hatespeech}"
# The target texts are tokenized separately

prompt_templates = [
    "generate a counterspeech with intent '{ct}': {hs}",
    "write a response reflecting '{ct}' to: {hs}",
    "produce a counterspeech guided by intent '{ct}' for: {hs}",
]

# Function to randomly inject noise or corruption
# into the prompts to make the model more robust
# The function randomly selects a corruption mode and applies it to the prompt
def corrupt_prompt(prompt):
    corruption_modes = [
        lambda x: re.sub(r"\b(generate|write|produce)\b", "<unk>", x),
        lambda x: x.replace(":", " __ "),
        lambda x: x.lower(),
        lambda x: x + " ???",
        lambda x: x.replace("counterspeech", "countersppech"),
    ]
    if random.random() < 0.4:  # 50% chance to corrupt
        corruption = random.choice(corruption_modes)
        return corruption(prompt)
    return prompt

# Preprocessing function to tokenize the dataset
# The function takes a batch of data and tokenizes the input and target texts
# It also creates the input prompts for the model
def preprocess(batch):
    prompts = []
    for ct, hs in zip(batch["csType"], batch["hatespeech"]):
        template = random.choice(prompt_templates)
        prompt = template.format(ct=ct, hs=hs)
        noisy_prompt = corrupt_prompt(prompt)
        prompts.append(noisy_prompt)

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
    model_inputs["raw_prompts"] = prompts  # for debugging/analysis
    return model_inputs

tokenized_dataset = ds.map(preprocess, batched=True)

# training arguments defined for the model trainer
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_counterspeech",
    eval_strategy="epoch",
    report_to=[],  # No external logging
    learning_rate=1e-4,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    save_strategy="epoch",
    generation_num_beams=5,
)

# Data Collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Evaluation metrics
# Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# compute_metrics function for evaluation on trainer
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

# Trainer Initialization
trainer = Seq2SeqTrainer(
    model=model,
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

# Evaluation on test Dataset
test_results = trainer.evaluate(tokenized_dataset["test"])
print(test_results)

# Evaluation on validation Dataset
metrics = trainer.evaluate()
print(metrics)


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