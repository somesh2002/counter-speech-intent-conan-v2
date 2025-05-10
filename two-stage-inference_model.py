#
# Importing necessary libraries
# The main aim of this code is to train a T5 model for generating counter-speech responses to hate speech.
# The script includes data loading, preprocessing, model training, and evaluation.
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
from torch.optim import AdamW


# Load Dataset
ds = load_dataset("Aswini123/IntentCONANv2")

# Model Initialization
model_name = "t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Preprocessing 
max_input_length = 128
max_target_length = 256

# Preprocessing function to tokenize the dataset
# The function takes a batch of data and tokenizes the input and target texts
# It also creates the input prompts for the model
# The input prompts are formatted as "generate {hatespeech}"
def preprocess_task1(batch):
    prompts = [f"generate counter_speech: {hs}" for hs in batch["hatespeech"]]
    
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

tokenized_initial_dataset = ds.map(preprocess_task1, batched=True)


# Training arguments defined for the model trainer
# The training arguments include parameters such as learning rate, batch size, number of epochs, etc.

training_args_task1 = Seq2SeqTrainingArguments(
    output_dir="./t5_counterspeech",
    report_to=[],
    learning_rate=6e-4,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    save_strategy="epoch",
    generation_num_beams=5,
)

# Data Collator for T5
# The data collator is used to create batches of data for training.
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Evaluation metrics

trainer_task1 = Seq2SeqTrainer(
    model=model,
    args=training_args_task1,
    train_dataset=tokenized_initial_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer_task1.train()
model.save_pretrained("task1_final")

# Preprocessing (with csType)
# The preprocessing function tokenizes the dataset and creates input prompts for the model.
# The input prompts are formatted as "generate {csType}: {hatespeech}"
def preprocess_task2(batch):
    prompts = [f"generate {ct}: {hs}" for ct, hs in zip(batch["csType"], batch["hatespeech"])]
    
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

tokenized_dataset = ds.map(preprocess_task2, batched=True)

# Training arguments defined for the model trainer
# The training arguments include parameters such as learning rate, batch size, number of epochs, etc.
training_args_task2 = Seq2SeqTrainingArguments(
    output_dir="./t5_counterspeech",
    eval_strategy="epoch",
    report_to=[],
    learning_rate=9e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    save_strategy="epoch",
    generation_num_beams=5,
)

# Metrics
# Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

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

# combine train and validation datasets
# The combined dataset is used for training the updated model.
combined_train_dataset = concatenate_datasets([
    tokenized_dataset["train"], 
    tokenized_dataset["validation"]
])


# Trainer Initialization
# The Seq2SeqTrainer is used to train the model
trainer_task2 = Seq2SeqTrainer(
    model=model,
    args=training_args_task2,
    train_dataset=combined_train_dataset,
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
trainer_task2.train()
trainer_task2.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# Evaluation 
test_results = trainer_task2.evaluate(tokenized_dataset["test"])
print(test_results)
