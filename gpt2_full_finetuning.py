# 
# This script is designed to train a GPT-2 model for generating counterspeech responses to hate speech. It uses the Hugging Face Transformers library and the datasets library to load and preprocess the data, train the model, and evaluate its performance.
# The script includes functions for tokenizing the dataset, training the model, and generating counterspeech responses.
# It also includes a function for computing evaluation metrics such as ROUGE, BLEU, and BERTScore.
# The model is trained using the Trainer class from the Transformers library, and the training arguments are defined using the TrainingArguments class.
# The script also includes a function for generating counterspeech responses given an intent and hate speech input.
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import evaluate

# Dataset
ds = load_dataset("Aswini123/IntentCONANv2")

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and Tokenizer
# Using GPT-2 for generating counterspeech
# The model is initialized with a set of special tokens for the intents and the end-of-sequence token.
# The special tokens are added to the tokenizer, and the model is resized to accommodate the new tokens.
# The model is moved to the appropriate device (GPU or CPU).
# The special tokens include additional tokens for different intents and a token for the end of the sequence.
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[Informative]", "[Denouncing]", "[Question]", "[Positive]", "[SEP]"
    ],
    "eos_token": "[EOS]"
}

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(SPECIAL_TOKENS)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# 5. Preprocessing
max_input_length = 256
max_target_length = 256
# The preprocessing function tokenizes the dataset and creates input prompts for the model.
# The input prompts are formatted as "generate {intent}: {hatespeech} [SEP] {counterspeech} [EOS]"
# The function takes a batch of data and tokenizes the input and target texts.
# It also creates the input prompts for the model.
def tokenize(batch):
    inputs = []
    for intent, hs, cs in zip(batch["csType"], batch["hatespeech"], batch["counterspeech"]):
        intent_token = f"[{intent.capitalize()}]"
        prompt = f"generate {intent_token}: {hs} [SEP] {cs} [EOS]"
        inputs.append(prompt)
    
    tokens = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_input_length + max_target_length,
        return_tensors="pt"
    )
    
    labels = tokens.input_ids.clone()
    for i, text in enumerate(inputs):
        sep_idx = tokenizer(text.split("[SEP]")[0], truncation=True)["input_ids"]
        sep_len = len(sep_idx)
        labels[i][:sep_len] = -100  # Mask the prompt part of the label
    
    tokens["labels"] = labels
    return tokens

tokenized_dataset = ds.map(tokenize, batched=True)

# Data Collator for GPT-2
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)
# Data Collator for Language Modeling
# The data collator is used to create batches of data for training.
# It pads the input sequences to the maximum length in the batch and creates attention masks.
training_args = TrainingArguments(
    output_dir="./gpt2_counterspeech",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    report_to=[],  # disable logging to external services like WandB
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    weight_decay=0.01,
    learning_rate=5e-5,
    save_total_limit=2,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True
)

# Evaluation metrics
# The evaluation metrics are used to evaluate the performance of the model during training.
# The metrics include ROUGE, BLEU, and BERTScore.
# The ROUGE metric is used to evaluate the quality of the generated text by comparing it to reference texts.

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
# Trainer Initialization
# The Trainer class is used to train the model.
# It takes the model, training arguments, training dataset, evaluation dataset, data collator, and tokenizer as input.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

# Training
trainer.train()
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# Evaluation
# The evaluation is performed on the test dataset.
# The evaluation metrics are computed using the compute_metrics function.
eval_results = trainer.evaluate(tokenized_dataset["test"])
print(eval_results)

def generate_counterspeech(intent, hate_text, max_length=128, use_beam=True):
    model.eval()
    intent_token = f"[{intent.capitalize()}]"
    input_text = f"{intent_token} {hate_text} [SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            eos_token_id=tokenizer.convert_tokens_to_ids("[EOS]"),
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5 if use_beam else 1,
            do_sample=not use_beam,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            early_stopping=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


example = generate_counterspeech("Denouncing", "You people ruin everything!")
print("Generated:", example)
