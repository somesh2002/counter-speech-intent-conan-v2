# Importing necessary libraries
import torch
# import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from torch.optim import AdamW
import torch
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
      
INTENTS = ["informative", "denouncing", "question", "positive"]
# Define the intents


# Intent Contrastive Collator
# This collator will handle the batching and processing of the data
# for the contrastive learning task.
# It will create pairs of target and distractor inputs for the model.
# It will also handle the tokenization and padding of the inputs and labels.

class IntentContrastiveCollator:
    def __init__(self, tokenizer, max_input_length=256, max_target_length=256):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.intents = INTENTS # ["informative", "denouncing", "question", "positive"]

    def __call__(self, batch):
        processed_batch = {
            "target_input_ids": [], "target_attention_mask": [], "target_labels": [],
            "distractor_input_ids": [], "distractor_attention_mask": [],
        }

        original_hate_speech_texts = []
        original_target_intents = []
        original_gold_cs_texts = []

        for item in batch:
            hate_speech = item['hatespeech']
            target_intent = item['csType']
            gold_counterspeech = item['counterspeech']

            original_hate_speech_texts.append(hate_speech)
            original_target_intents.append(target_intent)
            original_gold_cs_texts.append(gold_counterspeech)

            target_prompt = f"generate {target_intent}: {hate_speech}"
            tokenized_target_input = self.tokenizer(
                target_prompt,
                max_length=self.max_input_length,
                padding="max_length", # Or handle padding later for the whole batch
                truncation=True,
                return_tensors="pt"
            )
            processed_batch["target_input_ids"].append(tokenized_target_input.input_ids.squeeze())
            processed_batch["target_attention_mask"].append(tokenized_target_input.attention_mask.squeeze())

            with self.tokenizer.as_target_tokenizer():
                tokenized_target_labels = self.tokenizer(
                    gold_counterspeech,
                    max_length=self.max_target_length,
                    padding="max_length", # Or handle padding later
                    truncation=True,
                    return_tensors="pt"
                )
            # Replace padding token id in labels with -100
            labels = tokenized_target_labels.input_ids.squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
            processed_batch["target_labels"].append(labels)


            # 2. Distractor processing
            possible_distractor_intents = [i for i in self.intents if i != target_intent]
            distractor_intent = random.choice(possible_distractor_intents)

            distractor_prompt = f"generate {distractor_intent}: {hate_speech}"
            tokenized_distractor_input = self.tokenizer(
                distractor_prompt,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            processed_batch["distractor_input_ids"].append(tokenized_distractor_input.input_ids.squeeze())
            processed_batch["distractor_attention_mask"].append(tokenized_distractor_input.attention_mask.squeeze())

        # Stack tensors
        for key in processed_batch:
            processed_batch[key] = torch.stack(processed_batch[key])

        return processed_batch


# Hyperparameters
learning_rate = 5e-5
num_epochs = 3
batch_size = 8 
alpha = 0.5
beta = 0.5   

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Data Loader
train_dataset = ds["train"]
train_dataset = Dataset.from_dict(train_dataset)
collator = IntentContrastiveCollator(tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    total_loss_epoch = 0
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Move batch to device
        target_input_ids = batch["target_input_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)
        target_labels = batch["target_labels"].to(device) # These are the gold CS token_ids

        distractor_input_ids = batch["distractor_input_ids"].to(device)
        distractor_attention_mask = batch["distractor_attention_mask"].to(device)

        # 1. Primary Generation Loss (L_target)
        target_outputs = model(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            labels=target_labels
        )
        L_target = target_outputs.loss # Standard cross-entropy loss from T5

        with torch.no_grad():
 
            distractor_gold_outputs = model(
                input_ids=distractor_input_ids,
                attention_mask=distractor_attention_mask,
                labels=target_labels # IMPORTANT: using GOLD labels here
            )

        distractor_outputs_for_logits = model(
            input_ids=distractor_input_ids,
            attention_mask=distractor_attention_mask,
        )
        L_distractor_away_from_gold = -distractor_gold_outputs.loss # (Calculated with torch.no_grad() above)
        combined_loss = L_target + beta * L_distractor_away_from_gold
        combined_loss.backward()
        optimizer.step()

        total_loss_epoch += combined_loss.item()
        if (batch_idx + 1) % 100 == 0: # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                  f"Combined Loss: {combined_loss.item():.4f}, L_target: {L_target.item():.4f}, "
                  f"L_dist_away: {L_distractor_away_from_gold.item() * beta :.4f} (raw NLL for dist: {-L_distractor_away_from_gold.item()/beta:.4f})")


    print(f"Epoch {epoch+1} Average Loss: {total_loss_epoch / len(train_dataloader)}")


model.eval()
hate_speech_instance = "The world is total shit."
desired_intent = "informative"
prompt = f"generate {desired_intent}: {hate_speech_instance}"

input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
generated_cs = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated '{desired_intent}' CS: {generated_cs}")