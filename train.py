import numpy as np
import torch
from datasets import load_dataset
from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer)
from torch.utils.data import Dataset

MODEL_NAME = "roberta-base"
OUTPUT_DIR = './model_output'
NUM_LABELS = 28

dataset = load_dataset('google-research-datasets/go_emotions', 'simplified')
print('Dataset loaded')
label_names = dataset['train'].features['labels'].feature.names

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    tokens = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    tokens['labels'] = [
        [1.0 if i in batch['labels'][j] else 0.0 for i in range(NUM_LABELS)]
        for j in range(len(batch['text']))
    ]
    return tokens

print('Tokenizing...')
tokenized = dataset.map(tokenize, batched=True, remove_columns=['text', 'id'])
print('Tokenization complete')
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Wrap in a dataset class that returns labels as float32
class EmotionDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': torch.tensor(self.data[idx]['labels'], dtype=torch.float32)
        }

train_dataset = EmotionDataset(tokenized['train'])
val_dataset = EmotionDataset(tokenized['validation'])

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type='multi_label_classification'
)

# Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    fp16=True,
    logging_steps=50,
    save_total_limit=1,   
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print('Starting training...')
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print('Model saved to', OUTPUT_DIR)