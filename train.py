import numpy as np
from datasets import load_dataset
from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer)
import torch

MODEL_NAME = "roberta-base"
OUTPUT_DIR = './model_output'
NUM_LABELS = 28

dataset = load_dataset('go_emotions', 'simplified')
label_names = dataset['train'].features['labels'].feature.names

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    tokens = tokenizer(batch['text'], truncation=True, padding='max_length', max_length='128')
    labels = [[1.0 if i in batch['labels'][j] else 0.0 for i in range(NUM_LABELS)] for j in range(len(batch['text']))]
    tokens['labels'] = labels
    return tokens

tokenized = dataset.map(tokenize, batched=True, remove_columns=['text', 'id'])
tokenized.set_format('torch')

model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_lables=NUM_LABELS, problem_type='multi_label_classification')

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    fp16=True,
    logging_steps=50
    )

trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'])

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print('Model saved to', OUTPUT_DIR)