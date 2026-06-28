import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
import torch

MODEL_DIR = 'ShiChenLee/EmotionClassification'
NUM_LABELS = 28
THRESHOLDS = [0.3, 0.4, 0.5]

dataset = load_dataset('google-research-datasets/go_emotions', 'simplified')
label_names = dataset['train'].features['labels'].feature.names
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

val = dataset['validation']
all_probs, all_labels = [], []

for i in range(0, len(val), 64):
    batch = val[i:i+64]
    enc = tokenizer(batch['text'], truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    multi_hot_labels = np.zeros((len(batch['text']), NUM_LABELS))
    for j, labels in enumerate(batch['labels']):
        for l in labels:
            multi_hot_labels[j, l] = 1
    all_probs.append(probs)
    all_labels.append(multi_hot_labels)

all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

print(f"{'Label':<20} {'Thresh':>6} {'P':>6} {'R':>6} {'F1':>6}")
print('-' * 40)
for threshold in THRESHOLDS:
    preds = (all_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average=None, zero_division=0)
    for i, name in enumerate(label_names):
        print(f"{name:<20} {threshold:>6.2f} {precision[i]:>6.3f} {recall[i]:>6.3f} {f1[i]:>6.3f}")

    print()