import time
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertTokenizerFast
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from dataset import CustomDataset
from torch.nn import CrossEntropyLoss
from calc_measures import compute_measures, print_measures

data_path = 'data\\train.csv'
data = pd.read_csv(data_path)

texts = data['text'].tolist()
labels = data['label'].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_texts, train_labels)
test_dataset = CustomDataset(test_texts, test_labels)

# Using all the training data
# train_dataset = CustomDataset(texts, labels)

# model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

model_path = "e:\\nlp_projects\\fnd_competition_2023\\fine_tune_chinesebert.pt"
model = torch.load(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device', device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training Loop

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

criterion = CrossEntropyLoss()

epochs = 10
start = time.time()
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0
    train_logits = []
    train_cur_labels = []
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        train_cur_labels.extend(labels.cpu().numpy())

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        train_curr_loss = criterion(outputs.logits, labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
        train_logits.extend(preds)
        logits = logits.cpu().detach().numpy()
        label_ids = labels.to('cpu').numpy()

        total_train_loss += loss
        total_train_accuracy += flat_accuracy(logits, label_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    test_logits = []
    eval_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            eval_labels.extend(label.cpu().numpy())

            loss = criterion(outputs.logits, label)
            test_curr_loss = criterion(outputs.logits, label)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
            test_logits.extend(preds)
            logits = logits.cpu().detach().numpy()
            label_ids = label.to('cpu').numpy()

            total_eval_loss += loss.item()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

    
    print(f"Epochs: {epoch} | Train Loss: {total_train_loss / len(train_dataloader): .3f} | Train Accuracy: {total_train_accuracy / len(train_dataloader): .3f} | Test Loss: {total_eval_loss / len(test_dataloader): .3f} | Test Accuracy: {total_eval_accuracy / len(test_dataloader): .3f}")
    train_mesures = compute_measures(train_logits, train_cur_labels)
    test_mesures = compute_measures(test_logits, eval_labels)
    print_measures((epoch + 1), total_train_loss / len(train_dataloader), train_mesures, 'train')
    print_measures((epoch + 1), total_eval_loss / len(test_dataloader), test_mesures, 'test')
end = time.time()
elapsed = end - start
print(f"Time elapsed {elapsed/60:.2f} min")
