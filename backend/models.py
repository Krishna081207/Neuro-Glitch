import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Load DistilBERT model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

class MentalHealthDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['label']
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(label)
        return item

def predict_journal(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

def fine_tune_model(data_path):
    try:
        df = pd.read_csv(data_path)
    except Exception:
        return "Dataset not found."
    dataset = MentalHealthDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Batch loss: {loss.item()}")
    model.eval()
    print("Fine-tuning complete.")
