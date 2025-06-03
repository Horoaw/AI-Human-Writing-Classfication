import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from geomloss import SamplesLoss

class WassersteinClassifier(nn.Module):
    def __init__(self):
        super(WassersteinClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.fc(x)

def get_token_embeddings(texts, tokenizer, model, device, batch_size=6, max_length=256):
    all_embeddings = []
    model = model.to(device)
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts in batches"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.last_hidden_state: (batch_size, seq_len, hidden_dim)
            for j in range(outputs.last_hidden_state.size(0)):
                token_embeddings = outputs.last_hidden_state[j].cpu()
                all_embeddings.append(token_embeddings)
    return all_embeddings

def main():
    df = pd.read_csv("balanced_100k_dataset.csv")
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "distilbert-base-uncased"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if os.path.exists('bert_embeddings.pt'):
        print("Loading cached BERT embeddings...")
        embeddings_list = torch.load('bert_embeddings.pt')
    else:
        print("Extracting BERT embeddings...")
        embeddings_list = get_token_embeddings(texts, tokenizer, model, device, batch_size=6)
        torch.save(embeddings_list, 'bert_embeddings.pt')
        print("Saved BERT embeddings to 'bert_embeddings.pt'.")

    if os.path.exists("wasserstein_features.csv"):
        print("Loading cached Wasserstein features...")
        df_out = pd.read_csv("wasserstein_features.csv")
        wasserstein_features = df_out['wasserstein'].tolist()
        labels = df_out['label'].tolist()
    else:
        print("Extracting human corpus...")
        human_embeddings = [emb for emb, label in zip(embeddings_list, labels) if label == 0]
        human_all_tokens = torch.cat(human_embeddings, dim=0)

        MAX_TOKENS = 3000
        if human_all_tokens.size(0) > MAX_TOKENS:
            indices = torch.randperm(human_all_tokens.size(0))[:MAX_TOKENS]
            human_all_tokens = human_all_tokens[indices]
        print(f"Total tokens in human corpus (after limit): {human_all_tokens.shape}")

        sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

        wasserstein_features = []

        print("Calculating Wasserstein (Sinkhorn) distances...")
        for emb in tqdm(embeddings_list, desc="Sinkhorn distances"):
            x_i = emb
            y_j = human_all_tokens
            dist = sinkhorn_loss(x_i, y_j).item()
            wasserstein_features.append(dist)

        df_out = pd.DataFrame({'wasserstein': wasserstein_features, 'label': labels})
        df_out.to_csv("wasserstein_features.csv", index=False)
        print("Saved Wasserstein features to 'wasserstein_features.csv'.")

    X = torch.tensor(wasserstein_features).unsqueeze(1)
    y = torch.tensor(labels)

    dataset = TensorDataset(X, y)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model_cls = WassersteinClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_cls.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model_cls.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)
            logits = model_cls(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_cls.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).float()
                logits = model_cls(batch_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_y.numpy())
        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {acc:.4f}")

if __name__ == "__main__":
    main()
