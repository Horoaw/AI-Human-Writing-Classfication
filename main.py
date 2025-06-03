from GetData import load_data
from TokenizProcess import tokenize_and_embed
from Training_the_model import load_embeddings_labels, build_voting_classifier
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":

    df = load_data("data/raw_data.csv")

    embeddings = tokenize_and_embed(df["text"].tolist())
    torch.save(embeddings, "bert_embeddings.pt")

    labels = df["label"].values
    np.save("data/labels.npy", labels)

    X, y = embeddings.cpu().numpy(), labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = build_voting_classifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
