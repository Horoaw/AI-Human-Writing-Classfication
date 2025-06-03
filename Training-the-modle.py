import numpy as np
import pandas as pd
import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from datetime import datetime
from xgboost import plot_tree as xgb_plot_tree
from sklearn.tree import plot_tree as rf_plot_tree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    df = pd.read_csv("balanced_100k_dataset.csv")
    return df['text'].tolist(), df['label'].tolist()

class BertFeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def get_embeddings(self, texts):
        cache_path = "bert_embeddings.pt"
        if os.path.exists(cache_path):
            print("Loading cached embeddings...")
            return torch.load(cache_path, map_location='cpu')
        
        print("Generating BERT embeddings...")
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                ).to(device)
                
                outputs = self.model(**inputs)
                cls_embed = outputs.last_hidden_state[:, 0, :]
                mean_embed = outputs.last_hidden_state.mean(dim=1)
                combined = torch.cat([cls_embed, mean_embed], dim=1)
                embeddings.append(combined.squeeze().cpu())

                del inputs, outputs
                torch.cuda.empty_cache()
        
        torch.save(embeddings, cache_path)
        return embeddings

class ClassificationSystem:
    def __init__(self):
        self.models = {}
        self.history = {}
        self.results = {}
        self.metric_details = {}

    def initialize_models(self):
        self.models = {
            'LogisticRegression': GridSearchCV(
                LogisticRegression(max_iter=1000, class_weight='balanced'),
                {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'saga']},
                cv=3,
                n_jobs=-1
            ),
            'RandomForest': GridSearchCV(
                RandomForestClassifier(class_weight='balanced_subsample'),
                {'n_estimators': [100, 200], 'max_depth': [None, 10]},
                cv=3,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                learning_rate=0.1,
                n_estimators=1000,
                eval_metric=['logloss', 'error'],
                tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
                early_stopping_rounds=50,
                use_label_encoder=False
            )
        }

    def train(self, X_train, y_train, X_val, y_val):
        for name, model in self.models.items():
            print(f"\n=== Training {name} ===")
            start = time.time()

            if name == 'XGBoost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=50
                )
                self.history[name] = model.evals_result()
            else:
                model.fit(X_train, y_train)
                if hasattr(model, 'best_params_'):
                    print(f"Best params for {name}: {model.best_params_}")

            pred = model.predict(X_val)
            acc = accuracy_score(y_val, pred)
            prec = precision_score(y_val, pred)
            rec = recall_score(y_val, pred)
            f1 = f1_score(y_val, pred)

            self.results[name] = acc
            self.metric_details[name] = {'precision': prec, 'recall': rec, 'f1': f1}
            
            print(f"{name} Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            self._save_model(model, name)
            self._plot_confusion_matrix(y_val, pred, name)
            print(f"Training time for {name}: {time.time() - start:.2f}s")

        return self.results

    def _save_model(self, model, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(f"{model_name}_confusion_matrix.png")
        plt.close()

    def plot_metrics_comparison(self):
        df = pd.DataFrame(self.metric_details).T
        df.plot(kind='bar', figsize=(10, 6))
        plt.title("Model Metric Comparison")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("model_metrics_comparison.png")
        plt.show()

    def plot_accuracy_curve(self):
        plt.figure(figsize=(12, 6))
        if 'XGBoost' in self.history:
            evals = self.history['XGBoost']
            acc = [1 - e for e in evals['validation_0']['error']]
            loss = evals['validation_0']['logloss']
            plt.plot(acc, label="XGBoost Accuracy")
            plt.plot(loss, label="XGBoost Logloss")
        for model, acc in self.results.items():
            if model != 'XGBoost':
                plt.axhline(y=acc, linestyle='--', label=f'{model} Acc')
        plt.title("Model Accuracy & Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("accuracy_loss_curve.png")
        plt.show()

    def save_results(self):
        with open("model_metrics.json", "w") as f:
            json.dump(self.metric_details, f, indent=4)
        print("Model metrics saved to model_metrics.json")

        df = pd.DataFrame([
            {
                "Model": model,
                "Accuracy": self.results[model],
                "Precision": self.metric_details[model]['precision'],
                "Recall": self.metric_details[model]['recall'],
                "F1": self.metric_details[model]['f1']
            }
            for model in self.results
        ])
        df.to_csv("model_results.csv", index=False)
        print("Model results saved to model_results.csv")

        with open("model_report.md", "w") as f:
            f.write("# the results will be like: \n\n")
            for model in self.results:
                f.write(f"## {model}\n")
                f.write(f"- Accuracy: {self.results[model]:.4f}\n")
                f.write(f"- Precision: {self.metric_details[model]['precision']:.4f}\n")
                f.write(f"- Recall: {self.metric_details[model]['recall']:.4f}\n")
                f.write(f"- F1 Score: {self.metric_details[model]['f1']:.4f}\n\n")
        print("Model report saved to model_report.md")

    def plot_tree_structure(self, model_name="XGBoost", tree_index=0):
        model = self.models[model_name]
    
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
    
        if model_name == "XGBoost":
            plt.figure(figsize=(30, 12))
            xgb_plot_tree(model, num_trees=tree_index)
            plt.title(f"XGBoost Tree #{tree_index}")
            plt.savefig(f"{model_name}_tree_{tree_index}.png", dpi=300)
            plt.close()
            print(f"XGBoost tree plot saved to {model_name}_tree_{tree_index}.png")
    
        elif model_name == "RandomForest":
            estimator = model.estimators_[tree_index]
            plt.figure(figsize=(30, 12))
            rf_plot_tree(estimator, filled=True, fontsize=8, max_depth=3)
            plt.title(f"RandomForest Tree #{tree_index}")
            plt.savefig(f"{model_name}_tree_{tree_index}.png", dpi=300)
            plt.close()
            print(f"RandomForest tree plot saved to {model_name}_tree_{tree_index}.png")
    
        else:
            print(f"Tree plotting not supported for model: {model_name}")

    def ensemble_predict(self, X, weights=None):
        probas = []
        for name, model in self.models.items():
            if hasattr(model, 'best_estimator_'):
                m = model.best_estimator_
            else:
                m = model
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)
            else:
                pred = m.predict(X)
                onehot = np.zeros((len(pred), 2))
                onehot[np.arange(len(pred)), pred] = 1
                p = onehot
            probas.append(p)
        # default weights is 1
        if weights is None:
            weights = [1] * len(probas)
        weights = np.array(weights)
        weights = weights / weights.sum()
        combined = np.zeros_like(probas[0])
        for w, p in zip(weights, probas):
            combined += w * p
        # it will return the PM over than 0.5
        if combined.shape[1] == 2:
            return (combined[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(combined, axis=1)

def main():
    texts, labels = load_data()

    feature_extractor = BertFeatureExtractor()
    embeddings = feature_extractor.get_embeddings(texts)
    
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings)
    X = embeddings.cpu().numpy()
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.1,
        stratify=y,
        random_state=42
    )

    system = ClassificationSystem()
    system.initialize_models()
    system.train(X_train, y_train, X_val, y_val)

    system.save_results()
    ensemble_pred = system.ensemble_predict(X_val, weights=[5,2,3]) #the weights could be changed if u wannna
    ensemble_acc = accuracy_score(y_val, ensemble_pred)
    print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")

    system.plot_metrics_comparison()
    system.plot_accuracy_curve()
    system.plot_tree_structure("XGBoost", tree_index=0) 
    system.plot_tree_structure("RandomForest", tree_index=0)  

if __name__ == "__main__":
    main()
