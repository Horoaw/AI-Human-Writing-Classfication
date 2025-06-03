AI-Generated Text Detection Pipeline
====================================

This project is structured into three main stages for detecting AI-generated text:

1. GetData.py             --> Loads and cleans dataset.
2. TokenizProcess.py      --> Tokenizes and embeds text using BERT.
3. Training-the-model.py  --> Trains and evaluates a classifier using extracted features.

--------------------------------------
📁 Project Structure
--------------------------------------

├── GetData.py
├── TokenizProcess.py
├── Training-the-model.py
├── main.py 
├── data/
│   ├── raw_data.csv         # Input CSV file with text and label columns
│   ├── labels.npy           # Saved labels for training
│   └── bert_embeddings.pt   # Saved BERT embeddings
├── README.txt               # You are here
└── requirements.txt         # Python dependencies

--------------------------------------
🚀 Quick Start
--------------------------------------

Step 1: Install Dependencies
----------------------------
Make sure you have Python 3.8+ and CUDA GPU. Then install dependencies:

    pip install -r requirements.txt

Step 2: Prepare Your Dataset
----------------------------
Put your CSV file (with columns like "text" and "label") into the `data/` directory:

    data/raw_data.csv

Step 3: Run the Pipeline
------------------------

    python GetData.py
    python TokenizProcess.py
    python Training-the-model.py

Or run everything from the entry-point script:

    python main.py

--------------------------------------
📌 Notes
--------------------------------------

- `TokenizProcess.py` uses HuggingFace Transformers to compute BERT embeddings on GPU.
- `Training-the-model.py` loads BERT features and trains a `VotingClassifier` (XGBoost + SVM).
- Models and features are saved locally for reuse.

--------------------------------------
🧪 Requirements
--------------------------------------

- torch
- transformers
- scikit-learn
- xgboost
- numpy
- pandas

For GPU training:
- CUDA-compatible GPU
- cudatoolkit version matching your GPU and PyTorch

--------------------------------------
📝 Author
--------------------------------------
Horoaw - 2025
