# Multi-Interest User Representation for Deep News Recommender Systems

This repository contains the implementation of my Master's thesis titled **"Interest Representations in Deep News Recommender Systems"** completed at the Institut für Maschinelle Sprachverarbeitung (IMS), Universität Stuttgart.

## 📌 Thesis Abstract

Traditional news recommender systems often represent users using a single vector, which can fail to capture the diversity of user interests. This project proposes a **multi-interest user representation model** enhanced with a **disentanglement objective** to ensure distinct and interpretable interest vectors. The model is evaluated on the MIND dataset with a focus on improving:

- Click prediction accuracy  
- Recommendation fairness and diversity  
- Reducing systemic biases in embedding space  

## 🧠 Project Structure

```
MastersCode/
├── config/                 # Configuration files (e.g. mind_standard)
├── notebooks/             # Jupyter notebooks for analysis and visualization
├── nrs/                   # Core package: data loaders, encoders, training, scoring
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   ├── utils.py
│   └── __init__.py
├── train_mind             # Main training script
├── train_mind2            # Variant training script
├── requirements           # Python dependencies
```

## 🧪 Experiments Conducted

1. **Click Prediction**  
   - Multi-interest models significantly outperform traditional single-vector models across AUC, NDCG, MRR.

2. **Disentanglement**  
   - Cosine similarity-based loss encourages separation of interest vectors.
   - Visualized using t-SNE and cosine similarity histograms.

3. **Fairness & Diversity**  
   - Score distribution and standard deviation analyzed.
   - Multi-interest models reduce bias but need tuning to retain diversity.

## 📊 Dataset

- **Microsoft News Dataset (MIND)**
- Includes user click logs, impression logs, and metadata (titles, categories)
- Extensive preprocessing: tokenization, indexing, embedding generation, and negative sampling

## ⚙️ Tech Stack

- **Language**: Python 3.8+
- **Framework**: PyTorch
- **Libraries**: NumPy, Pandas, scikit-learn, Matplotlib, OmegaConf, WandB
- **Backbone Models**: BERT (for news encoding)

## 🚀 Getting Started

```bash
# (optional) create virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements

# run training
python train_mind

# optional: run variant
python train_mind2
```

## 📌 Highlights

- Modular and research-grade codebase
- Supports multiple interest vector configurations (N = 1, 5, 10, 25, 50)
- Visual analysis integrated: t-SNE plots, score distributions, similarity histograms
- Evaluation: AUC, NDCG@10, MRR, Precision, Recall, CTR@1

## 📬 Contact

For questions or collaborations, feel free to connect via [LinkedIn](https://www.linkedin.com/in/aditi-godbole/) or open an issue in this repository.

---

> This repository accompanies the official submission of my Master's thesis (Sept 2024) at the University of Stuttgart.
