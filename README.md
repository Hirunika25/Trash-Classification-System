# Trash Classification : MobileNetV3-Large 

A deep learning project that classifies waste images into 6 categories using MobileNetV3-Large trained on the :
[Recyclable and Household Waste Classification Dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

---


## Dataset

- **Source:** Kaggle — Recyclable and Household Waste Classification (Alistair King)
- **Size:** 15,000 images · 256×256 px · PNG format
- **Original structure:** 30 sub-categories, each with `default/` (studio) and `real_world/` images
- **Our approach:** Merge all sub-categories into 6 main classes, merge default + real_world per class

### 6 output classes
 
| Index | Class | Sub-categories included |
|-------|-------|------------------------|
| 0 | Plastic | plastic bottles, bags, straws, cup lids, cutlery, containers, styrofoam cups |
| 1 | Paper | newspaper, office paper, magazines, cardboard, paper cups |
| 2 | Glass | beverage bottles, food jars, cosmetic containers |
| 3 | Metal | aluminum soda cans, food cans, steel food cans, aerosol cans |
| 4 | Organic | food waste, fruit peels, vegetable scraps, eggshells, coffee grounds, tea bags |
| 5 | Textile | clothing, shoes |
 
### Dataset split 
 
```
Train : 70%  →  10,500 images   (used for model training)
Val   : 15%  →   2,250 images   (used for tuning and early stopping)
Test  : 15%  →   2,250 images   (used only the for final ensemble evaluation)
Seed  : 42
```
 
Split is stratified — every class gets the exact same 70/15/15 ratio.
 
---
 

## Project Structure

```
trash-classification-system/
│
├── data/
│   ├── raw/
│   │   └── images/                        # unzip Kaggle dataset here 
│   │       ├── Plastic water bottles/
│   │       │   ├── default/     *.png
│   │       │   └── real_world/  *.png
│   │       └── ...  (30+ sub-category folders)
│   ├── processed/
│   └── splits.csv                         
│
├── notebooks/
│   ├── eda/
│   │   ├── eda.ipynb
│   │   └── plots/
│   └── preprocessing.ipynb   # runs this once → generates splits.csv
│
├── models/
│   ├── checkpoints/
│   │   └── best_model.pth         # Download from Google drive
│   ├── demo_samples/              # place sample images here for the demo
│   ├── mobilenetV3.ipynb          # training notebook (Kaggle)
│   ├── demo.ipynb                 # Gradio demo notebook
│   ├── training_history.png
│   └── confusion_matrix.png
│
├── predict.py                     # single image inference script
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
``` 
[Download model checkpoint from google drive](https://drive.google.com/drive/folders/1_XVDZH1qiQtMfTnFCTyWEA9VV5gjdTUs?usp=sharing)

---

### Training strategy

- **Stage 1 (epochs 1–10):** Backbone frozen, head trained only — LR=1e-3
- **Stage 2 (epochs 11–20):** Top backbone blocks unfrozen, fine-tuned — LR=1e-5
- **Scheduler:** Cosine annealing
- **Imbalance handling:** WeightedRandomSampler + weighted CrossEntropyLoss
- **Platform:** Kaggle T4 GPU with AMP mixed precision

---

## Results

| Metric | Value |
|--------|-------|
| Val Accuracy | ~85% |
| Test Loss | — |
| Training platform | Kaggle T4 GPU |
| Total epochs | 20 (10 Stage 1 + 10 Stage 2) |

### Per-class performance (test set)

| Class | Correct | Total | Recall |
|-------|---------|-------|--------|
| Plastic | 728 | 975 | 74.7% |
| Paper | 411 | 450 | 91.3% |
| Glass | 141 | 150 | 94.0% |
| Metal | 288 | 300 | 96.0% |
| Organic | 218 | 225 | 96.9% |
| Textile | 144 | 150 | 96.0% |

**Key finding:** Plastic has the lowest recall (74.7%) due to its 9 visually diverse
sub-categories. All other classes exceed 91% recall.

---


## Quickstart
 
### 1. Clone the repo and install dependencies
 
```bash
git clone https://github.com/Hirunika25/Trash-Classification-System.git
cd Trash-Classification-System
pip install -r requirements.txt
```
 
### 2. Download the dataset and place it in the necessary folder
 
### 3. Run EDA 
file: notebooks/eda.ipynb

### 4. Run preprocessing 
file : notebooks/preprocessing.ipynb

### 5. Train the model and Get the checkpoint
file : models/mobilenetV3.ipynb
get the `best_model.pth` from the shared Google Drive folder and place at:
```
models/checkpoints/best_model.pth
```
### 6.Run inference on a single image

```bash
ppython predict.py --image test_images/test1.jpg
```
### 7. Run the Gradio demo

```bash
pip install gradio>=4.0.0
jupyter notebook models/demo.ipynb
```


 ---

## Configuration

All shared settings in `config.yaml`. Key values that must never change:

| Key | Value | Reason |
|-----|-------|--------|
| `data.seed` | 42 | Changing breaks split reproducibility |
| `data.num_classes` | 6 | Changing invalidates checkpoint |
| `classes` mapping | see config | Indices must match checkpoint |
| `data.image_size` | 224 | Changing invalidates checkpoint |

---

## Checkpoints

`best_model.pth` is gitignored (too large for GitHub). It contains:

```python
{
    "epoch"       : int,    # epoch with best val accuracy
    "val_acc"     : float,  # best validation accuracy
    "model_state" : dict,   # all learned weights
    "model_name"  : str,    # "mobilenetv3_large_100"
    "num_classes" : int,    # 6
}
```

Shared via Google Drive — link in group chat.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch` + `torchvision` | Model training and transforms |
| `timm` | MobileNetV3-Large pretrained weights |
| `pandas` | splits.csv handling |
| `scikit-learn` | Class weights · metrics · confusion matrix |
| `PyYAML` | config.yaml loading |
| `matplotlib` + `seaborn` | Training plots |
| `Pillow` | Image loading |
| `gradio` | Demo UI — `pip install gradio>=4.0.0` |

---

## License

Dataset: MIT License — Alistair King
([kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification))
