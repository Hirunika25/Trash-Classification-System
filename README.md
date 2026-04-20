# Trash Classification : Ensemble Deep Learning Project


[Recyclable and Household Waste Classification Dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
and combines them into a soft-voting ensemble for robust trash classification.

---

## Team & Model Ownership

| Member | Model | 
|--------|-------|
| Member 1 | EfficientNet-B3 |
| Member 2 | ResNet-50 | 
| Member 3 | MobileNetV3-Large | 
| Member 4 | Swin-T | 

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
trash-classifier/
│
├── data/
│   ├── raw/
│   │   └── images/                        # unzip Kaggle dataset here (gitignored)
│   │       ├── Plastic water bottles/
│   │       │   ├── default/     *.png
│   │       │   └── real_world/  *.png
│   │       └── ...  (30+ sub-category folders)
│   ├── processed/
│   └── splits.csv                         # same for everyone
│
├── notebooks/
│   ├── eda/
│   │   ├── eda.ipynb                      
│   │   ├── class_distribution.png
│   │   ├── subcategory_distribution.png
│   │   ├── default_vs_realworld.png
│   │   ├── sample_grid.png
│   │   └── pixel_stats.png
│   └── preprocessing.ipynb               # runs this once → generates splits.csv
│
├── models/
│   ├── member_1_efficientnet/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── checkpoints/                  # best_model.pth (gitignored)
│   ├── member_2_resnet/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── checkpoints/
│   ├── member_3_mobilenet/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── checkpoints/
│   └── member_4_swint/
│       ├── train.py
│       ├── model.py
│       └── checkpoints/
│
├── ensemble/
│   ├── ensemble.py                        # loads 4 checkpoints · soft voting
│   └── evaluate.py                        # metrics · confusion matrix
│
├── config.yaml                            
├── requirements.txt                       
├── .gitignore
└── README.md
```
 
---
 
## Quickstart
 
### 1. Clone the repo
 
```bash
git clone https://github.com/Hirunika25/Trash-Classification-System.git
cd Trash-Classification-System
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt

```
 
### 3. Download the dataset 

Dataset: [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

Download from Kaggle and unzip so the structure is:
```
data/raw/images/Plastic water bottles/default/image1.png
data/raw/images/Plastic water bottles/real_world/image1.png
...
```
 
### 4. Run EDA 
 
Open `notebooks/eda/eda.ipynb` and run all cells.
 
Check console output — must say **"All sub-categories mapped successfully"** with no Unknown warnings.
Share the 5 plots in `notebooks/eda/` with the team before proceeding.
 
### 5. Run preprocessing ( run once by one member)

 Get the splits by others;
```bash
git add data/splits.csv
git commit -m "add splits.csv  seed=42  70/15/15"
git push
```
 
### 6. Pull the split (Members 2, 3, 4)
 
```bash
git pull
```
 
Everyone now has the identical split. Nobody runs preprocessing again.
 
### 7. Train your model on Kaggle (each member independently)
 

 
### 8. Share your checkpoint
 
After training, download `best_model.pth`.
Upload to the team's shared Google Drive folder.
Name it clearly: `member1_efficientnet.pth`, `member2_resnet.pth` etc.
 
### 9. Run the ensemble 
 
1. Download all 4 `.pth` files from Drive
2. Place each at the path in `config.yaml` → `ensemble.checkpoint_paths`
3. Run:
 
```bash
python ensemble/ensemble.py
python ensemble/evaluate.py
```
 
---
 
## Configuration (`config.yaml`)
 
All shared settings live in one file. Load it in any script:
 
```python
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
```
 
### Settings that must never change after first run
 
| Key | Value | Reason |
|-----|-------|--------|
| `data.seed` | 42 | Changing breaks split reproducibility |
| `data.num_classes` | 6 | Changing invalidates all checkpoints |
| `classes` mapping | see config | Indices must match across all 4 models |
| `data.image_size` | 224 | Changing invalidates all checkpoints |
 
### Settings members can tune (open a PR)
 
`training.lr` · `training.batch_size` · `training.num_epochs` · `optimizer.weight_decay`
 
---
 
## Model Checkpoints
 
A checkpoint (`best_model.pth`) is a saved snapshot of a model's learned weights
at the epoch with the best validation accuracy.
 
- Checkpoints are **gitignored** — they are 50–200 MB each, too large for GitHub
- Share via the team **Google Drive folder**
- Always name it `best_model.pth` so the ensemble script finds it automatically
- Place it at the exact path listed in `config.yaml` under `ensemble.checkpoint_paths`
- If you retrain and improve, just overwrite and re-upload to Drive
 
---
 
## Ensemble Strategy
 
Each model outputs a softmax probability vector of shape `[6]`.
The ensemble averages all 4 vectors and takes the highest probability class:
 
```
Final class = argmax( (p1 + p2 + p3 + p4) / 4 )
```
 
This is **soft voting** — it considers prediction confidence, not just the winning class,
which consistently outperforms hard (majority) voting.
 
---
 
## Results
 
*Fill in after training is complete.*
 
| Model | Val Accuracy | Val F1 | Test Accuracy |
|-------|-------------|--------|---------------|
| EfficientNet-B3 (Member 1) | — | — | — |
| ResNet-50 (Member 2) | — | — | — |
| MobileNetV3-Large (Member 3) | — | — | — |
| Swin-T (Member 4) | — | — | — |
| **Ensemble (soft vote)** | **—** | **—** | **—** |
 
---
 
## Git Workflow
 

```
 
- Never commit directly to `main` — always open a pull request
- Never commit `data/raw/`, `data/processed/`, or `*.pth` files
- Always commit `data/splits.csv` after Member 1 generates it
- Commit `config.yaml` changes separately with a clear message
 
---
 
## Known Dataset Notes
 
- The raw dataset contains 30+ sub-category folders with varied naming conventions
  (some use underscores e.g. `food_waste`, `office_paper`)
- All sub-categories are mapped to 6 main classes via keyword matching in both
  `eda.ipynb` and `preprocessing.ipynb` — keep `KEYWORD_TO_CLASS` in sync across both files
- `paper_cups` is mapped to **Paper** (not Plastic)
- `styrofoam_cups` is mapped to **Plastic**
 
---
 
## Dependencies
 
See `requirements.txt`. Core libraries:
 
| Library | Purpose |
|---------|---------|
| `torch` + `torchvision` | Model training |
| `timm` | Pretrained model zoo — all 4 models in one library |
| `pandas` | splits.csv handling |
| `scikit-learn` | Stratified split · metrics · confusion matrix |
| `PyYAML` | Loading config.yaml |
| `matplotlib` + `seaborn` | EDA plots |
| `Pillow` | Image loading |
| `gradio` | Demo app — uncomment in requirements.txt when ready |
 
---
 
## License
 
Dataset: MIT License — Alistair King
([kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification))
