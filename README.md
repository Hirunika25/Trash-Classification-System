# Trash Classification — Deep Learning Project


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

**6 output classes:**

| Index | Class | Sub-categories included |
|-------|-------|------------------------|
| 0 | Plastic | water bottles, soda bottles, detergent bottles, bags, containers, cutlery, straws, cup lids |
| 1 | Paper | newspaper, office paper, magazines, cardboard boxes, cardboard packaging |
| 2 | Glass | beverage bottles, food jars, cosmetic containers |
| 3 | Metal | aluminum soda cans, food cans, steel food cans, aerosol cans |
| 4 | Organic | food waste, fruit peels, vegetable scraps, eggshells, coffee grounds, tea bags |
| 5 | Textile | clothing, shoes |

**Split (fixed seed = 42, never change):**

```
Train: 70% — 10,500 images
Val:   15% —  2,250 images
Test:  15% —  2,250 images   ← locked, only used for final ensemble evaluation
```

---

## Project Structure

