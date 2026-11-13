# Explainable Pneumonia Detection from Chest X-rays

**U-Net lung segmentation → DenseNet121 (RadImageNet) classifier → image + demographics fusion → Grad-CAM & SHAP explanations → Gradio web demo**

Student research project (Ho Chi Minh City University of Technology, VNU-HCM), Apr–May 2026.

> ⚠️ Research and education only. This is **not** a medical device and must not be used for clinical decisions.

**Live demo (Hugging Face Space):** https://huggingface.co/spaces/babyG36/multimodal_pneumonia_unet_AI

---

## Why this project

A CNN can score very high accuracy on chest X-rays while looking at the wrong things: text markers, medical devices, image borders (the *"Clever Hans"* effect). The aim was a classifier that is accurate **and** whose evidence lies inside the lungs:

1. Train a strong image-only baseline.
2. Use Grad-CAM to inspect *where* it looks. Some heat-maps fell on the image edges, on foreign objects and below the diaphragm.
3. Add a **U-Net lung segmentation** step so that the classifier only sees lung tissue, then retrain.
4. Add **patient demographics (age, sex)** through a late-fusion network and explain their effect with **SHAP**.

## Pipeline

```
Chest X-ray ──► U-Net (256×256) ──► lung mask ──► morphology clean-up ──► lung-only image
                                     (keep 2 largest components,              │
                                      fill holes, smooth edges)               ▼
                                                    DenseNet121 (RadImageNet weights, 224×224)
                                                    conv5_block16_concat → GAP → 1024-d
                                                                              │
Age, Sex ──► MLP (Dense 64 → BN → LeakyReLU → Dense 32) ──► 32-d ─────► Concatenate (1056-d)
                                                                              │
                                                    Dense 128 → Dropout 0.3 → Dense 64 → Sigmoid
                                                                              │
                                     Grad-CAM (masked to the lungs, smoothed) + SHAP (age/sex)
```

## Results

| Model | Data / split | Accuracy | Recall (Pneumonia) | Recall (Normal) | Notes |
|---|---|---|---|---|---|
| DenseNet121 (RadImageNet), full image | Kaggle *chest-xray-covid19-pneumonia*, held-out test (1,172 images) | **97.2%** | 97.1% | 97.5% | High score, but Grad-CAM showed attention outside the lungs |
| DenseNet121 (RadImageNet), **U-Net lung-masked** | same test set | **95.3%** | 96.1% | 93.1% | ~2-point accuracy cost; evidence now restricted to lung tissue |
| Image + demographics fusion | CheXpert subset (10,000 sampled), 20% validation (1,768) | 71% (AUC 0.707) | 90% | 29% | Work in progress — see *Limitations* |

Confusion matrices, classification reports and training curves are in [`results/`](results/).

### Training details
- **Class imbalance** (≈1 : 2.6 Normal : Pneumonia): train + val merged and re-split 80/20 with stratification; original test set kept untouched; `compute_class_weight('balanced')`.
- **Medically safe augmentation**: rotation ±10°, zoom 10%, brightness 0.9–1.1. **No flips**, because they would move the heart to the wrong side.
- **Two-stage transfer learning**: warm-up, then fine-tuning of the last 20 layers at LR 1e-5 with BatchNorm layers frozen. EarlyStopping (val_loss, patience 7), ReduceLROnPlateau, checkpoint on best val_loss.
- **U-Net**: encoder–decoder with BatchNorm, Dice + BCE loss, 85/15 split, trained on *chest-xray-masks-and-labels* (radiologist masks).

## Repository structure

```
classifier/     4-stage image pipeline: models · data utils · training · evaluation · run_pipeline.py
segmentation/   train_unet.py — U-Net lung segmentation (Dice + BCE)
fusion/         train_fusion.py — U-Net masking of CheXpert + image/demographics late-fusion model
explain/        gradcam_shap.py — lung-masked Grad-CAM + SHAP waterfall for age/sex
app/            app.py — Gradio web app (upload X-ray + age + sex → diagnosis + heat-map)
results/        metrics, confusion matrices, training logs
weights/        (not tracked) put the .h5 model files here, see weights/README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1) Image-only classifier (expects DATA_PATH/{train,val,test}/{NORMAL,PNEUMONIA})
DATA_PATH=path/to/data RADIMAGENET_WEIGHTS=weights/RadImageNet-DenseNet121_notop.h5 \
  python classifier/run_pipeline.py

# 2) U-Net lung segmentation (downloads the dataset with kagglehub)
python segmentation/train_unet.py

# 3) Fusion model on CheXpert (downloads with kagglehub, masks lungs with the U-Net)
python fusion/train_fusion.py

# 4) Web demo
cd app && UNET_PATH=../weights/unet_lung_extractor.h5 FUSION_PATH=../weights/multimodal_chexpert_best.h5 python app.py
```

All paths can be set with environment variables (`DATA_PATH`, `SAVE_DIR`, `UNET_PATH`, `DENSENET_PATH`, `FUSION_PATH`, `TEST_IMAGE`).

## Datasets
- Image classifier: [Chest X-ray (COVID-19 & Pneumonia)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia), using the NORMAL and PNEUMONIA classes
- Lung masks: [Chest X-ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
- Image + demographics: [CheXpert (Kaggle mirror)](https://www.kaggle.com/datasets/mimsadiislam/chexpert)
- Pre-trained weights: [RadImageNet](https://github.com/BMEII-AI/RadImageNet)

The datasets are not redistributed in this repository.

## Limitations & next steps
- **Fusion model**: adding age and sex lowered Normal recall a lot. The likely causes are a random 10k CheXpert sample with a skewed age distribution, a frozen image backbone, and the noisy CheXpert pneumonia labels. Next: stratify the sample by age, re-weight the classes, try a smaller weight for the clinical branch, and add a separate held-out test set.
- **U-Net** fails on large foreign objects and on lungs that merge with the diaphragm. Next: more diverse masks and more augmentation.
- Grad-CAM sometimes concentrates at the lung bases even when the prediction is correct. Next: compare with LRP and other attribution methods.

## Author
**Nguyễn Minh Nguyên** — B.Sc. Data Science & AI, HCMUT (VNU-HCM)
