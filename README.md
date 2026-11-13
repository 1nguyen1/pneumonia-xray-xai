# Explainable Pneumonia Detection from Chest X-rays

**U-Net lung segmentation → DenseNet121 (RadImageNet) → image + age/sex fusion → Grad-CAM & SHAP → Gradio web demo**

Student research project (Ho Chi Minh City University of Technology, VNU-HCM), Apr–May 2026.

**Live demo (Hugging Face Space):** https://huggingface.co/spaces/babyG36/multimodal_pneumonia_unet_AI

---

## The idea in one paragraph

A model can score high on chest X-rays while looking at the wrong things, such as text markers, medical devices or image borders. This project builds a pneumonia classifier that **only looks at the lungs** and **shows where it looked**. A U-Net cuts out the lungs first. A DenseNet121 then classifies the lung-only image. Finally, the image features are combined with the patient's age and sex. Grad-CAM shows which lung areas drove the decision, and SHAP shows how much age and sex mattered.

## How it was built (same order as `notebooks/final_pipeline.ipynb`)

| Step | What | Script | Result |
|---|---|---|---|
| 1 | Train an image-only DenseNet121 (RadImageNet weights) and inspect it with Grad-CAM | `classifier/`, `explain/gradcam_image_only.py` | 97.2% accuracy, but some heat-maps fell outside the lungs |
| 2 | Train a U-Net to segment the lungs | `segmentation/train_unet.py` | validation Dice **0.956** |
| 3 | Cut out the lungs in the whole dataset, then retrain DenseNet121 on lung-only images | `preprocessing/mask_dataset.py`, `classifier/` | 5,856 images masked; **95.3%** test accuracy |
| 4 | Combine lung-only image + age + sex (late fusion) with class weights | `fusion/train_fusion.py` | **AUC 0.715**; Normal recall 29% → 70% |
| 5 | Explain and deploy | `explain/gradcam_shap.py`, `app/app.py` | Lung-masked Grad-CAM, SHAP, Gradio app |

## Pipeline

```
Chest X-ray ──► U-Net (256×256) ──► lung mask ──► clean-up ──► lung-only image
                                     (keep 2 largest areas,          │
                                      fill holes, smooth edges)      ▼
                                          DenseNet121 (RadImageNet, 224×224) → 1024 features
                                                                     │
Age, Sex ──► small MLP (64 → 32) ──────────────────────────────► Concatenate (1056)
                                                                     │
                                          Dense 128 → Dropout 0.3 → Dense 64 → Sigmoid
                                                                     │
                                     Grad-CAM (only inside the lungs) + SHAP (age/sex)
```

## Results

| Model | Data | Accuracy | Recall Pneumonia | Recall Normal |
|---|---|---|---|---|
| DenseNet121, full image | Kaggle *chest-xray-covid19-pneumonia* test set (1,172 images) | 97.2% | 97.1% | 97.5% |
| DenseNet121, lung-only image | same test set | 95.3% | 96.1% | 93.1% |
| Fusion, no class weights | CheXpert subset (10,000 sampled), validation 1,768 | 71% (AUC 0.707) | 90% | 29% |
| **Fusion, class-weighted (final model)** | same validation set | 65.3% (**AUC 0.715**) | 63.2% | **70.0%** |

- **U-Net:** validation Dice 0.956 (704 image–mask pairs, 85/15 split).
- Masking the lungs costs about 2 points of accuracy, but the model's evidence now comes from lung tissue instead of the image borders.
- Without class weights the fusion model labelled almost everyone as Pneumonia. Class weights (Normal 1.58, Pneumonia 0.73) made it balanced.

Plots are in [`results/`](results/). For the final model, see `fusion_final_confusion_roc.png` and `fusion_final_training_curves.png`.

### Training details
- **Class imbalance** (≈1 : 2.6 Normal : Pneumonia): train and val were merged and re-split 80/20 with stratification. The original test set was kept untouched. Classes were weighted with `compute_class_weight('balanced')`.
- **Safe augmentation for X-rays**: rotation ±10°, zoom 10%, brightness 0.9–1.1. **No flips**, because a flip would put the heart on the wrong side.
- **Two-stage transfer learning**: warm-up, then fine-tuning of the last 20 layers at LR 1e-5. Uses EarlyStopping, ReduceLROnPlateau and keeps the best checkpoint.
- **U-Net**: encoder–decoder with BatchNorm, Dice + BCE loss, trained on radiologist-drawn lung masks.
- **Fusion**: the DenseNet branch is frozen. Age is standardised and sex encoded as 0/1, with missing sex set to −1. EarlyStopping is on validation AUC.

## Repository structure

```
notebooks/      final_pipeline.ipynb — the final Colab notebook (text logs kept, images removed)
classifier/     image-only DenseNet121 pipeline: models · data · training · evaluation · run_pipeline.py
segmentation/   train_unet.py — U-Net lung segmentation
preprocessing/  mask_dataset.py — cut out the lungs for the whole dataset (resumable)
fusion/         train_fusion.py — image + age/sex fusion model with class weights (final model)
explain/        gradcam_image_only.py — Grad-CAM for the image-only model
                gradcam_shap.py — lung-masked Grad-CAM + SHAP for the fusion model
app/            app.py — Gradio web app (X-ray + age + sex → diagnosis + heat-map)
results/        metrics, confusion matrices, ROC curves, training logs
weights/        (not tracked) put the .h5 model files here, see weights/README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1) Image-only classifier (expects DATA_PATH/{train,val,test}/{NORMAL,PNEUMONIA})
DATA_PATH=path/to/data RADIMAGENET_WEIGHTS=weights/RadImageNet-DenseNet121_notop.h5 python classifier/run_pipeline.py

# 2) U-Net lung segmentation (downloads the dataset with kagglehub)
python segmentation/train_unet.py

# 3) Cut out the lungs for the whole dataset, then retrain step 1 on the masked folder
DATA_PATH=path/to/data MASKED_DATA_PATH=path/to/masked_data python preprocessing/mask_dataset.py
DATA_PATH=path/to/masked_data python classifier/run_pipeline.py

# 4) Final fusion model on CheXpert (downloads with kagglehub, masks lungs with the U-Net)
python fusion/train_fusion.py

# 5) Web demo
cd app && UNET_PATH=../weights/unet_lung_extractor.h5 FUSION_PATH=../weights/multimodal_chexpert_best.h5 python app.py
```

All paths can be set with environment variables (`DATA_PATH`, `MASKED_DATA_PATH`, `SAVE_DIR`, `UNET_PATH`, `DENSENET_PATH`, `FUSION_PATH`, `TEST_IMAGE`).

## Datasets
- Image classifier: [Chest X-ray (COVID-19 & Pneumonia)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia). Only NORMAL and PNEUMONIA are used; COVID19 is skipped.
- Lung masks: [Chest X-ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
- Image + demographics: [CheXpert (Kaggle mirror)](https://www.kaggle.com/datasets/mimsadiislam/chexpert)
- Pre-trained weights: [RadImageNet](https://github.com/BMEII-AI/RadImageNet)

The datasets are not redistributed in this repository.

## Limitations & next steps
- **Fusion model**: class weights balanced the two classes, but overall discrimination is still modest (AUC ≈ 0.715). Likely causes are a random 10k CheXpert sample with a skewed age distribution, a frozen image branch and noisy CheXpert labels. Next: sample by age group, unfreeze the last DenseNet blocks, tune the decision threshold and add a separate test set.
- **U-Net** struggles with large foreign objects and with lungs that blend into the diaphragm. Next: more varied masks and more augmentation.
- Grad-CAM sometimes focuses on the lung bases even when the prediction is correct. Next: compare with LRP and other explanation methods.

## Author
**Nguyễn Minh Nguyên** — B.Sc. Data Science, HCMUT (VNU-HCM)
