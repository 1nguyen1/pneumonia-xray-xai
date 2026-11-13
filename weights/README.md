# Model weights

The weight files are not stored in git because of their size. Place them in this folder:

| File | Produced by | Size |
|---|---|---|
| `RadImageNet-DenseNet121_notop.h5` | [RadImageNet](https://github.com/BMEII-AI/RadImageNet) pre-trained backbone | ~28 MB |
| `unet_lung_extractor.h5` | `segmentation/train_unet.py` | ~23 MB |
| `DenseNet121_RadImageNet_MaskedData_Finetuned_best.h5` | `classifier/run_pipeline.py` on lung-masked images | ~32 MB |
| `multimodal_chexpert_best.h5` | `fusion/train_fusion.py` | ~31 MB |

The U-Net and fusion weights used by the live demo can also be downloaded from the Files tab of the
[Hugging Face Space](https://huggingface.co/spaces/babyG36/multimodal_pneumonia_unet_AI).
