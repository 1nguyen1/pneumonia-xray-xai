import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. KẾT NỐI VÀ TẢI DỮ LIỆU TỰ ĐỘNG BẰNG KAGGLEHUB
# ==========================================
print("-> Đang cài đặt thư viện kagglehub (nếu chưa có)...")
# pip install kagglehub   (run once before this script)

import kagglehub

print("\n-> Đang tải bộ dữ liệu từ Kaggle thông qua kagglehub...")
# Lệnh này sẽ tự động tải file zip, giải nén và lưu vào thư mục cache ẩn của Colab
DATASET_PATH = kagglehub.dataset_download("nikhilpandey360/chest-xray-masks-and-labels")

print(f"-> Đã tải dữ liệu thành công! Đường dẫn gốc: {DATASET_PATH}\n")

# ==========================================
# 2. XỬ LÝ ĐƯỜNG DẪN ẢNH VÀ MẶT NẠ (MASKS)
# ==========================================
# Đường dẫn mới sẽ dựa trên DATASET_PATH
IMG_DIR = os.path.join(DATASET_PATH, 'Lung Segmentation', 'CXR_png')
MASK_DIR = os.path.join(DATASET_PATH, 'Lung Segmentation', 'masks')

image_paths = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))
mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, '*.png')))

print(f"Tổng số ảnh gốc tìm thấy: {len(image_paths)}")
print(f"Tổng số ảnh mặt nạ tìm thấy: {len(mask_paths)}")

valid_images = []
valid_masks = []

for img_path in image_paths:
    base_name = os.path.basename(img_path).replace('.png', '')
    mask_match = [m for m in mask_paths if base_name in os.path.basename(m)]
    if mask_match:
        valid_images.append(img_path)
        valid_masks.append(mask_match[0])

print(f"-> Đã ghép nối thành công {len(valid_images)} cặp ảnh/mặt nạ.")

# ==========================================
# 3. HÀM DATA GENERATOR CHUẨN U-NET (TIẾT KIỆM RAM)
# ==========================================
IMG_SIZE = 256

def read_image_and_mask(img_path, mask_path):
    # Đọc và chuẩn hóa ảnh gốc (Đưa về thang xám 1 kênh)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)

    # Đọc và chuẩn hóa mặt nạ (0 là nền, 1 là phổi)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return img, mask

# Chia tập Train/Val
X_train_paths, X_val_paths, y_train_paths, y_val_paths = train_test_split(
    valid_images, valid_masks, test_size=0.15, random_state=42
)

def data_generator(img_paths, mask_paths, batch_size=8):
    while True:
        for i in range(0, len(img_paths), batch_size):
            batch_imgs = img_paths[i:i+batch_size]
            batch_masks = mask_paths[i:i+batch_size]

            X, Y = [], []
            for img_p, mask_p in zip(batch_imgs, batch_masks):
                img, mask = read_image_and_mask(img_p, mask_p)
                X.append(img)
                Y.append(mask)

            yield np.array(X), np.array(Y)

# ==========================================
# 4. KIẾN TRÚC MẠNG U-NET & DICE LOSS
# ==========================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_bce_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    return bce + dice_loss

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = Input(input_shape)

    # ENCODER
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # BOTTLENECK
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # DECODER
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=dice_bce_loss, metrics=[dice_coef, 'accuracy'])
    return model

print("-> Đang khởi tạo U-Net...")
model = build_unet()

# ==========================================
# 5. HUẤN LUYỆN MÔ HÌNH VÀ LƯU VÀO GOOGLE DRIVE
# ==========================================
# Mount Google Drive
try:  # Google Colab: save to Drive; elsewhere: save locally
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except ImportError:
    pass

# THIẾT LẬP NƠI LƯU (ĐẢM BẢO THƯ MỤC NÀY TỒN TẠI TRÊN DRIVE CỦA BẠN)
SAVE_DIR = os.environ.get('SAVE_DIR', 'weights')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
SAVE_PATH = os.path.join(SAVE_DIR, 'unet_lung_extractor.h5')

callbacks = [
    ModelCheckpoint(SAVE_PATH, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_dice_coef', mode='max', patience=3, factor=0.5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_dice_coef', mode='max', patience=8, restore_best_weights=True)
]

BATCH_SIZE = 8
train_steps = max(1, len(X_train_paths) // BATCH_SIZE)
val_steps = max(1, len(X_val_paths) // BATCH_SIZE)

print(f"\n{'='*50}")
print(" BẮT ĐẦU HUẤN LUYỆN U-NET")
print(f"{'='*50}\n")

history = model.fit(
    data_generator(X_train_paths, y_train_paths, BATCH_SIZE),
    steps_per_epoch=train_steps,
    validation_data=data_generator(X_val_paths, y_val_paths, BATCH_SIZE),
    validation_steps=val_steps,
    epochs=30,
    callbacks=callbacks
)

print("\n-> CHÚC MỪNG! ĐÃ TRAIN XONG U-NET VÀ LƯU TẠI GOOGLE DRIVE!")