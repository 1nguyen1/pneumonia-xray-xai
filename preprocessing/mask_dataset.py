"""Cắt phổi cho toàn bộ dataset bằng U-Net (chỉ lấy NORMAL/PNEUMONIA, bỏ COVID19).
Có thể chạy lại giữa chừng: ảnh đã cắt sẽ được bỏ qua.
Lần chạy cuối: 5,856 ảnh NORMAL/PNEUMONIA được cắt, 576 ảnh COVID19 bị bỏ qua.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

RAW_DATA_DIR = os.environ.get("DATA_PATH", "data")            # thư mục gốc: train/val/test/{NORMAL,PNEUMONIA}
MASKED_DATA_DIR = os.environ.get("MASKED_DATA_PATH", "masked_data")  # nơi lưu ảnh đã cắt phổi
UNET_MODEL_PATH = os.environ.get("UNET_PATH", "weights/unet_lung_extractor.h5")

print("-> Đang tải mô hình U-Net...")
unet_model = load_model(UNET_MODEL_PATH, compile=False)

print(f"-> Bắt đầu phân tích dữ liệu từ {RAW_DATA_DIR} sang {MASKED_DATA_DIR}...\n")

images_to_process = []
skipped_cut_count = 0   # Đếm số ảnh ĐÃ ĐƯỢC CẮT (Resume)
ignored_class_count = 0 # Đếm số ảnh rác KHÔNG PHẢI Normal/Pneumonia

for root, _, files in os.walk(RAW_DATA_DIR):
    current_folder = os.path.basename(root)

    # ==========================================
    # BỘ LỌC 1: CHỈ LẤY THƯ MỤC NORMAL & PNEUMONIA
    # ==========================================
    if current_folder not in ['NORMAL', 'PNEUMONIA']:
        ignored_class_count += sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        continue

    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            relative_path = os.path.relpath(img_path, RAW_DATA_DIR)
            save_path = os.path.join(MASKED_DATA_DIR, relative_path)

            # ==========================================
            # BỘ LỌC 2: BỎ QUA NHỮNG FILE ĐÃ ĐƯỢC CẮT
            # ==========================================
            if os.path.exists(save_path):
                skipped_cut_count += 1
            else:
                images_to_process.append((img_path, save_path))

# In báo cáo rà soát trước khi chạy AI
print(f"-> Đã chủ động bỏ qua {ignored_class_count} ảnh rác (VD: COVID19).")

if skipped_cut_count > 0:
    print(f"-> Đã tìm thấy {skipped_cut_count} ảnh ĐÃ CẮT TỪ TRƯỚC. Sẽ tự động bỏ qua để tiết kiệm thời gian!")

# Quyết định chạy AI
if len(images_to_process) == 0:
    print("-> Tất cả dữ liệu cần thiết đều đã được cắt xong! Không cần chạy lại AI.")
else:
    print(f"-> Bắt đầu cắt {len(images_to_process)} ảnh mới...\n")

    for img_path, save_path in tqdm(images_to_process, desc="Đang cắt nền phổi"):
        img = cv2.imread(img_path)
        if img is None: continue

        original_shape = img.shape[:2]

        # Tiền xử lý cho U-Net
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (256, 256))
        gray_normalized = gray_resized / 255.0
        unet_input = np.expand_dims(gray_normalized, axis=(0, -1))

        # U-Net Dự đoán
        mask = unet_model.predict(unet_input, verbose=0)[0]

        # Áp mặt nạ
        mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.float32)
        binary_mask_3c = np.stack((binary_mask,)*3, axis=-1)

        masked_img = (img * binary_mask_3c).astype(np.uint8)

        # Tạo thư mục và Lưu file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, masked_img)

    print("\n-> HOÀN TẤT! Đã cập nhật đầy đủ dataset chuẩn.")
