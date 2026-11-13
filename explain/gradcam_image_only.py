"""Grad-CAM cho mô hình chỉ dùng ảnh (DenseNet121 RadImageNet): bốc ngẫu nhiên ảnh test,
vẽ heat-map, tô xanh khi đoán đúng / đỏ khi đoán sai. Dùng để phát hiện mô hình nhìn sai chỗ.
"""
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TÙY CHỈNH
# ==========================================
# 1. Thư mục dữ liệu gốc (train/val/test/{NORMAL,PNEUMONIA})
DATA_EXTRACT_DIR = os.environ.get("DATA_PATH", "data")

# 2. Mô hình chỉ dùng ảnh đã train xong
MODEL_PATH = os.environ.get("DENSENET_PATH", "weights/DenseNet121_RadImageNet_Finetuned_best.h5")

# 3. Số lượng ảnh muốn bốc ngẫu nhiên để vẽ
NUM_IMAGES = 20

# Tìm đường dẫn chính xác của thư mục test
TEST_DIR = os.path.join(DATA_EXTRACT_DIR, "test") 

# ==========================================
# BƯỚC 2: TẢI MÔ HÌNH VÀ TÌM LỚP CONV CUỐI CÙNG
# ==========================================
print(f"-> Đang nạp mô hình từ: {MODEL_PATH}")
model = load_model(MODEL_PATH)

last_conv_layer_name = None
for layer in reversed(model.layers):
    try:
        # Sử dụng layer.output.shape (chuẩn Keras 3) thay vì layer.output_shape
        if len(layer.output.shape) == 4 and layer.name != 'input_1':
            last_conv_layer_name = layer.name
            break
    except Exception:
        # Bỏ qua nếu lớp đó không có thuộc tính shape (như Dropout, v.v.)
        continue

print(f"-> Lớp Convolutional cuối cùng được dùng cho Grad-CAM: {last_conv_layer_name}")
# ==========================================
# BƯỚC 3: HÀM TÍNH TOÁN GRAD-CAM (CHO MẠNG NHỊ PHÂN)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Tạo mô hình trung gian trả về [Đầu ra lớp Conv cuối, Đầu ra phân loại]
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # Vì là binary classification (1 nơ ron sigmoid), ta lấy trực tiếp giá trị này
        class_channel = preds[0, 0]

    # Đạo hàm của output dự đoán theo feature map của lớp conv cuối
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Tính trung bình gradient (Global Average Pooling cho Gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Nhân từng kênh của feature map với trọng số gradient tương ứng
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Chỉ giữ lại các giá trị có ảnh hưởng tích cực (ReLU)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ==========================================
# BƯỚC 4: HÀM CHỒNG HEATMAP LÊN ẢNH GỐC
# ==========================================
def overlay_gradcam(img_path, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Đọc ảnh gốc bằng OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Chuyển heatmap về [0, 255] và đổi kích thước bằng ảnh gốc
    heatmap = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Phủ màu cho heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)

    # Chồng ảnh
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)

    # Đổi BGR (OpenCV) sang RGB (Matplotlib) để in ra không bị sai màu
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# ==========================================
# BƯỚC 5: CHỌN ẢNH NGẪU NHIÊN VÀ IN HÀNG LOẠT
# ==========================================
# Lấy toàn bộ đường dẫn ảnh từ thư mục test (Bỏ qua COVID19)
image_paths = []
for root, dirs, files in os.walk(TEST_DIR):
    # Lấy tên thư mục hiện tại đang quét
    current_folder = os.path.basename(root)

    # BỘ LỌC: CHỈ CHO PHÉP 2 THƯ MỤC NÀY ĐI QUA
    if current_folder not in ['NORMAL', 'PNEUMONIA']:
        continue

    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))

# Kiểm tra xem có đủ ảnh để bốc không
if len(image_paths) < NUM_IMAGES:
    print(f"Cảnh báo: Chỉ tìm thấy {len(image_paths)} ảnh, ít hơn {NUM_IMAGES} ảnh yêu cầu.")
    NUM_IMAGES = len(image_paths)

# Bốc ngẫu nhiên X ảnh
selected_paths = random.sample(image_paths, NUM_IMAGES)

# Setup lưới hiển thị Matplotlib
cols = 4
rows = (NUM_IMAGES + cols - 1) // cols
plt.figure(figsize=(15, 4 * rows))

for i, img_path in enumerate(selected_paths):
    # Lấy nhãn thực tế
    true_label = os.path.basename(os.path.dirname(img_path))

    # Chuẩn bị ảnh để đưa vào model
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = preprocess_input(img_array)

    # 1. AI Dự đoán
    pred_prob = model.predict(img_array_preprocessed, verbose=0)[0][0]
    pred_label = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"

    # Đổi màu tiêu đề: Đúng -> Xanh lá, Sai -> Đỏ
    title_color = 'green' if pred_label == true_label else 'red'

    # 2. Tạo Heatmap
    heatmap = make_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name)

    # 3. Chồng Heatmap lên ảnh
    cam_img = overlay_gradcam(img_path, heatmap)

    # 4. Đưa lên lưới (Grid)
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(cam_img)
    ax.set_title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})", color=title_color, fontsize=11, fontweight='bold')
    ax.axis("off")

plt.tight_layout()
plt.show()
print("-> HOÀN TẤT VẼ GRAD-CAM!")
