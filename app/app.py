import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# 1. KHỞI TẠO VÀ TẢI MÔ HÌNH (Chạy 1 lần khi server bật)
# ==========================================
# Cấu hình đường dẫn trên máy chủ Hugging Face (File để chung thư mục với app.py)
UNET_PATH = 'unet_lung_extractor.h5'
FUSION_MODEL_PATH = 'multimodal_chexpert_best.h5'
LAST_CONV_LAYER = 'conv5_block16_concat'

print("⏳ Đang tải mô hình AI...")
unet_model = load_model(UNET_PATH, compile=False)
fusion_model = load_model(FUSION_MODEL_PATH)
print("✅ Hệ thống sẵn sàng!")

# ==========================================
# 2. CÁC HÀM XỬ LÝ LÕI (Từ Colab mang qua)
# ==========================================
def extract_lungs_robust(img_bgr, unet):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (256, 256))
    img_input = np.expand_dims(img_resized / 255.0, axis=(0, -1))

    mask = unet.predict(img_input, verbose=0)[0]
    mask = (mask > 0.5).astype(np.uint8) * 255 

    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened, connectivity=8)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_labels = areas.argsort()[-2:][::-1] + 1 
        clean_mask = np.zeros_like(mask_opened)
        for label in largest_labels:
            clean_mask[labels == label] = 255
    else:
        clean_mask = mask_opened

    clean_mask = cv2.dilate(clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(clean_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    final_mask = (filled_mask / 255.0).astype(np.float32)
    mask_resized = cv2.resize(final_mask, (img_bgr.shape[1], img_bgr.shape[0]))
    mask_3ch = np.stack((mask_resized,)*3, axis=-1)

    return (img_bgr * mask_3ch).astype(np.uint8), mask_resized

def get_multimodal_gradcam(img_array, clinical_array, model, last_conv_layer, binary_mask, orig_shape):
    grad_model = tf.keras.models.Model(
        inputs=[model.input[0], model.input[1]], 
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model([img_array, clinical_array])
        class_channel = preds[:, 0]
        
        epsilon = 1e-7
        safe_prob = tf.clip_by_value(class_channel, epsilon, 1.0 - epsilon)
        logit = tf.math.log(safe_prob / (1.0 - safe_prob))

    grads = tape.gradient(logit, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    
    heatmap_resized = cv2.resize(heatmap, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_masked = heatmap_resized * binary_mask 
    heatmap_smoothed = cv2.GaussianBlur(heatmap_masked, (51, 51), 0)
    heatmap_smoothed = np.maximum(heatmap_smoothed, 0)
    
    if np.max(heatmap_smoothed) > 0:
        heatmap_smoothed /= np.max(heatmap_smoothed)
        
    return heatmap_smoothed, preds[0][0]

# ==========================================
# 3. HÀM CẦU NỐI VỚI GIAO DIỆN GRADIO
# ==========================================
def predict_pneumonia(img_rgb, age, sex_str):
    if img_rgb is None:
        return None, "Vui lòng tải ảnh lên."
    
    # Gradio mặc định trả ảnh RGB, opencv cần BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Xử lý Lâm sàng
    age_norm = (age - 60.0) / 15.0 
    sex_encoded = 0.0 if sex_str == 'Nam (Male)' else 1.0
    clinical_input = np.array([[age_norm, sex_encoded]], dtype=np.float32)

    # 2. Xử lý Ảnh
    masked_bgr, binary_mask = extract_lungs_robust(img_bgr, unet_model)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(masked_rgb, (224, 224))
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(np.expand_dims(img_resized, axis=0))

    # 3. Chẩn đoán & Bản đồ nhiệt
    heatmap, prob = get_multimodal_gradcam(
        img_preprocessed, clinical_input, fusion_model, 
        LAST_CONV_LAYER, binary_mask, img_rgb.shape
    )
    
    # 4. Vẽ Heatmap
    heatmap_color = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    # 5. Định dạng kết quả
    pred_label = "🚨 VIÊM PHỔI (PNEUMONIA)" if prob > 0.5 else "✅ BÌNH THƯỜNG (NORMAL)"
    result_text = f"Kết luận: {pred_label}\nXác suất Viêm phổi: {prob:.2%}"
    
    return superimposed_rgb, result_text

# ==========================================
# 4. THIẾT KẾ GIAO DIỆN WEB BẰNG GRADIO BLOCKS
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 Trợ lý AI Chẩn Đoán Viêm Phổi (Đa Phương Thức)")
    gr.Markdown("Hệ thống kết hợp X-quang cắt lớp tự động (U-Net) và dữ liệu lâm sàng (Tuổi, Giới tính) để chẩn đoán bằng mô hình Fusion DenseNet121.")
    
    with gr.Row():
        # CỘT TRÁI: Nhập liệu
        with gr.Column(scale=1):
            gr.Markdown("### Thông tin Bệnh nhân")
            in_age = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Độ tuổi")
            in_sex = gr.Radio(choices=["Nam (Male)", "Nữ (Female)"], value="Nam (Male)", label="Giới tính")
            
            gr.Markdown("### Ảnh X-quang Ngực")
            in_image = gr.Image(type="numpy", label="Tải ảnh X-quang lên đây")
            
            submit_btn = gr.Button("Phân tích AI", variant="primary")
            
        # CỘT PHẢI: Kết quả
        with gr.Column(scale=1):
            gr.Markdown("### Báo cáo Y khoa")
            out_text = gr.Textbox(label="Kết quả Chẩn đoán", lines=2)
            out_image = gr.Image(label="Bản đồ nhiệt Grad-CAM (Vùng tổn thương)")

    # Nút bấm kích hoạt hàm
    submit_btn.click(
        fn=predict_pneumonia,
        inputs=[in_image, in_age, in_sex],
        outputs=[out_image, out_text]
    )

# Chạy server
demo.launch()