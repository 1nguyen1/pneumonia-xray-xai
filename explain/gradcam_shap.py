import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap # BẮT BUỘC PHẢI CÀI ĐẶT: !pip install shap

# ==========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN MÔ HÌNH
# ==========================================================
UNET_PATH = os.environ.get('UNET_PATH', 'weights/unet_lung_extractor.h5')
FUSION_MODEL_PATH = os.environ.get('FUSION_PATH', 'weights/multimodal_chexpert_best.h5')

print("⏳ Đang khởi động Bác sĩ AI Đa phương thức...")
unet_model = load_model(UNET_PATH, compile=False)
fusion_model = load_model(FUSION_MODEL_PATH)
last_conv_layer_name = 'conv5_block16_concat'
print("✅ Hệ thống chẩn đoán đã sẵn sàng!")

# ==========================================================
# 2. HÀM CẮT PHỔI (U-NET)
# ==========================================================
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

    masked_img_bgr = (img_bgr * mask_3ch).astype(np.uint8)
    return masked_img_bgr, mask_resized

# ==========================================================
# 3. HÀM GRAD-CAM (MÔ HÌNH FUSION TOÀN PHẦN)
# ==========================================================
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

# ==========================================================
# 4. GIAO DIỆN CHẨN ĐOÁN (KẾT HỢP X-QUANG & LÂM SÀNG)
# ==========================================================
def diagnose_patient(img_path, age, sex_str, true_label="Không rõ"):
    if not os.path.exists(img_path):
        print("❌ Không tìm thấy ảnh!")
        return

    # Tiền xử lý
    age_norm = (age - 60.0) / 15.0
    sex_encoded = 0.0 if sex_str.lower() == 'male' else 1.0
    clinical_input = np.array([[age_norm, sex_encoded]], dtype=np.float32)

    orig_bgr = cv2.imread(img_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    masked_bgr, binary_mask = extract_lungs_robust(orig_bgr, unet_model)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(masked_rgb, (224, 224))
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(np.expand_dims(img_resized, axis=0))

    # Chạy Grad-CAM
    heatmap, prob = get_multimodal_gradcam(
        img_preprocessed, clinical_input, fusion_model,
        last_conv_layer_name, binary_mask, orig_rgb.shape
    )

    pred_name = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    # Vẽ Heatmap
    heatmap_color = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Hiển thị
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_rgb)
    plt.title(f"1. X-quang Gốc\nBệnh nhân: {sex_str.capitalize()}, {age} tuổi")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(masked_rgb)
    plt.title("2. Ảnh Đã Cắt Phổi\n(Đầu vào chẩn đoán)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    color = 'green' if (true_label.upper() == pred_name) else 'red'
    plt.title(f"3. Báo cáo AI: {pred_name}\nXác suất Viêm Phổi: {prob:.2%}", color=color, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================================
# 5. TÍCH HỢP SHAP ĐỂ PHÂN TÍCH NHÁNH LÂM SÀNG
# ==========================================================
# ==========================================================
# 5. TÍCH HỢP SHAP ĐỂ PHÂN TÍCH NHÁNH LÂM SÀNG (ĐÃ FIX LỖI)
# ==========================================================
def analyze_clinical_impact_with_shap(img_path, target_age, target_sex_str):
    print("\n🔍 Đang phân tích sức ảnh hưởng của Tuổi & Giới tính bằng SHAP...")

    orig_bgr = cv2.imread(img_path)
    masked_bgr, _ = extract_lungs_robust(orig_bgr, unet_model)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(masked_rgb, (224, 224))
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(np.expand_dims(img_resized, axis=0))

    def predict_with_fixed_image(clinical_array):
        num_samples = clinical_array.shape[0]
        batch_images = np.repeat(img_preprocessed, num_samples, axis=0)
        preds = fusion_model.predict([batch_images, clinical_array], verbose=0)
        return preds

    background_ages = np.linspace(20, 80, num=10)
    background_sexes = [0.0, 1.0]

    bg_data = []
    for age in background_ages:
        for sex in background_sexes:
            age_norm = (age - 62.0) / 15.0
            bg_data.append([age_norm, sex])
    background_clinicals = np.array(bg_data, dtype=np.float32)

    explainer = shap.KernelExplainer(predict_with_fixed_image, background_clinicals)

    target_age_norm = (target_age - 62.0) / 15.0
    target_sex_encoded = 0.0 if target_sex_str.lower() == 'male' else 1.0
    patient_clinical = np.array([[target_age_norm, target_sex_encoded]], dtype=np.float32)

    shap_values = explainer.shap_values(patient_clinical)

    # -- SỬA LỖI TẠI ĐÂY: XỬ LÝ ĐA CHIỀU CỦA SHAP VALUES --
    # Đối với bài toán phân loại nhị phân bằng Keras/TensorFlow,
    # KernelExplainer thường trả về một mảng hoặc danh sách chứa mảng đa chiều.
    # Ta cần truy cập chính xác vào phần tử đầu tiên (đại diện cho bệnh nhân hiện tại).

    if isinstance(shap_values, list):
        # Nếu model có 2 class output (ví dụ softmax), lấy class 1
        s_vals_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        # Nếu model trả ra 1 số (sigmoid)
        s_vals_array = shap_values

    # Đảm bảo s_vals_array chỉ là một mảng 1 chiều chứa 2 giá trị (cho Tuổi và Giới tính)
    # Dùng .flatten() hoặc np.squeeze() để phá vỡ cấu trúc ma trận (2, 1) nếu có
    final_s_vals = np.squeeze(s_vals_array)

    # Lấy base_value (giá trị cơ sở)
    expected_val = explainer.expected_value
    if isinstance(expected_val, list) or isinstance(expected_val, np.ndarray):
         expected_val = float(expected_val[0]) # Ép kiểu về số thực (float)

    feature_names = ['Tuổi (Chuẩn hóa)', 'Giới tính (0:Nam, 1:Nữ)']

    print("\n✅ HOÀN TẤT! Vẽ biểu đồ...")
    plt.figure(figsize=(10, 4))

    # -- SỬA LỖI CÁCH GỌI HÀM VẼ --
    # Khởi tạo đối tượng Explanation chuẩn của SHAP
    shap_exp = shap.Explanation(
        values=final_s_vals,
        base_values=expected_val,
        data=patient_clinical[0],
        feature_names=feature_names
    )

    # Vẽ biểu đồ thác nước
    shap.waterfall_plot(shap_exp)

# ==========================================================
# THỰC THI KIỂM THỬ TỔNG HỢP (GRAD-CAM + SHAP)
# ==========================================================
TEST_IMAGE_PATH = os.environ.get('TEST_IMAGE', 'sample_xray.jpg')  # path to any chest X-ray image
PATIENT_AGE = 46
PATIENT_SEX = 'Male'
TRUE_LABEL = 'NORMAL'

# 1. Chạy chẩn đoán hình ảnh (Grad-CAM)
diagnose_patient(TEST_IMAGE_PATH, PATIENT_AGE, PATIENT_SEX, TRUE_LABEL)

# 2. Chạy phân tích Metadata (SHAP)
analyze_clinical_impact_with_shap(TEST_IMAGE_PATH, PATIENT_AGE, PATIENT_SEX)