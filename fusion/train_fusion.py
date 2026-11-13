import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import kagglehub
import gc

# ==========================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================================
UNET_PATH = os.environ.get('UNET_PATH', 'weights/unet_lung_extractor.h5')
DENSENET_PATH = os.environ.get('DENSENET_PATH', 'weights/DenseNet121_RadImageNet_MaskedData_Finetuned_best.h5')
LOCAL_MASKED_DIR = os.environ.get('MASKED_DIR', 'chexpert_masked_images')
LOCAL_CSV_PATH = os.environ.get('MASKED_CSV', 'chexpert_masked_train.csv')
NEW_SAVE_PATH = os.environ.get('FUSION_PATH', 'weights/multimodal_chexpert_best.h5')
HISTORY_CSV_PATH = os.environ.get('HISTORY_CSV_PATH', 'results/training_history_fusion.csv') # Nơi lưu nhật ký

os.makedirs(LOCAL_MASKED_DIR, exist_ok=True)
MAX_SAMPLES = 10000 # Đổi thành None nếu muốn train toàn bộ

# ==========================================================
# PHẦN 1: HÀM CẮT PHỔI & CHUẨN BỊ DATA
# ==========================================================
def extract_lungs(img_bgr, unet_model):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (256, 256))
    img_input = np.expand_dims(img_resized / 255.0, axis=(0, -1))
    mask = unet_model.predict(img_input, verbose=0)[0]
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
    return (img_bgr * mask_3ch).astype(np.uint8)

def download_and_preprocess_data():
    print("📥 [1/3] Đang tải Dataset CheXpert từ Kaggle...")
    dataset_path = kagglehub.dataset_download("mimsadiislam/chexpert")
    csv_candidates = [os.path.join(root, name) for root, dirs, files in os.walk(dataset_path) for name in files if name == 'train.csv']
    if not csv_candidates: raise ValueError("❌ Không tìm thấy train.csv!")
    original_csv = csv_candidates[0]
    df = pd.read_csv(original_csv)
    df = df[['Path', 'Age', 'Sex', 'Pneumonia']]
    df = df[df['Pneumonia'].isin([0.0, 1.0])].copy()
    if MAX_SAMPLES:
        df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42).reset_index(drop=True)
    print(f"\n✂️ [2/3] BẮT ĐẦU CẮT {len(df)} PHỔI BẰNG U-NET...")
    unet = load_model(UNET_PATH, compile=False)
    new_data = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        orig_img_path = os.path.join(dataset_path, row['Path'])
        safe_filename = row['Path'].replace('/', '_')
        new_img_path = os.path.join(LOCAL_MASKED_DIR, safe_filename)
        img = cv2.imread(orig_img_path)
        if img is not None:
            try:
                masked_img = extract_lungs(img, unet)
                cv2.imwrite(new_img_path, masked_img)
                new_data.append({'Masked_Path': new_img_path, 'Age': row['Age'], 'Sex': row['Sex'], 'Pneumonia': row['Pneumonia']})
            except Exception as e:
                pass
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(LOCAL_CSV_PATH, index=False)
    del unet
    tf.keras.backend.clear_session()
    gc.collect()
    print("\n✅ Đã cắt xong và giải phóng bộ nhớ U-Net!")

# ==========================================================
# PHẦN 2: CỖ MÁY GENERATOR & MODEL
# ==========================================================
class FastMultimodalGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, target_size=(224, 224), shuffle=True):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle: np.random.shuffle(self.indices)

    def __len__(self): return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        X_images = np.empty((len(batch_df), *self.target_size, 3), dtype=np.float32)
        X_clinicals = np.empty((len(batch_df), 2), dtype=np.float32)
        Y_labels = np.empty((len(batch_df), 1), dtype=np.float32)
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img = cv2.imread(row['Masked_Path'])
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, self.target_size)
                img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_resized.astype(np.float32))
            else:
                img_preprocessed = np.zeros((*self.target_size, 3), dtype=np.float32)
            X_images[i] = img_preprocessed
            X_clinicals[i] = [row['Age_norm'], row['Sex']]
            Y_labels[i] = row['Pneumonia']
        return (X_images, X_clinicals), Y_labels

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

def build_fusion_model(densenet_path):
    # 1. NHÁNH LÂM SÀNG (Giữ nguyên)
    clinical_input = Input(shape=(2,), name="clinical_input")
    x_clin = Dense(64, name="clin_dense_1")(clinical_input)
    x_clin = BatchNormalization(name="clin_bn_1")(x_clin)
    x_clin = LeakyReLU(alpha=0.1, name="clin_leaky_1")(x_clin)
    clinical_features = Dense(32, activation='relu', name="clinical_features")(x_clin)

    # 2. NHÁNH HÌNH ẢNH (DENSENET)
    base_cnn = load_model(densenet_path)
    base_cnn.trainable = False
    
    # ---------------------------------------------------
    # BƯỚC PHẪU THUẬT: SỬA LỖI CỔ CHAI SIGMOID
    # ---------------------------------------------------
    image_input = base_cnn.input
    
    # Lấy output từ lớp Convolution cuối cùng thay vì lớp Sigmoid cuối cùng
    # Lớp 'conv5_block16_concat' của DenseNet121 chứa chính xác 1024 channels không gian (spatial features)
    x_img_features = base_cnn.get_layer('conv5_block16_concat').output
    
    # Ép 1024 channels (có dạng ma trận không gian 7x7) thành véc-tơ 1024 chiều bằng GAP
    image_features = GlobalAveragePooling2D(name="fusion_gap")(x_img_features)
    # ---------------------------------------------------

    # 3. HỢP NHẤT VÀ PHÂN LOẠI
    # Trộn 1024 đặc trưng (ảnh) + 32 đặc trưng (lâm sàng) = Véc-tơ 1056 chiều
    fused = Concatenate(name="fusion_concat")([image_features, clinical_features])
    
    x = Dense(128, activation='relu', name="fusion_dense_last_1")(fused)
    x = Dropout(0.3, name="fusion_dropout_last")(x)
    x = Dense(64, activation='relu', name="fusion_dense_last_2")(x)
    
    # Lớp Sigmoid MỚI (Dành riêng cho kết luận Viêm phổi của mô hình Fusion)
    output = Dense(1, activation='sigmoid', name="final_prediction")(x)
    
    model = Model(inputs=[image_input, clinical_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# ==========================================================
# TRUNG TÂM ĐIỀU KHIỂN (TRAIN & ĐÁNH GIÁ)
# ==========================================================
if __name__ == "__main__":
    # --- BƯỚC A: TẢI DATA ---
    if not os.path.exists(LOCAL_CSV_PATH):
        download_and_preprocess_data()
    else:
        print("⏭️ Bỏ qua bước cắt phổi vì đã có dữ liệu.")

    # --- BƯỚC B: CHUẨN BỊ GENERATOR ---
    print("\n🧠 [3/3] KHỞI ĐỘNG QUÁ TRÌNH HUẤN LUYỆN FUSION MODEL...")
    df = pd.read_csv(LOCAL_CSV_PATH)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    scaler = StandardScaler()
    df['Age_norm'] = scaler.fit_transform(df[['Age']])
    df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1}).fillna(-1)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Pneumonia'])
    train_gen = FastMultimodalGenerator(train_df, batch_size=32, target_size=(224, 224), shuffle=True)
    val_gen = FastMultimodalGenerator(val_df, batch_size=32, target_size=(224, 224), shuffle=False)

    # --- BƯỚC C: HUẤN LUYỆN (CÓ NHẬT KÝ) ---
    model = build_fusion_model(DENSENET_PATH)
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
    csv_logger = CSVLogger(HISTORY_CSV_PATH, append=False) # Ghi nhật ký vào CSV

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=[early_stop, csv_logger]
    )

    model.save(NEW_SAVE_PATH)
    print(f"\n🎉 ĐÃ LƯU MODEL TẠI: {NEW_SAVE_PATH}")

    # ==========================================================
    # KHÚC SAU TRAIN: VẼ BIỂU ĐỒ & ĐÁNH GIÁ TỰ ĐỘNG
    # ==========================================================
    print("\n📊 ĐANG XUẤT CÁC BÁO CÁO ĐÁNH GIÁ CHUYÊN MÔN...")

    # --- 1. VẼ BIỂU ĐỒ LỊCH SỬ HỌC TẬP (LOSS & AUC) ---
    hist_df = pd.read_csv(HISTORY_CSV_PATH)
    plt.figure(figsize=(14, 5))

    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist_df['epoch'], hist_df['loss'], label='Train Loss', marker='o')
    plt.plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss', marker='o')
    plt.title('Lịch sử Sai số (Loss)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Biểu đồ AUC
    plt.subplot(1, 2, 2)
    plt.plot(hist_df['epoch'], hist_df['auc'], label='Train AUC', marker='s')
    plt.plot(hist_df['epoch'], hist_df['val_auc'], label='Val AUC', marker='s')
    plt.title('Lịch sử AUC (Diện tích dưới đường cong)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 2. DỰ ĐOÁN ĐỂ VẼ MA TRẬN & ROC ---
    print("\n-> AI đang chấm điểm lại tập Validation để vẽ Ma trận...")
    val_gen.shuffle = False
    val_gen.on_epoch_end()
    y_true = val_gen.df['Pneumonia'].values
    y_pred_prob = model.predict(val_gen, steps=len(val_gen), verbose=0).flatten()
    y_pred_binary = (y_pred_prob > 0.5).astype(int)

    # --- 3. VẼ MA TRẬN NHẦM LẪN ---
    cm = confusion_matrix(y_true, y_pred_binary)
    class_names = ['NORMAL', 'PNEUMONIA']

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 15})
    plt.title('Ma trận Nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
    plt.ylabel('Đáp án Thực tế')
    plt.xlabel('AI Dự đoán')

    # --- 4. VẼ ĐƯỜNG CONG ROC ---
    plt.subplot(1, 2, 2)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Tỷ lệ dương tính giả)')
    plt.ylabel('True Positive Rate (Tỷ lệ dương tính thật)')
    plt.title('Đường cong ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # --- 5. BÁO CÁO DẠNG CHỮ ---
    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---")