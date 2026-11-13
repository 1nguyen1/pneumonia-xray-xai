import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV2, DenseNet121

# ==========================================
# HÀM HỖ TRỢ: BIÊN DỊCH MÔ HÌNH CHUNG
# ==========================================
def compile_binary_model(model, learning_rate=1e-3):
    """
    Biên dịch mô hình cho bài toán Nhị phân (NORMAL vs PNEUMONIA).
    Ưu tiên theo dõi chỉ số Recall (Độ nhạy) cho y tế.
    """
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision')
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model

# ==========================================
# MÔ HÌNH 1: BASELINE CNN (Train from scratch)
# ==========================================
def build_baseline_cnn(input_shape=(224, 224, 3)):
    """
    Mạng CNN tự xây dựng từ đầu. Sử dụng Kaiming He (he_normal) 
    cho các lớp ReLU để tránh triệt tiêu đạo hàm.
    """
    print("-> Đang khởi tạo Baseline CNN...")
    model = Sequential([
        # Khối 1
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Khối 2
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Khối 3
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Lớp Trích xuất & Phân loại
        GlobalAveragePooling2D(),
        Dropout(0.3), # Chống Overfitting nhẹ
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(1, activation='sigmoid', name='output_binary') # 1 nơ-ron cho nhị phân
    ], name="Baseline_CNN")
    
    return compile_binary_model(model)

# ==========================================
# MÔ HÌNH 2: MOBILENET V2 (ImageNet Weights)
# ==========================================
def build_mobilenet_v2(input_shape=(224, 224, 3)):
    """
    Sử dụng Transfer Learning từ ImageNet. Tối ưu cho thiết bị cấu hình yếu.
    """
    print("-> Đang khởi tạo MobileNetV2 (ImageNet)...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Đóng băng thân mạng (Warm-up phase)
    base_model.trainable = False 
    
    # Xây dựng phần đuôi
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid', name='output_binary')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="MobileNetV2_Transfer")
    return compile_binary_model(model)

# ==========================================
# MÔ HÌNH 3: DENSENET 121 (RadImageNet Weights)
# ==========================================
def build_radimagenet_densenet(weight_path, input_shape=(224, 224, 3)):
    """
    Sử dụng Transfer Learning từ bộ dữ liệu y khoa RadImageNet.
    Yêu cầu file trọng số .h5 local.
    """
    print(f"-> Đang khởi tạo DenseNet121 (RadImageNet) từ: {weight_path}")
    # 1. Khởi tạo khung mạng rỗng
    base_model = DenseNet121(
        weights=None,  # Ép Keras bỏ qua bước kiểm tra đường dẫn
        include_top=False, 
        input_shape=input_shape
    )
    
    # 2. Tự tay nạp trọng số
    base_model.load_weights(weight_path)
    
    # Xây dựng phần đuôi
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) # Bịt mắt 50% nơ-ron để ép mô hình học kỹ vệt mờ viêm phổi
    outputs = Dense(1, activation='sigmoid', name='output_binary')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="DenseNet121_RadImageNet")
    return compile_binary_model(model)

# ==========================================
# HÀM HỖ TRỢ: UNFREEZE ĐỂ FINE-TUNING
# ==========================================
def unfreeze_model_for_finetuning(model, unfreeze_layers=20, learning_rate=1e-5):
    """
    Hàm dùng cho Giai đoạn 2 của huấn luyện: Rã đông các lớp cuối 
    để mô hình thích nghi sâu hơn với X-quang.
    """
    print(f"-> Đang rã đông {unfreeze_layers} lớp cuối cùng để Fine-Tuning...")
    # Tắt trainable toàn bộ trước
    for layer in model.layers:
        layer.trainable = False
        
    # Bật trainable cho N lớp cuối (không tính các lớp BatchNormalization để tránh nhiễu)
    for layer in model.layers[-unfreeze_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            
    # Compile lại với Learning Rate cực nhỏ (để không phá hủy trọng số cũ)
    return compile_binary_model(model, learning_rate=learning_rate)