import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

def build_callbacks(model_name, save_dir="saved_models"):
    """
    Tạo bộ 4 hệ thống giám sát (Callbacks) cho quá trình huấn luyện.
    """
    print(f"-> Đang thiết lập hệ thống giám sát cho {model_name}...")
    
    # Tạo thư mục lưu trữ nếu chưa có
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    callbacks = []

    # 1. MODEL CHECKPOINT (Lưu điểm Save Game)
    # Trong y tế, Val Loss là thước đo ổn định nhất để lưu mô hình.
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.h5")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',      # Theo dõi sai số trên tập Validation
        mode='min',              # Càng nhỏ càng tốt
        save_best_only=True,     # CHỈ lưu khi epoch hiện tại tốt hơn epoch trước
        verbose=1
    )
    callbacks.append(checkpoint)

    # 2. EARLY STOPPING (Hệ thống Phanh khẩn cấp chống học vẹt)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,              # Nếu sau 7 epoch liên tiếp mà Val Loss KHÔNG giảm -> DỪNG
        restore_best_weights=True, # Tự động load lại bộ tạ của epoch tốt nhất khi dừng
        verbose=1
    )
    callbacks.append(early_stopping)

    # 3. REDUCE LR ON PLATEAU (Hộp số tự động)
    # Khi mô hình không tiến bộ được nữa, giảm tốc độ học để "đi rà" vào vùng tối ưu
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,              # Giảm tốc độ học xuống còn 30%
        patience=3,              # Chỉ chịu đựng 3 epoch không giảm
        min_lr=1e-6,             # Không bao giờ giảm xuống dưới mức này
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # 4. CSV LOGGER (Hộp đen ghi nhật ký - Khuyến nghị thêm)
    # Lưu lại lịch sử Loss/Accuracy ra file Excel để vẽ biểu đồ sau này
    log_path = os.path.join(save_dir, f"{model_name}_training_log.csv")
    csv_logger = CSVLogger(log_path, append=False)
    callbacks.append(csv_logger)

    return callbacks

# ==========================================
# HÀM HUẤN LUYỆN TỔNG LỰC (THE MAIN TRAINING LOOP)
# ==========================================
def train_model(model, train_gen, val_gen, class_weights, model_name, epochs=30, save_dir="saved_models"):
    """
    Hàm kích hoạt quá trình học, kết hợp Model, Data và Callbacks.
    """
    print(f"\n{'='*50}")
    print(f" BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH: {model_name.upper()}")
    print(f"{'='*50}\n")
    
    # Lấy bộ Callbacks
    my_callbacks = build_callbacks(model_name, save_dir=save_dir)
    
    # Kích hoạt hàm fit
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=my_callbacks
    )
    
    print("\n-> HUẤN LUYỆN HOÀN TẤT!")
    return history