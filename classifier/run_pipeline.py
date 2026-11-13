import os
import tensorflow as tf

# Import các Trạm (Stages)
from stage01_models import unfreeze_model_for_finetuning
from stage01_models import build_radimagenet_densenet
from stage02_dataset_utils import prepare_dataframes, get_class_weights, build_data_generators
from stage03_train_pipeline import train_model
from stage04_inference_eval import evaluate_model

# ==========================================
# CẤU HÌNH DỰ ÁN CHO COLAB (DÙNG ZIP RIÊNG)
# ==========================================
# 1. Nơi lưu model tốt nhất sau khi train
SAVE_DIR = os.environ.get("SAVE_DIR", "saved_models")

# 2. File trọng số RadImageNet bạn đã để trong AI_DATA
WEIGHTS_PATH = os.environ.get("RADIMAGENET_WEIGHTS", "weights/RadImageNet-DenseNet121_notop.h5")

# 3. ĐƯỜNG DẪN DỮ LIỆU VỪA GIẢI NÉN
# Giả sử trong file zip của bạn là 3 folder train/val/test nằm trực tiếp
DATA_PATH = os.environ.get("DATA_PATH", "data")  # folder containing train/ val/ test/ subfolders

MODEL_NAME = "DenseNet121_RadImageNet"
EPOCHS = 30
BATCH_SIZE = 32

def main():
    print(f"\n{'='*50}")
    print(" BẮT ĐẦU PIPELINE (DÙNG DATA ZIP TỪ DRIVE)")
    print(f"{'='*50}\n")
    
    print(f"-> Đang lấy dữ liệu trực tiếp từ: {DATA_PATH}")
    
    # --- STAGE 02: DỮ LIỆU ---
    train_df, val_df, test_df = prepare_dataframes(DATA_PATH)
    weights_dict = get_class_weights(train_df)
    
    preprocess_fn = tf.keras.applications.densenet.preprocess_input
    
    train_gen, val_gen, test_gen = build_data_generators(
        train_df, val_df, test_df, 
        preprocessing_fn=preprocess_fn, 
        batch_size=BATCH_SIZE
    )
    
    # --- STAGE 01: KHỞI TẠO KIẾN TRÚC ---
    model = build_radimagenet_densenet(WEIGHTS_PATH)
    
    # --- STAGE 03: HUẤN LUYỆN 2 GIAI ĐOẠN ---
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print("\n>>> BƯỚC 1: WARM-UP (Huấn luyện lớp phân loại cuối) <<<")
    # Chạy Warm-up khoảng 10-15 epochs là đủ để lớp cuối ổn định
    train_model(
        model=model, 
        train_gen=train_gen, 
        val_gen=val_gen, 
        class_weights=weights_dict, 
        model_name=MODEL_NAME + "_Warmup", 
        epochs=15, # Giảm epoch xuống
        save_dir=SAVE_DIR
    )
    
    print("\n>>> BƯỚC 2: FINE-TUNING SÂU (Rã đông các lớp cuối) <<<")
    # Mở khóa 20 lớp cuối cùng của DenseNet121 để học sâu hơn
    model_finetuned = unfreeze_model_for_finetuning(model, unfreeze_layers=20, learning_rate=1e-5)
    
    # Tiếp tục huấn luyện mô hình đã rã đông với Learning Rate rất nhỏ
    train_model(
        model=model_finetuned, 
        train_gen=train_gen, 
        val_gen=val_gen, 
        class_weights=weights_dict, 
        model_name=MODEL_NAME + "_Finetuned", 
        epochs=15, # Chạy nốt số epoch còn lại
        save_dir=SAVE_DIR
    )
    
    # --- STAGE 04: ĐÁNH GIÁ ---
    # Đánh giá bằng mô hình đã được Fine-tune
    best_model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_Finetuned_best.h5")
    evaluate_model(best_model_path, test_gen)

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()