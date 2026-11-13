import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. THU THẬP VÀ PHÂN CHIA DỮ LIỆU (STRATIFIED SPLIT)
# ==========================================
def prepare_dataframes(data_path):
    """
    Thu thập đường dẫn ảnh, lọc 2 nhãn NORMAL/PNEUMONIA.
    Đặc biệt: Gom tập train và val gốc (vì val gốc quá ít) để chia lại 
    bằng Stratified Split, đảm bảo tỷ lệ bệnh luôn chuẩn xác.
    """
    print("-> Đang xây dựng DataFrames và phân chia tập dữ liệu...")
    data = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                label = os.path.basename(root)
                split_folder = os.path.basename(os.path.dirname(root))
                
                # Chỉ lấy 2 nhãn cốt lõi
                if label in ['NORMAL', 'PNEUMONIA'] and split_folder in ['train', 'test', 'val']:
                    data.append({
                        'path': os.path.join(root, file),
                        'label': '0' if label == 'NORMAL' else '1', # Ép kiểu chuỗi số cho Binary Classification
                        'split_origin': split_folder
                    })
    
    df_all = pd.DataFrame(data)
    
    # Giữ nguyên tập Test gốc để đánh giá công bằng
    test_df = df_all[df_all['split_origin'] == 'test'].copy()
    
    # Gom Train và Val gốc lại để chia mới
    train_val_df = df_all[df_all['split_origin'].isin(['train', 'val'])].copy()
    
    # STRATIFIED SPLIT: Chia 80% Train - 20% Validation (Ép buộc giữ đúng tỷ lệ 1:2.6)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_val_df['label'] # Điểm mấu chốt để không bị hụt ảnh NORMAL
    )
    
    print(f"   + Tập Train: {len(train_df)} ảnh")
    print(f"   + Tập Val: {len(val_df)} ảnh")
    print(f"   + Tập Test: {len(test_df)} ảnh")
    
    return train_df, val_df, test_df

# ==========================================
# 2. GIẢI QUYẾT NGHỊCH LÝ MẤT CÂN BẰNG Y KHOA
# ==========================================
def get_class_weights(train_df):
    """
    Tính toán tự động trọng số lớp dựa trên số lượng ảnh thực tế.
    Phạt nặng mô hình nếu đoán sai nhãn thiểu số (NORMAL).
    """
    print("-> Đang tính toán Class Weights...")
    
    y_train = train_df['label'].astype(int).values
    classes = np.unique(y_train)
    
    # Sử dụng thuật toán cân bằng của scikit-learn
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    
    # In ra để kiểm chứng (Thường NORMAL sẽ quanh mức ~1.3-1.4, PNEUMONIA ~0.5)
    # Tỷ lệ giữa hai số này chính là khoảng 2.6 lần!
    print(f"   + Trọng số phạt cho NORMAL (0): {class_weights_dict[0]:.2f}")
    print(f"   + Trọng số phạt cho PNEUMONIA (1): {class_weights_dict[1]:.2f}")
    
    return class_weights_dict

# ==========================================
# 3. TRẠM TIỀN XỬ LÝ VÀ DATA AUGMENTATION CHUẨN Y KHOA
# ==========================================
def build_data_generators(train_df, val_df, test_df, preprocessing_fn=None, batch_size=32):
    """
    Tạo bộ cung cấp dữ liệu. 
    Lưu ý: preprocessing_fn là biến động, sẽ nhận hàm tiền xử lý của từng model khác nhau.
    """
    print("-> Đang thiết lập Data Generators...")
    
    # BỘ BIẾN ĐỔI CHO TẬP TRAIN (Augmentation có chọn lọc)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_fn, # Tiền xử lý theo yêu cầu của Model
        rotation_range=10,                       # Xoay nhẹ +-10 độ
        zoom_range=0.1,                          # Thu phóng nhẹ 10%
        brightness_range=[0.9, 1.1],             # Thay đổi độ sáng để mô phỏng tia X cứng/mềm
        horizontal_flip=False,                   # TUYỆT ĐỐI KHÔNG: Lật ngang làm sai vị trí Tim
        vertical_flip=False,                     # TUYỆT ĐỐI KHÔNG: Lật dọc làm ngược giải phẫu
        fill_mode='constant', cval=0             # Điền màu đen cho các góc bị khuyết khi xoay
    )
    
    # BỘ BIẾN ĐỔI CHO TẬP VAL/TEST (Tuyệt đối KHÔNG Augmentation, chỉ tiền xử lý)
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_fn
    )
    
    # CÁC THAM SỐ CHUNG 
    # Mặc định ép về 224x224 và chuyển thành RGB (3 kênh) cho mọi models
    generator_params = {
        'x_col': 'path',
        'y_col': 'label',
        'target_size': (224, 224),
        'color_mode': 'rgb',
        'class_mode': 'binary', # Dùng binary để output là mảng 1D (hợp với BCE Loss)
        'batch_size': batch_size
    }
    
    print("   + Khởi tạo Train Generator...")
    train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, shuffle=True, **generator_params)
    
    print("   + Khởi tạo Val Generator...")
    val_gen = val_test_datagen.flow_from_dataframe(dataframe=val_df, shuffle=False, **generator_params)
    
    print("   + Khởi tạo Test Generator...")
    test_gen = val_test_datagen.flow_from_dataframe(dataframe=test_df, shuffle=False, **generator_params)
    
    return train_gen, val_gen, test_gen