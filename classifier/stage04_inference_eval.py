import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_gen):
    """
    Load mô hình tốt nhất, chạy dự đoán trên tập Test và in ra báo cáo y khoa.
    """
    print(f"\n{'='*50}")
    print(" BẮT ĐẦU INFERENCE & EVALUATION")
    print(f"{'='*50}\n")
    
    # 1. Tải mô hình
    print(f"-> Đang nạp mô hình từ: {model_path}")
    model = load_model(model_path)
    
    # BẮT BUỘC: Reset generator để đảm bảo thứ tự ảnh không bị xáo trộn
    test_gen.reset()
    
    # 2. INFERENCE (Làm bài thi)
    print("-> AI đang chẩn đoán trên tập Test...")
    # Lấy xác suất dự đoán (ra các con số như 0.1, 0.9...)
    predictions = model.predict(test_gen, steps=len(test_gen), verbose=1)
    
    # Chuyển xác suất thành nhãn: >= 0.5 là PNEUMONIA (1), < 0.5 là NORMAL (0)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes # Đáp án chuẩn từ thư mục
    
    # 3. EVALUATION (Chấm điểm)
    class_names = ['NORMAL', 'PNEUMONIA']
    
    print("\n--- BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
    # In ra F1-Score, Recall, Precision
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # (Tùy chọn) Vẽ biểu đồ Ma trận nhầm lẫn
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Đáp án thực tế (True)')
    plt.xlabel('AI Dự đoán (Predicted)')
    plt.show()
    
    return y_true, y_pred