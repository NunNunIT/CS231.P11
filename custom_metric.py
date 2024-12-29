import tensorflow.keras.backend as K


def custom_accuracy(y_true, y_pred, threshold=0.5):
    """
    Tính độ chính xác tùy chỉnh (custom accuracy) cho bài toán đa nhãn với ngưỡng có thể điều chỉnh.
    
    Args:
        y_true: Tensor của nhãn thực tế (ground truth labels).
        y_pred: Tensor của dự đoán xác suất từ mô hình.
        threshold: Ngưỡng để chuyển xác suất thành nhãn nhị phân (mặc định: 0.5).
    
    Returns:
        Tensor: Độ chính xác trung bình trên tất cả các nhãn.
    """
    # Chuyển y_true về dạng float để tính toán
    # Áp dụng ngưỡng cho predictions để chuyển thành nhãn nhị phân
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred >= threshold, K.floatx())  
    
    # Tính đúng sai từng nhãn
    correct = K.cast(K.equal(y_true, y_pred), K.floatx())
    
    # Tính trung bình chính xác của từng nhãn và sau đó lấy trung bình trên tất cả các nhãn
    label_accuracies = K.mean(correct, axis=0)  # Chính xác trên từng nhãn
    overall_accuracy = K.mean(label_accuracies)  # Trung bình trên tất cả nhãn
    
    return overall_accuracy


def custom_hamming_loss(y_true, y_pred, threshold=0.5):
    """
    Tính Hamming Loss cho bài toán phân loại đa nhãn.
    Hamming Loss là tỷ lệ nhãn bị dự đoán sai trên tổng số nhãn.
    
    Args:
        y_true: Tensor của nhãn thực tế
        y_pred: Tensor của dự đoán xác suất
        threshold: Ngưỡng để chuyển xác suất thành nhãn
    
    Returns:
        Tensor: Hamming Loss
    """
    y_pred = K.cast(y_pred >= threshold, K.dtype(y_true))
    y_true = K.cast(y_true, K.dtype(y_pred))
    
    return K.mean(K.cast(K.not_equal(y_true, y_pred), K.floatx()))

def custom_exact_match_ratio(y_true, y_pred, threshold=0.5):
    """
    Tính tỷ lệ các mẫu có tất cả các nhãn được dự đoán đúng.
    
    Args:
        y_true: Tensor của nhãn thực tế
        y_pred: Tensor của dự đoán xác suất
        threshold: Ngưỡng để chuyển xác suất thành nhãn
    
    Returns:
        Tensor: Tỷ lệ exact match
    """
    y_pred = K.cast(y_pred >= threshold, K.dtype(y_true))
    y_true = K.cast(y_true, K.dtype(y_pred))
    
    # Kiểm tra xem tất cả các nhãn có đúng không
    all_correct = K.all(K.equal(y_true, y_pred), axis=-1)
    return K.mean(K.cast(all_correct, K.floatx()))