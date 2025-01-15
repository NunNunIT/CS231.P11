# [CS231.P11] - MÔ HÌNH XÁC ĐỊNH VẬT PHẨM THỜI TRANG TRONG HÌNH ẢNH

## Giới thiệu
Dự án này là một phần trong học phần "Nhập môn Thị giác Máy tính" (CS231), với mục tiêu nghiên cứu và xây dựng hệ thống tự động hóa quy trình nhận diện và phân loại sản phẩm thời trang trong hình ảnh. Hệ thống được thiết kế để hỗ trợ các ứng dụng thực tiễn trong thương mại điện tử và logistics.

Dựa trên các phương pháp tiên tiến của học máy (Machine Learning) và học sâu (Deep Learning), dự án triển khai các mô hình hiện đại như Mạng Nơ-ron Tích Chập (CNN) và thuật toán YOLO để tối ưu hóa khả năng phân tích dữ liệu hình ảnh phức tạp. Qua đó, hệ thống không chỉ giải quyết bài toán thực tiễn mà còn đóng góp vào nghiên cứu học thuật, mở ra nhiều cơ hội áp dụng trong các lĩnh vực như chuỗi cung ứng, bán lẻ thông minh, và quản lý hàng hóa.

## Danh sách thành viên
| STT | Họ tên                | MSSV     | Chức vụ     |
|:---:|:---------------------:|:--------:|:-----------:|
| 1   | Nguyễn Thị Hồng Nhung | 21522436 | Nhóm trưởng |
| 2   | Lê Trung Hiếu         | 21520850 | Thành viên  |

## Dataset
Dataset gốc:
```
https://www.kaggle.com/datasets/sovedi1356/fewsion-setup
```

Dataset cho phân loại:
```
https://www.kaggle.com/datasets/hongnhung21522436/8-labels-cloth-classification
```

Dataset cho yolo:
```
https://www.kaggle.com/datasets/hongnhung21522436/8-classes-cloth-yolo
```

## Kết quả Model đã huấn luyện
Vì giới hạn dung lượng tải lên của Github, một số tệp còn thiếu sót.

Bổ sung các model còn thiếu bằng cách truy cập drive. Tải về và giải nén tại thư mục /Model này

https://drive.google.com/drive/folders/182ZBSGI6Y2b9sJOrtHEQrzKrX2G11muS


## **Lưu ý:** 
Để đảm bảo tính ổn định và tránh các lỗi phát sinh do sự không tương thích giữa các phiên bản thư viện, khuyến nghị sử dụng các phiên bản sau:
```
scikit-multilearn-ng==0.0.8
keras==3.4.1
tensorflow==2.17.0
streamlit
ultralytics==8.3.44
scikit-image==0.24.0
streamlit-image-select==0.6.0
```
Trong trường hợp không muốn cài đặt trực tiếp các phiên bản được đề xuất, có thể tạo môi trường ảo để chạy.
## Bước 1: Tạo môi trường
```
python -m venv venv
```

## Bước 2: Kích hoạt môi trường
- Với Linux/MacOS
```
venv/bin/activate
```

- Với Windows
```
venv\Scripts\activate
```

## Bước 3: Cài đặt thư viện
``` 
pip install -r requirements.txt
 ```

## Bước 4: Hủy kích hoạt
``` 
deactivate
```


## Chạy ứng dụng
Sau khi cài đặt các thư viện cần thiết, chạy đoạn mã sau để khởi động ứng dụng
```
python -m streamlit run .\streamlit_app.py
```

Sử dùng trình duyệt bất kỳ, truy cập ```http://localhost:8501/```

Hình ảnh trực quan hóa:

![image](https://github.com/user-attachments/assets/33c4d4bd-18bb-4183-a7b8-ac34be4c767b)

![image](https://github.com/user-attachments/assets/02796ef0-8fb2-4bd2-9f0f-7fd0be09e5ef)

