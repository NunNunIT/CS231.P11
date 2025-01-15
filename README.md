
# **Lưu ý:** 
Để đảm bảo tính ổn định và tránh các lỗi phát sinh do sự không tương thích giữa các phiên bản thư viện, khuyến nghị sử dụng các phiên bản sau:
```
keras==3.4.1
tensorflow==2.17.0
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


# Chạy ứng dụng
Sau khi cài đặt các thư viện cần thiết, chạy đoạn mã sau để khởi động ứng dụng
```
streamlit run .\streamlit_app.py
```

Sử dùng trình duyệt bất kỳ, truy cập ```http://127.0.0.1:5000```

Hình ảnh trực quan hóa:

![image](https://github.com/user-attachments/assets/33c4d4bd-18bb-4183-a7b8-ac34be4c767b)

![image](https://github.com/user-attachments/assets/02796ef0-8fb2-4bd2-9f0f-7fd0be09e5ef)

