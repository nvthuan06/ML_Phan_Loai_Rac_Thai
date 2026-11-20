# Hệ Thống Phân Loại Rác Tự Động (MobileNetV2 + Machine Learning)

Hệ thống phân loại hình ảnh rác thải thành **3 nhóm** (**Hữu cơ**, **Tái chế**, **Vô cơ**) kết hợp trích xuất đặc trưng bằng mạng học sâu tiền huấn luyện **MobileNetV2** (1.280 chiều) và các thuật toán Học máy cổ điển (**SVM**, **KNN**, **Random Forest**), triển khai dưới dạng REST API bằng **FastAPI**.

---

## Kiến trúc & Quy trình xử lý

Ảnh đầu vào ──► Tiền xử lý (bo_loc: resize 224×224, chuẩn hóa)
──► MobileNetV2 Extractor (vector 1.280 đặc trưng)
──► StandardScaler + SVM (RBF)
──► Nhãn dự đoán + Xác suất chi tiết 3 lớp (JSON / Web UI)

---

## Bộ dữ liệu (Dataset)

Tổng cộng **16.110 ảnh** thuộc 3 lớp (`Huu_Co`, `Tai_Che`, `Vo_Co`), được làm sạch và chia phân tầng (_Stratified Split_) theo tỷ lệ **80% Train / 20% Test**:

| Tập dữ liệu | Số lượng ảnh | Kích thước ma trận đặc trưng | Dung lượng file `.npy` | Thời gian trích xuất (CPU) |
| :---------- | :----------: | :--------------------------: | :--------------------: | :------------------------: |
| **Train**   |    12.888    |       `(12888, 1280)`        |        66.0 MB         |           416.5s           |
| **Test**    |    3.222     |        `(3222, 1280)`        |        16.5 MB         |           107.4s           |

---

## Kết quả thực nghiệm & Lựa chọn mô hình

| Mô hình                 | Tham số tối ưu                                            | CV F1-Macro | Test Accuracy | Test F1-Macro | Test Precision | Test Recall | Suy luận (ms/ảnh) | Kích thước (MB) | Thời gian train (s) |
| :---------------------- | :-------------------------------------------------------- | :---------: | :-----------: | :-----------: | :------------: | :---------: | :---------------: | :-------------: | :-----------------: |
| **SVM**                 | `kernel='rbf'`, `C=10`, `gamma='scale'`                   | **0.9657**  |  **0.9677**   |  **0.9680**   |   **0.9691**   | **0.9669**  |       3.183       |    **42.39**    |        599.2        |
| **KNN**                 | `metric='cosine'`, `n_neighbors=7`, `weights='distance'`  |   0.9643    |    0.9662     |    0.9661     |     0.9675     |   0.9650    |       0.238       |      63.06      |        58.5         |
| **Random Forest**       | `n_estimators=300`, `max_depth=25`, `min_samples_split=2` |   0.9396    |    0.9410     |    0.9410     |     0.9450     |   0.9376    |       0.045       |      48.42      |        210.1        |
| **Logistic Regression** | `C=0,002`                                                 |   0.9513    |     0.955     |    0.9551     |     0.9561     |   0.9543    |       0.007       |      0.05       |        78.1         |

- **Mô hình được chọn triển khai (`SVM - RBF`):** Đạt điểm cao nhất ở toàn bộ chỉ số chất lượng (`Test F1-Macro = 0.9680`, `Test Accuracy = 0.9677`), không bị overfitting (điểm Test cao hơn nhẹ so với CV) và có dung lượng file nhẹ nhất (`42.39 MB`).

---

## Cấu trúc thư mục

```text
ML_bao_cao/
├── api/
│   ├── __init__.py
│   ├── main.py                      # Controller FastAPI (routes, CORS, kiểm tra file)
│   └── service.py                   # Service ML (nạp model, trích xuất & dự đoán)
├── data/
│   └── preprocessor/
│       └── bo_loc_anh.py            # Hàm tiền xử lý ảnh bo_loc() dùng chung cho Train & API
├── features/                        # Đặc trưng 1.280 chiều đã trích xuất (.npy)
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   └── y_test.npy
├── frontend/
│   └── index.html                   # Giao diện web kéo-thả / dán ảnh (Ctrl + V)
├── model/                           # Mô hình & siêu dữ liệu phục vụ suy luận
│   ├── bo_trich_xuat_mobilenetv2.keras
│   ├── model_tot_nhat.pkl
│   ├── model_tot_nhat.json
│   ├── knn_model.pkl
│   ├── rf_model.pkl
│   ├── svm_model.pkl
    ├── lr_model.pkl
│   └── labels.json
├── report/                          # Kết quả đánh giá, CSV và biểu đồ báo cáo
│   ├── cv_knn.json
│   ├── cv_rf.json
│   ├── cv_svm.json
    ├── cv_lr.json
│   ├── bang_so_sanh_test.csv
│   ├── confusion_matrix_3_mo_hinh.png
│   └── so_sanh_mo_hinh.png
└── train/                           # Các notebook huấn luyện và đánh giá
    ├── 01_eda_tien_xu_ly.ipynb
    ├── 02_trich_xuat_dac_trung.ipynb
    ├── 03a_huan_luyen_knn.ipynb
    ├── 03b_huan_luyen_rf.ipynb
    ├── 03c_huan_luyen_svm.ipynb
    ├── 03d_huan_luyen_lr.ipynb
    └── 04_danh_gia_mo_hinh.ipynb
```

---

## Cài đặt & Khởi chạy

### 1. Tạo môi trường ảo và cài đặt thư viện

Yêu cầu **Python 3.10 – 3.12**. Đứng tại thư mục gốc `ML_bao_cao`:

```bash
python -m venv .venv

# Kích hoạt trên Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install tensorflow scikit-learn numpy pandas matplotlib joblib pillow opencv-python fastapi "uvicorn[standard]" python-multipart
```

### 2. Chạy server Backend (FastAPI)

Đảm bảo thư mục `model/` đã có đủ `bo_trich_xuat_mobilenetv2.keras`, `model_tot_nhat.pkl` và `labels.json`, sau đó chạy:

```bash
uvicorn api.main:app --reload
```

Sau khi terminal báo `Khởi động xong ...`:

- **Giao diện Web:** Mở `http://127.0.0.1:8000/` (hoặc mở trực tiếp file `frontend/index.html`).
- **Tài liệu API (Swagger UI):** Mở `http://127.0.0.1:8000/docs`.

### 3. Huấn luyện lại từ đầu (tùy chọn)

Nếu muốn tái lập toàn bộ quá trình huấn luyện, chạy lần lượt các notebook trong thư mục `train/` bằng kernel `.venv`:

1. `01_eda_tien_xu_ly.ipynb` — Khám phá dữ liệu, làm sạch và chia tập Train/Test phân tầng.
2. `02_trich_xuat_dac_trung.ipynb` — Trích xuất vector 1.280 chiều bằng MobileNetV2 và lưu vào `features/`.
3. `03a_huan_luyen_knn.ipynb`, `03b_huan_luyen_rf.ipynb`, `03c_huan_luyen_svm.ipynb` — Huấn luyện và dò tham số 3 mô hình (có thể chạy song song).
4. `04_danh_gia_mo_hinh.ipynb` — Chấm điểm tập Test, vẽ Confusion Matrix, biểu đồ so sánh và xuất mô hình tốt nhất.

---

## Tài liệu API

| Phương thức | Endpoint   | Mô tả                                                                                                             |
| :---------: | :--------- | :---------------------------------------------------------------------------------------------------------------- |
|    `GET`    | `/`        | Hiển thị giao diện web phân loại rác                                                                              |
|   `POST`    | `/predict` | Nhận file ảnh (`multipart/form-data`, khóa `file`, hỗ trợ `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, tối đa 10 MB) |

**Ví dụ phản hồi từ `POST /predict`:**

```json
{
  "filename": "chai_nhua.jpg",
  "predicted_label": "Tai_Che",
  "display_name": "Tái Chế",
  "confidence": 0.9842,
  "confidence_percent": "98.42%",
  "probabilities": {
    "Huu_Co": 0.0081,
    "Tai_Che": 0.9842,
    "Vo_Co": 0.0077
  },
  "inference_time_ms": 48.6
}
```

---

## Công nghệ sử dụng

- **Deep Learning & Computer Vision:** TensorFlow / Keras (MobileNetV2), OpenCV, Pillow
- **Machine Learning & Data:** Scikit-Learn, NumPy, Pandas, Matplotlib, Joblib
- **Backend & Frontend:** FastAPI, Uvicorn, HTML5 / CSS3 / Vanilla JavaScript
