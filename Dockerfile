FROM python:3.11-slim

# Tối ưu log Python và tắt cảnh báo thừa của TensorFlow
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Cài thư viện hệ thống tối thiểu cho OpenCV và Scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cài dependencies trước để tận dụng cache khi sửa code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy các module phục vụ chạy thực tế
COPY api/ ./api/
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]