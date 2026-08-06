# استفاده ازイメージ پایه پایتون ۳.۱۰ (سبک)
FROM python:3.10-slim

# نصب کتابخانه‌های سیستمی مورد نیاز برای کامپایل برخی کتابخانه‌ها
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libz-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# کپی فایل requirements و نصب کتابخانه‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی بقیه فایل‌های پروژه
COPY . .

# اجرای اپ
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]