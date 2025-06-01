# Action Recognition System

Hệ thống nhận diện hành động sử dụng cảm biến gia tốc và con quay hồi chuyển.

## Cấu trúc thư mục

```
/
├── deployment/           # Backend code
│   ├── test.py          # Server chính
│   ├── requirements.txt # Python dependencies
│   ├── render.yaml     # Render deployment config
│   ├── models/         # ML models
│   └── preprocessing/  # Data files
└── frontend/           # React frontend
    ├── src/           # Source code
    └── build/         # Production build
```

## Cài đặt

### Backend

```bash
cd deployment
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run build
```

## Chạy ứng dụng

1. Khởi động backend:
```bash
cd deployment
python test.py
```

2. Frontend sẽ được serve tự động bởi backend tại `http://localhost:8080`

## Deploy

Ứng dụng được deploy trên Render.com. Cấu hình deploy được định nghĩa trong `deployment/render.yaml`. 