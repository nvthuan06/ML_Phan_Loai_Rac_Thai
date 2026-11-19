from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.service import du_doan
from fastapi.responses import FileResponse

DINH_DANG_HOP_LE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GIOI_HAN_MB = 10

@asynccontextmanager
async def lifespan(app: FastAPI):
    du_doan.load_models()
    yield
    du_doan.unload_models()


app = FastAPI(
    title="API Phân Loại Rác Tự Động",
    description="Hệ thống phân loại rác 3 nhóm (Hữu cơ, Tái chế, Vô cơ) dùng MobileNetV2 + SVM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/", include_in_schema=False)
def giao_dien():
    return FileResponse(Path(__file__).resolve().parent.parent / "frontend" / "index.html")

@app.post("/predict", tags=["Dự đoán"])
async def du_doan_anh_rac(file: UploadFile = File(...)):
    ten_file = file.filename or "upload.jpg"
    duoi_file = Path(ten_file).suffix.lower()

    if duoi_file not in DINH_DANG_HOP_LE:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{duoi_file}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(sorted(DINH_DANG_HOP_LE))}",
        )

    noi_dung = await file.read()
    if not noi_dung:
        raise HTTPException(status_code=400, detail="File ảnh bị rỗng.")
    if len(noi_dung) > GIOI_HAN_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Dung lượng ảnh vượt quá {GIOI_HAN_MB} MB.")

    try:
        return du_doan.du_doan(noi_dung,ten_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể xử lý ảnh: {e}")


