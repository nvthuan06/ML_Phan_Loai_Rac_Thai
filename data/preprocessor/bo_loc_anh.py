from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input

KICH_THUOC_ANH = (224,224)
DINH_DANG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def kiem_tra_dinh_dang_anh(file:Path) -> bool:
    if file.suffix.lower() not in DINH_DANG:
        return False
    try:
        with Image.open(file) as img :
            img.verify()
        return True
    except Exception:
        return False

def bo_loc (file)->np.ndarray:
    if isinstance(file, (str, Path)):
        img = Image.open(file)
    elif isinstance(file, (bytes, bytearray)):
        img = Image.open(BytesIO(file))
    elif isinstance(file, Image.Image):
        img = file
    else:
        raise TypeError("Định dạng đầu vào không được hỗ trợ.")

    img = img.convert("RGB").resize(KICH_THUOC_ANH, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return preprocess_input(arr)