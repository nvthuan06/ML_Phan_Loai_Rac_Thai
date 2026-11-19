import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from keras.models import load_model

THU_MUC_GOC = Path(__file__).resolve().parent.parent
if str(THU_MUC_GOC) not in sys.path:
    sys.path.insert(0, str(THU_MUC_GOC))

from data.preprocessor.bo_loc_anh import bo_loc

THONG_TIN_NHAN = {
    "Huu_Co": {"ten": "Hữu cơ"},
    "Tai_Che": {"ten": "Tái Chế"},
    "Vo_Co": {"ten": "Vô Cơ"},
}

THU_MUC_MODEL = THU_MUC_GOC / "model"


class RacONhiem:
    def __init__(self, model_dir: Path = THU_MUC_MODEL):
        self.model_dir = model_dir
        self.bo_loc_trich_xuat = None
        self.model_du_doan = None
        self.labels_map: dict[str, str] = {}
        self.thong_tin_model: dict[str, Any] = {}

    def san_sang(self) -> bool:
        return self.bo_loc_trich_xuat is not None and self.model_du_doan is not None

    def load_models(self) -> None:
        print("Đang nạp bộ lọc đặc trưng MobileNetV2 và SVM...")
        bat_dau = time.perf_counter()

        self.bo_loc_trich_xuat = load_model(self.model_dir / "bo_trich_xuat_mobilenetv2.keras")
        self.model_du_doan = joblib.load(self.model_dir / "model_tot_nhat.pkl")

        with open(self.model_dir / "labels.json", "r", encoding="utf-8") as f:
            self.labels_map = json.load(f)

        info_path = self.model_dir / "model_tot_nhat.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                self.thong_tin_model = json.load(f)

        anh_gia = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.bo_loc_trich_xuat(anh_gia, training=False)
        print(f"Khởi động xong trong {time.perf_counter() - bat_dau:.2f}s ...")

    def unload_models(self) -> None:
        self.bo_loc_trich_xuat = None
        self.model_du_doan = None
        self.labels_map.clear()
        self.thong_tin_model.clear()

    def lay_thong_tin(self) -> dict[str, Any]:
        return self.thong_tin_model or {"labels": self.labels_map}

    def du_doan(self, noi_dung_anh: bytes, ten_anh: str) -> dict[str, Any]:
        if not self.san_sang():
            raise RuntimeError("Mô hình chưa được nạp vào bộ nhớ.")

        bat_dau = time.perf_counter()
        duoi_file = Path(ten_anh).suffix.lower() or ".jpg"
        duong_dan_tam: Path | None = None

        try:
            # Ghi và đóng file tạm trước khi gọi bo_loc()
            with tempfile.NamedTemporaryFile(delete=False, suffix=duoi_file) as tmp:
                tmp.write(noi_dung_anh)
                duong_dan_tam = Path(tmp.name)

            anh_da_xu_ly = bo_loc(duong_dan_tam)
            batch_tensor = np.expand_dims(anh_da_xu_ly, axis=0)
            dac_trung = np.asarray(self.bo_loc_trich_xuat(batch_tensor, training=False), dtype=np.float32)

            xac_suat = self.model_du_doan.predict_proba(dac_trung)[0]
            vi_tri_cao_nhat = int(np.argmax(xac_suat))
            nhan_idx = int(self.model_du_doan.classes_[vi_tri_cao_nhat])
            ten_nhan = self.labels_map[str(nhan_idx)]
            do_tu_tin = float(xac_suat[vi_tri_cao_nhat])

            chi_tiet_xac_suat = {
                self.labels_map[str(int(cls_idx))]: round(float(prob), 4)
                for cls_idx, prob in zip(self.model_du_doan.classes_, xac_suat)
            }

            thoi_gian_ms = round((time.perf_counter() - bat_dau) * 1000, 1)
            goi_y = THONG_TIN_NHAN.get(ten_nhan, {})

            return {
                "filename": ten_anh,
                "predicted_label": ten_nhan,
                "display_name": goi_y.get("ten", ten_nhan),
                "confidence": round(do_tu_tin, 4),
                "confidence_percent": f"{do_tu_tin * 100:.2f}%",
                "probabilities": chi_tiet_xac_suat,
                "inference_time_ms": thoi_gian_ms,
            }
        finally:
            if duong_dan_tam and duong_dan_tam.exists():
                duong_dan_tam.unlink(missing_ok=True)


du_doan = RacONhiem()