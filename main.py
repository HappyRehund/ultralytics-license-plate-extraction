import base64
import cv2
import requests
import ultralytics
from openai import OpenAI
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from typing import Optional
import os
from dotenv import load_dotenv
import tempfile
import uuid
import math
import hashlib
from datetime import datetime
from collections import defaultdict

load_dotenv()

# Cek environment ultralytics
ultralytics.checks()

# Load YOLO model (ganti dengan model Anda kalau custom)
MODEL_PATH = os.getenv("ANPR_MODEL_PATH", "anpr-demo-model.pt")
MODEL_URL = os.getenv(
    "ANPR_MODEL_URL",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-model.pt",
)

def ensure_model() -> str:
    """Pastikan file model ada. Jika belum ada, download dari MODEL_URL."""
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    print(f"[INFO] Model '{MODEL_PATH}' tidak ditemukan. Mengunduh dari {MODEL_URL} ...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    print(f"[INFO] Model disimpan di: {MODEL_PATH}")
    return MODEL_PATH

# Setup OpenAI client
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

client = OpenAI(api_key=key)

# Prompt OCR
PROMPT = """
Can you extract the vehicle number plate text inside the image?
If you are not able to extract text, please respond with None.
Only output text, please.
If any text character is not from the English language, replace it with a dot (.).
"""

def extract_text(base64_encoded_data: str) -> Optional[str]:
    """Panggil GPT OCR untuk ekstraksi teks plat nomor dari gambar base64."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_data}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def download_video(video_url: str, save_path: Optional[str] = None) -> str:
    """Download video dari URL ke file lokal sementara (auto cleanup di caller)."""
    # gunakan file sementara jika save_path tidak diberikan
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="anpr-input-", suffix=".mp4", delete=False)
        save_path = tmp.name
        tmp.close()

    with requests.get(video_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return save_path

def process_video(
    video_path: str, 
    output_path: str = "anpr-output.avi",
    ocr_every_n_frames: Optional[int] = None,  # auto if None
    max_ocr_calls: int = 10,               
    plates_per_window: int = 1,
    ):
    """Proses video: deteksi plat nomor + OCR + simpan hasil video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error reading video file: {video_path}"

    # Setup video writer
    w, h, fps = (int(cap.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    video_writer = cv2.VideoWriter(output_path,
                                   fourcc,
                                   fps if fps > 0 else 30, (w, h))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_eff = fps if fps > 0 else 30
    if ocr_every_n_frames is None:
        if frame_count > 0 and max_ocr_calls > 0:
            ocr_every_n_frames = max(1, math.ceil(frame_count / max_ocr_calls))
        else:
            # fallback: sekitar setiap 0.5 detik
            ocr_every_n_frames = max(1, math.ceil(fps_eff * 0.5))

    windows = max(1, max_ocr_calls)
    window_size = max(1, math.ceil((frame_count or (fps_eff * 5)) / windows))
    window_ocr_counts: defaultdict[int, int] = defaultdict(int)
    
    # Load YOLO model
    mp = ensure_model()
    model = YOLO(mp)
    padding = 10

    # NEW: collect detections
    detections = []
    frame_idx = 0
    ocr_calls = 0
    seen_hashes: set[str] = set()

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        results = model.predict(im0)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()

        ann = Annotator(im0, line_width=3)

        for cls, box in zip(clss, boxes):
            height, width, _ = im0.shape
            x1 = max(int(box[0]) - padding, 0)
            y1 = max(int(box[1]) - padding, 0)
            x2 = min(int(box[2]) + padding, width)
            y2 = min(int(box[3]) + padding, height)

            # Crop plat
            crop = im0[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Encode ke base64 (pastikan tipe bytes untuk b64encode)
            ok, buf = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            img_bytes = buf.tobytes()
            crop_hash = hashlib.md5(img_bytes).hexdigest()

            stride_ok = (ocr_every_n_frames is None) or (frame_idx % ocr_every_n_frames == 0)
            win_id = min(windows - 1, frame_idx // window_size)
            window_ok = window_ocr_counts[win_id] < plates_per_window

            response_text = ""
            if stride_ok and window_ok and (crop_hash not in seen_hashes) and (ocr_calls < max_ocr_calls):
                base64_im0 = base64.b64encode(img_bytes).decode("utf-8")
                response_text = (extract_text(base64_im0) or "").strip()
                print(f"Extracted text: {response_text}")
                seen_hashes.add(crop_hash)
                ocr_calls += 1
                window_ocr_counts[win_id] += 1

            ann.box_label(box, label=response_text if response_text else "", color=colors(int(cls), True))

            # NEW: append structured detection
            detections.append({
                "frame": int(frame_idx),
                "time_sec": float(frame_idx / (fps or 30)),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "text": response_text,
                "class_id": int(cls),
            })

        # Tulis frame
        video_writer.write(im0)
        frame_idx += 1

    cap.release()
    video_writer.release()
    print(f"Video hasil disimpan di: {output_path}")
    print(f"[INFO] OCR calls={ocr_calls}/{max_ocr_calls} | windows={windows} | window_size={window_size} | plates_per_window={plates_per_window}")

    unique_plates = sorted(list(set(d["text"] for d in detections if d["text"])))

    return {
        "video_path": output_path,
        "fps": fps_eff,
        "width": w,
        "height": h,
        "detections": detections,
        "unique_plates": unique_plates,
        "ocr_calls": ocr_calls,
        "ocr_stride": ocr_every_n_frames,
        "window_size": window_size,
        "plates_per_window": plates_per_window,
    }

def gcp_trigger(video_url: str, output_dir: str = "outputs"):
    """Ambil video dari URL, proses, lalu hapus file input lokal."""
    print(f"[INFO] Mulai proses video dari URL: {video_url}")
    os.makedirs(output_dir, exist_ok=True)

    # buat nama output unik agar tidak overwrite
    out_name = f"anpr-output-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(output_dir, out_name)

    local_path = download_video(video_url)  # simpan sementara
    try:
        result = process_video(local_path, output_path, ocr_every_n_frames=None, max_ocr_calls=45, plates_per_window=1)
        print("[INFO] Selesai memproses video.")

        detected_plates = result.get("unique_plates", [])
        print(f"[INFO] Plat nomor yang berhasil diekstrak: {detected_plates}")

        return result
    
    finally:
        # pastikan input sementara dibersihkan
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                print(f"[INFO] Hapus file input sementara: {local_path}")
        except Exception as e:
            print(f"[WARN] Gagal menghapus file sementara: {e}")
            
# ==== Contoh pemanggilan (simulate trigger) ====
if __name__ == "__main__":
    test_url = os.getenv(
        "ANPR_VIDEO_URL",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-video.mp4",
    )
    gcp_trigger(test_url)
