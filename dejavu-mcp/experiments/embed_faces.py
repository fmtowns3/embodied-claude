"""顔ベクトルの弁別力を測る最小実験。

images/ に <名前>_<連番>.jpg の形で写真を置くと、
  - 同一人物どうしの類似度
  - 別人どうしの類似度
を出して、その差（マージン）が開いているかを見る。

「別人のベクトルが違う」だけでは何も言えない。どんな2枚でもベクトルは違う。
同一人物が別人より明確に近いことが確かめられて初めて、この設計は成立する。

使い方:
    .venv\\Scripts\\python.exe embed_faces.py
"""

import sys
from itertools import combinations
from pathlib import Path

# Windows のコンソールは既定が cp932 で、日本語の出力が化ける
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "images"
FACE_MODEL = ROOT / "models" / "blaze_face_short_range.tflite"
DINO_NAME = "facebook/dinov2-with-registers-base"

# 顔の切り出しに付ける余白。髪や輪郭が入らないと個人差が出にくいが、
# 広げすぎると襟元＝服の色が入り、「服で当てている」交絡になる。
# 第1引数で上書きできる: python embed_faces.py 0.10
CROP_MARGIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
# DINOv2-with-registers はレジスタトークンを 4 本持つ。パッチ平均を取るとき除外する
NUM_REGISTERS = 4


def detect_face(detector, path: Path):
    """最も確信度の高い顔を1つ返す。余白を付けた切り出し済み PIL 画像。

    顔が見つからなければ None。
    """
    # mp.Image.create_from_file は webp を読めないので PIL で開いて渡す
    rgb = np.asarray(Image.open(path).convert("RGB"))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.detections:
        return None

    best = max(result.detections, key=lambda d: d.categories[0].score)
    box = best.bounding_box
    h, w = rgb.shape[:2]

    mx, my = int(box.width * CROP_MARGIN), int(box.height * CROP_MARGIN)
    x0 = max(0, box.origin_x - mx)
    y0 = max(0, box.origin_y - my)
    x1 = min(w, box.origin_x + box.width + mx)
    y1 = min(h, box.origin_y + box.height + my)

    return Image.fromarray(rgb[y0:y1, x0:x1]), best.categories[0].score


def embed(model, processor, image: Image.Image, device):
    """CLS トークンとパッチ平均の2種類を返す。どちらが効くかは実測しないと分からない。"""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)

    hidden = out.last_hidden_state[0]  # (1 + registers + patches, dim)
    cls = hidden[0]
    patches = hidden[1 + NUM_REGISTERS:].mean(dim=0)

    def l2(v):
        return (v / v.norm()).cpu().numpy()

    return l2(cls), l2(patches)


def main() -> int:
    paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(paths) < 2:
        print(f"images/ に画像が {len(paths)} 枚しかありません。")
        print("<名前>_<連番>.jpg の形で、2人 x 各2枚以上を置いてください。")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"モデル読み込み: {DINO_NAME}")
    processor = AutoImageProcessor.from_pretrained(DINO_NAME)
    model = AutoModel.from_pretrained(DINO_NAME).to(device).eval()

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )

    records = []
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for path in paths:
            found = detect_face(detector, path)
            if found is None:
                print(f"  [skip] {path.name}: 顔を検出できませんでした")
                continue
            face, score = found
            cls_vec, patch_vec = embed(model, processor, face, device)
            person = path.stem.rsplit("_", 1)[0]
            records.append((person, path.name, cls_vec, patch_vec))
            print(f"  [ok]   {path.name}: person={person} 検出スコア={score:.3f} 切出={face.size}")

    if len(records) < 2:
        print("顔を検出できた画像が足りません。")
        return 1

    for label, index in (("CLS トークン", 2), ("パッチ平均", 3)):
        same, diff = [], []
        print(f"\n=== {label} ===")
        for a, b in combinations(records, 2):
            sim = float(np.dot(a[index], b[index]))
            is_same = a[0] == b[0]
            (same if is_same else diff).append(sim)
            mark = "同一" if is_same else "別人"
            print(f"  {mark}  {a[1]:<20} x {b[1]:<20} cos={sim:+.4f}")

        if same and diff:
            s_min, d_max = min(same), max(diff)
            print(f"  ---- 同一人物 平均 {np.mean(same):+.4f} (最小 {s_min:+.4f}) / "
                  f"別人 平均 {np.mean(diff):+.4f} (最大 {d_max:+.4f})")
            margin = s_min - d_max
            verdict = "分離できている" if margin > 0 else "★重なっている（このままでは弁別できない）"
            print(f"  ---- マージン（同一の最小 − 別人の最大）= {margin:+.4f}  → {verdict}")
        else:
            print("  ---- 同一人物ペアか別人ペアのどちらかが無いので比較できません。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
