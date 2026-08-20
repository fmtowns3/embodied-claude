"""顔検出器を使わずに「誰か」を当てられるか。

embed_faces.py は MediaPipe で顔を検出 → クロップ → 1ベクトル、という順序だった。
その結果 人物A_2 と 人物B_2（どちらも横向き）で検出が落ち、そもそも比較の土俵に
乗らなかった。

ここでは検出器を外し、画像全体を 224x224 にして DINOv2 の 16x16=256 パッチを取り、
各パッチと「その人の顔ベクトル」との類似度の最大値で判定する。
顔がどこにあるかを検出器に訊かず、特徴空間の近さで探す。

参照ベクトル（ref_face）は、顔検出に成功した画像の顔クロップから作る。
つまり「一度でも正面が撮れていれば、以後は横向きでも追える」かを試している。

使い方:
    .venv\\Scripts\\python.exe patch_search.py
"""

import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "images"
FACE_MODEL = ROOT / "models" / "blaze_face_short_range.tflite"
DINO_NAME = "facebook/dinov2-with-registers-base"

CROP_MARGIN = 0.35
NUM_REGISTERS = 4
GRID = 16  # 224 / 14

# 第1引数に 3crop を渡すと、問い合わせ側だけ3クロップ・ステッチにする
USE_3CROP = len(sys.argv) > 1 and sys.argv[1] == "3crop"

# AutoImageProcessor は shortest_edge=256 → center_crop=224 で端を切り落とす。
# 画像全体をパッチで覆いたいので、アスペクト比を無視して 224x224 に潰す。
TO_TENSOR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def patches_of(model, image: Image.Image, device) -> np.ndarray:
    """画像 → L2正規化済みパッチ特徴 [256, 768]"""
    tensor = TO_TENSOR(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pixel_values=tensor)
    patches = out.last_hidden_state[0, 1 + NUM_REGISTERS:].cpu().numpy()
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    return patches / np.clip(norms, 1e-10, None)


def three_crop_patches(model, image: Image.Image, device, overlap: float = 0.25) -> np.ndarray:
    """左・中央・右の3クロップを重ねて撮り、パッチ面を横に繋ぐ。

    224x224 に潰すと横長の画像は解像度が足りず、遠い顔が数パッチに潰れる。
    幅半分ずつ3枚に分ければ、横方向の実効解像度が上がる。
    重なった列は平均する（境目に段差を作らないため）。
    """
    w, h = image.size
    crop_w = w // 2
    step = int(crop_w * (1 - overlap))
    x_starts = [0, step, max(0, w - crop_w)]

    batch = torch.stack([
        TO_TENSOR(image.crop((x, 0, x + crop_w, h))) for x in x_starts
    ]).to(device)
    with torch.no_grad():
        out = model(pixel_values=batch)
    grids = out.last_hidden_state[:, 1 + NUM_REGISTERS:].cpu().numpy().reshape(3, GRID, GRID, -1)

    overlap_cols = 4
    unique = GRID - overlap_cols
    total = GRID + 2 * unique
    acc = np.zeros((GRID, total, grids.shape[-1]), dtype=np.float32)
    cnt = np.zeros((GRID, total, 1), dtype=np.float32)
    for i, start in enumerate((0, unique, 2 * unique)):
        acc[:, start:start + GRID] += grids[i]
        cnt[:, start:start + GRID] += 1.0

    patches = (acc / cnt).reshape(-1, grids.shape[-1])
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    return patches / np.clip(norms, 1e-10, None)


def query_patches(model, image: Image.Image, device) -> np.ndarray:
    return three_crop_patches(model, image, device) if USE_3CROP else patches_of(model, image, device)


def detect_face(detector, path: Path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.detections:
        return None
    best = max(result.detections, key=lambda d: d.categories[0].score)
    box = best.bounding_box
    h, w = rgb.shape[:2]
    mx, my = int(box.width * CROP_MARGIN), int(box.height * CROP_MARGIN)
    x0, y0 = max(0, box.origin_x - mx), max(0, box.origin_y - my)
    x1 = min(w, box.origin_x + box.width + mx)
    y1 = min(h, box.origin_y + box.height + my)
    return Image.fromarray(rgb[y0:y1, x0:x1])


def main() -> int:
    paths = sorted(p for p in IMAGE_DIR.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO_NAME).to(device).eval()

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )

    # 1) 顔検出に成功した画像から、人物ごとの参照ベクトルを作る
    face_vecs: dict[str, list[np.ndarray]] = {}
    detected: set[str] = set()
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for path in paths:
            person = path.stem.rsplit("_", 1)[0]
            crop = detect_face(detector, path)
            if crop is None:
                continue
            detected.add(path.name)
            face_vecs.setdefault(person, []).append(patches_of(model, crop, device).mean(axis=0))

    refs = {}
    for person, vecs in face_vecs.items():
        v = np.mean(vecs, axis=0)
        refs[person] = v / np.linalg.norm(v)
    print(f"問い合わせ方式: {'3クロップ・ステッチ' if USE_3CROP else '単発224'}")
    print(f"参照ベクトルを作れた人物: {sorted(refs)}")
    print(f"顔検出に失敗した画像: {sorted(p.name for p in paths if p.name not in detected)}\n")

    # 2) 全画像を「検出器なし」で判定する
    people = sorted(refs)
    head = f"{'file':<20} {'正解':<11} " + " ".join(f"{p:>11}" for p in people) + f"  {'判定':<11} {'結果'}"
    print(head)
    print("-" * len(head))

    hit = miss = 0
    for path in paths:
        truth = path.stem.rsplit("_", 1)[0]
        patches = query_patches(model, Image.open(path).convert("RGB"), device)
        scores = {p: float((patches @ refs[p]).max()) for p in people}
        guess = max(scores, key=scores.get)
        ok = guess == truth
        hit, miss = hit + ok, miss + (not ok)
        mark = "○" if ok else "×"
        flag = "" if path.name in detected else "  ←検出失敗した画像"
        cells = " ".join(f"{scores[p]:>11.4f}" for p in people)
        print(f"{path.name:<20} {truth:<11} {cells}  {guess:<11} {mark}{flag}")

    print(f"\n正解 {hit} / {hit + miss}")

    # 3) 参照が自分自身を含む分の下駄を外して、検出失敗画像だけを見る
    unseen = [p for p in paths if p.name not in detected]
    if unseen:
        print("\n=== 顔検出が落ちた画像だけ（検出器なしで拾えたか）===")
        for path in unseen:
            truth = path.stem.rsplit("_", 1)[0]
            patches = query_patches(model, Image.open(path).convert("RGB"), device)
            scores = sorted(((float((patches @ refs[p]).max()), p) for p in people), reverse=True)
            top, second = scores[0], scores[1]
            print(f"  {path.name}: 正解={truth} / 1位={top[1]}({top[0]:.4f}) "
                  f"2位={second[1]}({second[0]:.4f}) 差={top[0] - second[0]:+.4f} "
                  f"{'○' if top[1] == truth else '×'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
