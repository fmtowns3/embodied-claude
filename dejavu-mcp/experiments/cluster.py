"""名前を一切与えずに「同じ人」を束ねられるか。

これが dejavu の核心。ラベルを教えずにベクトルだけで群れを作り、
出来た群れが結果として人物と一致するかを見る。一致するなら
「名前が付く前の見覚え」が成立していることになる。

束ね方は heishio と同じ貪欲な逐次クラスタリング:
  新しい観測を既存クラスタの重心と比べ、
  最大類似度がしきい値以上ならそこへ入れて重心を更新、
  未満なら新しいクラスタを作る。

2種類のベクトルで比べる:
  person … 人物マスク内パッチの平均（服を含む。顔検出が要らないので全16枚で作れる）
  face   … 顔クロップのパッチ平均（服が入らない。顔検出できた14枚のみ）

person は同一セッションだと服が同じなので当たって当然。そこは差し引いて読む。

使い方:
    .venv\\Scripts\\python.exe cluster.py
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
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
SEG_MODEL = ROOT / "models" / "selfie_segmenter.tflite"
DINO_NAME = "facebook/dinov2-with-registers-base"
CROP_MARGIN = 0.35
NUM_REGISTERS = 4
GRID = 16

TO_TENSOR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def patches_of(model, image, device) -> np.ndarray:
    tensor = TO_TENSOR(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pixel_values=tensor)
    p = out.last_hidden_state[0, 1 + NUM_REGISTERS:].cpu().numpy()
    return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-10, None)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def cover_16(segmenter, image) -> np.ndarray:
    rgb = np.asarray(image)
    result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    mask = result.confidence_masks[0].numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return cv2.resize((mask > 0.5).astype(np.float32), (GRID, GRID),
                      interpolation=cv2.INTER_AREA).reshape(-1)


def face_crop(detector, rgb: np.ndarray):
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.detections:
        return None
    b = max(result.detections, key=lambda d: d.categories[0].score).bounding_box
    h, w = rgb.shape[:2]
    mx, my = int(b.width * CROP_MARGIN), int(b.height * CROP_MARGIN)
    return Image.fromarray(rgb[max(0, b.origin_y - my):min(h, b.origin_y + b.height + my),
                               max(0, b.origin_x - mx):min(w, b.origin_x + b.width + mx)])


def greedy_cluster(items: list[tuple[str, str, np.ndarray]], threshold: float):
    """items: [(ファイル名, 正解ラベル, ベクトル)]。順番に見て群れを作る。"""
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    members: list[list[tuple[str, str]]] = []
    for name, truth, vec in items:
        if centroids:
            sims = np.array([float(vec @ c) for c in centroids])
            best = int(sims.argmax())
            if sims[best] >= threshold:
                # 重心を件数で重み付けして更新する
                centroids[best] = unit(centroids[best] * counts[best] + vec)
                counts[best] += 1
                members[best].append((name, truth))
                continue
        centroids.append(vec.copy())
        counts.append(1)
        members.append([(name, truth)])
    return members


def score(members, n_people: int) -> tuple[int, float, int]:
    """クラスタ数・純度・正解ラベルが分裂した人数を返す。"""
    total = sum(len(m) for m in members)
    pure = sum(Counter(t for _, t in m).most_common(1)[0][1] for m in members)
    seen = Counter()
    for m in members:
        for label in {t for _, t in m}:
            seen[label] += 1
    split = sum(1 for label in seen if seen[label] > 1)
    return len(members), pure / total, split


def main() -> int:
    paths = sorted(p for p in IMAGE_DIR.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO_NAME).to(device).eval()

    face_opt = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, min_detection_confidence=0.5)
    seg_opt = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(SEG_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, output_confidence_masks=True)

    person_items, face_items = [], []
    with mp.tasks.vision.FaceDetector.create_from_options(face_opt) as detector, \
         mp.tasks.vision.ImageSegmenter.create_from_options(seg_opt) as segmenter:
        for path in paths:
            image = Image.open(path).convert("RGB")
            truth = path.stem.rsplit("_", 1)[0]
            patches = patches_of(model, image, device)
            cover = cover_16(segmenter, image)
            inside = patches[cover > 0.5]
            if len(inside):
                person_items.append((path.name, truth, unit(inside.mean(axis=0))))
            crop = face_crop(detector, np.asarray(image))
            if crop is not None:
                face_items.append((path.name, truth, unit(patches_of(model, crop, device).mean(axis=0))))

    truth_people = {t for _, t, _ in person_items}
    print(f"正解の人数: {len(truth_people)} 人 {sorted(truth_people)}")
    print(f"person ベクトル: {len(person_items)} 枚 / face ベクトル: {len(face_items)} 枚\n")

    for label, items in (("person（人物マスク内・服を含む）", person_items),
                         ("face（顔クロップ・服が入らない）", face_items)):
        print(f"########## {label} ##########")
        print(f"{'しきい値':>8} {'クラスタ数':>10} {'純度':>8} {'分裂した人数':>12}   判定")
        best = None
        for th in np.arange(0.60, 0.96, 0.025):
            members = greedy_cluster(items, float(th))
            n, purity, split = score(members, len(truth_people))
            perfect = n == len(truth_people) and purity == 1.0
            if perfect and best is None:
                best = (float(th), members)
            print(f"{th:>8.3f} {n:>10} {purity:>8.3f} {split:>12}   {'★完全一致' if perfect else ''}")
        if best:
            th, members = best
            print(f"\n  しきい値 {th:.3f} での群れ:")
            for i, m in enumerate(members):
                names = ", ".join(n for n, _ in m)
                print(f"    群れ{i + 1}: {names}")
        else:
            print("\n  完全一致するしきい値は無かった。")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
