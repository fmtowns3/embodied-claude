"""Stage 1（人物領域で絞る）を挟むと、当たりが変わるか。

patch_search.py は全パッチをいきなり顔重心と突き合わせていた。heishio の
test_twostage.py は2段階になっている:

    Stage 1: 全パッチ × 人物重心 → max の 70% 以上のパッチだけ残す
    Stage 2: 残ったパッチ × 各人の顔重心 → max で「誰か」を決める

人物重心は「特定の誰か」ではなく、全員のマスク内パッチを平均した
「人らしさ」の汎用ベクトル。髪や背景のパッチが顔重心に誤マッチする余地を
先に削るのが狙い。

Stage 1 あり／なしを同じ参照・同じ素材で比べる。

使い方:
    .venv\\Scripts\\python.exe stage2_search.py        # 単発224
    .venv\\Scripts\\python.exe stage2_search.py 3crop  # 3クロップ（Stage1は単発のみ）
"""

import sys
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
TOP_RATIO = 0.7  # heishio と同じ: 人物類似度が max の 70% 以上のパッチを残す

TO_TENSOR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def patches_of(model, image: Image.Image, device) -> np.ndarray:
    tensor = TO_TENSOR(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pixel_values=tensor)
    p = out.last_hidden_state[0, 1 + NUM_REGISTERS:].cpu().numpy()
    return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-10, None)


def person_mask_16(segmenter, image: Image.Image) -> np.ndarray:
    """人物マスクを 16x16 のパッチ格子に落とす。各セルの人物被覆率を返す。"""
    rgb = np.asarray(image)
    result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    mask = result.confidence_masks[0].numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask > 0.5).astype(np.float32)
    # パッチ格子に合わせて面積平均で縮小する（最近傍だと細い部分が消える）
    return cv2.resize(binary, (GRID, GRID), interpolation=cv2.INTER_AREA).reshape(-1)


def detect_face_crop(detector, rgb: np.ndarray):
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.detections:
        return None
    best = max(result.detections, key=lambda d: d.categories[0].score)
    b = best.bounding_box
    h, w = rgb.shape[:2]
    mx, my = int(b.width * CROP_MARGIN), int(b.height * CROP_MARGIN)
    return Image.fromarray(rgb[max(0, b.origin_y - my):min(h, b.origin_y + b.height + my),
                               max(0, b.origin_x - mx):min(w, b.origin_x + b.width + mx)])


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

    cache: dict[str, np.ndarray] = {}
    person_vecs = []
    face_src: dict[str, list[np.ndarray]] = {}
    detected: set[str] = set()

    with mp.tasks.vision.FaceDetector.create_from_options(face_opt) as detector, \
         mp.tasks.vision.ImageSegmenter.create_from_options(seg_opt) as segmenter:

        for path in paths:
            image = Image.open(path).convert("RGB")
            patches = patches_of(model, image, device)
            cover = person_mask_16(segmenter, image)
            cache[path.name] = patches
            # 人物重心の材料：マスク内パッチの平均
            inside = patches[cover > 0.5]
            if len(inside):
                v = inside.mean(axis=0)
                person_vecs.append(v / np.linalg.norm(v))
            # 顔参照の材料：顔検出できた画像の顔クロップ
            crop = detect_face_crop(detector, np.asarray(image))
            if crop is not None:
                detected.add(path.name)
                face_src.setdefault(path.stem.rsplit("_", 1)[0], []).append(
                    patches_of(model, crop, device).mean(axis=0))

    ref_person = np.mean(person_vecs, axis=0)
    ref_person /= np.linalg.norm(ref_person)
    refs = {p: v / np.linalg.norm(v)
            for p, v in ((p, np.mean(vs, axis=0)) for p, vs in face_src.items())}
    people = sorted(refs)

    print(f"人物重心の材料: {len(person_vecs)} 枚 / 顔参照: {people}")
    print(f"顔検出に失敗: {sorted(p.name for p in paths if p.name not in detected)}\n")

    summary = {}
    for mode in ("Stage1 なし（全パッチ）", "Stage1 あり（人物領域で絞る）"):
        hit = 0
        rows = []
        for path in paths:
            patches = cache[path.name]
            used = patches
            n_used = len(patches)
            if mode.startswith("Stage1 あり"):
                psims = patches @ ref_person
                keep = psims >= psims.max() * TOP_RATIO
                used = patches[keep]
                n_used = int(keep.sum())
            scores = {p: float((used @ refs[p]).max()) for p in people}
            guess = max(scores, key=scores.get)
            truth = path.stem.rsplit("_", 1)[0]
            hit += guess == truth
            rows.append((path.name, truth, scores, guess, n_used))
        summary[mode] = rows

        print(f"=== {mode} ===  正解 {hit} / {len(rows)}")
        head = f"{'file':<20} {'正解':<11} " + " ".join(f"{p:>10}" for p in people) + f" {'使用P':>6}  結果"
        print(head)
        for name, truth, scores, guess, n_used in rows:
            cells = " ".join(f"{scores[p]:>10.4f}" for p in people)
            flag = "" if name in detected else "  ←検出失敗"
            print(f"{name:<20} {truth:<11} {cells} {n_used:>6}  {'○' if guess == truth else '× ' + guess}{flag}")
        print()

    print("=== Stage1 で動いた差（正解と2位の差）===")
    a = summary["Stage1 なし（全パッチ）"]
    b = summary["Stage1 あり（人物領域で絞る）"]
    improved = worsened = 0
    for (name, truth, sa, _, _), (_, _, sb, _, n_used) in zip(a, b):
        others = [p for p in sa if p != truth]
        ga = sa[truth] - max(sa[p] for p in others)
        gb = sb[truth] - max(sb[p] for p in others)
        improved += gb > ga
        worsened += gb < ga
        arrow = "改善" if gb > ga else ("悪化" if gb < ga else "同じ")
        print(f"  {name:<20} {ga:+.4f} → {gb:+.4f}  ({arrow}, 使用パッチ {n_used:>3}/{GRID*GRID})")
    print(f"\n  改善 {improved} 件 / 悪化 {worsened} 件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
