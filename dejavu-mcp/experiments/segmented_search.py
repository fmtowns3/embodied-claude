"""背景をグレーで塗ってから探すと、当たりが変わるか。

patch_search.py は画像全体のパッチを顔重心と突き合わせていた。人物A_2 が
人物B に 0.0079 差で負けたのは、髪や背景のパッチが効いた可能性がある。

heishio 版の server.py は embed の前に必ず前処理を挟んでいた:
  - MediaPipe selfie_segmenter で人物マスクを取る
  - 人物以外をニュートラルグレー(128,128,128)で塗る（person ベクトル用）
  - LAB の L チャンネルだけ平行移動し、人物領域の平均輝度を 128 に揃える

素材が白背景・均一照明なので輝度正規化はほぼ効かないはずだが、
背景を消す効果は出るかもしれない。前処理あり／なしを同じ土俵で比べる。

使い方:
    .venv\\Scripts\\python.exe segmented_search.py        # 単発224
    .venv\\Scripts\\python.exe segmented_search.py 3crop  # 3クロップ
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
NEUTRAL_GRAY = 128
TARGET_L = 128
USE_3CROP = len(sys.argv) > 1 and sys.argv[1] == "3crop"

TO_TENSOR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def normalize_brightness(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """人物領域の平均輝度を TARGET_L に揃える。L チャンネルだけ平行移動する。"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    person_l = lab[:, :, 0][mask > 0]
    if person_l.size == 0:
        return rgb
    lab[:, :, 0] = np.clip(lab[:, :, 0] + (TARGET_L - person_l.mean()), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def foreground_of(segmenter, image: Image.Image) -> tuple[Image.Image, float]:
    """人物以外をグレーで塗り、輝度を正規化した画像を返す。"""
    rgb = np.asarray(image)
    result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    mask = result.confidence_masks[0].numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask > 0.5).astype(np.uint8)
    ratio = float(binary.sum() / binary.size)
    if ratio == 0.0:
        return image, 0.0
    normed = normalize_brightness(rgb, binary)
    fill = np.full_like(rgb, NEUTRAL_GRAY)
    fg = np.where(np.dstack([binary] * 3), normed, fill)
    return Image.fromarray(fg), ratio


def patches_of(model, image: Image.Image, device) -> np.ndarray:
    tensor = TO_TENSOR(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pixel_values=tensor)
    p = out.last_hidden_state[0, 1 + NUM_REGISTERS:].cpu().numpy()
    return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-10, None)


def three_crop_patches(model, image: Image.Image, device, overlap: float = 0.25) -> np.ndarray:
    w, h = image.size
    crop_w = w // 2
    step = int(crop_w * (1 - overlap))
    batch = torch.stack([
        TO_TENSOR(image.crop((x, 0, x + crop_w, h)))
        for x in (0, step, max(0, w - crop_w))
    ]).to(device)
    with torch.no_grad():
        out = model(pixel_values=batch)
    grids = out.last_hidden_state[:, 1 + NUM_REGISTERS:].cpu().numpy().reshape(3, GRID, GRID, -1)
    unique = GRID - 4
    acc = np.zeros((GRID, GRID + 2 * unique, grids.shape[-1]), dtype=np.float32)
    cnt = np.zeros((GRID, GRID + 2 * unique, 1), dtype=np.float32)
    for i, start in enumerate((0, unique, 2 * unique)):
        acc[:, start:start + GRID] += grids[i]
        cnt[:, start:start + GRID] += 1.0
    p = (acc / cnt).reshape(-1, grids.shape[-1])
    return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-10, None)


def query_patches(model, image, device):
    return three_crop_patches(model, image, device) if USE_3CROP else patches_of(model, image, device)


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

    print(f"問い合わせ方式: {'3クロップ' if USE_3CROP else '単発224'}  /  前処理: 背景グレー塗り + 輝度正規化\n")

    results = {}
    with mp.tasks.vision.FaceDetector.create_from_options(face_opt) as detector, \
         mp.tasks.vision.ImageSegmenter.create_from_options(seg_opt) as segmenter:

        # 参照は「素の画像から顔検出したクロップ」で作る（前回と同条件）
        refs_src: dict[str, list[np.ndarray]] = {}
        detected: set[str] = set()
        for path in paths:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            crop = detect_face_crop(detector, rgb)
            if crop is None:
                continue
            detected.add(path.name)
            person = path.stem.rsplit("_", 1)[0]
            refs_src.setdefault(person, []).append(patches_of(model, crop, device).mean(axis=0))
        refs = {p: v / np.linalg.norm(v) for p, v in
                ((p, np.mean(vs, axis=0)) for p, vs in refs_src.items())}

        # 問い合わせ側だけ前処理あり／なしで比較する
        for mode in ("前処理なし", "前処理あり"):
            people = sorted(refs)
            hit = 0
            rows = []
            for path in paths:
                image = Image.open(path).convert("RGB")
                ratio = 1.0
                if mode == "前処理あり":
                    image, ratio = foreground_of(segmenter, image)
                patches = query_patches(model, image, device)
                scores = {p: float((patches @ refs[p]).max()) for p in people}
                guess = max(scores, key=scores.get)
                truth = path.stem.rsplit("_", 1)[0]
                hit += guess == truth
                rows.append((path.name, truth, scores, guess, ratio))
            results[mode] = (hit, rows)

            print(f"=== {mode} ===")
            head = f"{'file':<20} {'正解':<11} " + " ".join(f"{p:>11}" for p in people) + f" {'人物率':>7}  結果"
            print(head)
            for name, truth, scores, guess, ratio in rows:
                cells = " ".join(f"{scores[p]:>11.4f}" for p in people)
                flag = "" if name in detected else "  ←検出失敗"
                print(f"{name:<20} {truth:<11} {cells} {ratio:>7.3f}  {'○' if guess == truth else '× ' + guess}{flag}")
            print(f"  正解 {hit} / {len(rows)}\n")

    # 差分だけ抜き出す
    print("=== 前処理で動いた差 ===")
    _, before = results["前処理なし"]
    _, after = results["前処理あり"]
    for (name, truth, sb, gb, _), (_, _, sa, ga, ratio) in zip(before, after):
        others = [p for p in sb if p != truth]
        gap_b = sb[truth] - max(sb[p] for p in others)
        gap_a = sa[truth] - max(sa[p] for p in others)
        arrow = "改善" if gap_a > gap_b else ("悪化" if gap_a < gap_b else "変化なし")
        print(f"  {name:<20} 正解との差 {gap_b:+.4f} → {gap_a:+.4f}  ({arrow}, 人物率 {ratio:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
