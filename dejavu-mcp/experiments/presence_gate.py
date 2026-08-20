"""「人がいる／いない」の門番は成立するか。

stage2_search.py で分かったのは、Stage 1 のパッチ絞り込みは max 検索の
結果を1つも変えないということ。ならば Stage 1 の本当の役割は
heishio のコードにある通り、

    person_likely = delta_max >= DELTA_THRESHOLD

＝ 人がいないフレームを顔照合まで進めずに捨てること。10秒ごとに撮り続けるなら
ほとんどのフレームに人はいないので、ここで落とせるかが実用性を決める。

人物なし画像をわざわざ用意しなくても測れる。同じ画像の中で
  人物マスクの内側のパッチ  … 「人がいる」フレーム相当
  人物マスクの外側のパッチ  … 「人がいない」フレーム相当
に分けて、それぞれの max を比べればよい。両者が分離していれば門番は作れる。

使い方:
    .venv\\Scripts\\python.exe presence_gate.py
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
SEG_MODEL = ROOT / "models" / "selfie_segmenter.tflite"
DINO_NAME = "facebook/dinov2-with-registers-base"
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


def cover_16(segmenter, image) -> np.ndarray:
    rgb = np.asarray(image)
    result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    mask = result.confidence_masks[0].numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    binary = (mask > 0.5).astype(np.float32)
    return cv2.resize(binary, (GRID, GRID), interpolation=cv2.INTER_AREA).reshape(-1)


def main() -> int:
    paths = sorted(p for p in IMAGE_DIR.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    seg_opt = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(SEG_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, output_confidence_masks=True)

    data = []
    with mp.tasks.vision.ImageSegmenter.create_from_options(seg_opt) as segmenter:
        for path in paths:
            image = Image.open(path).convert("RGB")
            data.append((path.name, patches_of(model, image, device), cover_16(segmenter, image)))

    # 人物重心は「マスク内側パッチの平均」を全枚数で平均したもの
    vecs = []
    for _, patches, cover in data:
        inside = patches[cover > 0.5]
        if len(inside):
            v = inside.mean(axis=0)
            vecs.append(v / np.linalg.norm(v))
    ref_person = np.mean(vecs, axis=0)
    ref_person /= np.linalg.norm(ref_person)
    print(f"人物重心の材料: {len(vecs)} 枚\n")

    print(f"{'file':<20} {'内側P':>5} {'外側P':>5} {'内側max':>9} {'外側max':>9} {'差':>9}")
    print("-" * 62)
    inside_maxes, outside_maxes = [], []
    for name, patches, cover in data:
        # 境界パッチ（人物と背景が混ざるセル）は両方から外す。
        # cover 0.5 で切ると、輪郭のセルが「外側」に入りながら人物のピクセルを含み、
        # 外側 max を押し上げてしまう。
        ins, outs = patches[cover >= 0.9], patches[cover <= 0.1]
        if not len(ins) or not len(outs):
            print(f"{name:<20} 片側が空なのでスキップ（内側 {len(ins)} / 外側 {len(outs)}）")
            continue
        i_max = float((ins @ ref_person).max())
        o_max = float((outs @ ref_person).max())
        inside_maxes.append(i_max)
        outside_maxes.append(o_max)
        print(f"{name:<20} {len(ins):>5} {len(outs):>5} {i_max:>9.4f} {o_max:>9.4f} {i_max - o_max:>+9.4f}")

    lo_in, hi_out = min(inside_maxes), max(outside_maxes)
    print(f"\n  人がいる側  最小 {lo_in:.4f} / 平均 {np.mean(inside_maxes):.4f}")
    print(f"  いない側    最大 {hi_out:.4f} / 平均 {np.mean(outside_maxes):.4f}")
    margin = lo_in - hi_out
    if margin > 0:
        print(f"  ★分離できている。マージン {margin:+.4f}")
        print(f"    しきい値は {hi_out:.4f} 〜 {lo_in:.4f} の間に置ける（中点 {(lo_in + hi_out) / 2:.4f}）")
    else:
        print(f"  ★重なっている。マージン {margin:+.4f} → 単一しきい値では門番にならない")
    print(f"  参考: heishio の DELTA_THRESHOLD = 0.35")
    return 0


if __name__ == "__main__":
    sys.exit(main())
