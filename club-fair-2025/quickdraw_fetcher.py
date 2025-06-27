# quickdraw_fetcher.py  – drop-in replacement for download_quickdraw()
import os, pathlib
from tqdm import tqdm
from PIL import Image
from quickdraw import QuickDrawDataGroup          # ← small on-demand API :contentReference[oaicite:1]{index=1}

CLASS_LABELS          = ["flower", "car", "snowman", "fish"]
NUM_SAMPLES_PER_CLASS = 2000
ROOT_DIR              = "quickdraw_4cls"
IMG_SIZE              = (28, 28)                  # same as your CNN

def download_quickdraw_light(root_dir=ROOT_DIR,
                             n_per_class=NUM_SAMPLES_PER_CLASS,
                             recognized=True):
    """
    Grab n_per_class 28×28 PNG bitmaps for each class and save to
    root_dir/train/<class>/...   (~50 kB per image, ~13 MB per class).
    """
    root = pathlib.Path(root_dir, "train")
    root.mkdir(parents=True, exist_ok=True)

    for cls in CLASS_LABELS:
        out_dir = root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(out_dir.glob("*.png")))
        if existing >= n_per_class:
            print(f"✓ {cls}: already have {existing} images")
            continue

        print(f"⤵  Fetching {n_per_class-existing} “{cls}” drawings …")
        qdg = QuickDrawDataGroup(cls, max_drawings=None, recognized=recognized)
        saved = existing

        for drawing in tqdm(qdg.drawings):
            if saved >= n_per_class:
                break
            img = drawing.get_image(stroke_width=2)   # 255×255 PIL
            img = img.resize(IMG_SIZE).convert("L")   # 28×28 grayscale
            img.save(out_dir / f"{cls}_{saved:04d}.png")
            saved += 1

        print(f"✓ {cls}: total {saved} images ready")

if __name__ == "__main__":
    download_quickdraw_light()
