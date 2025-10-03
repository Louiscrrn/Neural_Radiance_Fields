import argparse
from pathlib import Path
from utils import load_images, show_set_images
from PIL import Image
import numpy as np

if __name__ == "__main__":

    path = "outputs/lego_orbit/renders/lego/view_058.png"

    img = Image.open(path)
    print(type(img))         # <class 'PIL.PngImagePlugin.PngImageFile'>
    print(img.size)          # (width, height)
    print(img.mode)  
    print(np.all(np.array(img) == 0)
)

    """parser = argparse.ArgumentParser(description="Visualisation d'un ensemble d'images")
    parser.add_argument(
        "path",
        type=str,
        help="Chemin vers le dossier contenant les images (png/jpg)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Image Viewer",
        help="Titre affiché dans la fenêtre de visualisation"
    )
    args = parser.parse_args()

    img_path = Path(args.path)
    if not img_path.exists() or not img_path.is_dir():
        raise FileNotFoundError(f"Le dossier {img_path} n'existe pas ou n'est pas un dossier valide")

    images = load_images(img_path)
    if len(images) == 0:
        raise RuntimeError(f"Aucun fichier image trouvé dans {img_path}")

    print(f"{len(images)} images chargées depuis {img_path}")

    img = images[0]

    print(img.shape)
    print(img)

    show_set_images(images, title=args.title)
"""