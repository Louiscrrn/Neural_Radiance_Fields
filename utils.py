import numpy as np
from PIL import Image
import cv2

def show_set_images(images, title="Images Viewer", delay=500):
    """
    Affiche les images dans une fenêtre OpenCV avec overlay texte.
    
    Contrôles :
    - Flèche gauche : image précédente
    - Flèche droite : image suivante
    - Espace : bascule lecture auto / pause
    - q : quitter
    
    Args:
        images (list): liste d'images (PIL ou NumPy).
        title (str): titre affiché en haut de la fenêtre.
        delay (int): délai entre les images en mode auto (ms).
    """
    images_np = [np.array(img) for img in images]
    idx = 0
    autoplay = False
    total = len(images_np)

    while True:
        # Copie de l'image pour ajouter du texte
        img_disp = cv2.cvtColor(images_np[idx], cv2.COLOR_RGB2BGR).copy()

        # Texte du haut (titre)
        cv2.putText(
            img_disp, title, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

        # Texte du bas (progression + autoplay)
        status = f"image {idx+1}/{total} - autoplay {'ON' if autoplay else 'OFF'}"
        h = img_disp.shape[0]
        cv2.putText(
            img_disp, status, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
        )

        # Affiche l'image
        cv2.imshow("Image Viewer", img_disp)

        if cv2.getWindowProperty("Image Viewer", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Mode autoplay → avance automatiquement
        if autoplay:
            key = cv2.waitKey(delay)
            idx = (idx + 1) % total
        else:
            key = cv2.waitKey(0)

        # Gestion des touches
        if key == ord("q"):
            break
        elif key == 2:  # flèche gauche
            idx = (idx - 1) % total
        elif key == 3:  # flèche droite
            idx = (idx + 1) % total
        elif key == 32:  # espace = toggle autoplay
            autoplay = not autoplay

    cv2.destroyAllWindows()


def load_images(file_path) :
    images_files = sorted([f for f in file_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    images = [Image.open(f) for f in images_files]
    return images

