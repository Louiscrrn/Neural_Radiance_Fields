import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os


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
        img_disp = cv2.cvtColor(images_np[idx], cv2.COLOR_RGB2BGR).copy()

        cv2.putText(
            img_disp, title, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

        status = f"image {idx+1}/{total} - autoplay {'ON' if autoplay else 'OFF'}"
        h = img_disp.shape[0]
        cv2.putText(
            img_disp, status, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow("Image Viewer", img_disp)

        if cv2.getWindowProperty("Image Viewer", cv2.WND_PROP_VISIBLE) < 1:
            break

        if autoplay:
            key = cv2.waitKey(delay)
            idx = (idx + 1) % total
        else:
            key = cv2.waitKey(0)

        if key == ord("q"):
            break
        elif key == 2:  
            idx = (idx - 1) % total
        elif key == 3:  
            idx = (idx + 1) % total
        elif key == 32:  
            autoplay = not autoplay

    cv2.destroyAllWindows()


def load_images(file_path) :
    images_files = sorted([f for f in file_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    images = [Image.open(f) for f in images_files]
    return images



def plot_training_curves(history, save_dir=None):
    """
    Trace 3 figures séparées (Loss, PSNR, SSIM),
    chacune avec 2 sous-graphiques (train / val).

    Args:
        history (pd.DataFrame): doit contenir les colonnes :
            - epoch
            - train_loss, train_psnr, train_ssim
            - val_loss, val_psnr, val_ssim (facultatif)
        save_dir (str, optional): dossier où sauvegarder les figures PNG
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    epochs = history['epoch']

    # ======================== LOSS =========================
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig1.suptitle('MSE Loss')

    # Train Loss
    axes[0].plot(epochs, history['train_loss'], marker='o', color='blue')
    axes[0].set_title('Train')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid()

    # Val Loss
    if 'val_loss' in history and history['val_loss'].notna().any():
        axes[1].plot(epochs, history['val_loss'], marker='o', color='orange')
        axes[1].set_title('Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss')
        axes[1].grid()
    else:
        axes[1].set_visible(False)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        fig1.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300)
    plt.show()

    # ======================== PSNR =========================
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle('PSNR')

    axes[0].plot(epochs, history['train_psnr'], marker='o', color='blue')
    axes[0].set_title('Train')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('dB')
    axes[0].grid()

    if 'val_psnr' in history and history['val_psnr'].notna().any():
        axes[1].plot(epochs, history['val_psnr'], marker='o', color='orange')
        axes[1].set_title('Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('dB')
        axes[1].grid()
    else:
        axes[1].set_visible(False)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        fig2.savefig(os.path.join(save_dir, 'psnr_curves.png'), dpi=300)
    plt.show()

    # ======================== SSIM =========================
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle('SSIM')

    axes[0].plot(epochs, history['train_ssim'], marker='o', color='blue')
    axes[0].set_title('Train')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Structural Similarity')
    axes[0].grid()

    if 'val_ssim' in history and history['val_ssim'].notna().any():
        axes[1].plot(epochs, history['val_ssim'], marker='o', color='orange')
        axes[1].set_title('Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Structural Similarity')
        axes[1].grid()
    else:
        axes[1].set_visible(False)

    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        fig3.savefig(os.path.join(save_dir, 'ssim_curves.png'), dpi=300)
    plt.show()

    return fig1, fig2, fig3
