import yaml
from pathlib import Path
from utils import load_images, show_set_images
import os
from dotenv import load_dotenv


if __name__ == "__main__":

    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_path = data_path / Path(config["path"]["train"])    
    val_path = data_path / Path(config["path"]["val"])    
    test_path = data_path / Path(config["path"]["test"])    

    train_images = load_images(train_path)
    val_images = load_images(val_path)
    test_images = load_images(test_path)

    show_set_images(train_images, title="Train Images")
    show_set_images(val_images, title="Val Images")
    show_set_images(test_images, title="Test Images")



