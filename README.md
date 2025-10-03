# NeRF Implementation (PyTorch)

This project is a clean PyTorch implementation of a Neural Radiance Field (NeRF) 
trained and tested on the Lego dataset (Blender). Includes positional encoding, hierarchical sampling, and volumetric rendering.

## 📂 Project structure
- `config.yaml` : training and dataset configuration (image size, rays per batch, epochs, etc.)
- `dataset.py` : PyTorch `Dataset` + `collate_rays` to sample rays
- `main.py` : entry point for training
- `model.py` : NeRF model, positional encoding, sampling & volume rendering
- `train.py` : `NeRFTrainer` class with training loop
- `visualization.py` : functions to render full images from the trained NeRF model
- `utils.py` : helper functions

## ⚙️ Requirements
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 🚀 Training

Set your dataset path in `.env`:
```bash
echo "DATA_PATH=/path/to/nerf_synthetic" > .env
```

Configure parameters in `config.yaml` (`img_size`, `N_rays`, `lr`, `epochs`, etc.)

**Run training:**
```bash
python main.py
```

## 📊 Notes

- Work in progress

---

✅ Inspired by the original NeRF paper: *Mildenhall et al., ECCV 2020*
