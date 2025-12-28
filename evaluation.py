# reload generator, re-save weights (Keras3), and export 5x5 grid

# Purpose:
#   - Load the saved generator architecture and weights (no training).
#   - Ensure weights are saved under the Keras 3–compliant filename:
#         "generator.weights.h5"
#     while also keeping a compatibility copy:
#         "generator.h5"
#   - Generate a 5×5 conditional sample grid for quick visual verification.
# Notes:
#   - This script is safe to run repeatedly; it does not alter the model graph.


import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models

# Make sampling deterministic for reproducibility (same z/labels -> same grid)
np.random.seed(504); tf.random.set_seed(504)
z_dim = 100; img_rows=28; img_cols=28

# I/O locations
MODELS_DIR = "models"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)  # create outputs/ if missing

# 1) Load architecture
#    The training script saved the generator structure as a JSON string.
with open(os.path.join(MODELS_DIR, "generator.json"), "r") as f:
    generator = models.model_from_json(f.read())

# 2) Load existing weights 
#    Prefer the Keras3-compliant filename if present, otherwise fall back to legacy.
w_new = os.path.join(MODELS_DIR, "generator.weights.h5")
w_old = os.path.join(MODELS_DIR, "generator.h5")
if os.path.exists(w_new):
    generator.load_weights(w_new)
elif os.path.exists(w_old):
    generator.load_weights(w_old)
else:
    # If neither file is found, the training step likely hasn't produced weights.
    raise FileNotFoundError("Could not find existing weights: generator.weights.h5 or generator.h5")

# 3) Re-save weights with Keras 3 convention
#    This guarantees the canonical filename while preserving the legacy copy.
generator.save_weights(w_new)
# Optional: also copy to legacy name for compatibility 
shutil.copyfile(w_new, w_old)
print("✅ Re-saved to:", w_new, "and copied to:", w_old)

# 4) Generate a 5×5 sample grid 
#    Label pattern covers digits 0..9 across rows to visually confirm conditioning.
labels_5x5 = np.array([0,1,2,3,4, 5,6,7,8,9,
                       0,1,2,3,4, 5,6,7,8,9,
                       0,1,2,3,4], dtype=np.int32).reshape(-1,1)

# Sample a fresh latent batch (25 vectors) and synthesize images
z = np.random.normal(size=(25, z_dim)).astype("float32")
gen = generator.predict([z, labels_5x5], verbose=0).astype("float32")

def save_grid(images, path, n=5):
    """Save an n×n image grid to disk; expects images scaled in [-1, 1]."""
    # Convert from [-1,1] back to [0,1] for display
    images = (images + 1.0) / 2.0
    h, w = img_rows, img_cols

    # Pre-allocate a blank canvas and tile images row-by-row
    canvas = np.zeros((n*h, n*w))
    for i in range(n):
        for j in range(n):
            canvas[i*h:(i+1)*h, j*w:(j+1)*w] = images[i*n+j, :, :, 0]

    # Render and save without axes or padding borders
    plt.figure(figsize=(4,4)); plt.axis("off"); plt.imshow(canvas, cmap="gray"); plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0); plt.close()

# Write the evaluation grid for quick manual inspection
save_grid(gen, os.path.join(OUT_DIR, "eval_generated.png"), n=5)
print("Saved outputs/eval_generated.png")
