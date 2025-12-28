# Conditional GAN on MNIST — training script

# - Uses only the 60k MNIST training split (unsupervised/generative setting).
# - Generator (G) is conditioned on class labels via an embedding.
# - Discriminator (D) receives image + a per-pixel label mask channel.
# - Losses are Binary Cross-Entropy from logits (D outputs logits, not probs).

import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Reproducibility: fix random seeds (helps make runs comparable)
SEED = 504
np.random.seed(SEED); tf.random.set_seed(SEED)

# Core hyperparameters
z_dim = 100                      # latent vector size for G
num_classes = 10                 # digits 0..9
img_rows, img_cols, channels = 28, 28, 1
batch_size = 128                
epochs = 30                      # enough to reach crisp digits on MNIST
lr_g = 2e-4; lr_d = 2e-4; beta_1 = 0.5  # Adam settings commonly used in GANs

# Output folders (models for saved nets, outputs for figures)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Data
# Load only the training split as required by the project 
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

# Scale images to [-1, 1] so they match the generator's tanh output range
x_train = (x_train.astype("float32")/127.5) - 1.0
x_train = np.expand_dims(x_train, -1)        # -> (N, 28, 28, 1)
y_train = y_train.astype("int32")            # labels kept as int32 for Embedding

# Build a simple tf.data pipeline (shuffle each epoch, fixed batch size)
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(60000, seed=SEED).batch(batch_size, drop_remainder=True))


# Models
def build_generator(z_dim=100, num_classes=10):
    # Conditional generator: inputs are noise z and class label y
    noise = layers.Input(shape=(z_dim,), name="noise")
    label = layers.Input(shape=(1,), dtype="int32", name="class_label")

    # Class embedding -> flatten -> concat with noise (standard cGAN trick)
    emb = layers.Embedding(num_classes, 50)(label) 
    emb = layers.Flatten()(emb)
    x = layers.Concatenate()([noise, emb])

    # Project + reshape to a low-res feature map (7x7)
    x = layers.Dense(7*7*256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((7,7,256))(x)

    # Two stride-2 upsampling stages to reach 28x28
    x = layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Final 28x28x1 image with tanh activation in [-1,1]
    out = layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same",
                                 use_bias=False, activation="tanh")(x)
    return models.Model([noise, label], out, name="Generator")

def build_discriminator(num_classes=10):
    # Conditional discriminator: inputs are image x and class label y
    img = layers.Input(shape=(img_rows, img_cols, channels), name="img")
    label = layers.Input(shape=(1,), dtype="int32", name="class_label")

    # Embed the label into a 28*28 mask and concatenate as an extra channel
    emb = layers.Embedding(num_classes, img_rows*img_cols)(label)
    emb = layers.Flatten()(emb)
    emb = layers.Reshape((img_rows, img_cols, 1))(emb)
    x = layers.Concatenate(axis=-1)([img, emb])  # (28,28,2) => image + label mask

    # Downsampling conv blocks (typical DCGAN-style D)
    x = layers.Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Flatten -> single logit for real/fake (BCE uses from_logits=True)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)  # from_logits=True
    return models.Model([img, label], out, name="Discriminator")

# Instantiate G and D
generator = build_generator(z_dim, num_classes)
discriminator = build_discriminator(num_classes)

# Loss/optimizers (BCE expects logits; Adam with beta_1=0.5 is common in GANs)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(lr_g, beta_1=beta_1)
d_opt = tf.keras.optimizers.Adam(lr_d, beta_1=beta_1)

@tf.function
def train_step(real_imgs, real_labels):
    # One adversarial step:
    # 1) update D to classify real as 1 and fake as 0.
    # 2) Update G to make D classify fake as 1.

    b = tf.shape(real_imgs)[0]

    # Discriminator update 
    # Sample noise and random labels to form a fake batch
    noise = tf.random.normal([b, z_dim])
    fake_labels = tf.random.uniform([b, 1], minval=0, maxval=num_classes, dtype=tf.int32)
    with tf.GradientTape() as d_tape:
        # Generate fake images and compute D logits on real and fake
        fake_imgs = generator([noise, fake_labels], training=True)
        real_logits = discriminator([real_imgs, tf.expand_dims(real_labels, -1)], training=True)
        fake_logits = discriminator([fake_imgs, fake_labels], training=True)

        # D wants real->1, fake->0 (BCE from logits)
        d_loss = bce(tf.ones_like(real_logits), real_logits) + \
                 bce(tf.zeros_like(fake_logits), fake_logits)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # Generator update
    # new noise + labels so G can't overfit to the previous batch
    noise = tf.random.normal([b, z_dim])
    sampled_labels = tf.random.uniform([b, 1], minval=0, maxval=num_classes, dtype=tf.int32)
    with tf.GradientTape() as g_tape:
        gen_imgs = generator([noise, sampled_labels], training=True)
        fake_logits = discriminator([gen_imgs, sampled_labels], training=True)

        # G wants D to output 1 on fake (i.e., fool D)
        g_loss = bce(tf.ones_like(fake_logits), fake_logits)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    return d_loss, g_loss

def save_grid(images, path, n=5):
    # utility: save an n×n grid of images; expects inputs in [-1,1]
    images = (images + 1.0) / 2.0        # back to [0,1] for plotting
    h, w = img_rows, img_cols
    canvas = np.zeros((n*h, n*w))
    for i in range(n):
        for j in range(n):
            canvas[i*h:(i+1)*h, j*w:(j+1)*w] = images[i*n+j, :, :, 0]
    plt.figure(figsize=(4,4)); plt.axis("off")
    plt.imshow(canvas, cmap="gray"); plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0); plt.close()


# Training loop

hist_d, hist_g = [], []
for epoch in range(1, epochs+1):
    # Iterate over the whole training set once per epoch
    for real_imgs, real_labels in train_dataset:
        d_loss, g_loss = train_step(real_imgs, real_labels)

    # Track epoch-end losses (last mini-batch is representative for MNIST)
    hist_d.append(float(d_loss)); hist_g.append(float(g_loss))
    print(f"Epoch {epoch:03d}/{epochs} | D_loss={float(d_loss):.4f} | G_loss={float(g_loss):.4f}")

    # Produce a fixed 5×5 grid covering digits 0..9 (helps monitor quality over time)
    labels_5x5 = np.array([0,1,2,3,4, 5,6,7,8,9, 0,1,2,3,4, 5,6,7,8,9, 0,1,2,3,4], dtype=np.int32).reshape(-1,1)
    z = np.random.normal(size=(25, z_dim)).astype("float32")
    gen = generator.predict([z, labels_5x5], verbose=0)
    save_grid(gen, os.path.join("outputs", f"generated_epoch_{epoch:03d}.png"), n=5)


# Final Artifacts
# final grid with a fresh noise sample (same label pattern)
z = np.random.normal(size=(25, z_dim)).astype("float32")
labels_5x5 = np.array([0,1,2,3,4, 5,6,7,8,9, 0,1,2,3,4, 5,6,7,8,9, 0,1,2,3,4], dtype=np.int32).reshape(-1,1)
gen_final = generator.predict([z, labels_5x5], verbose=0)
save_grid(gen_final, os.path.join("outputs", "generated.png"), n=5)

# Plot the training losses 
plt.figure()
plt.plot(hist_d, label="Discriminator")
plt.plot(hist_g, label="Generator")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join("outputs", "loss_plot.png")); plt.close()

# Save generator architecture and weights
#  In Keras 3, save_weights typically requires ".weights.h5".
with open(os.path.join("models", "generator.json"), "w") as f:
    f.write(generator.to_json())
generator.save_weights(os.path.join("models", "generator.h5"))

# Save the noise/labels used for the last grid (useful for deterministic re-eval)
np.save(os.path.join("outputs", "z_eval.npy"), z)
np.save(os.path.join("outputs", "labels_eval.npy"), labels_5x5)
print("Training complete. Artifacts saved to ./models and ./outputs")
