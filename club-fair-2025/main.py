# quickdraw_cnn_pipeline.py
# ---------------------------------------------------------------
# Train a 28×28 CNN on 4 doodle classes.
# Step 1: Pull 500 samples/class from Quick Draw! (first run only)
# Step 2: Train the model using those PNGs.
# ---------------------------------------------------------------
import os, pathlib, random, sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, callbacks
from PIL import Image

# --------------------------- CONFIG ----------------------------
CLASS_LABELS            = sorted(["flower", "car", "snowman", "fish"])   # keep length = 4
NUM_SAMPLES_PER_CLASS   = 500
ROOT_DIR                = "quickdraw_4cls"                       # change if you like
IMG_SIZE                = (28, 28)
BATCH_SIZE              = 128
EPOCHS                  = 100
SEED                    = 1337

# ------------------ STEP 2: DATASET LOADING --------------------
def get_datasets(root_dir=ROOT_DIR, val_split=0.2, batch_size=BATCH_SIZE):
    """
    Returns train_ds, val_ds from the PNG folders.
    Uses Keras' built-in 80/20 split via validation_split parameter.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(root_dir, "train"),
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=val_split,
        subset="training"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(root_dir, "train"),
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=val_split,
        subset="validation"
    )
    # Normalize to 0-1 floats
    norm = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    return train_ds.map(norm).prefetch(tf.data.AUTOTUNE), \
           val_ds.map(norm).prefetch(tf.data.AUTOTUNE)

# ------------------- STEP 3: BUILD THE CNN ---------------------
def make_model():
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (1,)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_LABELS), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --------------------------- MAIN ------------------------------
def main():
    global tf

    train_ds, val_ds = get_datasets()                 # load PNGs
    model = make_model()
    model.summary()

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,            # stop after 8 epochs of no val_loss drop
        restore_best_weights=True,
        verbose=1)

    rlr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1)
    
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS,
              callbacks=[ckpt, rlr])

    # ---------------------------------------------------------------
    #  Evaluate on the validation set: confusion matrix + report
    # ---------------------------------------------------------------
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import tensorflow as tf

    # 1) Load the best model on disk (safer than relying on in-memory weights)
    best_model = tf.keras.models.load_model("best_model.keras", compile=False)

    # 2) Gather ALL validation images & labels into numpy arrays
    val_images, val_labels = [], []
    for batch_imgs, batch_labs in val_ds.unbatch():        # unbatch = 1 element at a time
        val_images.append(batch_imgs.numpy())
        val_labels.append(batch_labs.numpy())
    val_images = np.stack(val_images)                      # shape (N, 28, 28, 1)
    val_labels = np.array(val_labels)                      # shape (N,)

    # 3) Predict and build the confusion matrix
    pred_probs  = best_model.predict(val_images, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    cm = confusion_matrix(val_labels, pred_labels, labels=range(len(CLASS_LABELS)))
    print("Confusion matrix (rows = true, cols = pred):\n", cm)

    print("\nDetailed metrics:\n",
        classification_report(val_labels, pred_labels,
                                target_names=CLASS_LABELS, digits=3))

    # 4) Nicely visualise the matrix (optional)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(CLASS_LABELS)), CLASS_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_ylabel("True label");  ax.set_xlabel("Predicted label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.show()


    model.save("final_model.keras")
    print("✓ Training complete — model saved to final_model.keras")

if __name__ == "__main__":
    main()
