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
