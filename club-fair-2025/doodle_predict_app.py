# doodle_demo.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np, tensorflow as tf

CLASS_LABELS = ["car", "fish", "flower", "snowman"]
MODEL_PATH   = "final_model.keras"      # load the checkpoint you like
IMG_SIZE     = (28, 28)

@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

model = load_model(MODEL_PATH)

st.set_page_config(page_title="Doodle Classifier", layout="centered")
st.title("🎨 Draw a doodle and click **Predict**")

# --- canvas ---
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=9,
    stroke_color="#000000",
    background_color="#FFFFFF",        # white bg, black ink  ⇒ matches training
    height=280, width=280,
    drawing_mode="freedraw",
    key="canvas"
)

col1, col2 = st.columns([1, 2])

def preprocess_canvas(img_arr):
    """Take RGBA numpy array (280×280×4) → 1×28×28×1 float tensor"""
    img = Image.fromarray(img_arr).convert("L")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1)

if col1.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Draw something first!")
    else:
        x = preprocess_canvas(canvas_result.image_data)
        probs = model.predict(x, verbose=0)[0]
        top = int(np.argmax(probs))

        col2.markdown(f"### **{CLASS_LABELS[top].capitalize()}**  ({probs[top]:.1%})")
        st.bar_chart(
            {cls: float(p) for cls, p in zip(CLASS_LABELS, probs)},
            use_container_width=True
        )

if col1.button("Clear"):
    st.experimental_rerun()
