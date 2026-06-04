import os
import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_a.keras")

_model = None

def _load_model():
    global _model
    if _model is None:
        from keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model

def resize_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255
    return arr.reshape(1, 28, 28, 1)

def predict(input_data: dict) -> dict:
    model = _load_model()

    if "image_path" in input_data:
        img = Image.open(input_data["image_path"])

    elif "image" in input_data:
        arr = np.array(input_data["image"], dtype="uint8")
        img = Image.fromarray(arr)

    else:
        raise ValueError("Input format not recognized")

    tensor = resize_image(img)
    scores = model.predict(tensor, verbose=0)[0]
    digit = int(np.argmax(scores))
    prob = float(scores[digit])

    return {"digit":digit,
            "probability":prob,
            "scores":[round(float(s), 4) for s in scores]}
