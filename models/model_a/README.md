# Model A — MNIST Digit Classifier

A Convolutional Neural Network (CNN) trained on the MNIST dataset that classifies handwritten digits (0–9).

## Performance
- **Test accuracy**: ~99%
- **Loss function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam

## Architecture
```
Input (28x28x1)
→ Conv2D(32, 3x3, relu)
→ MaxPooling2D(2x2)
→ Conv2D(16, 3x3, relu)
→ MaxPooling2D(2x2)
→ Flatten
→ Dense(128, relu)
→ Dense(64, relu)
→ Dense(10, softmax)
```

## Files
| File | Description |
|---|---|
| `train.py` | Trains the model and saves it as `model_a.keras` |
| `predict.py` | Loads the model and runs inference |
| `model_a.keras` | Saved model weights (download separately) |
| `requirements.txt` | Python dependencies |

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```
Trains for 10 epochs with batch size 128 and saves `model_a.keras` in the same folder.

## Running Inference

**From a file path:**
```python
from predict import predict

result = predict({"image_path": "digit.png"})
print(result)
# {"digit": 7, "probability": 0.9991, "scores": [0.0, ..., 0.9991, 0.0]}
```

**From a 2D pixel array:**
```python
import numpy as np
from predict import predict

pixels = np.array(image).tolist()  # any size, will be resized to 28x28
result = predict({"image": pixels})
```

## Input Format
| Key | Type | Description |
|---|---|---|
| `image_path` | `str` | Path to any image file (.png, .jpg, etc.) |
| `image` | `list[list[int]]` | 2D pixel array, any size, values 0–255 |

Images of any size are automatically converted to grayscale and resized to 28×28.

## Output Format
```json
{
  "digit": 7,
  "probability": 0.9991,
  "scores": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9991, 0.0, 0.0]
}
```
| Key | Type | Description |
|---|---|---|
| `digit` | `int` | Predicted digit (0–9) |
| `probability` | `float` | Confidence of the prediction (0.0–1.0) |
| `scores` | `list[float]` | Softmax probability for each of the 10 classes |
