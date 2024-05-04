import onnxruntime as ort
from PIL import Image
import numpy as np
from utills import transform,class_lables
import os
# Load the model and create InferenceSession
model_path = "models/ONNX_model.onnx"
class_lables = {value: key for key, value in class_lables.items()}

session = ort.InferenceSession(model_path)
#loop on each image in the test folder
for img in os.listdir("testimages"):
    image = Image.open(f"testimages/{img}")
    preprocess_image = transform(image)
    # Run inference
    outputs = session.run(None, {"input.1": [preprocess_image]})
    class_idx = np.argmax(outputs)
    print(f"Predicted class for {img}:", class_lables[class_idx])
