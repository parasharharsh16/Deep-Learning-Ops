import torch.onnx
import torchvision.models as models

# Load model
model = torch.load("models/pytorch_model.pth")
model.to('cpu')
# Set the model to evaluation mode
model.eval()

# Input example, adjust according to your model input
dummy_input = torch.randn(1, 3, 224, 224)

# Specify the path where you want to save the ONNX model
onnx_path = "models/ONNX_model.onnx"

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

print("Model converted to ONNX format and saved as:", onnx_path)