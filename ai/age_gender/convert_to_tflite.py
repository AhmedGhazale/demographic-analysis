from model import AgeGenderModel
import torch
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx2tf import convert  # from onnx2tf==1.26.3
import tensorflow as tf


age_gender_model = AgeGenderModel().eval()

age_gender_model.load_state_dict(torch.load("best_model_efficientnet_b0.pth"))

age_gender_model.cpu()

# Step 2: Export PyTorch model to ONNX
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input shape
onnx_path = "model.onnx"
torch.onnx.export(
    age_gender_model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=13
)

# Step 3: Convert ONNX to TensorFlow using onnx2tf
# (This replaces the old onnx-tf approach)
convert(
    input_onnx_file_path=onnx_path,
    output_folder_path="tf_model",
    copy_onnx_input_output_names_to_tflite=True,  # Preserve I/O names
    non_verbose=True
)

# Step 4: Convert TensorFlow SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('age_gender_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite conversion complete!")