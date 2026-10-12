"""
Unit tests for the XAI Diagnostic Dashboard backend logic.
Author: Om Awalekar | PRN: 32458020002
"""

import pytest
import torch
import numpy as np
from PIL import Image
from model_utils import get_model, preprocess_image, generate_gradcam, overlay_heatmap

@pytest.fixture
def dummy_image():
    return Image.new('RGB', (300, 300), color=(73, 109, 137))

@pytest.fixture
def dummy_model():
    return get_model()

def test_model_initialization(dummy_model):
    assert dummy_model is not None, "Model failed to initialize."
    assert not dummy_model.training, "Model should be in eval mode, not training mode."
    assert dummy_model.base_model.fc.out_features == 2, "Output layer should have exactly 2 classes (Normal vs Pneumonia)."

def test_preprocess_image(dummy_image):
    tensor = preprocess_image(dummy_image)
    assert isinstance(tensor, torch.Tensor), "Output must be a PyTorch Tensor."
    assert tensor.shape == (1, 3, 224, 224), f"Unexpected tensor shape: {tensor.shape}"

def test_generate_gradcam(dummy_model, dummy_image):
    input_tensor = preprocess_image(dummy_image)
    heatmap, pred_idx, probabilities = generate_gradcam(dummy_model, input_tensor)
    
    assert isinstance(pred_idx, int), "Prediction index must be an integer."
    assert pred_idx in [0, 1], "Prediction index must be 0 (Normal) or 1 (Pneumonia)."
    
    assert probabilities.shape == (2,), "Probabilities should contain 2 values."
    assert np.isclose(np.sum(probabilities), 1.0), "Probabilities must sum to 1."
    
    assert isinstance(heatmap, np.ndarray), "Heatmap must be a NumPy array."
    assert heatmap.shape == (7, 7), f"Heatmap shape should be (7, 7), got {heatmap.shape}"
    assert np.min(heatmap) >= 0 and np.max(heatmap) <= 1.0, "Heatmap values must be normalized between 0 and 1."

def test_overlay_heatmap(dummy_image):
    dummy_heatmap = np.zeros((7, 7), dtype=np.float32)
    overlaid_img = overlay_heatmap(dummy_image, dummy_heatmap, opacity=0.5)
    
    assert isinstance(overlaid_img, np.ndarray), "Overlaid image must be a NumPy array."
    assert overlaid_img.shape == (300, 300, 3), "Overlaid image dimensions should match the original input."