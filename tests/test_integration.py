

import pytest
import numpy as np
from PIL import Image
from classes.model_trainer import ModelTrainer
from classes.predictor import Predictor
from classes.data_handler import DataHandler

@pytest.fixture
def sample_data():
    """Generate sample training data for integration tests."""
    return (
        np.random.rand(10, 1, 32, 32, 1).astype('float32'),  # X_train
        np.random.randint(0, 28, 10)  # y_train (28 classes)
    )

def test_train_save_predict_workflow(tmp_path, sample_data):
    """
    End-to-end test: Train → Save → Load → Predict (Path Coverage).
    Covers interaction between ModelTrainer and Predictor.
    """
    X_train, y_train = sample_data
    
    # 1. Train model
    model = ModelTrainer.build_model()
    history = ModelTrainer.train_model(model, X_train, y_train, epochs=1)
    assert history is not None  # Training occurred
    
    # 2. Save and load
    model_path = str(tmp_path / "model.keras")
    model.save(model_path)
    loaded_model = ModelTrainer.load_model(model_path)
    assert loaded_model is not None  # Loading succeeded
    
    # 3. Predict
    pred_class, confidence, _ = Predictor.predict_image(loaded_model, X_train[0])
    assert pred_class in range(28)  # Valid class prediction
    assert 0 <= confidence <= 1  # Confidence normalized

def test_canvas_to_prediction_integration():
    """
    Test UI-to-prediction workflow (Condition + Path Coverage).
    Simulates Streamlit canvas → DataHandler → Predictor pipeline.
    """
    # 1. Simulate canvas output (RGBA)
    canvas_data = np.zeros((32, 32, 4), dtype=np.uint8)
    canvas_data[..., 3] = 255  # Opaque alpha channel
    
    # 2. Convert to grayscale PIL Image
    img = Image.fromarray(canvas_data).convert("L")
    
    # 3. Mock model (always predicts class 1 with 90% confidence)
    class MockModel:
        def predict(self, x, verbose=0):
            return np.array([[0.9 if i == 1 else 0.01 for i in range(28)]])
    
    # 4. Full prediction workflow
    pred_class, confidence, processed_img = Predictor.predict_image(MockModel(), img)
    
    # Assertions. Mock model's forced prediction
    assert pred_class == 1  
    # 90% confidence
    assert confidence == pytest.approx(0.9, 0.01)  
    # Correct preprocessing
    assert processed_img.shape == (1, 1, 32, 32, 1)  

def test_invalid_image_handling():
    """Test error handling for corrupt images (Condition Testing)."""
    with pytest.raises(ValueError):
        # Invalid input
        Predictor.predict_image(None, "not_an_image")  