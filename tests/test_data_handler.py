
"""
DataHandler.preprocess_image() Validation
"""
import unittest
from PIL import Image
import numpy as np
import sys
import os
from io import StringIO
from classes.data_handler import DataHandler
from classes.predictor import Predictor
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
class TestDataHandler(unittest.TestCase):
    """
    Comprehensive tests for DataHandler class for both FYP and Quality Assurance and Testing Module.
    """
    
    def setUp(self):
        self.test_img = Image.new('L', (64, 64))  # 64x64 grayscale image
        
    def test_preprocess_image_resizing(self):
        """UT-01: Verify image resizing to 32x32"""
        _, processed_img = DataHandler.preprocess_image(self.test_img)
        self.assertEqual(processed_img.size, (32, 32))
        
    def test_preprocess_image_normalization(self):
        """UT-02: Validate pixel normalization [0,1]"""
        processed_arr, _ = DataHandler.preprocess_image(self.test_img)
        self.assertTrue(np.all(processed_arr <= 1.0))
        self.assertTrue(np.all(processed_arr >= 0.0))
        
    def test_preprocess_image_tensor_shape(self):
        """UT-03: Check output tensor dimensions"""
        processed_arr, _ = DataHandler.preprocess_image(self.test_img)
        self.assertEqual(processed_arr.shape, (1, 1, 32, 32, 1))
    

    # Quality Assurance Module White-box tests cases
    # Test Case UT-1
    def test_load_data_valid(self):
        """
        Test valid CSV loading with branch coverage.
        """
        # Arrange: Mock CSV with 1024 features (32x32) 1024 columns and label=1
        features_csv = "0," * 1024[:-1]  
        labels_csv = "1" 
        # Act: Call load_data()
        X, y = DataHandler.load_data(
            StringIO(features_csv), 
            StringIO(labels_csv)
        )
        # Assert: Branch 1: Valid reshape. Check reshaping and label range. 
        self.assertEqual(X.shape[1:], (1, 32, 32, 1))  
        # Branch 2: Label normalisation in [0,27]
        self.assertTrue(y[0] in range(28))  

    # Test Case UT-2
    def test_predict_image():
        # Create a 32x32 grayscale image
        img_array = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Mock model (simulates softmax output)
        class MockModel:
            def predict(self, x, verbose=0):
                # Dummy probabilities
                return np.array([[0.01]*28])  
        
        pred_class, confidence, _ = Predictor.predict_image(MockModel(), img)
        assert pred_class in range(28), "Class out of bounds"
        assert 0 <= confidence <= 1, "Confidence not normalized"


if __name__ == '__main__':
    unittest.main()