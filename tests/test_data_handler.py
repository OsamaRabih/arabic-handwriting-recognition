
"""
DataHandler Module Validation covers all branches in DataHandler 
(file validation, reshaping, normalization).
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
    Unit tests for focusing on branch and condition coverage.
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
    

    def test_data_handler_load_data_valid(self):
        """
        Test Case UT-1: DataHandler.load_data() with valid input
        Description: Ensure CSV files are loaded and reshaped correctly
        Coverage Criteria: Branch coverage (valid file handling)
        """
        # Mock CSV data (1024 features + 1 label)
        features_csv = "0," * 1024[:-1]  # 1024 columns
        labels_csv = "1"
        
        # Execute the method
        X, y = DataHandler.load_data(
            StringIO(features_csv), 
            StringIO(labels_csv)
        )
        
        # Assertions
        self.assertEqual(X.shape[1:], (1, 32, 32, 1), "Data not reshaped correctly to (1,32,32,1)")
        self.assertTrue(y[0] in range(28), "Label not in expected range [0,27]")
        self.assertTrue(np.all(X >= 0) and np.all(X <= 1), "Pixel values not normalized to [0,1]")

    def test_predictor_predict_image(self):
        """
        Test Case UT-2: Predictor.predict_image()
        Description: Verify image preprocessing and prediction
        Coverage Criteria: Condition testing (pixel normalization)
        """
        # Create a 32x32 grayscale image with random pixels
        img_array = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Mock model that returns dummy probabilities
        class MockModel:
            def predict(self, x, verbose=0):
                # Uniform probabilities
                return np.array([[0.01]*28])  
        
        # Execute prediction
        pred_class, confidence, processed_img = Predictor.predict_image(MockModel(), img)
        
        # Assertions
        self.assertIn(pred_class, range(28), "Predicted class out of bounds [0,27]")
        self.assertTrue(0 <= confidence <= 1, "Confidence value not in range [0,1]")
        self.assertEqual(processed_img.size, (32, 32), "Processed image wrong dimensions")

if __name__ == '__main__':
    unittest.main()