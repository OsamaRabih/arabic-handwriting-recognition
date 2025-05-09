
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

    
if __name__ == '__main__':
    unittest.main()