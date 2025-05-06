
"""
DataHandler.preprocess_image() Validation
"""
import unittest
from PIL import Image
import numpy as np
from classes.data_handler import DataHandler

class TestDataHandler(unittest.TestCase):
    """
    Unit tests for DataHandler image preprocessing.
    """
    
    def setUp(self):
        self.test_img = Image.new('L', (64, 64))  # 64x64 grayscale image
        
    def test_preprocess_image_resizing(self):
        """
        Verify image resizing to 32x32 pixels.
        """
        processed_arr, processed_img = DataHandler.preprocess_image(self.test_img)
        self.assertEqual(processed_img.size, (32, 32))
        
    def test_preprocess_image_normalization(self):
        """
        Validate pixel value normalization [0,1].
        """
        processed_arr, _ = DataHandler.preprocess_image(self.test_img)
        self.assertTrue(np.all(processed_arr <= 1.0) and np.all(processed_arr >= 0.0))
        
    def test_preprocess_image_shape(self):
        """
        Check output tensor shape (1,1,32,32,1).
        """
        processed_arr, _ = DataHandler.preprocess_image(self.test_img)
        self.assertEqual(processed_arr.shape, (1, 1, 32, 32, 1))

if __name__ == '__main__':
    unittest.main()