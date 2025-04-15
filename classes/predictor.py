
"""
Predictorr Module
"""
# Import required libraries
import numpy as np
import streamlit as st
from classes.data_handler import DataHandler
from PIL import Image

# Prediction operations
class Predictor:
    
    # Class attribute with Arabic character labels
    characters =[ 'أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز',
        'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك',
        'ل', 'م', 'ن', 'ه', 'و', 'ي'
    ]

    def __init__(self, model):
        self.model = model

    @staticmethod
    def predict_image(model, image):
        """
        Makes prediction on a single image 
        Args:
            model (tf.keras.Model): Trained model 
            image (PIL.Image): Input image    
        Returns:
            tuple: (predicted_class, confidence, processed_img)
        """
        try:
            processed_array, processed_img = DataHandler.preprocess_image(image)
            if processed_array is None:
                return None, None, None
                
            prediction = model.predict(processed_array, verbose=0)
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction)
            return pred_class, confidence, processed_img
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, None

    @staticmethod
    def display_results(image, pred_class, confidence, characters):
        """
        Displays prediction results with visualization
        
        Args:
            image (PIL.Image): Processed input image
            pred_class (int): Predicted class index
            confidence (float): Prediction confidence
            characters (list): List of class labels
        """
        try:
            st.image(image.resize((128, 128)), caption="Processed Image")
            st.write(f"**Predicted:** {characters[pred_class]}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
        except Exception as e:
            st.error(f"Result display error: {str(e)}")