
"""
DataHandler Module
"""
# Import required libraries
import numpy as np
import pandas as pd
import streamlit as st

# Data handling operations
class DataHandler:
    """
    Handles data loading and preprocessing operations 
    """   
    @staticmethod
    def load_data(features_file, labels_file):
        """
        Loads and preprocesses CSV data    
        Args:
            features_file (UploadedFile): Streamlit uploaded file object for features
            labels_file (UploadedFile): Streamlit uploaded file object for labels          
        Returns:
            tuple: (X, y) preprocessed features and labels
        """
        
        # Try loading and processing data
        try:
            # Get Streamlit secrets and set default max upload size = 10MB
            max_upload_size = int(st.secrets.get("MAX_UPLOAD_SIZE", 10)) 
            # Validate file size
            if features_file.size > max_upload_size * 1024 * 1024:
                raise ValueError(f"File exceeds maximum size of {max_upload_size}MB")        
            # Validate file objects existence
            if features_file is None or labels_file is None:
                raise ValueError("No files uploaded")   
            # Validate file extensions (only for direct file uploads)
            if hasattr(features_file, 'name') and hasattr(labels_file, 'name'):
                if not (features_file.name.endswith('.csv') and labels_file.name.endswith('.csv')):
                    raise ValueError("Only CSV files are supported")  
            # Read files with validation
            try:
                X = pd.read_csv(features_file, header=None)
                y = pd.read_csv(labels_file, header=None)
            # Handle any errors
            except pd.errors.EmptyDataError:
                # Raise error value and Show error message
                raise ValueError("Uploaded CSV files are empty")
            # Handle any errors
            except pd.errors.ParserError:
                # Raise error value and Show error message
                raise ValueError("Invalid CSV format")
            # Validate numeric data
            if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
                raise ValueError("Non-numeric data detected in features")
            # Check Dimension validation
            # Check if shape is 32x32=1024
            if X.shape[1] != 1024:  
                # Raise error value and Show error message
                raise ValueError(f"Expected 1024 features, got {X.shape[1]}")
            # Check the labels and images are in equal samples counts 
            if len(X) != len(y):
                # Raise error value and Show error message
                raise ValueError(f"Mismatched samples: {len(X)} features vs {len(y)} labels")
            
            # Processing (only reached if all checks pass)
            X = X.values.reshape(-1, 1, 32, 32, 1).astype('float32') / 255.0
            # Flatten and convert to zero based labels
            y = y.values.flatten() - 1
            # Return processed data
            return X, y
            
        # Handle any errors
        except Exception as e:
            # Show error message
            st.error(f"Data loading error: {str(e)}")
            # Return empty values
            return None, None

    @staticmethod
    def preprocess_image(image):
        """
        Processes images for model prediction
        Args:
            image (PIL.Image): Input image to process  
        Returns:
            tuple: (processed_array, processed_img) 
                   normalized array and resized PIL Image
        """
        try:
            # Convert to grayscale and resize
            img = image.convert("L").resize((32, 32))
            # Invert the image (255 - pixel value) and Convert to numpy array 
            img_array = 255 - np.array(img)
            # Transpose array
            img_array = np.transpose(img_array)
            # Reshape and normalise
            img_array = img_array.reshape(1, 1, 32, 32, 1).astype('float32') / 255.0
            # Return processed array and image
            return img_array, img
        # Handle any errors
        except Exception as e:
            # Show error message
            st.error(f"Image processing error: {str(e)}")
            # Return empty values
            return None, None
