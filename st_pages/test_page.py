
"""
Testing Page Module
"""
# Import required libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from classes.data_handler import DataHandler
from classes.predictor import Predictor

# Define page display function
def show():
    # Set page title with emoji
    st.title("🧪 Model Testing")
    
    # Check if model is available
    if st.session_state.model is None:
        # Show warning if no model
        st.warning("⚠️ Please train a model first!")
        return
        
    # Display testing instructions
    st.info("""
    **Testing Instructions:**
    1. Upload test data (features and labels)
    2. Show sample test images
    3. Run testing
    """)
    
    # Section 1: Data Upload
    st.header("1. Upload Testing Data 🗂️")
    # Create two columns for file uploaders
    col1, col2 = st.columns(2)
    with col1:
        # Upload test features
        test_features = st.file_uploader("Test Images (CSV)", 
                                        type=['csv'], 
                                        key='test_features')
    with col2:
        # Upload test labels
        test_labels = st.file_uploader("Test Labels (CSV)", 
                                     type=['csv'], 
                                     key='test_labels')
    
    # Check if both files are uploaded
    if test_features and test_labels:
        try:
            # Load and preprocess test data
            X_test, y_test = DataHandler.load_data(test_features, test_labels)
            
            # Section 2: Data Preview
            #st.header("2. Data Preview ")
            # Display dataset statistics
           # st.write(f"📊 Test samples: {len(X_test)}")
            
            # show sample images
            st.subheader("2. Sample of Preprocessed Images (12/3360) 👀")
            # Create grid of sample images
            fig, axes = plt.subplots(3, 4, figsize=(10, 5))
            for i, ax in enumerate(axes.flat):
                # Reshape and display each image
                img = np.transpose(X_test[i].reshape(32, 32))
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Label: {Predictor.characters[y_test[i]]}")
                ax.axis('off')
            # Display the plot in Streamlit
            st.pyplot(fig)
            
            # Run testing button
            if st.button("🧪 Run Testing"):
                # Show loading spinner
                with st.spinner("Testing in progress..."):
                    # Evaluate model on test data using the saved session state model
                    test_loss, test_acc = st.session_state.model.evaluate(X_test, y_test, verbose=0)
                    # Store test metrics in session state
                    st.session_state.test_metrics = {
                        'loss': test_loss,
                        'accuracy': test_acc
                    }
                    # Show success message with accuracy and loss
                    st.success(f"✅ Test Accuracy: {test_acc* 100:.2f}") 
                    st.success(f"📉 Test Loss: {test_loss:.4f}")
                    st.write(f"Now, you can test our model by trying the prediction process") 
                    
        # Handle any errors
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")