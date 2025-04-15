
"""
Training Page Module
"""
# Import required libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from classes.data_handler import DataHandler
from classes.model_trainer import ModelTrainer
from classes.predictor import Predictor
from io import BytesIO

def show():
    # Set page title with emoji
    st.title("🎓 Model Training")
    # Display instructions for training
    st.info("""
    **Training Instructions:**
    1. Upload training data (features and labels)
    3. Attention Mechanism Option
    4. Start training
    """)  
    # Initialise session state for file persistence
    if 'train_features_data' not in st.session_state:
        st.session_state.train_features_data = None
    if 'train_labels_data' not in st.session_state:
        st.session_state.train_labels_data = None   
    # Section 1: Data Upload
    st.header("1. Upload Training Data 🗂️")
    # Create two columns for file uploaders
    col1, col2 = st.columns(2)
    with col1:
        # Upload training features with session persistence
        train_features = st.file_uploader("Training Images (CSV)", 
                                        type=['csv'], 
                                        key='train_features')
        if train_features:
            st.session_state.train_features_data = train_features.getvalue()  
    with col2:
        # Upload training labels with session persistence
        train_labels = st.file_uploader("Training Labels (CSV)", 
                                      type=['csv'], 
                                      key='train_labels')
        if train_labels:
            st.session_state.train_labels_data = train_labels.getvalue() 
        
    # Check if we have data (either newly uploaded or from session state)
    has_data = (st.session_state.train_features_data is not None and 
            st.session_state.train_labels_data is not None)
    if has_data:
        try:
            # Use the uploaded files if available, otherwise use session state
            if train_features is not None and train_labels is not None:
                # Use the newly uploaded files directly
                X_train, y_train = DataHandler.load_data(train_features, train_labels)
            else:
                # Create file-like objects from session state data
                features_file = BytesIO(st.session_state.train_features_data)
                labels_file = BytesIO(st.session_state.train_labels_data)
                X_train, y_train = DataHandler.load_data(features_file, labels_file)
                
            # Only proceed if data was loaded successfully
            if X_train is not None and y_train is not None:
                st.subheader("📷 Sample of Preprocessed Images (12/13360) 👀")
                # Create grid of sample images
                fig, axes = plt.subplots(3, 4, figsize=(10, 5))
                for i, ax in enumerate(axes.flat):
                    # Reshape and display each image
                    img = np.transpose(X_train[i].reshape(32, 32))
                    ax.imshow(img, cmap='gray')
                    # Show the numeric label (1-28)
                    ax.set_title(f"Label: {Predictor.characters[y_train[i]]}")
                    ax.axis('off')
                # Display the plot in Streamlit
                st.pyplot(fig)

            # Section 3: Attention Mechanism
            st.header("3. Attention Mechanism 🎯")
            # Add attention mechanism radio button
            use_attention = st.radio(
                "Use Attention Mechanism (Experimental)",
                options=["No", "Yes"],
                index=0
            ) == "Yes"
            # Display information about Attention layer if is used
            if use_attention:
                st.info("""
                Attention mechanism will be added after the LSTM layer.
                This may improve performance but will increase training time.
                """)
            
            # Start training button
            if st.button("🚀 Start Training"):
                # Show loading spinner
                with st.spinner("Training in progress..."):
                    # Build model with optional attention
                    model = ModelTrainer.build_model(use_attention=use_attention)
                    # Train model
                    history = ModelTrainer.train_model(model, X_train, y_train, epochs = 20, batch_size = 128)
                    # If training successful
                    if history:
                        # Save model and update session state
                        model_path = ModelTrainer.save_model(model)
                        st.session_state.model = model
                        st.session_state.train_history = history
                        # Show success message
                        st.success(f"✅ Training complete! Model saved temporarily at: {model_path}, NOW THE MODEL IS READY FOR TESTINIG")
                        # Add download button for the trained model
                        with open(model_path, "rb") as f:
                            model_bytes = f.read()
                        st.download_button(
                            label="📥 Download Trained Model",
                            data=model_bytes,
                            file_name=model_path.split("/")[-1],
                            mime="application/octet-stream"                           
                        )
                        # Update session state if download button clicked
                        st.session_state.model = model
                        st.session_state.train_history = history

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")