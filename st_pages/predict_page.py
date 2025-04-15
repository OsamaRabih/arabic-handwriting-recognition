
"""
Prediction Page Module
"""
# Import required libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
from classes.predictor import Predictor
from streamlit_drawable_canvas import st_canvas

def show():
    # Set page title with emoji
    st.title("🔮 Character Prediction")
    
    # Model upload option
    st.header("1. Load Model")
    uploaded_model = st.file_uploader("Upload Trained Model (.keras)", 
                                    type=['keras'],
                                    accept_multiple_files=False
                                    )
    
    if uploaded_model is not None:
        try:
            # Save uploaded model temporarily
            with open("temp_model.keras", "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            # Load the model with False safe mode 
            st.session_state.model = tf.keras.models.load_model("temp_model.keras", safe_mode=False)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    # Check if model is available
    if st.session_state.model is None:
        # Show warning if no model
        st.warning("⚠️ Please train or upload a model first!")
        return
        
    # Display prediction instructions
    st.info("""
    **Prediction Instructions:**
    1. Choose input method
    2. Draw or upload character
    3. Get prediction
    """)
    
    # Horizontal radio buttons
    input_method = st.radio("Select Input Method", 
                          ["🖌️ Draw Character", "📁 Upload Image"],
                          index=0,
                          horizontal=True)
    
    # Drawing canvas option
    if input_method == "🖌️ Draw Character":
        st.header("2. Draw Your Character ✍️")
        # Use a unique key for the canvas widget
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = "canvas_initial"
        # Create drawing canvas with the current key
        canvas = st_canvas(
            stroke_width=25,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=320,
            width=320,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key
        )  

        # Create action buttons
        col1, col2 = st.columns(2)
        with col1:
            # Predict button for drawn image
            if st.button("🔮 Predict Drawing"):
                if canvas.image_data is not None:
                    # Convert canvas to image
                    img = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA')
                    # Preprocess drew character and make prediction
                    pred_class, confidence, processed_img = Predictor.predict_image(
                        st.session_state.model, img
                    )                 
                    # Display results if prediction successful
                    if pred_class is not None:
                        st.subheader("🎯 Prediction Result")
                        Predictor.display_results(processed_img, pred_class, 
                                                 confidence, Predictor.characters)
                else:
                    st.warning(f"⚠️ Please draw a character on the canvas first")
        with col2:
            # Clear canvas button
            if st.button("🧹 Clear Canvas"):
                # Reset the canvas by changing its key
                st.session_state.canvas_key = f"canvas_{hash(st.session_state.canvas_key)}"
                # Refresh the page
                st.rerun()  
    # Image upload option
    else:
        st.header("2. Upload Character Image In (jpg / png/ jpeg) Format📤")
        # File uploader
        uploaded_file = st.file_uploader("Choose an image", 
                                       type=["jpg", "png", "jpeg"])
        
        # If file is uploaded
        if uploaded_file is not None:
            # Open and display image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=200)
            
            # Predict button for uploaded image
            if st.button("🔮 Predict Uploaded Image"):
                # Error handler
                try:
                    # Make prediction
                    pred_class, confidence, processed_img = Predictor.predict_image(
                        st.session_state.model, img
                    )
                    
                    # Display results if prediction successful
                    if pred_class is not None:
                        st.subheader("🎯 Prediction Result")
                        Predictor.display_results(processed_img, pred_class,
                                                confidence, Predictor.characters)
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
        else:
            st.warning(f"Pleas, upload an image of a character first")

