
"""
Main Arabic Character Recognition Application
"""

# Import required libraries
import streamlit as st
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Set page title and icon
st.set_page_config(page_title="Arabic Handwriting Recognition", page_icon="🖋️")
# Create necessary directories if they don't exist
os.makedirs("assets/models", exist_ok=True)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    .stRadio label {
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialise session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_save_path = st.secrets.get("MODEL_SAVE_PATH", "/tmp")
if 'train_history' not in st.session_state:
    st.session_state.train_history = None
if 'test_metrics' not in st.session_state:
    st.session_state.test_metrics = None
if 'train_features_data' not in st.session_state:
    st.session_state.train_features_data = None
if 'train_labels_data' not in st.session_state:
    st.session_state.train_labels_data = None

def main():
    # Create sidebar navigation
    st.sidebar.title("🧭 Navigation")
    # Create radio buttons for page selection
    app_mode = st.sidebar.radio("Choose Page", [
        "🎓 Train Model",
        "🧪 Test Model", 
        "📊 View Results",
        "🔮 Make Predictions",
        "🚪 Exit"
    ])  
    # Route to selected page
    if app_mode == "🎓 Train Model":
        # Import and show training page
        from st_pages import train_page
        train_page.show()
    elif app_mode == "🧪 Test Model":
        # Import and show testing page
        from st_pages import test_page
        test_page.show()
    elif app_mode == "📊 View Results":
        # Import and show results page
        from st_pages import results_page
        results_page.show()
    elif app_mode == "🔮 Make Predictions":
        # Import and show prediction page
        from st_pages import predict_page
        predict_page.show()
    elif app_mode == "🚪 Exit":
        # Import and show exit page
        from st_pages import exit_page
        exit_page.show()
    
    # Display instructions for navigation
    st.sidebar.info("""
    **How to use:**
    1. Start with 🎓 Train Model
    2. Test with 🧪 Test Model
    3. View 📊 Results
    4. Make 🔮 Predictions
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app is designed to recognize Arabic handwritten characters using a deep learning model."
    )
# Run main function when script is executed
if __name__ == "__main__":
    main()