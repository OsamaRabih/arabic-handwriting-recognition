
"""
Results Page Module
"""
import streamlit as st
import matplotlib.pyplot as plt
import io
import zipfile

def show():
    # Set page title
    st.title("Training Results")
    
    # Check if training history exists
    if not st.session_state.train_history:
        # Show warning if no training data
        st.warning("No training results available! Train and Test The Model First!")
        return
        
    # Create header for training metrics section
    st.header("Training Metrics")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training and validation loss on first subplot
    ax1.plot(st.session_state.train_history.history['loss'], label='Train Loss')
    ax1.plot(st.session_state.train_history.history['val_loss'], label='Val Loss')
    
    # Set title and legend for loss plot
    ax1.set_title('Loss Curve')
    ax1.legend()
    
    # Plot training and validation accuracy on second subplot
    ax2.plot(st.session_state.train_history.history['accuracy'], label='Train Acc')
    ax2.plot(st.session_state.train_history.history['val_accuracy'], label='Val Acc')
    
    # Set title and legend for accuracy plot
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    
    # Display the matplotlib figure in Streamlit
    st.pyplot(fig)
    
    # Create export results button
    if st.button("Export Results"):
        # Create in-memory bytes buffer
        buf = io.BytesIO()
        
        # Create zip file in memory
        with zipfile.ZipFile(buf, 'w') as zipf:
            # Save plots to PNG file
            fig.savefig("training_metrics.png")
            
            # Add plot image to zip
            zipf.write("training_metrics.png")
            
            # Check if test metrics exist
            if st.session_state.test_metrics:
                # Create text file with test results
                with open("test_results.txt", "w") as f:
                    # Write test loss to file
                    f.write(f"Test Loss: {st.session_state.test_metrics['loss']:.4f}\n")
                    # Write test accuracy to file
                    f.write(f"Test Accuracy: {st.session_state.test_metrics['accuracy']:.4f}")
                
                # Add test results file to zip
                zipf.write("test_results.txt")
            
        # Reset buffer position
        buf.seek(0)
        
        # Create download button for zip file
        st.download_button(
            label="Download Results",
            data=buf,
            file_name="model_results.zip",
            mime="application/zip"
        )
        