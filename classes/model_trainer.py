
"""
ModelTrainner Module
"""
# Import required libraries
import os
import streamlit as st
import tensorflow as tf
import tempfile

# Model training operations
class ModelTrainer:
    """
    Handles model building and training operations with optional attention
    """
    
    @staticmethod
    def build_model(input_shape=(1, 32, 32, 1), num_classes=28, use_attention=False):
        """
        Constructs the LSTM-CNN hybrid model architecture with optional attention       
        Args:
            input_shape (tuple): Input tensor shape
            num_classes (int): Number of output classes
            use_attention (bool): Whether to add attention mechanism          
        Returns:
            tf.keras.Model: Compiled TensorFlow model
        """
        # Try building model
        try:   
            # Check GPU availability via secrets
            if st.secrets.get("ENABLE_GPU", "False").lower() == "true":
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Set GPU memory limit (4GB)
                    memory_limit = int(st.secrets.get("GPU_MEMORY_LIMIT", 4096))
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
            # Define input layer
            inputs = tf.keras.layers.Input(shape=input_shape)
            # Remove singleton dimension
            ## Remove the extra sequence dimension (of size 1) so that the input becomes (32,32,1)
            x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), output_shape=(32, 32, 1))(inputs)
            # CNN Feature Maps Blocks
            ## First CNN block
            x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # Poolint to (16,16,32)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            # Second CNN block
            x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # Pooling to (8,8,64)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            
            # Third CNN block
            x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # Now (8,8,128)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Prepare for LSTM  Sequence Formation (batch, height, width, channels)
            ## Consider a width dimension as time steps.
            shape = tf.keras.backend.int_shape(x) 
            x = tf.keras.layers.Reshape((shape[2], shape[1]*shape[3]))(x)
            # LSTM Layers to capture sequential dependencies
            x = tf.keras.layers.LSTM(128, return_sequences = use_attention)(x)
            # Add attention mechanism if selected
            if use_attention:
                # Attention mechanism
                attention = tf.keras.layers.Attention()([x, x])
                x = tf.keras.layers.Concatenate()([x, attention])
                x = tf.keras.layers.GlobalAveragePooling1D()(x)
            else:
                x = tf.keras.layers.Dropout(0.3)(x)
            
            # Fully Connected Layer outputs the predicted probability distribution over the 28 classes.
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
            # Create and compile model
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            # Verify model can be saved and loaded with safe_mode
            try:
                test_path = "test_model.keras"
                model.save(test_path)
                # Explicit safe_mode
                tf.keras.models.load_model(test_path, safe_mode=False)  
                os.remove(test_path)
            except Exception as e:
                st.error(f"Model save/load verification failed: {str(e)}")
                return None
            # Return compiled model
            return model
        # Handle any errors
        except Exception as e:
            # Show error message
            st.error(f"Model building error: {str(e)}")
            # Return empty value
            return None
        
    @staticmethod
    def train_model(model, X_train, y_train, epochs=20, batch_size=128):
        """
        Trains the model with progress tracking       
        Args:
            model (tf.keras.Model): Model to train
            X_train (np.array): Training features
            y_train (np.array): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size   
        Returns:
            tf.keras.History: Training history object
        """
        try:
            # Configure training callbacks
            callbacks = [
                # Stop training if no improvement after 5 epochs
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                # Reduce learning rate if plateau detected
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
            ]         
            # Initialise progress indicators 
            # Visual progress bar
            progress_bar = st.progress(0)
            # Dynamic text display
            status_text = st.empty()  
            metrics_container = st.empty()        
            class TrainingCallback(tf.keras.callbacks.Callback):
                """
                    Custom callback to update Streamlit UI during training.
                    This executes at the end of each epoch to:
                    1. Update the progress bar
                    2. Display current metrics
                    3. Provide real-time feedback
                """
                def on_epoch_end(self, epoch, logs=None):
                    """
                    Called at the end of each training epoch.
                    
                    Args:
                        epoch (int): Current epoch index (0-based)
                        logs (dict): Metrics dictionary containing:
                            - loss: Training loss
                            - accuracy: Training accuracy
                            - val_loss: Validation loss 
                            - val_accuracy: Validation accuracy
                    """
                    # Calculate progress percentage (0-1)
                    progress = (epoch + 1) / epochs
                    # Update progress bar
                    progress_bar.progress(progress)
                    # Update status text with formatted metrics
                    status_text.text(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Loss: {logs['loss']:.4f}, "
                        f"Acc: {logs['accuracy']:.4f}, "
                        f"Val Loss: {logs['val_loss']:.4f}, "
                        f"Val Acc: {logs['val_accuracy']:.4f}"
                    )   
            # Start model training
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                # Combine default and custom callbacks and Suppress default logging
                callbacks=callbacks + [TrainingCallback()], verbose=0
            )  
            # Clean up UI elements after training completes         
            progress_bar.empty()
            status_text.empty()
            # Add validation metrics display
            # Display final metrics
            with metrics_container.expander("📊 Final Training Metrics", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Accuracy", f"{history.history['accuracy'][-1]:.2%}")
                    st.metric("Training Loss", f"{history.history['loss'][-1]:.4f}")
                with col2:
                    st.metric("Validation Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
                    st.metric("Validation Loss", f"{history.history['val_loss'][-1]:.4f}")
            return history
        except Exception as e:
            # Clean up UI elements if error occurs
            progress_bar.empty()
            status_text.empty()
            st.error(f"Training error: {str(e)}")
            return None

    @staticmethod
    def save_model(model):
        """
        Saves the trained model to a temporary directory in Streamlit Cloud.
        Args:
            model (tf.keras.Model): The trained model to be saved.
            Returns:
                str: The path where the model is saved, or None if an error occurs.
        """
        try:
            # Creates a temporary directory using a temporary directory in Streamlit Cloud (files will be lost after the session ends)
            model_dir = tempfile.mkdtemp()  
            # Define the model name and format and join them with the temparary directory
            model_path = os.path.join(model_dir, "model.keras")
            # Save the model to the temporary path
            model.save(model_path)
            # Return the path where the model is saved
            return model_path
        except Exception as e:
            # Log the error message
            st.error(f"Error saving model: {str(e)}")
            # Return None to indicate failure
            return None