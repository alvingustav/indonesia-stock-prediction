import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import streamlit as st

def load_model_safe(model_path):
    """
    Safely load Keras model dengan handling untuk berbagai compatibility issues
    """
    try:
        # Method 1: Load dengan custom objects
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
            'MeanSquaredError': tf.keras.metrics.MeanSquaredError(),
            'adam': tf.keras.optimizers.Adam(),
            'Adam': tf.keras.optimizers.Adam()
        }
        
        model = load_model(model_path, custom_objects=custom_objects)
        st.success("✅ Model loaded dengan custom objects")
        return model
        
    except Exception as e1:
        st.warning(f"⚠️ Custom objects failed: {e1}")
        
        try:
            # Method 2: Load tanpa compile, lalu compile ulang
            model = load_model(model_path, compile=False)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError()]
            )
            
            st.success("✅ Model loaded dan dikompilasi ulang")
            return model
            
        except Exception as e2:
            st.warning(f"⚠️ Recompile failed: {e2}")
            
            try:
                # Method 3: Force load dengan TF compatibility mode
                with tf.keras.utils.custom_object_scope({'mse': 'mean_squared_error'}):
                    model = load_model(model_path)
                
                st.success("✅ Model loaded dengan compatibility mode")
                return model
                
            except Exception as e3:
                st.error(f"❌ All loading methods failed: {e3}")
                return None

def check_tensorflow_version():
    """Check TensorFlow version compatibility"""
    tf_version = tf.__version__
    st.info(f"🐍 TensorFlow version: {tf_version}")
    
    # Recommend version jika ada masalah
    if tf_version.startswith('2.18'):
        st.success("✅ TensorFlow version compatible")
    else:
        st.warning(f"⚠️ Recommended TensorFlow 2.18.x, current: {tf_version}")
    
    return tf_version