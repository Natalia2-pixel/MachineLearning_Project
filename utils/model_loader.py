import joblib
import tensorflow as tf
import os

def load_model(model_type):
    model_paths = {
        "SVM": "Models/svm_model.pkl",
        "GBM": "Models/gbm_model.pkl",
        "CNN": "Models/cnn_model.h5",
        "Xception": "Models/xception_model.h5",
        "GAN": "Models/generator_model.h5"
    }

    if model_type not in model_paths:
        raise ValueError(f"Invalid model type: {model_type}. Choose from: {list(model_paths.keys())}")

    model_path = model_paths[model_type]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type in ["SVM", "GBM"]:
        return joblib.load(model_path)
    else:
        return tf.keras.models.load_model(model_path)
