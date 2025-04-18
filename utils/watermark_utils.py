import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import joblib
import tensorflow as tf

# Load models
svm_model = joblib.load("Models/svm_model.pkl")
gbm_model = joblib.load("Models/gbm_model.pkl")
pca_x = joblib.load("Models/pca_x.pkl")
pca_y = joblib.load("Models/pca_y.pkl")
cnn_model = tf.keras.models.load_model("Models/cnn_model.h5", compile=False)
cnn_model.compile(optimizer='adam', loss='mse')
xception_model = tf.keras.models.load_model("Models/xception_model.h5", compile=False)
xception_model.compile(optimizer='adam', loss='mse')
cgan_generator = tf.keras.models.load_model("Models/cgan_generator_model.h5", compile=False)
cgan_generator.compile(optimizer='adam', loss='mse')

def apply_watermark_with_model(image_path=None, model=None, cover_image=None, custom_watermark_img=None):
    # Resize and convert to grayscale 64x64
    cover_image = cover_image.convert("L").resize((64, 64))
    custom_watermark_img = custom_watermark_img.convert("L").resize((64, 64)) if custom_watermark_img else Image.new("L", (64, 64), color=0)

    # Convert to numpy arrays
    cover_np = np.array(cover_image).astype(np.float32) / 255.0
    watermark_np = np.array(custom_watermark_img).astype(np.float32) / 255.0

    # Force correct shapes
    if cover_np.shape != (64, 64):
        raise ValueError(f"Cover image shape incorrect: {cover_np.shape}, expected (64, 64)")
    if watermark_np.shape != (64, 64):
        raise ValueError(f"Watermark image shape incorrect: {watermark_np.shape}, expected (64, 64)")

    # Blend as ground truth
    blended_gt = (0.7 * cover_np + 0.3 * watermark_np)

    # Flatten
    cover_flat = cover_np.flatten()
    mark_flat = watermark_np.flatten()

    if model == "SVM" or model == "GBM":
        combined_input = np.hstack([cover_flat, mark_flat])  # Expecting length = 4096 + 4096 = 8192
        expected_input_size = pca_x.components_.shape[1]
        if combined_input.shape[0] != expected_input_size:
            raise ValueError(f"Input size {combined_input.shape[0]} does not match PCA expected input size {expected_input_size}")
    
        reduced_input = pca_x.transform([combined_input])
    
        if model == "SVM":
            predicted_compressed = svm_model.predict(reduced_input)
        else:
            predicted_compressed = gbm_model.predict(reduced_input)

        predicted_flat = pca_y.inverse_transform(predicted_compressed)[0]

    elif model == "GBM":
        combined_input = np.hstack([cover_flat, mark_flat])
        if combined_input.shape[0] != pca_x.n_features_:
            raise ValueError(f"GBM input length {combined_input.shape[0]} doesn't match PCA expected {pca_x.n_features_}")
        reduced_input = pca_x.transform([combined_input])
        predicted_compressed = gbm_model.predict(reduced_input)
        predicted_flat = pca_y.inverse_transform(predicted_compressed)[0]

    elif model == "CNN":
        stacked_input = np.stack([cover_np, watermark_np], axis=-1)
        input_tensor = np.expand_dims(stacked_input, axis=0)
        predicted = cnn_model.predict(input_tensor, verbose=0)
        predicted_flat = predicted[0].squeeze()

    elif model == "Xception":
        input_tensor = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cover_np[..., np.newaxis]))
        input_tensor = tf.image.resize(input_tensor, [71, 71])
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        predicted = xception_model.predict(input_tensor, verbose=0)
        predicted_flat = tf.image.resize(predicted[0], [64, 64]).numpy().squeeze()

    elif model == "GAN":
        stacked_input = np.stack([cover_np, watermark_np], axis=-1)
        input_tensor = tf.convert_to_tensor(stacked_input[np.newaxis, ...])
        predicted = cgan_generator.predict(input_tensor, verbose=0)
        predicted_flat = predicted[0].squeeze()

    else:
        raise ValueError("Unsupported model selected")

    # Compute metrics
    mse = np.mean((blended_gt.flatten() - predicted_flat.flatten()) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    ssim_score = ssim(blended_gt, predicted_flat, data_range=1.0)

    # Convert predicted image back to PIL
    pred_img = Image.fromarray((predicted_flat * 255).astype(np.uint8))

    return pred_img, {"mse": mse, "psnr": psnr, "ssim": ssim_score}
