import numpy as np
from spectral import open_image
import hdf5storage
import h5py
import os
from PIL import Image
import cv2
# 图像保存时的通道顺序为RGB读取后也为RGB
def read_origin_hyper(origin_path):
    origin = open_image(origin_path)
    img = origin.load()
    np_tensor = np.float32(img)
    return np_tensor

def normalize_band(band):
    band_min, band_max = band.min(), band.max()
    if band_max > band_min:  # Avoid division by zero
        return ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
    else:
        return (band * 255).astype(np.uint8)

def apply_gamma_correction(image, gamma=2.2):
    """Apply gamma correction to an image array."""
    # Ensure image is in range [0, 1] for gamma correction
    image_normalized = image / 255.0
    # Apply gamma correction
    gamma_corrected = np.power(image_normalized, 1.0 / gamma)
    # Scale back to [0, 255] and convert to uint8
    return (gamma_corrected * 255).astype(np.uint8)

# RGBI = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)
#
# R, G, B, A = cv2.split(RGBI)
# RGB = cv2.merge((R, G, B))
# cv2.imwrite('output_rgb.jpg', RGB)

def RGBA_generate(signal_hdr_path, RGBA_sava_path):
    data = read_origin_hyper(signal_hdr_path)

    band_17 = data[:, :, 19]
    band_41 = data[:, :, 53]
    band_75 = data[:, :, 87]
    band_155 = data[:, :, 154]
    # "test_gamma_corrected.jpg"

    output_image = RGBA_sava_path
    rgb = np.stack([band_75, band_41, band_17], axis=-1)

    rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(rgb.shape[-1]):
        rgb_normalized[..., i] = normalize_band(rgb[..., i])

    rgb_normalized = np.squeeze(rgb_normalized)

    print("RGB normalized array shape:", rgb_normalized.shape)

    if rgb_normalized.ndim != 3 or rgb_normalized.shape[-1] != 3:
        raise ValueError(f"Expected 3D array with shape (height, width, 3), got shape {rgb_normalized.shape}")

    gamma = 2.2  # Standard gamma value for display (adjust as needed)
    rgb_gamma_corrected = apply_gamma_correction(rgb_normalized, gamma=gamma)

    band_155_normalized = normalize_band(band_155)
    if band_155_normalized.ndim != 2:
        band_155_normalized = np.squeeze(band_155_normalized)
    rgba = np.zeros((rgb_normalized.shape[0], rgb_normalized.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb_gamma_corrected  # RGB channels with gamma correction
    rgba[..., 3] = band_155_normalized  # Alpha channel (band_155, no gamma correction)


    rgba_image = Image.fromarray(rgba, mode='RGBA')
    rgba_image.save(output_image, "PNG")
    print(f"RGBA image saved as {output_image}")

if __name__ == "__main__":
    origin_path = "D:/potato_hyper_dataset/origin/"
    save_path = "D:/potato_hyper_dataset/Train_RGB/"
    hyper_fold_names = os.listdir(origin_path)
    for hyper_fold_name in hyper_fold_names:
        hyper_hdr_path = os.path.join(origin_path, hyper_fold_name, 'results', 'REFLECTANCE_'+hyper_fold_name+'.hdr')
        RGBA_save_path = os.path.join(save_path, 'REFLECTANCE_'+hyper_fold_name+'.png')
        RGBA_generate(hyper_hdr_path, RGBA_save_path)

