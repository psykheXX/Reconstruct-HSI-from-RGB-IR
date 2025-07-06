import numpy as np
from spectral import open_image
import hdf5storage
import h5py
import os

origin_path = "D:/potato_hyper_dataset/origin/"
save_path = "D:/potato_hyper_dataset/downsampling/"

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def resample_spectrum(data, original_wavelengths, target_resolution=10):
    """
    下采样光谱图像到指定分辨率（nm），并舍去最后一个波段。

    参数：
    data: ndarray, 形状为 (height, width, bands)
    original_wavelengths: ndarray, 原始波长数组
    target_resolution: float, 目标分辨率（nm）

    返回：
    resampled_data: 下采样后的光谱数据，形状为 (height, width, n_new_bands-1)
    new_wavelengths: 下采样后的波长数组（舍去最后一个波段）
    """
    height, width, bands = data.shape

    data_2d = data.reshape(-1, bands)

    start_wl = original_wavelengths[0]
    end_wl = original_wavelengths[-1]

    new_wavelengths = np.arange(start_wl, end_wl + target_resolution, target_resolution)

    new_wavelengths = new_wavelengths[:-1]

    n_pixels = data_2d.shape[0]
    n_new_bands = len(new_wavelengths)
    resampled_data_2d = np.zeros((n_pixels, n_new_bands), dtype=np.float32)

    for i in range(n_pixels):
        resampled_data_2d[i, :] = np.interp(
            new_wavelengths,
            original_wavelengths,
            data_2d[i, :],
            left=np.nan,
            right=np.nan
        )

    # (height, width, n_new_bands)
    resampled_data = resampled_data_2d.reshape(height, width, n_new_bands)

    return resampled_data, new_wavelengths

def read_origin_hyper(origin_path):
    origin = open_image(origin_path)
    img = origin.load()
    np_tensor = np.float32(img)
    data = np_tensor[:, :, 35:171]

    return data

def preprocess_test():
    start_wl = 498.8
    end_wl = 902.12
    n_bands = 136
    original_wavelengths = np.linspace(start_wl, end_wl, n_bands)

    signal_dat_path = "E:/2024年乌兰察布马铃薯试验/6.25数据/0高光谱数据/1345/results/REFLECTANCE_1345.dat"
    signal_hdr_path = signal_dat_path.replace(".dat", ".hdr")

    # 加载高光谱图像
    hyperspectral_image = open_image(signal_hdr_path)
    image_data = hyperspectral_image.load()

    # 转换为 NumPy 数组并选择波段
    np_tensor = np.float32(image_data)
    data = np_tensor[:, :, 35:171]  # 选择第35到170波段（共136波段）

    # 调用下采样函数
    resampled_data, new_wavelengths = resample_spectrum(data, original_wavelengths, target_resolution=10)

    # 打印结果
    print(f"New wavelengths: {new_wavelengths}")
    print(f"Number of new bands: {len(new_wavelengths)}")
    print(f"Resampled data shape: {resampled_data.shape}")
    var_name = 'cube'
    mat_dir = 'REFLECTANCE_1345'
    save_matv73(mat_dir, var_name, resampled_data)

if __name__ == "__main__":
    hyper_fold_names = os.listdir(origin_path)
    for hyper_fold_name in hyper_fold_names:
        hyper_dat_path = os.path.join(origin_path, hyper_fold_name, 'results', 'REFLECTANCE_'+hyper_fold_name+'.dat')
        hyper_hdr_path = os.path.join(origin_path, hyper_fold_name, 'results', 'REFLECTANCE_'+hyper_fold_name+'.hdr')

        origin_hyper = read_origin_hyper(hyper_hdr_path)

        start_wl = 498.8
        end_wl = 902.12
        n_bands = 136
        original_wavelengths = np.linspace(start_wl, end_wl, n_bands)
        resampled_hyper, _ = resample_spectrum(origin_hyper, original_wavelengths, target_resolution=10)

        var_name = 'cube'
        mat_dir = os.path.join(save_path, 'REFLECTANCE_'+hyper_fold_name)
        save_matv73(mat_dir, var_name, resampled_hyper)
        print(f"REFLECTANCE_{hyper_fold_name} success!")