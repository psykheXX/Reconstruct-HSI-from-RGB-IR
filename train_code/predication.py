import torch
import torch.backends.cudnn as cudnn
import os
from architecture.MST_Plus_Plus import MST_Plus_Plus
import cv2
import numpy as np
import itertools
import hdf5storage

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def main(test_data, mat_name):
    cudnn.benchmark = True
    model = MST_Plus_Plus().cuda()
    checkpoint = torch.load('net_20epoch.pth')
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                          strict=True)
    outf = os.path.join("D:/potato_hyper_dataset/res_Spec", mat_name)
    result = mst_test(model, test_data, outf)
    return result

def mst_test(model, test_data, save_path):
    var_name = 'cube'
    rgb = test_data
    rgb = np.float32(rgb)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    rgb = torch.from_numpy(rgb).float().cuda()
    print(f'Reconstructing camera data')
    with torch.no_grad():
        result = forward_ensemble(rgb, model, 'mean')
    result = result.cpu().numpy() * 1.0
    result = np.transpose(np.squeeze(result), [1, 2, 0])
    result = np.minimum(result, 1.0)
    result = np.maximum(result, 0)

    mat_dir = os.path.join(save_path)
    save_matv73(mat_dir, var_name, result)
    print(f'The reconstructed hyper spectral image are saved as {mat_dir}.')
    return result

def forward_ensemble(x, forward_func, ensemble_mode = 'mean'):
    def _transform(data, xflip, yflip, transpose, reverse=False):
        if not reverse:  # forward transform
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # reverse transform
            if transpose:
                data = torch.transpose(data, 2, 3)
            if yflip:
                data = torch.flip(data, [2])
            if xflip:
                data = torch.flip(data, [3])
        return data

    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    for xflip, yflip, transpose in opts:
        data = x.clone()
        data = _transform(data, xflip, yflip, transpose)
        data = forward_func(data)
        outputs.append(
            _transform(data, xflip, yflip, transpose, reverse=True))
    if ensemble_mode == 'mean':
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        return torch.stack(outputs, 0).median(0)[0]


if __name__ == '__main__':
    file_path = "D:/potato_hyper_dataset/Valid_RGB"
    file_names = os.listdir(file_path)
    for file_name in file_names:
        img_path = os.path.join(file_path, file_name)
        if img_path.endswith('.png'):
            frame = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            if frame is None:
                print("Failed to load image.")
            else:
                mat_name = file_name.replace('.png', '.mat')
                r = main(frame, mat_name)