import os
import numpy as np
from glob import glob
from tifffile import imsave


def save_stn(name, img):
    assert len(img.shape) == 3
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    imsave(name, img)


def save_tensor(name, data):
    data = data[0, 0].cpu().numpy()
    imsave(name, data)


def get_filenames(p1, p2=None):
    all1 = sorted(glob(p1 + "*.tiff") + glob(p1 + "*.tif"))

    if p2 is not None:
        all2 = sorted(glob(p2 + "*.tiff") + glob(p2 + "*.tif"))
        assert len(all1) == len(all2), "different number of source and target images"

        for i in range(len(all1)):
            assert os.path.basename(all1[i]) == os.path.basename(
                all2[i]
            ), f"Filename mismatch: {all1[i]}, {all2[i]}"
        return all1, all2
    else:
        return all1
