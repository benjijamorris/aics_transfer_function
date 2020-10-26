import sys
from typing import Tuple, Optional
import numpy as np


def simple_norm(
    struct_img0: np.ndarray,
    stat_method: str = "middle_otsu",
    scaling_param: Tuple[float, float] = (1, 1),
    out_range: Tuple[float, float] = (-1.0, 1.0),
    bulk_params: Optional[
        Tuple[str, float, float, Optional[float], Optional[float]]
    ] = None,
    inplace: bool = False,
):
    if not inplace:
        struct_img = np.copy(struct_img0)
    else:
        struct_img = struct_img0

    # check if parameters are passed in as bulk_params
    if bulk_params is not None:
        bulk_params = eval(bulk_params)
        stat_method = bulk_params[0]
        scaling_param = (bulk_params[1], bulk_params[2])
        if len(bulk_params) == 5:
            out_range = (bulk_params[3], bulk_params[4])

    if stat_method == "middle_otsu":
        # take middle chunk based on otsu results
        from scipy.ndimage import gaussian_filter
        from skimage.filters import threshold_otsu

        # do otsu thresholding
        img_smooth = gaussian_filter(
            struct_img.astype(np.float32), sigma=1.0, mode="nearest", truncate=3.0
        )
        img_bw = img_smooth > threshold_otsu(img_smooth)

        # find the middle body chunk
        low_chunk = 0
        high_chunk = struct_img.shape[0]
        for zz in range(struct_img.shape[0]):
            if np.count_nonzero(img_bw[zz, :, :] > 0) > 50:
                if zz > 0:
                    low_chunk = zz - 1
                break

        for zz in range(struct_img.shape[0]):
            if np.count_nonzero(img_bw[struct_img.shape[0] - zz - 1, :, :] > 0) > 50:
                if zz > 0:
                    high_chunk = struct_img.shape[0] - zz
                break

        structure_img0 = struct_img[low_chunk:high_chunk, :, :]

        # use the middle body chunk to estimate mean and std
        m = np.mean(structure_img0)
        s = np.std(structure_img0)

    elif stat_method == "full":
        from scipy.stats import norm

        m, s = norm.fit(struct_img.flat)

    else:
        print("unsupported stat_method for simple norm")
        sys.exit(0)

    lower = max(m - scaling_param[0] * s, struct_img.min())
    upper = min(m + scaling_param[1] * s, struct_img.max())
    struct_img[struct_img < lower] = lower
    struct_img[struct_img > upper] = upper
    struct_img = (struct_img - lower + 1e-8) / (upper - lower + 1e-8)

    assert out_range[1] > out_range[0], "output range is invalid"
    out_range_span = out_range[1] - out_range[0]
    struct_img = struct_img * out_range_span + out_range[0]

    if not inplace:
        return struct_img.astype("float32")
