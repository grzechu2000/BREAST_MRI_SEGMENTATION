# Testing different image denoising algorithms and comparing them by using different image denoising quality metrics
# Filters for testing: BM3D, MEDIAN FILTER, GAUSSIAN FILTER, WIENER FILTER, BILATERAL FILTER, NON LOCAL MEANS FILTER,
# IMAGE QUALITY METRICS: PSNR (Peak Signal to Noise Ratio), SSIM (Structural Similarity Index), MSE (Mean Squared Error)
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
from skimage.filters import median

import cv2
import bm3d
import numpy as np
from skimage import exposure

from skimage.util import random_noise


original_image = cv2.imread("contrast_image.jpg", 0)
bm3d_image = bm3d.bm3d(original_image, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

nlm_image = cv2.fastNlMeansDenoising(original_image)


# median_image =
# gaussian_image =
# wiener_image =
# bilateral_image =
# nlm_image =
#
#
mse = mean_squared_error(original_image, nlm_image)
# ssim =
# psnr =

plt.imshow(bm3d_image, cmap='gray')
plt.imshow(original_image, cmap='gray')
plt.show()
