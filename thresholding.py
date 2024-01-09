from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu
import cv2
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk, rectangle, square
from skimage import exposure
from skimage.filters import median, gaussian
from skimage.filters import try_all_threshold, threshold_minimum, threshold_yen
from skimage import feature
from skimage import filters


img = cv2.imread("contrast_image_2.jpg", 0)
img2 = cv2.imread("contrast_image.jpg", 0)

# Preprocessing for thresholding includes, heavy gamma adjustment, value > 2 and
# gaussian blur of image results below


footprint = rectangle(1, 3)


img2 = exposure.adjust_gamma(img2, 3)
img2 = gaussian(img2, sigma=1)
plt.imshow(img2, cmap='gray')
plt.show()

edges1 = feature.canny(img2)
plt.imshow(edges1, cmap='gray')
plt.show()
edges2 = feature.canny(img2, sigma=3)
plt.imshow(edges2, cmap='gray')
plt.show()

edge_roberts = filters.roberts(img2)
plt.imshow(edge_roberts, cmap='gray')
plt.show()
edge_sobel = filters.sobel(img2)
plt.imshow(edge_sobel, cmap='gray')
plt.show()
edge_laplace = filters.laplace(img2)
plt.imshow(edge_laplace, cmap='gray')
plt.show()

thresh = threshold_yen(img2)
bin = img2 > thresh
plt.imshow(bin, cmap='gray')
plt.show()
fig2, ax2 = try_all_threshold(img2, figsize=(10, 8), verbose=False)
plt.show()




# plt.imshow(img, cmap='gray')
# plt.show()
# plt.hist(median.flat, bins=100, range=(100,255))  #.flat returns the flattened numpy array (1D)
# thresholds = threshold_multiotsu(img, classes=3)
# thresh = threshold_otsu(img)
# binary = img > thresh
# plt.imshow(binary, cmap='gray')
# plt.show()
# regions = np.digitize(img, bins=thresholds)
# plt.imshow(regions)
#
# segm1 = (regions == 0)
# segm2 = (regions == 1)
# segm3 = (regions == 2)
# segm4 = (regions == 3)
# # segm5 = (regions == 4)
#
#
# from scipy import ndimage as nd
#
# segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
# segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))
#
# segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
# segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))
#
# segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
# segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))
#
# segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
# segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))
#
# # segm5_opened = nd.binary_opening(segm5, np.ones((3,3)))
# # segm5_closed = nd.binary_closing(segm5_opened, np.ones((3,3)))
#
# all_segments_cleaned = np.zeros((img.shape[0], img.shape[1], 3))
#
# all_segments_cleaned[segm1_closed] = (1,0,0)
# all_segments_cleaned[segm2_closed] = (0,1,0)
# all_segments_cleaned[segm3_closed] = (0,0,1)
# all_segments_cleaned[segm4_closed] = (1,1,0)
# # all_segments_cleaned[segm5_closed] = (1,0,1)
#
# plt.imshow(all_segments_cleaned)