from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu
import cv2


img = cv2.imread("img1.jpg", 0)

from skimage.restoration import denoise_tv_chambolle
denoised_img = denoise_tv_chambolle(img, weight=0.1, eps=0.0002, max_num_iter=200, channel_axis=False)

median = cv2.medianBlur(img, 3)
edges = cv2.Canny(median, 160, 200)
plt.imshow(edges, cmap="gray")


plt.imshow(median, cmap='gray')
# plt.hist(median.flat, bins=100, range=(100,255))  #.flat returns the flattened numpy array (1D)
thresholds = threshold_multiotsu(median, classes=4)
regions = np.digitize(median, bins=thresholds)
plt.imshow(regions)

segm1 = (regions == 0)
segm2 = (regions == 1)
segm3 = (regions == 2)
segm4 = (regions == 3)
# segm5 = (regions == 4)


from scipy import ndimage as nd

segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))

segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))

# segm5_opened = nd.binary_opening(segm5, np.ones((3,3)))
# segm5_closed = nd.binary_closing(segm5_opened, np.ones((3,3)))

all_segments_cleaned = np.zeros((median.shape[0], median.shape[1], 3))

all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)
all_segments_cleaned[segm4_closed] = (1,1,0)
# all_segments_cleaned[segm5_closed] = (1,0,1)

plt.imshow(all_segments_cleaned)