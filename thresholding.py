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
from skimage.filters import gaussian
from skimage.segmentation import active_contour, flood_fill, flood, watershed
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.morphology import disk

img = cv2.imread("contrast_image_2.jpg", 0)
img2 = cv2.imread("contrast_image.jpg", 0)

plt.imshow(img2, cmap='gray')
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Original Image')
plt.show()

from scipy import ndimage as ndi

img2 = exposure.adjust_gamma(img2, 3)
plt.imshow(img2, cmap='gray')
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Gamma Corrected Image y = 2')
plt.show()



s = np.linspace(0, 2*np.pi, 400)
r = 200 + 200*np.sin(s)
c = 200 + 200*np.cos(s)
init = np.array([r, c]).T
snake = active_contour(gaussian(img2, sigma=1), init, alpha=0.015, beta=10, gamma=0.001)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img2, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Active contour - snakes method')

plt.show()

from skimage.util import img_as_ubyte
from skimage.filters import rank

image = img_as_ubyte(img2)

img2 = gaussian(img2, sigma=1)

markers = rank.gradient(img2, disk(5)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(img2, disk(2))

# process the watershed
labels = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
ax[2].set_title("Markers")

ax[3].imshow(image, cmap=plt.cm.gray)
ax[3].imshow(labels, cmap=plt.cm.gray, alpha=.5)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()


plt.imshow(img2, cmap='gray')
plt.show()

geodesic = morphological_geodesic_active_contour(img2, 100)
plt.imshow(geodesic, cmap='gray')
plt.show()

edges1 = feature.canny(img2)
plt.imshow(edges1, cmap='gray')
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Canny threshold')
plt.show()
edges2 = feature.canny(img2, sigma=3)
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Canny threshold sigma = 3')
plt.imshow(edges2, cmap='gray')
plt.show()

edge_roberts = filters.roberts(img2)
plt.imshow(edge_roberts, cmap='gray')
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Roberts threshold')
plt.show()
edge_sobel = filters.sobel(img2)
plt.imshow(edge_sobel, cmap='gray')
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Sobel threshold')
plt.show()
edge_laplace = filters.laplace(img2)
plt.imshow(edge_laplace, cmap='gray')
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Laplace threshold')
plt.show()
from skimage.filters import threshold_otsu, threshold_isodata, threshold_li, threshold_minimum, threshold_mean
thresh_yen = threshold_yen(img2)
bin = img2 > thresh_yen
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Yen threshold')
plt.imshow(bin, cmap='gray')
plt.show()

thresh_otsu = threshold_otsu(img2)
bin = img2 > thresh_otsu
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Otsu threshold')
plt.imshow(bin, cmap='gray')
plt.show()

thresh_minimum = threshold_minimum(img2)
bin = img2 > thresh_minimum
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Minimum threshold')
plt.imshow(bin, cmap='gray')
plt.show()


thresh_li = threshold_li(img2)
bin = img2 > thresh_li
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Li threshold')
plt.imshow(bin, cmap='gray')
plt.show()


thresh_iso = threshold_isodata(img2)
bin = img2 > thresh_iso
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Isodata threshold')
plt.imshow(bin, cmap='gray')
plt.show()

thresh_mean = threshold_mean(img2)
bin = img2 > thresh_mean
plt.xlabel('Pixels X')
plt.ylabel('Pixels Y')
plt.title('Mean threshold')
plt.imshow(bin, cmap='gray')
plt.show()
