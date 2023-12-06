import pydicom as dicom
import matplotlib.pylab as plt

import numpy as np
import cv2
import pydicom as dicom
image_path = 'Breast_MRI_153/01-01-1990-NA-MRI BREAST BILATERAL WWO-78209/2.000000-ax dyn pre-75557/1-092.dcm'

ds = dicom.dcmread(image_path)
img = cv2.imread(image_path)
dcm_sample = ds.pixel_array*128
cv2.imshow('sample image dicom', img)

cv2.waitKey(0)

ret, thresh = cv2.threshold(dcm_sample, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret, thresh1 = cv2.threshold(dcm_sample, 127, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(dcm_sample, 100, 200)

cv2.imshow('sample image dicom thresh', edges)

cv2.waitKey(0)
plt.imshow(dcm_sample)
plt.show()

# specify your image path
# ds = dicom.dcmread(image_path)
#
# plt.imshow(ds.pixel_array)
# plt.show()