import numpy as np
from PIL import Image
import pydicom
image_path = 'Breast_MRI_153/01-01-1990-NA-MRI BREAST BILATERAL WWO-78209/2.000000-ax dyn pre-75557/1-092.dcm'
im1 = "Breast_MRI_153/01-01-1990-NA-MRI BREAST BILATERAL WWO-78209/4.000000-ax dyn 1st pass-83574/1-090.dcm"

im = pydicom.dcmread(im1)

im = im.pixel_array.astype(float)

rescaled_image = (np.maximum(im, 0)/im.max()) * 255
final_image = np.uint8(rescaled_image)

final_image = Image.fromarray(final_image)
final_image.show()
final_image.save('img1.jpg')
final_image.save('img1.png')
