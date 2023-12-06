import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('img1.jpg', cv.IMREAD_GRAYSCALE)
blur = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow("blur method", blur)

cv.waitKey(0)


edges = cv.Canny(blur, 160, 200)
ret2, th2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
ret,thresh4 = cv.threshold(img, 100,255,cv.THRESH_TOZERO)

ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

cv.imshow("Canny method", edges)

cv.waitKey(0)

cv.imshow("otsu", th2)
cv.waitKey(0)
cv.imshow("thtozero", thresh4)
cv.waitKey(0)
cv.imshow("th", thresh1)
cv.waitKey(0)





