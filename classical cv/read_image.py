import cv2

img1 = cv2.imread('lec_1\images\download.jpg')


cv2.imshow("The original mage",img1)


cv2.waitKey(0)

# save image

cv2.imwrite('saved_original_image.jpg',img1)