import cv2

img = cv2.imread('lec_1\images\download.jpg')


resizing = cv2.resize(img,(640,480))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blurring = cv2.GaussianBlur(img,(15,15),0)

edge = cv2.Canny(img,150,300)

cv2.imshow('Resized Image',resizing)
cv2.imshow('Gray Image',gray)
cv2.imshow('blurring Image',blurring)
cv2.imshow('Edge Image',edge)

cv2.waitKey(0)
cv2.destroyAllWindows()
