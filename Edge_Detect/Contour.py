import cv2


filename = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/images/rock3.jpg"
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gray, (3, 3), 5)

ret, binary = cv2.threshold(gauss, 127, 255, cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

newImg = cv2.imread("/Users/jiaqi970602/PycharmProjects/Opencv_detect/images/rock3.jpg")

cv2.drawContours(gray, contours, -1, (0, 0, 0), 3)
cv2.imshow("img", gray)
cv2.waitKey(0)
