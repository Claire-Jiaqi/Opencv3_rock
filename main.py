import cv2
import numpy as np


def main():
    # Read image
    img_path = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/images/rock1.jpg"
    img = cv2.imread(img_path, 0)

    # 创建一个窗口
    cv2.namedWindow("image", flags=cv2.WINDOW_AUTOSIZE)

    cv2.namedWindow("image_roi", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

    cv2.imshow("image", img)

    showCrosshair = False

    # 如果为Ture的话 , 则鼠标的其实位置就作为了roi的中心
    # False: 从左上角到右下角选中区域

    fromCenter = False

    # Select ROI
    rect = cv2.selectROI("image", img, showCrosshair, fromCenter)

    print("选中矩形区域")
    (x, y, w, h) = rect

    # Crop image
    imCrop = img[y: y + h, x:x + w]

    # Display cropped image
    cv2.imshow("image_roi", imCrop)
    cv2.imwrite("image_roi.png", imCrop)
    cv2.waitKey(0)


main()