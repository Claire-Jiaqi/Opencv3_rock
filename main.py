import cv2


def main():
    # Read image
    img_path = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/images/rock1.jpg"
    img = cv2.imread(img_path, 0)

    # use k-means method resize the original photo to 1/4

    img_test2 = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

    showCrosshair = False

    # 如果为True的话 , 则鼠标的其实位置就作为了roi的中心
    # False: 从左上角到右下角选中区域

    fromCenter = False

    # Select ROI
    rect = cv2.selectROI("image", img_test2, showCrosshair, fromCenter)

    print("选中矩形区域")
    (x, y, w, h) = rect

    # Crop image
    imCrop = img_test2[y: y + h, x: x + w]

    # Display cropped image
    cv2.imshow("image_roi", imCrop)
    cv2.imwrite("image_roi.png", imCrop)
    cv2.waitKey(0)


main()