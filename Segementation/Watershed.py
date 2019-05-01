import cv2
import numpy
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def waterShed(sourceDir):

    img = cv2.imread(sourceDir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU
    reval_O, dst_Otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Triangle
    reval_T, dst_Tri = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)

    plt.hist(gray.ravel(), 256, [0, 256])
    plt.show()

    cv2.imshow('show', dst_Tri)

    ret, thresh1 = cv2.threshold(gray, reval_T, 255, cv2.THRESH_BINARY)
    cv2.imshow('After threshold', thresh1)

    # noise removal
    kernel = numpy.ones((3, 3), numpy.uint8)

    # use closing to de-noise
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('After opening', closing)

    # erode the edge, sure background area
    dilate_bg = cv2.erode(closing, kernel, iterations=2)
    cv2.imshow('After erode', dilate_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
    dist_output = cv2.normalize(dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow('Distance transform', dist_output*70)


    # 阈值处理距离图像,获取图像前景图
    retval_D, dst_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    cv2.imshow('Find the seed', dst_fg)


    # 前景图格式转换
    dst_fg = numpy.uint8(dst_fg)

    # 未知区域计算:背景减去前景
    unknown = cv2.subtract(dilate_bg, dst_fg)
    cv2.imshow("Difference value", unknown)
    cv2.imwrite('/Users/jiaqi970602/PycharmProjects/Opencv_detect/unknown_reginon.png', unknown)

    # Marker labelling
    retval_C, marks = cv2.connectedComponents(dst_fg)
    cv2.imshow('Connect marks', marks)
    cv2.imwrite('/Users/jiaqi970602/PycharmProjects/Opencv_detect/connect_marks.png', marks)

    # Add one to all labels so that sure background is not 0, but 1
    marks = marks + 1
    # Now, mark the region of unknown with zero
    marks[unknown == 255] = 0
    cv2.imshow("marks undown", marks)

    # 分水岭算法分割
    marks = cv2.watershed(img, marks)

    # 绘制分割线
    img[marks == -1] = [255, 0, 0]

   # img[marks == -1] = [255, 0, 255]

    cv2.imshow("Watershed", img)
    cv2.imwrite('/Users/jiaqi970602/PycharmProjects/Opencv_detect/watershed.png', img)
    cv2.waitKey(0)


sourceDir = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/image_roi_1.png"
waterShed(sourceDir)
