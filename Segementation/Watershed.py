import cv2
import numpy


def waterShed(sourceDir):

    img = cv2.imread(sourceDir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU
    reval_O, dst_Otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Triangle
    reval_T, dst_Tri = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)

    #cv2.imshow('show', dst_Tri)
    #cv2.waitKey(0)

    # noise removal
    kernel = numpy.ones((3, 3), numpy.uint8)

    # 形态学处理:开处理,膨胀边缘
    opening = cv2.morphologyEx(dst_Tri, cv2.MORPH_OPEN, kernel, iterations=2)

    # 膨胀处理背景区域 sure background area
    dilate_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    # 计算开处理图像到邻域非零像素距离
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 正则处理
    norm = cv2.normalize(dist_transform, 0, 255, cv2.NORM_MINMAX)

    # 阈值处理距离图像,获取图像前景图
    retval_D, dst_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # 前景图格式转换
    dst_fg = numpy.uint8(dst_fg)

    # 未知区域计算:背景减去前景
    unknown = cv2.subtract(dilate_bg, dst_fg)
    cv2.imshow("Difference value", unknown)
    cv2.imwrite('/Users/jiaqi970602/PycharmProjects/Opencv_detect/unknown_reginon.png', unknown)

    # 处理连接区域
    retval_C, marks = cv2.connectedComponents(dst_fg)
    cv2.imshow('Connect marks', marks)
    cv2.imwrite('/Users/jiaqi970602/PycharmProjects/Opencv_detect/connect_marks.png', marks)

    # 处理掩模
    marks = marks + 1
    marks[unknown == 255] = 0
    cv2.imshow("marks undown", marks)

    # 分水岭算法分割
    marks = cv2.watershed(img, marks)

    # 绘制分割线
    img[marks == -1] = [255, 0, 255]
    cv2.imshow("Watershed", img)
    cv2.imwrite('/Users/jiaqi970602/PycharmProjects/Opencv_detect/watershed.png', img)
    cv2.waitKey(0)


sourceDir = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/image_roi.png"
waterShed(sourceDir)
