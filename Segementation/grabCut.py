import cv2
import numpy


def grab_cut(sourceDir):
    # 读取图片
    img = cv2.imread(sourceDir)
    # 图片宽度
    img_x = img.shape[1]
    # 图片高度
    img_y = img.shape[0]
    # 分割的矩形区域
    rect = (96,1, 359, 358)
    # 背景模式,必须为1行,13x5列
    bgModel = numpy.zeros((1, 65), numpy.float64)
    # 前景模式,必须为1行,13x5列
    fgModel = numpy.zeros((1, 65), numpy.float64)
    # 图像掩模,取值有0,1,2,3
    mask = numpy.zeros(img.shape[:2], numpy.uint8)
    # grabCut处理,GC_INIT_WITH_RECT模式
    cv2.grabCut(img, mask, rect, bgModel, fgModel, 4, cv2.GC_INIT_WITH_RECT)
    # grabCut处理,GC_INIT_WITH_MASK模式
    # cv2.grabCut(img, mask, rect, bgModel, fgModel, 4, cv2.GC_INIT_WITH_MASK)
    # 将背景0,2设成0,其余设成1
    mask2 = numpy.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    # 重新计算图像着色,对应元素相乘
    img = img*mask2[:, :, numpy.newaxis]
    cv2.imshow("Result", img)
    cv2.waitKey(0)


sourceDir = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/image_roi.png"
grab_cut(sourceDir)
