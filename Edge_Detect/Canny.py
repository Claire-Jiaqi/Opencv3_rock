import cv2
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def doCanny(img):
    # filename = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/image_roi.png"
    # img = cv2.imread(filename, 0)

    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def addaptiveCanny():
    filename = "/Users/jiaqi970602/PycharmProjects/Opencv_detect/image_roi.png"
    img = cv2.imread(filename, 0)

    img_1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    img_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    titles = ['GUASSIAN', 'MEAN', 'Simper']
    imgs = [img_1, img_2, img]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imgs[i], 'gray')
        plt.title(titles[i])
    plt.show()

