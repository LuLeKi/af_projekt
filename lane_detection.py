import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class LaneDetection:

    debug_image = None
    THRESHOLDING_POINT = 40

    def __init__(self):
        pass

    def normalize_floats(self, matrix):
        matrix = 255 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        return matrix


    def detect(self, state_image):
        # turn image to grayscale
        gray_img = np.dot(state_image[...,:3], [0.299, 0.587, 0.114])
        gray_normalized = self.normalize_floats(gray_img) 

        # now convolve the image
        # i want to use prewitt
        kx = np.array([[-1, 0, 1] for i in range(0,3)], dtype=np.float32)
        ky = np.array([[1 - i, 1 - i, 1 - i] for i in range(0, 3)], dtype=np.float32)
        # print(kx.shape, ky.shape)

        cx = convolve2d(gray_normalized, kx, mode="same", boundary="symm")
        cy = convolve2d(gray_normalized, ky, mode="same", boundary="symm")

        grad = np.sqrt(cx**2, cy**2)
        # print(grad)
        grad = self.normalize_floats(grad)

        #remove car from image
        carx = 48 
        cary = 64 
        carh = 13 
        grad[cary:cary + carh, carx - 3:carx + 3] = 0 
        #remove stats bottom from image
        grad[cary + carh:] = 0

        # thresholding
        grad = (grad > self.THRESHOLDING_POINT) * grad * (255 / grad)
 

        # print(cx, cy)

        # rescale in the end
        # gray_normalized = grad.astype(np.uint8)

        # plt.imshow(gray_normalized, cmap='jet')
        # plt.colorbar()  
        # mplt.title("Gradient Magnitude with Colormap")
        # plt.show()
        self.debug_image = grad 
       # self.debug_image = state_image
        pass
