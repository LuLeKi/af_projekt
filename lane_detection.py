import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class LaneDetection:

    debug_image = None
    # for randomize optim calculte histogram and then 
    # improve contrast
    THRESHOLDING_POINT = 50

    def __init__(self):
        pass

    def normalize_floats(self, matrix):
        matrix = 255 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        return matrix

    def binary_dilation(self, img, structure=None):
        img = img > 0
        if structure is None:
            structure = np.ones((3, 3), dtype=bool)

        # Perform convolution
        convolved = convolve2d(img.view(np.uint8), structure.view(np.uint8), mode='same', boundary='fill', fillvalue=0)

        # Any nonzero result means at least one neighbor was active
        return convolved > 0

    def grow_region(self, img, start, structure=None):
        bool_image = img > 0
        seed = np.zeros_like(bool_image, dtype=bool)
        seed[start] = True

        prev = None
        curr = seed.copy()

        while prev is None or not np.array_equal(prev, curr):
            prev = curr.copy()
            curr = self.binary_dilation(curr, structure) & bool_image
 
        return curr.astype(np.uint8) * 255


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

        left_x, right_x = -1, -1
        edges = np.where(grad[cary] > 0)[0]
        if len(edges) >= 2:
            left_x = edges[0]     # leftmost
            right_x = edges[-1]    # rightmost
        else:
            # fallback if not enough edges
            left_x, right_x = -1, -1

        left_lane = self.grow_region(grad, (cary, left_x), structure=np.ones((3, 3), dtype=bool))
        right_lane = self.grow_region(grad, (cary, right_x), structure=np.ones((3, 3), dtype=bool))

        # rescale in the end
        # gray_normalized = grad.astype(np.uint8)

        # plt.imshow(gray_normalized, cmap='jet')
        # plt.colorbar()
        # mplt.title("Gradient Magnitude with Colormap")
        # plt.show()
        left_lane = np.stack((left_lane,) * 3, axis=-1) * (100, 98, 200) 
        right_lane = np.stack((right_lane,) * 3, axis=-1) * (50, 200, 180) 
        self.debug_image = left_lane + right_lane 
        print(self.debug_image)
      # self.debug_image = state_image
        pass
