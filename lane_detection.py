import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt



class LaneDetection:

    debug_image = None
    left_lane = None
    right_lane = None
    # for randomize optim calculte histogram and then 
    # improve contrast
    THRESHOLDING_POINT = 50

    def __init__(self):
        pass

    def align_to_wrapper(self, detected_pts: np.ndarray) -> np.ndarray:
        """
        detected_pts: (N,2) array of (v,row, u,col) in [0..95] state‐pixel coords
        returns:      (N,2) array of (x, y) in same frame as
                    _get_lane_boundary_groundtruth  i.e. origin bottom‐left.
        """
        vs = detected_pts[:, 0].astype(float)
        us = detected_pts[:, 1].astype(float)

        xs = vs[::-1] 
        ys = us[::-1] 

        return np.stack((xs, ys), axis=1)


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

    def grow(self, img, start):
        mask = img > 0
        visited = np.zeros_like(mask, dtype=bool)
        stack = [start]
        height, width = mask.shape

        while stack:
            y, x = stack.pop()
            if not (0 <= y < height and 0 <= x < width):
                continue
            if visited[y, x] or not mask[y, x]:
                continue

            visited[y, x] = True
            # add four pints around current coordinate to stack
            # to then check if they are in the mask(image)
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

        return visited.astype(np.uint8) * 255

    def detect(self, state_image):
        # turn image to grayscale
        gray_img = np.dot(state_image[...,:3], [0.299, 0.587, 0.114])
        gray_normalized = self.normalize_floats(gray_img) 

        # now convolve the image
        # i want to use prewitt
        kx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]], dtype=np.float32)  # vertical filter

        ky = np.array([[ 1,  1,  1],
               [ 0,  0,  0],
               [-1, -1, -1]], dtype=np.float32)  
        # print(kx.shape, ky.shape)

        cx = convolve2d(gray_normalized, kx, mode="same", boundary="symm")
        cy = convolve2d(gray_normalized, ky, mode="same", boundary="symm")

        grad = np.sqrt(cx**2 + cy**2)
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

        #left_lane = self.grow_region(grad, (cary, left_x), structure=np.ones((3, 3), dtype=bool))
        #right_lane = self.grow_region(grad, (cary, right_x), structure=np.ones((3, 3), dtype=bool))
        left_lane = self.grow(grad, (cary, left_x))
        right_lane = self.grow(grad, (cary, right_x))

        # left_lane = self.reconstruct_from_seed(grad, (cary, left_x))
        # right_lane = self.reconstruct_from_seed(grad, (cary, right_x))
        # rescale in the end
        # gray_normalized = grad.astype(np.uint8)

        # plt.imshow(gray_normalized, cmap='jet')
        # plt.colorbar()
        # mplt.title("Gradient Magnitude with Colormap")
        # plt.show()
        left_lane_mask = left_lane.astype(bool)[:, :, np.newaxis]
        right_lane_mask = right_lane.astype(bool)[:, :, np.newaxis]
        initial_lanes_mask = (grad > 0).astype(bool)[:, :, np.newaxis]
        lanes_mask = left_lane_mask | right_lane_mask | initial_lanes_mask 
        self.debug_image = np.where(initial_lanes_mask, np.stack((grad,) * 3, axis=-1) * (1, 1, 1), self.debug_image)
        self.debug_image = np.where(lanes_mask, np.stack((left_lane,) * 3, axis=-1) * (0, 0, 1), state_image)
        self.debug_image = np.where(right_lane_mask, np.stack((right_lane,) * 3, axis=-1) * (0, 1, 0), self.debug_image)

        self.left_lane = left_lane
        self.right_lane = right_lane

        #return an array of coordinates where the values are non-zero for every lane
        left_lane_points = np.argwhere(left_lane > 0)
        right_lane_points = np.argwhere(right_lane > 0)

        unique_y = {}
        for y, x in left_lane_points:
            unique_y[y] = x

        left_lane_points = np.array([[x, y] for y, x in unique_y.items()])
        if left_lane_points.size == 0:
            left_lane_points = np.empty((0, 2))

        unique_y = {}
        for y, x in right_lane_points:
            if y not in unique_y or abs(x - unique_y[y]) > 1:
                unique_y[y] = x

        right_lane_points = np.array([[x, y] for y, x in unique_y.items()])
        if right_lane_points.size == 0:
            right_lane_points = np.empty((0, 2)) 

        if left_lane_points.size > 0:
            left_lane_points = left_lane_points[left_lane_points[:, 1].argsort()]
            new_y = np.linspace(left_lane_points[:, 1].min(), left_lane_points[:, 1].max(), 120)
            left_lane_x = np.interp(new_y, left_lane_points[:, 1], left_lane_points[:, 0])
            left_lane_points = np.stack((left_lane_x, new_y), axis=1)

        if right_lane_points.size > 0:
            right_lane_points = right_lane_points[right_lane_points[:, 1].argsort()]
            new_y = np.linspace(right_lane_points[:, 1].min(), right_lane_points[:, 1].max(), 120)
            right_lane_x = np.interp(new_y, right_lane_points[:, 1], right_lane_points[:, 0])
            right_lane_points = np.stack((right_lane_x, new_y), axis=1)
            

        return self.align_to_wrapper(left_lane_points), self.align_to_wrapper(right_lane_points)
