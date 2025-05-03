import numpy as np
import cv2
from scipy.signal import convolve2d


class LaneDetection:

    debug_image = None
    THRESHOLDING_POINT = 40

    def __init__(self):
        pass

    def normalize_floats(self, matrix):
        # Verhindert Division durch 0
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val == 0:
            return np.zeros_like(matrix)
        return 255 * (matrix - min_val) / (max_val - min_val)

    def detect(self, state_image):
        # Bild in Graustufen umwandeln
        gray_img = np.dot(state_image[...,:3], [0.299, 0.587, 0.114])
        gray_normalized = self.normalize_floats(gray_img)

        # Prewitt-Filter
        kx = np.array([[-1, 0, 1] for _ in range(3)], dtype=np.float32)
        ky = np.array([[1 - i, 1 - i, 1 - i] for i in range(3)], dtype=np.float32)

        cx = convolve2d(gray_normalized, kx, mode="same", boundary="symm")
        cy = convolve2d(gray_normalized, ky, mode="same", boundary="symm")

        grad = np.sqrt(cx**2 + cy**2)
        grad = self.normalize_floats(grad)

        # Fahrzeugmaske
        carx = 48 
        cary = 64 
        carh = 13 
        grad[cary:cary + carh, carx - 3:carx + 3] = 0 
        grad[cary + carh:] = 0

        # Schwellenwert anwenden
        grad = np.where(grad > self.THRESHOLDING_POINT, 255.0, 0.0)

        # Debug-Bild speichern
        self.debug_image = grad.astype(np.uint8)

        # Spurpunkte extrahieren: Für jede Zeile finde links und rechts jeweils die erste Kante
        left_lane = []
        right_lane = []

        for y in range(20, 84):  # sinnvoller Vertikalbereich (nach cary)
            row = grad[y]
            # Linke Spur = erste helle Kante links der Fahrzeugmitte
            left_candidates = np.where(row[:48] > 0)[0]
            if left_candidates.size > 0:
                left_lane.append([left_candidates[-1], y])  # letzter heller Pixel vor Fahrzeug

            # Rechte Spur = erste helle Kante rechts der Fahrzeugmitte
            right_candidates = np.where(row[49:] > 0)[0]
            if right_candidates.size > 0:
                right_lane.append([49 + right_candidates[0], y])  # erster heller Pixel nach Fahrzeug

        # In numpy-Arrays umwandeln
        left_lane = np.array(left_lane, dtype=np.float32)
        right_lane = np.array(right_lane, dtype=np.float32)

        return left_lane, right_lane
