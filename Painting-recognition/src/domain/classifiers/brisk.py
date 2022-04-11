from typing import List
from domain.classifiers.base_classifier import BaseClassifier
import numpy as np
import cv2

class BRISKClassifier(BaseClassifier):
    def get_features(self, image: np.ndarray) -> List:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        BRISK = cv2.BRISK_create()
        keypoints, descriptors = BRISK.detectAndCompute(gray, None)
        return descriptors