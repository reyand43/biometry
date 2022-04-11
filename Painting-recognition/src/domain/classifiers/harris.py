from typing import List
from domain.classifiers.base_classifier import BaseClassifier
import numpy as np
import cv2

class HarrisClassifier(BaseClassifier):
    def get_features(self, image: np.ndarray) -> List:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        return dst