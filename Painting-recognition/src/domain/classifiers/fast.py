from typing import List

import cv2
import numpy as np
from domain.classifiers.base_classifier import BaseClassifier

class FASTClassifier(BaseClassifier):
    def get_features(self, image: np.ndarray) -> List:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(0)
        fast.setThreshold(50)
        kp = fast.detect(gray ,None)
        for i in range(len(kp)):
            kp[i] = [kp[i].pt[0], kp[i].pt[1]]
        kp = np.array(kp).reshape(len(kp), 2)
        return kp
