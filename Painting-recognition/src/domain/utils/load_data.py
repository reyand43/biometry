
from typing import List, Tuple
from domain.config.settings import DATA_CONF
import os
import cv2


def load() -> List:
    data = []
    for style in DATA_CONF:
        files = os.listdir(DATA_CONF[style])
        files.sort()
        for filename in files:
            f = os.path.join(DATA_CONF[style], filename)
            img = cv2.imread(f)
            data.append(
                (
                    img,
                    style
                )
            )
    return data



