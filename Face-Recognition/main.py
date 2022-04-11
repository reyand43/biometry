import os
from typing import List

import cv2
from sklearn.datasets import fetch_olivetti_faces

DATABASE_CONF = {
    "ORL": {
        "number_group": 40,
        "number_img": 10,
        "img_path": "./data/ORL/s{g}/{im}.png",
    },
    "ORL_mask": {
        "number_group": 40,
        "number_img": 10,
        "img_path": "./data/ORL_mask/s{g}/{im}-with-mask.jpg",
        "cropped_img_path": "./data/ORL_mask/s{g}/{im}-with-mask-cropped.jpg",
    },
    "ORL_fawkes": {
        "number_group": 40,
        "number_img": 10,
        "img_path": "./data/ORL_fawkes/s{g}/{im}_cloaked.jpg",
        "cropped_img_path": "./data/ORL_fawkes/s{g}/{im}_cloaked_cropped.jpg",
    }
}

def upload(database):
    data_images = fetch_olivetti_faces()
    database_data = data_images["images"]
    for g_i in range(DATABASE_CONF[database]["number_group"]):
        os.mkdir("./data/ORL/s{g}".format(g=g_i + 1))
        for im_i in range(DATABASE_CONF[database]["number_img"]):
            print(
                DATABASE_CONF["ORL"]["img_path"]
                .format(g=g_i + 1, im=im_i + 1)
            )
            img = database_data[g_i * 10 + im_i] * 255
            cv2.imwrite(
                DATABASE_CONF["ORL"]["img_path"]
                .format(g=g_i + 1, im=im_i + 1),
                img
            )

def crop(database) -> List:
    width, height = 64, 64
    x, y = 14, 30
    for g_i in range(DATABASE_CONF[database]["number_group"]):
            database_group = []
            for im_i in range(DATABASE_CONF[database]["number_img"]):
                img = cv2.imread(
                    DATABASE_CONF[database]["img_path"]
                    .format(g=g_i + 1, im=im_i + 1),
                    -1
                )
                print(
                DATABASE_CONF[database]["cropped_img_path"]
                .format(g=g_i + 1, im=im_i + 1)
                )
                crop_img = img[y:y+height, x:x+width]
                cv2.imwrite(DATABASE_CONF[database]["cropped_img_path"]
                .format(g=g_i + 1, im=im_i + 1), crop_img)
                
crop("ORL_fawkes")