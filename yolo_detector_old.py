from dataclasses import dataclass
from time import time
import pathlib
from glob import glob
from typing import Union, Tuple
import pandas as pd
import torch
import numpy as np
import cv2

from custom_decorators import timeit
from custom_logger import logger


"""
To avoid 'cannot instantiate 'PosixPath' on your system. Cache may be out of date, try `force_reload=True`' error
for some reason i get this error on this model, it didn't happened using models I've trained in the past
"""
pathlib.PosixPath = pathlib.WindowsPath


@dataclass
class Detector:
    model_path: str
    conf_threshold: float = .1
    ultralitycs_path: str = "ultralytics/yolov5"
    model_type: str = "custom"
    force_reload: bool = True

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info(
            "Initing detector:\n"
            f"{self.model_path=}\n"
            f"{self.conf_threshold=}\n"
            f"{self.model_type=}\n"
            f"{self.force_reload=}\n"
        )
        self.model = torch.hub.load(self.ultralitycs_path, self.model_type, self.model_path, self.force_reload)
        self.model.conf = self.conf_threshold
        logger.info(f"Model loaded")

    @timeit(logger=logger)
    def detect(self, img: Union[str, np.array]) -> Tuple[np.array, pd.DataFrame]:
        results = self.model([img])

        return np.squeeze(results.render()), results.pandas().xyxy[0]


if __name__ == '__main__':
    detector = Detector(model_path=fr"C:\Users\table\PycharmProjects\MojeCos\DamagedRoad\models\yolov5_1.pt")
    # image = cv2.imread(r"C:\Users\table\PycharmProjects\MojeCos\DamagedRoad\TestData\photo_2025-04-04_08-24-13.jpg")
    # converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_draw, res = detector.detect(img=converted)
    # image_draw = cv2.resize(image_draw, (1280, 720))
    # cv2.imshow('MainWindow2', image_draw)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture(r"C:\Users\table\PycharmProjects\MojeCos\DamagedRoad\Videos\vid7.mp4")
    p_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        print(frame.shape)
        frame = cv2.resize(frame, (500, 283))
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_draw, res = detector.detect(img=converted)

        # image_draw = cv2.resize(image_draw, (500, 800))
        c_time = time()
        fps = int(1 / (c_time - p_time))
        p_time = c_time
        cv2.putText(image_draw, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
        cv2.imshow('MainWindow2', image_draw)
        cv2.waitKey(1)
