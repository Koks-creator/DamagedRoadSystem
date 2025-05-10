from dataclasses import dataclass
from time import time
from collections import defaultdict
from typing import Union, Tuple, Generator
from pathlib import Path
import numpy as np
import cv2

from sort_tracker import Sort
from yolo_detector import YoloDetector
from config import Config
from custom_decorators import timeit, log_call
from custom_logger import logger


@dataclass
class DamagedRoadSystem:
    model_path: Union[str, Path] = Config.MODEL_PATH
    classes_path: Union[str, Path] = Config.CLASSES_PATH
    device: str = Config.DEVICE # "cpu", "cuda:0"
    sort_max_age: int = Config.SORT_MAX_AGE
    sort_min_hits: int = Config.SORT_MIN_HITS
    sort_iou_threshold: float = Config.SORT_IOU_THRESHOLD

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info(
            "Initing damaged road system:\n" \
            f"{self.model_path=}\n"
            f"{self.classes_path=}\n"
            f"{self.device=}\n"
            f"{self.sort_max_age=}\n"
            f"{self.sort_min_hits=}\n"
            f"{self.sort_iou_threshold=}\n"
        )
        self.yolo_detector = YoloDetector(
            model_path=self.model_path,
            device=self.device,
            classes_path=self.classes_path
        )
        logger.info("Model loaded")

        self.sorttr = Sort(
            max_age=self.sort_max_age,
            min_hits=self.sort_min_hits,
            iou_threshold=self.sort_iou_threshold
        )
        logger.info("Sort alg loaded")
    
    def create_empty_summary(self) -> dict:
        return {class_name: set() for class_name in self.yolo_detector.classes_list}

    @timeit(logger=logger)
    def draw_bbox(self, img: np.ndarray, bbox: Tuple[int, int, int, int], class_name: str, 
                  obj_id: int, conf: float, colors: dict = None, font_size: float = 1.4,
                  font_thick: int = 2, bbox_thick: int = 2
                  ) -> None:
        if colors is None:
            colors = self.yolo_detector.colors
            
        color = colors[class_name]
        cv2.rectangle(img, bbox[:2], bbox[2:4], color, bbox_thick)
        cv2.putText(img, f"{class_name} {conf} - {obj_id}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN,
                    font_size, color, font_thick
                    )
    
    @staticmethod
    def draw_summary(frame: np.array, summary: dict, x: int, y: int, font_size: float = 1.4, 
                     font_thick: int = 2, color: tuple[int, int, int] = (200, 200, 200), y_step: int = 20) -> None:
        y_start = y
        for key, val in summary.items():
            cv2.putText(frame, f"{key}: {len(val)}", (x, y_start), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_thick)
            y_start += y_step

    @log_call(logger=logger, log_params=["use_sahi", "conf", "iou", "augment", "sahi_conf", 
                                         "sahi_slice_height", "sahi_slice_width", "sahi_overlap_height_ratio", 
                                         "sahi_overlap_width_ratio", "track", "draw_track", "min_det_frames", 
                                         "font_size", "font_thick", "bbox_thick"], hide_res=True)
    @timeit(logger=logger)
    def process_frame(self, frame: np.array, use_sahi: bool = True, conf: float = .2, iou: float = .35, 
                      augment: bool = True, sahi_conf: float = 0.2, sahi_slice_height: int = 256, 
                      sahi_slice_width: int = 256, sahi_overlap_height_ratio: float = 0.2, 
                      sahi_overlap_width_ratio: float = 0.2, detection_history: defaultdict[int] = None,
                      track: bool = True, draw_track: bool = True, min_det_frames: int = 5, 
                      font_size: float = 1.4, font_thick: int = 2, bbox_thick: int = 2
                      ) -> tuple[np.array, np.array, dict, np.array, Generator]:
        summary = self.create_empty_summary()
        if use_sahi:
            detection_res, detection_frame = self.yolo_detector.detect_with_sahi(images=[frame], conf=sahi_conf, 
                                                                                 slice_height=sahi_slice_height, 
                                                                                 slice_width=sahi_slice_width, 
                                                                                 overlap_height_ratio=sahi_overlap_height_ratio,
                                                                                 overlap_width_ratio=sahi_overlap_width_ratio
                                                                                )[0] # only 1 frame
            detection_res_gen = self.yolo_detector.yield_sahi_data(sahi_result=detection_res)
        else:
            detection_res, detection_frame = self.yolo_detector.detect(images=[frame], conf=conf, iou=iou, augment=augment)[0] # only 1 frame
            detection_res_gen = self.yolo_detector.yield_data(bbox=detection_res)
        #(2, 'block cracking', 0.44653183221817017, (769, 159, 966, 234)) sahi
        #(0, 'hole', 0.22668276727199554, (735, 166, 819, 191)) reg yolo
        track_data = []
        if track:
            for detection in detection_res_gen:
                class_id, _, conf, x1, y1, x2, y2 = *detection[:3], *detection[3]
                track_data.append([x1, y1, x2, y2, conf, class_id])

            updated_tracks = self.sorttr.update(track_data)
            current_detections = set()
            for track_data in updated_tracks:
                conf = round(track_data[-3], 2) # getting conf val like this cuz below i am casting all of this shit to int yes yes
                x1, y1, x2, y2, _, obj_id, class_id = track_data.astype(int)
                class_name = self.yolo_detector.classes_list[class_id]

                detection_key = f"{class_name}_{obj_id}"

                # checking if object was seen X times to avoid counting same shit
                current_detections.add(detection_key)
                detection_history[detection_key] += 1
                if detection_history[detection_key] >= min_det_frames:
                    summary[class_name].add(obj_id)

                if draw_track:
                    self.draw_bbox(img=frame, bbox=(x1, y1, x2, y2), class_name=class_name, obj_id=obj_id, 
                                   conf=conf, font_size=font_size, font_thick=font_thick, bbox_thick=bbox_thick)

            # resting counter for objects not seen in the current frame
            keys_to_remove = [key for key in detection_history if key not in current_detections]
            for key in keys_to_remove:
                del detection_history[key]
        return detection_frame, frame, summary, track_data, detection_res_gen

    @log_call(logger=logger, log_params=["vid_cap", "use_sahi", "conf", "iou", "augment", "sahi_conf", 
                                         "sahi_slice_height", "sahi_slice_width", "sahi_overlap_height_ratio", 
                                         "sahi_overlap_width_ratio", "track", "draw_track", "min_det_frames", 
                                         "font_size", "font_thick", "bbox_thick"])
    @timeit(logger=logger) 
    def main(self, vid_cap: Union[int, Path, str], use_sahi: bool = True, conf: float = .2, 
             iou: float = .35, augment: bool = True, sahi_conf: float = 0.2, 
             sahi_slice_height: int = 256, sahi_slice_width: int = 256, 
             sahi_overlap_height_ratio: float = 0.2, sahi_overlap_width_ratio: float = 0.2, 
             track: bool = True, draw_track: bool = True, min_det_frames: int = 5,
             font_size: float = 1.4, font_thick: int = 2, bbox_thick: int = 2
             ) -> None:
        
        summary = self.create_empty_summary()
        detection_hist = defaultdict(int)

        cap = cv2.VideoCapture(vid_cap)
        p_time = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            detection_frame, frame, summary_frame, track_data, detection_res_gen = self.process_frame(
                frame=frame, 
                use_sahi=use_sahi, 
                conf=conf, 
                iou=iou, 
                augment=augment, 
                sahi_conf=sahi_conf, 
                sahi_slice_height=sahi_slice_height, 
                sahi_slice_width=sahi_slice_width, 
                sahi_overlap_height_ratio=sahi_overlap_height_ratio, 
                sahi_overlap_width_ratio=sahi_overlap_width_ratio,
                detection_history=detection_hist,
                track=track,
                draw_track=draw_track,
                min_det_frames=min_det_frames,
                font_size=font_size,
                font_thick=font_thick,
                bbox_thick=bbox_thick
            )
            if track:
                # Update
                for key, val in summary_frame.items():
                    summary[key].update(val)
                
                self.draw_summary(frame=frame, summary=summary, x=30, y=30)
            else:
                frame = detection_frame
            
            c_time = time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time
            cv2.putText(frame, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)

            key = cv2.waitKey(1)
            if key == 27:
                break
            cv2.imshow("Result", frame)
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    mocne = DamagedRoadSystem(
        model_path=Config.MODEL_PATH,
        classes_path=Config.CLASSES_PATH
    )
    mocne.main(
        vid_cap=rf"{Config.VIDEOS_FOLDER}/vid4.mp4",
        use_sahi=Config.USE_SAHI,
        conf=Config.CONF_THRESH,
        iou=Config.IOU,
        augment=Config.AUGMENT,
        sahi_conf=Config.SAHI_CONF_THRESH,
        sahi_slice_height=Config.SAHI_SLICE_HEIGHT,
        sahi_slice_width=Config.SAHI_SLICE_WIDTH,
        sahi_overlap_height_ratio=Config.SAHI_OVERLAP_HEIGHT_RATIO,
        sahi_overlap_width_ratio=Config.SAHI_OVERLAP_WIDTH_RATIO,
        track=Config.TRACK,
        draw_track=Config.DRAW_TRACK,
        min_det_frames=Config.MIN_DET_FRAMES,
        font_size=Config.FONT_SIZE,
        font_thick=Config.FONT_THICK,
        bbox_thick=Config.BBOX_THICK
    )
