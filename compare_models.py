from dataclasses import dataclass
from glob import glob
import numpy as np
from pathlib import Path
from typing import Union
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
import cv2
import time

from yolo_detector_old import Detector
from yolo_detector import YoloDetector
from config import Config

class ModelTypes(Enum):
    OLD: str = "old"
    NEW: str = "new"
    NEW_SAHI: str = "new_sahi"


b = plt.get_backend()


@dataclass
class CompareModels:
    models_config: dict

    def __post_init__(self) -> None:
        matplotlib.use(b) # turbo important, needs to be used after initing models, plt.show() can work

    def compare_on_video(self, video_path: Union[str, Path], cols: int = 2, start_y: int = 40, x_padding: int = 20, 
                        y_padding: int = 40, display_width: int = 640, display_height: int = 320
                        ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        window_positions = {}
        multiplier = 0
        for i, model_name in enumerate(self.models_config.keys()):
            cv2.namedWindow(model_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(model_name, display_width, display_height)
            
            if i > 0:
                if i % cols == 0:
                    start_y += display_height + y_padding
                    multiplier = 0
                else:
                    multiplier += 1
            
            x_position = multiplier * (display_width + x_padding)
            # start_x += display_width
            cv2.moveWindow(model_name, x_position, start_y)
            
            window_positions[model_name] = (x_position, start_y)
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break
                
                for model_name, model_attr in self.models_config.items():
                    model_detector = model_attr["detector"]
                    model_func_param = model_attr.get("detector_func_params", {})
                    detector_type = model_attr["type"]
                    frame_copy = frame.copy()
                    
                    # measure processing time
                    start_time = time.time()
                    
                    if detector_type == ModelTypes.OLD.value:
                        res_frame = model_detector.detect(img=frame_copy)[0]
                    elif detector_type == ModelTypes.NEW.value:
                        res_frame = model_detector.detect(images=[frame_copy], **model_func_param)[0][1]
                    elif detector_type ==  ModelTypes.NEW_SAHI.value:
                        res_frame = model_detector.detect_with_sahi(images=[frame_copy], **model_func_param)[0][1]

                    processing_time = (time.time() - start_time) * 1000
                    
                    res_frame_resized = cv2.resize(res_frame, (display_width, display_height))
                    
                    cv2.putText(res_frame_resized, model_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(res_frame_resized, f"Proc time {processing_time:.2f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    
                    cv2.imshow(model_name, res_frame_resized)
            
            key = cv2.waitKey(1)
            
            if key == 27:
                print("Quiting...")
                break
                
            elif key == 32:
                paused = not paused
                status = "pause" if paused else "resumed"
                print(f"{status=}")

        cap.release()
        cv2.destroyAllWindows()

    def compare_on_images(self, image: np.array, figsize1: tuple = (15, 5), figsize2: tuple = (10, 5), rows: int = 2) -> None:
        images = []
        labels = []
        processing_times = []
        
        for model_name, model_attr in self.models_config.items():
            model_detector = model_attr["detector"]
            model_func_param = model_attr.get("detector_func_params", {})
            detector_type = model_attr["type"]

            frame_copy = image.copy()
            
            # measure processing time
            start_time = time.time()
                    
            if detector_type == ModelTypes.OLD.value:
                res_frame = model_detector.detect(img=frame_copy)[0]
            elif detector_type == ModelTypes.NEW.value:
                res_frame = model_detector.detect(images=[frame_copy], **model_func_param)[0][1]
            elif detector_type ==  ModelTypes.NEW_SAHI.value:
                res_frame = model_detector.detect_with_sahi(images=[frame_copy], **model_func_param)[0][1]
            

            processing_time = (time.time() - start_time) * 1000 
            
            res_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
            cv2.putText(res_frame, f"Proc time {processing_time:.2f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            images.append(res_frame)
            labels.append(f"{model_name} - {processing_time:.2f} ms")
            processing_times.append(processing_time)

        n = len(images)
        plt.figure(figsize=figsize1)
        rows = 2
        cols = (n + 1) // 2 

        for i, (image, title) in enumerate(zip(images, labels)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image)
            plt.title(title)
            plt.axis("off")

        plt.figure(figsize=figsize2)
        plt.bar(labels, processing_times, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Processing Time (ms)')
        plt.title('Model Processing Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    models = {
        "yolov5_1": {
            "detector": Detector(model_path=f"{Config.MODELS_PATH}/yolov5_1.pt", conf_threshold=.2),
            "type": ModelTypes.OLD.value
        },
        "yolo11_1": {
            "detector": YoloDetector(model_path=f"{Config.MODELS_PATH}/yolo11_1.pt", 
                                        classes_path=f"{Config.MODELS_PATH}/classes.txt"),
            "detector_func_params": {"conf": .2, "iou": .35, "augment": True, "agnostic_nms": True},
            "type": ModelTypes.NEW.value
        },
        "yolo11_2": {
            "detector": YoloDetector(model_path=f"{Config.MODELS_PATH}/yolo11_2.pt", 
                                        classes_path=f"{Config.MODELS_PATH}/classes.txt"),
            "detector_func_params": {"conf": .2, "iou": .35, "augment": True, "agnostic_nms": True},
            "type": ModelTypes.NEW.value
        },
        "yolo11_1_sahi": {
            "detector": YoloDetector(model_path=f"{Config.MODELS_PATH}/yolo11_1.pt", 
                                        classes_path=f"{Config.MODELS_PATH}/classes.txt"),
            "detector_func_params": {"slice_height": 480, "slice_width": 480},
            "type": ModelTypes.NEW_SAHI.value
        },
    }
    compare_models = CompareModels(
        models_config=models
    )
    for file_path in glob(f"{Config.VIDEOS_FOLDER}/*.*"):
        compare_models.compare_on_video(video_path=file_path)
    
    # for file_path in glob(f"{Config.TEST_DATA_FOLDER}/*.*"):
    #     image = cv2.imread(file_path)
    #     compare_models.compare_on_images(image=image)