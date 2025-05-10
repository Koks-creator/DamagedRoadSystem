# Road Damage Detector
this bullshit not finished yet

## Setup
  - ```pip install -r requirements.txt```

## Packages
```
filterpy==1.4.5
matplotlib==3.7.1
numpy==1.26.0
opencv_contrib_python==4.10.0.84
opencv_python==4.9.0.80
opencv_python_headless==4.9.0.80
pandas==2.0.2
Pillow==10.3.0
sahi==0.11.22
torch==2.0.1
tqdm==4.65.0
ultralytics==8.3.89
```
## Project Structure
```
├───DatasetPrepTools
│   ├───class_counts.py
│   ├───dataset_cleaner.py
│   ├───move_files.py
│   └───setup_dataset_folder.py
├───models
├───TestData
├───train_data
│   ├───images
│   │   ├───train
│   │   └───val
│   └───labels
│       ├───train
│       └───val
├───Videos
├───models
│   ├───classes.txt
│   ├───yolo11_1_tuned.pt
│   ├───yolo11_1.pt
│   └───yolov5_1.pt
│ 
│   compare_models.py
│   config.py
│   custom_logger.py
│   custom_decorators.py
│   config.py
│   sort_tracker.py
│   requirements.txt
│   yolo_detector_old.py
│   yolo_detector.py
```
## Detectors
### <u>***yolo_detector_old.py***</u>
Detector for yolov5, nothing fancy.
#### **Attributes:**
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | - | Path to the YOLOv5 model weights file (required) |
| `conf_threshold` | `float` | 0.1 | Confidence threshold for filtering detections |
| `ultralytics_path` | `str` | "ultralytics/yolov5" | Path to the ultralytics YOLOv5 implementation |
| `model_type` | `str` | "custom" | Type of model to load ("custom" or pre-trained) |
| `force_reload` | `bool` | True | Whether to force reload the model |

#### **Methods:**
`detect` - detection function that processes an image and returns detection results.
#### **Parameters:**
 - `img`: `Union[str, np.array]` - Numpy image or path to an image.

### **Basic usage:**
```python
yolo_predictor = YoloDetector(
        model_path=model_path,
)
image = cv2.imread(image_path)
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_draw, res = detector.detect(img=converted)

for detection in res:
    bbox, conf, obj_id, class_id = detection[:4], *detection[4:]
cv2.imshow("res", image_draw)
cv2.waitKey(0)
```

#### **Returns:**
- `Tuple[np.array, pd.DataFrame]` - Image with drawn detections, dataframe with detections.

### <u>***yolo_detector.py***</u>
Detector for yolov11, for both regular and sahi version.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | - | Path to the YOLOv11 model weights file (required) |
| `classes_path` | `str` | - | Path to model classes (required) |
| `device` | `str` |  `"cpu"` | Device ("cpu", "cuda:0") |

### **Methods:**
`detect` - detection function that processes an image and returns detection results.

#### **Parameters:**
 - `images`: `List[np.array]` - List of numpy images.
 - `conf`: `float` - Detection confidence threshold (0-1.0).
 - `iou`: `float` - Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
 - `augment`: `bool` - Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.
 - `agnostic_nms`: `float` - Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.

#### **Returns:**
  - `List[tuple]` - return list of tuples, each tuple contains pair of detection data and image with drawn detections.

 <br>
 <br>

`detect_with_sahi` - detection function that processes an image and returns detection results BUT with **SAHI** (Slicing Aided Hyper Inference - https://docs.ultralytics.com/guides/sahi-tiled-inference/), which helps a lot in detecting really small objects.

#### **Parameters:**
 - `images`: `List[np.ndarray]` - List of numpy images.
 - `conf`: `float` - Confidence threshold.
 - `slice_height`, `slice_width`: `int` - The larger the better detection of smaller objects but longer processing time.
 - `overlap_height_ratio`, `overlap_width_ratio`: `int` - Slice overlay.

#### **Returns:**
 - `List[tuple]` - return list of tuples, each tuple contains pair of detection data and image with drawn detections.

 <br>
 <br>

`yield_data` - Method for yielding detection data from `detect` method.
#### **Parameters:**
 - `bbox`: `Boxes` - Detection data from `detect`.
#### **Returns:**
 - `Generator` - processed detection data into: `cls_id, class_name, conf,  (x1, y1, x2, y2)`

 <br>
 <br>

`yield_sahi_data` - Method for yielding detection data from `detect_with_sahi` method.
#### **Parameters:**
 - `sahi_result`: `PredictionResult` - Detection data from `detect_with_sahi`.
#### **Returns:**
- `Generator` - processed detection data into: `cls_id, class_name, conf, (x1, y1, x2, y2)`

### **Basic usage:**
```python
yolo_predictor = YoloDetector(
        model_path=model_path,
        classes_path=classes_path
)
image = cv2.imread(image_path)
res, res_img = yolo_predictor.detect(images=[image])[0]
#res, res_img = self.yolo_detector.detect_with_sahi(images=[frame])[0]

detection_res_gen = self.yolo_detector.yield_data(bbox=res)
# detection_res_gen = self.yolo_detector.yield_sahi_data(sahi_result=res)
for detection in detection_res_gen:
    class_id, _, conf, x1, y1, x2, y2 = *detection[:3], *detection[3]
cv2.imshow("res", res_img)
cv2.waitKey(0)
```
## Model comparison
In order to copare models use `compare_models.py`. It compare on both images and videos.

#### **Model configuration**
Prepare model configuration dict:
```python
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
```
#### **Visual configuration**
Both image and video methods have some options that allow to customize visualiztion a little bit.

 - **compare_on_video:**
   - `cols`: `int`
   - `start_y`: `int`
   - `x_padding`: `int`
   - `y_padding`: `int`
   - `display_width`: `int`
   - `display_height`: `int`

- **compare_on_images:**
   - `figsize1`: `tuple` - size of main graph (with actual predictions)
   - `figsize2`: `tuple` - size of graph with execution times for each model
   - `rows`: `int`

#### **Results**

![alt text](assets/video_compare1.png)
![alt text](assets/video_compare2.png)
![alt text](assets/video_compare3.png)

Yolov11
