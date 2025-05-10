import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Optional, Union

def bbox_to_z(bbox: List[float]) -> np.ndarray:
    """
    Convert bounding box coordinates to state vector.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        State vector [x, y, s, r]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def z_to_bbox(z: np.ndarray, score: Optional[float] = None) -> np.ndarray:
    """
    Convert state vector to bounding box coordinates.
    
    Args:
        z: State vector [x, y, s, r]
        score: Detection confidence score
    
    Returns:
        Bounding box [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
    """
    val = z[2] * z[3]
    if val < 1e-6:
        val = 1e-6

    w = np.sqrt(val)
    h = z[2] / (w + 1e-6)
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

class KalmanBoxTracker:
    count: int = 0

    def __init__(self, bbox: List[float], conf: float, class_id: int):

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Initialize covariance matrices
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in velocity
        self.kf.P *= 10.0             # General uncertainty
        self.kf.R[2:, 2:] *= 10.0     # Measurement noise
        
        # Assign a unique ID to the tracker
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.conf = conf
        self.class_id = class_id

        # Initialize state with the bounding box
        self.kf.x[:4] = bbox_to_z(bbox)
        self.time_since_update: int = 0
        self.history: List[np.ndarray] = []
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0

    def update(self, bbox: List[float], conf: Optional[float] = None):
        """
        Update the tracker with a new bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
            conf: Confidence score
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        if conf is not None:
            self.conf = conf

        z = bbox_to_z(bbox)
        self.kf.update(z)

    def predict(self) -> np.ndarray:
        """
        Predict the next state.
        
        Returns:
            Predicted bounding box
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """
        Get the current bounding box estimate.
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return z_to_bbox(self.kf.x)

class Sort:
    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, dets: List[List[float]]) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            dets: List of detections [x1, y1, x2, y2, conf, class_id]
        
        Returns:
            Array of active tracks
        """
        predictions: List[np.ndarray] = []
        for t in self.trackers:
            pred = t.predict()
            predictions.append(pred)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, predictions)

        # Update matched trackers with assigned detections
        for m in matched:
            det_idx, trk_idx = m
            x1, y1, x2, y2, conf, class_id = dets[det_idx]
            self.trackers[trk_idx].update([x1, y1, x2, y2], conf=conf)

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            x1, y1, x2, y2, conf, class_id = dets[i]
            trk = KalmanBoxTracker([x1, y1, x2, y2], conf, class_id)
            self.trackers.append(trk)

        ret: List[List[Union[float, int]]] = []
        new_trackers: List[KalmanBoxTracker] = []
        for t in self.trackers:
            d = t.get_state()[0]  # [x1, y1, x2, y2]
            if (t.time_since_update < self.max_age) and (t.hit_streak >= self.min_hits):
                ret.append([
                    d[0], d[1], d[2], d[3],
                    t.conf,
                    t.id,
                    t.class_id
                ])

            if t.time_since_update < self.max_age:
                new_trackers.append(t)

        self.trackers = new_trackers
        return np.array(ret)

    @staticmethod
    def iou(bb_test: List[float], bb_gt: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bb_test: [x1, y1, x2, y2]
            bb_gt: [x1, y1, x2, y2]
        
        Returns:
            IoU value
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (
            (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
            (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh
        )
        return o

    def associate_detections_to_trackers(
        self, dets: List[List[float]], preds: List[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to trackers using IoU.
        
        Args:
            dets: List of detections
            preds: List of tracker predictions
        
        Returns:
            Matched indices, unmatched detection indices, unmatched tracker indices
        """
        if len(self.trackers) == 0 or len(dets) == 0:
            return [], list(range(len(dets))), list(range(len(self.trackers)))

        iou_matrix = np.zeros((len(dets), len(preds)), dtype=np.float32)
        for d, det in enumerate(dets):
            x1, y1, x2, y2, _, _ = det
            for t, pred in enumerate(preds):
                iou_matrix[d, t] = self.iou([x1, y1, x2, y2], pred[0])

        matched_indices: List[Tuple[int, int]] = []
        used_dets: set = set()
        used_trks: set = set()

        # Greedy matching based on highest IoU
        for _ in range(min(len(dets), len(preds))):
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            d_idx, t_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched_indices.append((d_idx, t_idx))
            used_dets.add(d_idx)
            used_trks.add(t_idx)
            iou_matrix[d_idx, :] = -1  # Mark as used
            iou_matrix[:, t_idx] = -1

        unmatched_dets = [i for i in range(len(dets)) if i not in used_dets]
        unmatched_trks = [i for i in range(len(preds)) if i not in used_trks]

        return matched_indices, unmatched_dets, unmatched_trks

if __name__ == "__main__":
    sort_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)

    # Detections format: [x1, y1, x2, y2, conf, class_id]
    frame1_dets = [
        [10, 10, 50, 50, 0.9, 0],   
        [100, 100, 150, 150, 0.8, 1]
    ]
    frame2_dets = [
        [12, 12, 52, 52, 0.95, 0],
        [99, 99, 149, 149, 0.85, 1]
    ]

    tracks_frame1 = sort_tracker.update(frame1_dets)
    print("Frame1 - Tracks:\n", tracks_frame1)
    
    tracks_frame2 = sort_tracker.update(frame2_dets)
    print("Frame2 - Tracks:\n", tracks_frame2)