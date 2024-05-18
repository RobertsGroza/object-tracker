from deep_sort.deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        encoder_model_filename = "deep_sort/model/mars-small128.pb"
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-2] for d in detections]
        classes = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        tracker_detections = []
        for bbox_id, bbox in enumerate(bboxes):
            tracker_detections.append(Detection(bbox, scores[bbox_id], features[bbox_id], classes[bbox_id]))

        self.tracker.predict()
        self.tracker.update(tracker_detections)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_id = track.class_id
            tracks.append(Track(track_id, bbox, class_id))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    class_id = None

    def __init__(self, id, bbox, class_id):
        self.track_id = id
        self.bbox = bbox
        self.class_id = class_id
