import json


class VideoSummary:
    distinct_objects = []
    frame_rate = 0

    def __init__(self, frame_rate):
        self.distinct_objects = []
        self.frame_rate = frame_rate

    def add_object(self, object_id, object_class):
        if not any(el["id"] == object_id for el in self.distinct_objects):
            self.distinct_objects.append({"id": object_id, "class": object_class})

    def prepend_summary_to_file(self, out_file_path):
        out_file = open(out_file_path, "r")
        original_content = out_file.read()
        out_file_w_summary = open(out_file_path, "w")
        out_file_w_summary.write(f'{json.dumps({"fps": self.frame_rate, "ids": self.distinct_objects})}\n')
        out_file_w_summary.write(original_content)
        out_file_w_summary.close()
        out_file.close()
