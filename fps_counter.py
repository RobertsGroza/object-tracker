import time


class FPSCounter:
    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    fps_sum = 0
    current_fps = 0

    def __init__(self):
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.frame_count = 0
        self.fps_sum = 0
        self.current_fps = 0

    def process_new_frame(self):
        self.new_frame_time = time.time()
        self.frame_count += 1
        self.current_fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.fps_sum += self.current_fps
        self.prev_frame_time = self.new_frame_time

    def print_average_fps(self):
        print(f'AVERAGE FPS: {self.fps_sum / self.frame_count}')
