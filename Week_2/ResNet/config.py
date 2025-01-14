import os
import torch

class AppConfig:
    def __init__(self):
        # Paths to weights
        self.resnet_weights_path = "weights/resnet_weights.pth"
        self.densenet_weights_path = "weights/densenet_weights.pth"

        # Paths to class label files
        self.weather_classes_file = os.path.join(os.path.dirname(__file__), "classes", "weather_cls.txt")
        self.scenes_classes_file = "classes/scenes_cls.txt"

        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_weather_classes(self):
        with open(self.weather_classes_file, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_scenes_classes(self):
        with open(self.scenes_classes_file, "r") as f:
            return [line.strip() for line in f.readlines()]