import torch
import numpy as np
import torch
import torch.nn.functional as F

class GlobalVars:
    def __init__(self):
        self.num_epochs = 500
        self.train_batch_size = 64
        self.test_batch_size = 16
        self.num_classes = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_image_probabilities = None

    def initialize_image_probabilities(self, batch_size, is_training=True):
        if is_training:
            self.log_image_probabilities = torch.ones(batch_size, self.num_classes, device=self.device)
        else:
            self.log_image_probabilities = torch.ones(batch_size, self.num_classes).numpy()

    def update_image_probabilities(self, judge, outputs):
        if isinstance(self.log_image_probabilities, np.ndarray):
            outputs_np = outputs.cpu().numpy()
            for i, class_indices in enumerate(judge):
                self.log_image_probabilities[:, class_indices] *= outputs_np[:, i][:, np.newaxis]
        else:
            for i, class_indices in enumerate(judge):
                self.log_image_probabilities[:, class_indices] *= outputs[:, i].unsqueeze(1)

global_vars = GlobalVars()  # 假设有10个标签