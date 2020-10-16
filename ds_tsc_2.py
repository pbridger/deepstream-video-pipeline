import sys
import torch


class TorchScriptPart(torch.nn.Module):
    def __init__(self, ssd_module):
        super().__init__()
        self.ssd_module = ssd_module

    def forward(self, image_batch):
        locs, labels = self.ssd_module.detector(image_batch.half())
        return self.ssd_module.postprocess(locs, labels)


