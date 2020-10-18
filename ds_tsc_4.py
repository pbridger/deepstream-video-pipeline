import sys
import torch


class TorchScriptPart(torch.nn.Module):
    def __init__(self, ssd_module):
        super().__init__()
        self.ssd_module = ssd_module

    def forward(self, locs, labels):
        return self.ssd_module.postprocess(locs.unsqueeze(0), labels.unsqueeze(0))


