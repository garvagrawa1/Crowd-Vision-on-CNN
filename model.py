"""CrowdVisionNet - our own face detector CNN, trained completely from scratch.

Design idea (viva explanation)
------------------------------
The network is a *fully convolutional* face / non-face classifier for 48x48 windows.

* During TRAINING it sees 48x48 crops and outputs a 1x1 prediction:
      - 2 class scores  (non-face, face)
      - 4 box offsets   (how to nudge the window so it fits the face exactly)
* During INFERENCE we feed the WHOLE frame. Because the network has no
  fully-connected layers (the "FC" layer is a 4x4 convolution), it produces a
  prediction map in ONE forward pass. Each cell of the map is exactly the
  answer for one 48x48 window, the windows being 8 pixels apart (stride 8).

So this is mathematically the same as a sliding-window detector, but tens of
times faster, because overlapping windows share their convolution work.
All convolutions are 'valid' (no padding), so full-frame results equal the
results on individual crops - no train / test mismatch.

No YOLO, no pretrained weights, no OpenCV face detector: every weight is
learned from random initialisation by train.py.
"""
import torch
from torch import nn

WINDOW = 48   # input window size the network was trained on
STRIDE = 8    # distance between neighbouring windows in the output map


def conv_block(cin, cout, k):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=k, padding=0, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CrowdVisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16, 3),        # 48 -> 46
            nn.MaxPool2d(2),             # 46 -> 23
            conv_block(16, 32, 3),       # 23 -> 21
            nn.MaxPool2d(2),             # 21 -> 10
            conv_block(32, 64, 3),       # 10 -> 8
            nn.MaxPool2d(2),             # 8  -> 4
            conv_block(64, 96, 4),       # 4  -> 1   ("fully connected" layer as a conv)
            nn.Dropout2d(0.25),
        )
        self.cls_head = nn.Conv2d(96, 2, kernel_size=1)    # face / non-face logits
        self.box_head = nn.Conv2d(96, 4, kernel_size=1)    # bounding-box regression

    def forward(self, x):
        f = self.features(x)
        return self.cls_head(f), self.box_head(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = CrowdVisionNet().eval()
    print(net)
    print("Trainable parameters:", count_parameters(net))
    c, b = net(torch.zeros(1, 3, 48, 48))
    print("48x48 crop  ->", tuple(c.shape), tuple(b.shape))
    c, b = net(torch.zeros(1, 3, 480, 640))
    print("640x480 frame ->", tuple(c.shape), "(one prediction per window)")
