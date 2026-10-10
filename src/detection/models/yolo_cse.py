import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f

from rbccps_od.domain.cse import CSEBlock


class CSEBottleneck(nn.Module):

    def __init__(self, channels):

        super().__init__()

        hidden = max(1, channels // 2)

        self.cv1 = Conv(channels, hidden, 1, 1)
        self.cv2 = Conv(hidden, hidden, 3, 1)
        self.cv3 = Conv(hidden, channels, 1, 1)

        self.cse = CSEBlock(channels)

    def forward(self, x):

        y = self.cv1(x)
        y = self.cv2(y)
        y = self.cv3(y)

        y = self.cse(y)

        return x + y
    
class CSEC2f(C2f):

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5
    ):

        super().__init__(c1, c2, n, shortcut, g, e)

        hidden = int(c2 * e)

        self.m = nn.ModuleList(
            CSEBottleneck(hidden)
            for _ in range(n)
        )
