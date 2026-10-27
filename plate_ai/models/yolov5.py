"""Forward operators for supported YOLOv5 detection checkpoints."""
import torch
from torch import nn


class Conv(nn.Module):
    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, "bn"):
            x = self.bn(x)
        return self.act(x)

    forward_fuse = forward


class Bottleneck(nn.Module):
    def forward(self, x):
        value = self.cv2(self.cv1(x))
        return x+value if self.add else value


class C3(nn.Module):
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def forward(self, x):
        first = self.cv1(x)
        second = self.m(first)
        third = self.m(second)
        return self.cv2(torch.cat((first, second, third, self.m(third)), dim=1))


class Concat(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=self.d)


class Detect(nn.Module):
    def forward(self, inputs):
        decoded = []
        for index, value in enumerate(inputs):
            value = self.m[index](value)
            batch, _, height, width = value.shape
            value = value.view(batch, self.na, self.no, height, width).permute(0, 1, 3, 4, 2)
            y, x = torch.meshgrid(torch.arange(height, device=value.device, dtype=value.dtype),
                                  torch.arange(width, device=value.device, dtype=value.dtype), indexing="ij")
            grid = torch.stack((x, y), dim=-1)[None, None]-.5
            stride = self.stride[index].to(device=value.device, dtype=value.dtype)
            anchors = (self.anchors[index]*stride).view(1, self.na, 1, 1, 2)
            center, size, confidence = value.sigmoid().split((2, 2, self.nc+1), dim=-1)
            prediction = torch.cat(((center*2+grid)*stride, (size*2).square()*anchors, confidence), dim=-1)
            decoded.append(prediction.reshape(batch, -1, self.no))
        return torch.cat(decoded, dim=1)


class DetectionModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        saved = []
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if i == -1 else saved[i] for i in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        return x


Model = DetectionModel


class Classify(nn.Module):
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


class ClassificationModel(DetectionModel):
    pass
