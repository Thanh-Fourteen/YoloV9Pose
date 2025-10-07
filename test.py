import torch
import torch.nn as nn
import math
from models.yolo import DDetect, DDetectPose

# Giả sử class DDetectPose của bạn đã được định nghĩa ở trên
# (nếu cần, bạn có thể copy class đó vào trước đoạn này)

# Tạo object DDetectPose
model = DDetectPose(nc=1, kpt_label=5, ch=[128, 256, 512])
model = DDetect(nc=1, ch=[128, 256, 512])

# Tạo input giả - 3 feature maps (batch=1)
x = [
    torch.randn(1, 128, 80, 80),  # P3
    torch.randn(1, 256, 40, 40),  # P4
    torch.randn(1, 512, 20, 20)   # P5
]

model.eval()
# Gọi forward()
with torch.no_grad():
    out = model(x)

# Kiểm tra kiểu output
if isinstance(out, tuple):
    pred, _ = out
else:
    pred = out

print("Output shape:", pred.shape)
