import torch
import torch.nn as nn

class SignalClassifier(nn.Module):
    def __init__(self):
        super(SignalClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # لایه ورودی (3 ویژگی) به مخفی اول (16 نورون)
        self.fc2 = nn.Linear(32, 16)  # مخفی اول به مخفی دوم (8 نورون)
        self.fc3 = nn.Linear(16, 3)   # مخفی دوم به خروجی (3 کلاس)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # فعال‌سازی tanh
        x = torch.tanh(self.fc2(x))  # فعال‌سازی tanh
        x = self.fc3(x)              # خروجی خام
        return x