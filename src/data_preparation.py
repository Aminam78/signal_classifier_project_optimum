import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def generate_synthetic_data_with_pattern(num_samples_per_class, num_features=3, batch_size=32, shuffle=True, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # تعریف میانگین‌ها برای هر کلاس
    means = {
        0: [0.3, 0.5, 0.7],  # طبیعی: فرکانس پایین، دامنه متوسط، طول موج بالا
        1: [0.7, 0.7, 0.3],  # مصنوعی: فرکانس بالا، دامنه بالا، طول موج پایین
        2: [0.5, 0.5, 0.5]   # ناشناخته: ویژگی‌های متوسط
    }
    std_dev = 0.1  # انحراف معیار

    X = []
    y = []
    for class_label in range(3):
        for _ in range(num_samples_per_class):
            features = [np.random.normal(loc=mean, scale=std_dev) for mean in means[class_label]]
            X.append(features)
            y.append(class_label)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_data_loaders(num_train_per_class=333, num_test_per_class=67, batch_size=32):
    train_loader = generate_synthetic_data_with_pattern(num_train_per_class, batch_size=batch_size, shuffle=True)
    test_loader = generate_synthetic_data_with_pattern(num_test_per_class, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader