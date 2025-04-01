from src.data_preparation import get_data_loaders
from src.model import SignalClassifier
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_signal

# تولید داده‌ها
train_loader, test_loader = get_data_loaders(num_train_per_class=333, num_test_per_class=67, batch_size=32)

# ایجاد مدل
model = SignalClassifier()

# آموزش مدل
model = train_model(model, train_loader, num_epochs=50, lr=0.003)

# ارزیابی مدل
evaluate_model(model, test_loader)

# پیش‌بینی روی سیگنال‌های جدید
new_signals = [
    [0.5, 0.5, 0.5],
    [0.3, 0.7, 0.7],
    [0.7, 0.5, 0.3]
]
for signal in new_signals:
    predict_signal(model, signal)