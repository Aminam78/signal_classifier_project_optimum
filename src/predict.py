import torch

def predict_signal(model, signal, class_names=["Natural", "Artificial", "Unknown"]):
    """
    پیش‌بینی کلاس یک سیگنال جدید.
    Args:
        model: مدل آموزش‌دیده
        signal: سیگنال جدید (لیست یا آرایه 3 تایی)
        class_names: نام کلاس‌ها
    """
    model.eval()
    with torch.no_grad():
        # تبدیل سیگنال به تنسور
        signal_tensor = torch.tensor(signal, dtype=torch.float32).reshape(1, -1)
        output = model(signal_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        print(f"Prediction for new signal {signal}: {predicted_class}")