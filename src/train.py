import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, num_epochs=50, lr=0.003): #تغییر نرخ آموزش به 0.001
    """
    آموزش مدل شبکه عصبی.
    Args:
        model: مدل شبکه عصبی
        train_loader: داده‌های آموزشی
        num_epochs: تعداد epochها
        lr: نرخ یادگیری
    Returns:
        مدل آموزش‌دیده
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # تغییر به Adam

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()       # صفر کردن گرادیان‌ها
            outputs = model(inputs)     # پیش‌بینی مدل
            loss = criterion(outputs, labels)  # محاسبه خطا
            loss.backward()             # محاسبه گرادیان‌ها
            optimizer.step()            # به‌روزرسانی وزن‌ها
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    return model