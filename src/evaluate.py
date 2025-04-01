import torch 

def evaluate_model(model, test_loader):
    """
    ارزیابی مدل روی داده‌های تست.
    Args:
        model: مدل آموزش‌دیده
        test_loader: داده‌های تست
    Returns:
        دقت مدل
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test data: {accuracy:.4f}")
    return accuracy