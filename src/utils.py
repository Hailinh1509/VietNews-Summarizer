# Chứa các hàm tiện ích và phụ trợ cho dự án

def compute_metrics(eval_pred):
    """
    Hàm tính toán ROUGE score cho dự án.
    Sẽ được cập nhật chi tiết trong quá trình huấn luyện và đánh giá.
    """
    pass

def plot_loss(train_losses, val_losses):
    """
    Vẽ biểu đồ biểu diễn loss của quá trình train và validation.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
