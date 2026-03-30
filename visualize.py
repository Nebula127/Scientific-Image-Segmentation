import matplotlib.pyplot as plt

def show_result(image, ground_truth, prediction, prediction_bin=None):
    n_cols = 4 if prediction_bin is not None else 3

    plt.figure(figsize=(15, 4))

    plt.subplot(1, n_cols, 1)
    plt.title("MRI")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, n_cols, 2)
    plt.title("Ground Truth")
    plt.imshow(ground_truth, cmap="gray")
    plt.axis("off")

    plt.subplot(1, n_cols, 3)
    plt.title("Prediction")
    plt.imshow(prediction, cmap="gray")
    plt.axis("off")

    if prediction_bin is not None:
        plt.subplot(1, n_cols, 4)
        plt.title("Binary Mask")
        plt.imshow(prediction_bin, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()