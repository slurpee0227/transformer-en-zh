import matplotlib.pyplot as plt


def save_loss_curve(train_losses, valid_losses, output_path='output/loss_curve.png'):
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.legend()

    plt.savefig(output_path)

    print(f'Loss curve saved to {output_path}')
