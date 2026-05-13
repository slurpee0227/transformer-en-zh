import matplotlib.pyplot as plt
import numpy as np


def visualize_attention(attention_matrix, output_path='output/attention_map.png'):
    plt.figure(figsize=(8, 8))

    plt.imshow(attention_matrix, cmap='viridis')

    plt.colorbar()

    plt.title('Attention Visualization')

    plt.savefig(output_path)

    print(f'Attention map saved to {output_path}')
