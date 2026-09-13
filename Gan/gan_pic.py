import gzip
import os
import struct
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)


class Generator(nn.Module):
    """
    生成器：从随机噪声生成 MNIST 图片。

    输入形状: [batch_size, noise_dim]
    输出形状: [batch_size, 784]
    """

    def __init__(self, noise_dim, image_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, image_dim),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    """
    判别器：判断输入图片是真实 MNIST 图片的概率。

    输入形状: [batch_size, 784]
    输出形状: [batch_size, 1]
    """

    def __init__(self, image_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, image):
        return self.net(image)


NOISE_DIM = 100
IMAGE_DIM = 28 * 28
HIDDEN_DIM = 128
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.0002
ADAM_BETAS = (0.5, 0.999)


def read_mnist_images(path):
    """读取 MNIST 图片 IDX 文件，返回 [N, 1, 28, 28]，像素范围转成 [-1, 1]。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        images = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).float()

    images = images.view(num_images, 1, rows, cols) / 255.0
    return images * 2 - 1


def read_mnist_labels(path):
    """读取 MNIST 标签 IDX 文件，返回 [N]。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()

    return labels.view(num_labels)


def get_mnist_loader(batch_size):
    """从本项目 data/MNIST/raw 目录加载训练集。"""
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"MNIST raw data directory not found: {raw_dir}")

    images = read_mnist_images(raw_dir / "train-images-idx3-ubyte")
    labels = read_mnist_labels(raw_dir / "train-labels-idx1-ubyte")
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def train(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion, device):
    """交替训练判别器和生成器。"""
    generator.train()
    discriminator.train()

    for epoch in range(NUM_EPOCHS):
        total_d_loss = 0
        total_g_loss = 0

        for real_images, _ in train_loader:
            real_images = real_images.view(BATCH_SIZE, -1).to(device)
            real_labels = torch.full((BATCH_SIZE, 1), 0.9, device=device)
            fake_labels = torch.zeros(BATCH_SIZE, 1, device=device)

            # 1. 训练判别器：真实图片 -> 1，生成图片 -> 0。
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            fake_images = generator(noise).detach()

            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images)
            d_loss_real = criterion(real_logits, real_labels)
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 2. 训练生成器：希望生成图片被判别器判断为真实图片。
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images)
            g_loss = criterion(fake_logits, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"D_loss: {total_d_loss / len(train_loader):.4f} | "
            f"G_loss: {total_g_loss / len(train_loader):.4f}"
        )


def save_generated_images(generator, device, num_images=25):
    """随机采样噪声，生成图片并保存为网格。"""
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_images, NOISE_DIM, device=device)
        generated_images = generator(noise).cpu()

    # 训练时图片范围是 [-1, 1]，保存前转回 [0, 1]。
    generated_images = (generated_images + 1) / 2
    generated_images = generated_images.clamp(0, 1).view(num_images, 28, 28)

    rows = 5
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i], cmap="gray")
        ax.axis("off")

    output_path = OUTPUT_DIR / "gan_generated_mnist.png"
    plt.suptitle("GAN Generated MNIST")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"生成图片已保存: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_mnist_loader(BATCH_SIZE)

    generator = Generator(NOISE_DIM, IMAGE_DIM, HIDDEN_DIM).to(device)
    discriminator = Discriminator(IMAGE_DIM, HIDDEN_DIM).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    criterion = nn.BCEWithLogitsLoss()

    print(f"使用设备: {device}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"噪声维度: {NOISE_DIM}")
    print(f"图片维度: {IMAGE_DIM}")
    print(f"生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")

    train(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion, device)
    save_generated_images(generator, device)


if __name__ == "__main__":
    main()
