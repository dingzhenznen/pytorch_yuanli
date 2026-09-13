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


class VAE(nn.Module):
    """
    变分自编码器：学习 MNIST 图片的潜在概率分布，并从潜在空间采样生成图片。
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + std * eps。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


INPUT_DIM = 784
HIDDEN_DIM = 256
LATENT_DIM = 20
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
BETA = 1.0


def read_mnist_images(path):
    """读取 MNIST 图片 IDX 文件，返回 [N, 1, 28, 28]，像素归一化到 [0, 1]。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        images = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).float()
    return images.view(num_images, 1, rows, cols) / 255.0


def read_mnist_labels(path):
    """读取 MNIST 标签 IDX 文件，返回 [N]。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()
    return labels.view(num_labels)


def get_mnist_loaders(batch_size):
    """从本项目 data/MNIST/raw 目录加载 MNIST。"""
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"MNIST raw data directory not found: {raw_dir}")

    train_images = read_mnist_images(raw_dir / "train-images-idx3-ubyte")
    train_labels = read_mnist_labels(raw_dir / "train-labels-idx1-ubyte")
    test_images = read_mnist_images(raw_dir / "t10k-images-idx3-ubyte")
    test_labels = read_mnist_labels(raw_dir / "t10k-labels-idx1-ubyte")

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """VAE 损失：重构损失 + beta * KL 散度。"""
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train(model, train_loader, optimizer, device, num_epochs):
    """训练 VAE。"""
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_samples = 0

        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            x_recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(x_recon, images, mu, logvar, BETA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_samples += batch_size

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Loss: {total_loss / total_samples:.4f} | "
            f"Recon: {total_recon / total_samples:.4f} | "
            f"KL: {total_kl / total_samples:.4f}"
        )


def test(model, test_loader, device):
    """测试 VAE 在测试集上的平均损失。"""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            x_recon, mu, logvar = model(images)
            loss, _, _ = vae_loss(x_recon, images, mu, logvar, BETA)
            total_loss += loss.item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


def save_image_grid(images, path, rows=5, cols=5, title=None):
    """把 [N, 784] 图片保存成网格图。"""
    images = images[: rows * cols].view(-1, 28, 28).cpu()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def generate_images(model, device, num_images=25):
    """从标准正态分布采样 z，并用 decoder 生成新图片。"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, LATENT_DIM, device=device)
        generated = model.decode(z)

    output_path = OUTPUT_DIR / "vae_generated_mnist.png"
    save_image_grid(generated, output_path, title="VAE Generated MNIST")
    print(f"生成图片已保存: {output_path}")
    return generated


def save_reconstruction_examples(model, test_loader, device, num_images=10):
    """保存原图和重构图对比。"""
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:num_images].view(num_images, -1).to(device)

    with torch.no_grad():
        x_recon, _, _ = model(images)

    comparison = torch.empty(num_images * 2, INPUT_DIM)
    comparison[0::2] = images.cpu()
    comparison[1::2] = x_recon.cpu()

    output_path = OUTPUT_DIR / "vae_reconstructions.png"
    save_image_grid(
        comparison,
        output_path,
        rows=num_images,
        cols=2,
        title="Original / Reconstruction",
    )
    print(f"重构对比图已保存: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"使用设备: {device}")
    print(f"输入维度: {INPUT_DIM}")
    print(f"潜在维度: {LATENT_DIM}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_loader, optimizer, device, NUM_EPOCHS)
    test(model, test_loader, device)
    save_reconstruction_examples(model, test_loader, device)
    generate_images(model, device)


if __name__ == "__main__":
    main()
