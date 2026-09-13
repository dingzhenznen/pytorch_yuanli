import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import struct
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# ── 自编码器模型 ─────────────────────────────────
class Autoencoder(nn.Module):
    """
    基础自编码器：对称结构
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # 瓶颈层
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        """编码：获取潜在表示"""
        return self.encoder(x)

    def decode(self, z):
        """解码：从潜在表示重构"""
        return self.decoder(z)


INPUT_DIM = 784   # 例如 MNIST 图像展开后
HIDDEN_DIM = 256
LATENT_DIM = 32   # 潜在空间维度，远小于输入维度
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3


def read_mnist_images(path):
    """读取 MNIST 图片 IDX 文件，返回 [N, 1, 28, 28]。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        images = torch.frombuffer(f.read(), dtype=torch.uint8).clone().float()
    return images.view(num_images, 1, rows, cols) / 255.0


def read_mnist_labels(path):
    """读取 MNIST 标签 IDX 文件，返回 [N]。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = torch.frombuffer(f.read(), dtype=torch.uint8).clone().long()
    return labels.view(num_labels)


def get_mnist_loaders(batch_size):
    """从本项目 data/MNIST/raw 目录加载 MNIST 数据集。"""
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


def train(model, train_loader, criterion, optimizer, device, num_epochs):
    """训练自编码器。"""
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)

            x_recon = model(images)
            loss = criterion(x_recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.6f}")


def test(model, test_loader, criterion, device):
    """在测试集上计算平均重构损失。"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            x_recon = model(images)
            loss = criterion(x_recon, images)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.6f}")
    return avg_loss


def show_reconstruction_example(model, test_loader, device):
    """打印一批测试数据的输入、潜在向量和重构形状。"""
    model.eval()

    images, _ = next(iter(test_loader))
    images = images.view(images.size(0), -1).to(device)

    # print(f"输入形状: {images[0].shape}")

    # z = model.encode(images[0])
    # print(f"潜在向量形状: ",z)

    with torch.no_grad():
        z = model.encode(images)
        x_recon = model.decode(z)

    print(f"输入形状: {images.shape}")
    print(f"潜在向量形状: {z.shape}")
    print(f"重构形状: {x_recon.shape}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    model = Autoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"设备: {device}")
    print(f"输入维度: {INPUT_DIM}")
    print(f"潜在维度: {LATENT_DIM}")
    print(f"压缩比: {INPUT_DIM / LATENT_DIM:.1f}x")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    train(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    test(model, test_loader, criterion, device)
    show_reconstruction_example(model, test_loader, device)


if __name__ == "__main__":
    main()
