from pathlib import Path
import os


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(42)


class Generator(nn.Module):
    """
    生成器：从随机噪声生成二维数据点。

    输入形状: [batch_size, noise_dim]
    输出形状: [batch_size, data_dim]
    """

    def __init__(self, noise_dim, data_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
    判别器：判断输入数据点是真实数据的概率。

    输入形状: [batch_size, data_dim]
    输出形状: [batch_size, 1]
    """

    def __init__(self, data_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


NOISE_DIM = 16
DATA_DIM = 2
HIDDEN_DIM = 128
BATCH_SIZE = 256
NUM_EPOCHS = 5000
LEARNING_RATE = 0.0002
ADAM_BETAS = (0.5, 0.999)
PRINT_INTERVAL = 500


def generate_real_data(batch_size, device):
    """生成环形分布的真实二维数据。"""
    angles = torch.rand(batch_size, device=device) * 2 * torch.pi
    radius = 1.0 + torch.randn(batch_size, device=device) * 0.1
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    return torch.stack([x, y], dim=1)


def train(generator, discriminator, g_optimizer, d_optimizer, criterion, device):
    """训练基础 GAN。"""
    generator.train()
    discriminator.train()
    d_losses = []
    g_losses = []

    for epoch in range(NUM_EPOCHS):
        # 1. 训练判别器：真实数据判为 1，生成数据判为 0。
        real_data = generate_real_data(BATCH_SIZE, device)
        real_labels = torch.full((BATCH_SIZE, 1), 0.9, device=device)
        fake_labels = torch.zeros(BATCH_SIZE, 1, device=device)

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        fake_data = generator(noise).detach()

        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data)

        d_loss_real = criterion(real_pred, real_labels)
        d_loss_fake = criterion(fake_pred, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 2. 训练生成器：希望生成数据被判别器判为 1。
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        fake_data = generator(noise)
        fake_pred = discriminator(fake_data)
        g_loss = criterion(fake_pred, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(
                f"Epoch {epoch + 1:4d} | "
                f"D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}"
            )

    return d_losses, g_losses


def visualize_results(generator, device, num_samples=1000):
    """保存生成数据和真实环形数据的对比图。"""
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, NOISE_DIM, device=device)
        generated_data = generator(noise).cpu()
        real_data = generate_real_data(num_samples, device).cpu()

    output_path = OUTPUT_DIR / "basic_gan.png"

    plt.figure(figsize=(6, 6))
    plt.scatter(
        real_data[:, 0],
        real_data[:, 1],
        alpha=0.35,
        s=10,
        c="orange",
        label="Real",
    )
    plt.scatter(
        generated_data[:, 0],
        generated_data[:, 1],
        alpha=0.5,
        s=10,
        c="blue",
        label="Generated",
    )
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Basic GAN Generated Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"生成结果已保存: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(NOISE_DIM, DATA_DIM, HIDDEN_DIM).to(device)
    discriminator = Discriminator(DATA_DIM, HIDDEN_DIM).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    criterion = nn.BCEWithLogitsLoss()

    print(f"使用设备: {device}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")

    train(generator, discriminator, g_optimizer, d_optimizer, criterion, device)
    visualize_results(generator, device)


if __name__ == "__main__":
    main()
