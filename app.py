import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------- 配置 ----------------------
st.set_page_config(page_title="自监督学习实验平台", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 数据加载 ----------------------
@st.cache_resource
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = load_data()

# ---------------------- 1. 图像变换自监督任务：旋转预测 ----------------------
class RotationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 4)  # 4个旋转角度：0°, 90°, 180°, 270°

    def forward(self, x):
        x = self.conv(x).view(-1, 128)
        return self.fc(x)

def rotate_image(img, k):
    return torch.rot90(img, k, dims=[2, 3])

def train_rotation_model(model, loader, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    loss_history = []
    acc_history = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data, _ in tqdm(loader):
            data = data.to(device)
            batch_size = data.size(0)
            k = torch.randint(0, 4, (batch_size,)).to(device)
            rotated_data = rotate_image(data, k)

            optimizer.zero_grad()
            outputs = model(rotated_data)
            loss = criterion(outputs, k)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += predicted.eq(k).sum().item()
            total += batch_size

        avg_loss = total_loss / total
        avg_acc = 100. * correct / total
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    return loss_history, acc_history

# ---------------------- 2. 简化版MAE（掩码重建） ----------------------
class MAE(nn.Module):
    def __init__(self, mask_ratio=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 28*28), nn.Tanh()
        )
        self.mask_ratio = mask_ratio

    def mask_input(self, x):
        batch_size, dim = x.shape
        mask = torch.rand(batch_size, dim).to(x.device) > self.mask_ratio
        masked_x = x * mask.float()
        return masked_x, mask

    def forward(self, x):
        x_flat = x.view(-1, 28*28)
        masked_x, mask = self.mask_input(x_flat)
        z = self.encoder(masked_x)
        recon = self.decoder(z)
        return recon, masked_x.view(-1, 1, 28, 28), mask

def train_mae(model, loader, epochs=3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in tqdm(loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, _, _ = model(data)
            loss = criterion(recon, data.view(-1, 28*28))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(loader.dataset)
        loss_history.append(avg_loss)
        print(f"MAE Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return loss_history

# ---------------------- Streamlit界面 ----------------------
st.title("🎨 自监督学习实验平台（图像变换与MAE重建）")
tab1, tab2, tab3 = st.tabs([
    "1. 旋转预测自监督任务",
    "2. MAE掩码重建实验",
    "3. 对比实验与结果分析"
])

# ---------------------- 模块1：旋转预测 ----------------------
with tab1:
    st.header("旋转预测自监督任务")
    epochs_rot = st.slider("训练轮数", 1, 5, 3, key="epochs_rot")
    if st.button("开始训练", key="train_rot"):
        with st.spinner("训练中..."):
            model_rot = RotationPredictor().to(device)
            loss_hist, acc_hist = train_rotation_model(model_rot, train_loader, epochs=epochs_rot)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
            ax1.plot(loss_hist, label='Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax2.plot(acc_hist, label='Accuracy')
            ax2.set_title('Training Accuracy (%)')
            ax2.legend()
            st.pyplot(fig)

            model_rot.eval()
            with torch.no_grad():
                test_imgs, _ = next(iter(test_loader))
                test_imgs = test_imgs[:5].to(device)
                k = torch.tensor([0,1,2,3,0]).to(device)
                rotated_imgs = rotate_image(test_imgs, k)
                preds = model_rot(rotated_imgs).argmax(1)

            fig, axes = plt.subplots(1,5, figsize=(15,3))
            for i in range(5):
                axes[i].imshow(rotated_imgs[i,0].cpu()*0.5+0.5, cmap='gray')
                axes[i].set_title(f"True:{k[i]*90}°\nPred:{preds[i]*90}°")
                axes[i].axis('off')
            st.pyplot(fig)

# ---------------------- 模块2：MAE掩码重建 ----------------------
with tab2:
    st.header("MAE掩码重建实验")
    mask_ratio = st.slider("掩码比例", 0.1, 0.9, 0.5, 0.1, key="mask_ratio")
    epochs_mae = st.slider("训练轮数", 1, 5, 3, key="epochs_mae")
    if st.button("开始训练MAE", key="train_mae"):
        with st.spinner("训练中..."):
            model_mae = MAE(mask_ratio=mask_ratio).to(device)
            loss_hist = train_mae(model_mae, train_loader, epochs=epochs_mae)

            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(loss_hist, label='MAE Loss')
            ax.set_title('Training Loss')
            ax.legend()
            st.pyplot(fig)

            model_mae.eval()
            with torch.no_grad():
                test_imgs, _ = next(iter(test_loader))
                test_imgs = test_imgs[:5].to(device)
                recon, masked_imgs, _ = model_mae(test_imgs)
                recon = recon.view(-1,1,28,28)

            fig, axes = plt.subplots(3,5, figsize=(15,6))
            for i in range(5):
                axes[0,i].imshow(test_imgs[i,0].cpu()*0.5+0.5, cmap='gray')
                axes[0,i].set_title("Original")
                axes[0,i].axis('off')
                axes[1,i].imshow(masked_imgs[i,0].cpu()*0.5+0.5, cmap='gray')
                axes[1,i].set_title("Masked")
                axes[1,i].axis('off')
                axes[2,i].imshow(recon[i,0].cpu()*0.5+0.5, cmap='gray')
                axes[2,i].set_title("Reconstructed")
                axes[2,i].axis('off')
            st.pyplot(fig)

# ---------------------- 模块3：对比实验 ----------------------
with tab3:
    st.header("对比实验：不同掩码比例/数据增强效果")
    st.subheader("1. 不同掩码比例对比")
    mask_ratios = [0.3, 0.5, 0.7]
    if st.button("对比不同掩码比例", key="compare_mask"):
        with st.spinner("对比中..."):
            fig, axes = plt.subplots(1, 3, figsize=(15,4))
            for idx, ratio in enumerate(mask_ratios):
                model = MAE(mask_ratio=ratio).to(device)
                loss_hist = train_mae(model, train_loader, epochs=2)
                axes[idx].plot(loss_hist, label=f"Mask Ratio {ratio}")
                axes[idx].set_title(f"Mask Ratio {ratio} Loss")
                axes[idx].legend()
            st.pyplot(fig)

    st.subheader("2. 训练前后效果对比")
    if st.button("对比训练前后重建效果", key="compare_train"):
        with st.spinner("对比中..."):
            model_before = MAE(mask_ratio=0.5).to(device)
            model_after = MAE(mask_ratio=0.5).to(device)
            train_mae(model_after, train_loader, epochs=3)

            test_imgs, _ = next(iter(test_loader))
            test_imgs = test_imgs[:3].to(device)

            with torch.no_grad():
                recon_before, masked, _ = model_before(test_imgs)
                recon_after, _, _ = model_after(test_imgs)

            fig, axes = plt.subplots(3, 3, figsize=(12,8))
            for i in range(3):
                axes[i,0].imshow(masked[i,0].cpu()*0.5+0.5, cmap='gray')
                axes[i,0].set_title("Masked Input")
                axes[i,0].axis('off')
                axes[i,1].imshow(recon_before[i].view(28,28).cpu()*0.5+0.5, cmap='gray')
                axes[i,1].set_title("Before Training")
                axes[i,1].axis('off')
                axes[i,2].imshow(recon_after[i].view(28,28).cpu()*0.5+0.5, cmap='gray')
                axes[i,2].set_title("After Training")
                axes[i,2].axis('off')
            st.pyplot(fig)

st.markdown("---")
st.caption("模式识别与图像处理 - A7 自监督学习实验 | 可直接提交GitHub")
