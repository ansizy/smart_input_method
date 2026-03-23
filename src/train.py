import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import InputMethodModel


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次
    :param model:   模型
    :param dataloader:  dataloader
    :param loss_fn: 损失函数
    :param optimizer:   优化器
    :param device:  cuda or cpu
    :return: epoch avg loss
    """
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():

    # 获取cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    dataloader = get_dataloader(train=True)

    # 获取词表大小
    with open(config.MODELS_DIR / "vocab.txt", 'r', encoding='utf-8') as f:
        # 读取每一行
        vocab_list = [line.strip() for line in f]
    vocab_size = len(vocab_list)

    # 模型
    model = InputMethodModel(vocab_size=vocab_size).to(device)

    # loss
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard 可视化
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 训练
    best_loss = float("inf")

    for epoch in range(config.EPOCHS):
        print("=" * 10, f"Epoch {epoch + 1}", "=" * 10)
        # 训练一个 epoch 逻辑
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"loss {loss}")

        # 记录训练结果
        writer.add_scalar("loss", loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / "best.pt")

    writer.close()


if __name__ == '__main__':
    train()