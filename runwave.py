import ray.train as train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
import torch
from torch import nn
import dataset as ds
from torch.utils.data import DataLoader, random_split
from config import get_config
from wavelet import WaveletModel
from tqdm import tqdm
import os
from functools import partial
from collections import OrderedDict


def get_dataset():
    if os.path.exists(DATA_FILE_PATH) and os.path.exists(LABELS_FILE_PATH):
        print("Loading data and labels from files...")
        data = torch.load(DATA_FILE_PATH)
        labels = torch.load(LABELS_FILE_PATH)
    else:
        print("GET THE DATA =====================================")
        segs = []
        for name in tqdm(config['dataset_names'], desc="Segmenting datasets"):
            segs.append(ds.segment_dataset(name))

        data, labels = ds.clean(segs, "delete")

        print("Saving data and labels to files...")
        torch.save(data, DATA_FILE_PATH)
        torch.save(labels, LABELS_FILE_PATH)
    return data,labels

def get_dataloader(data,labels,batch_size):
    dataset = ds.ECGDataset(data, labels)

    print("CREATE DATALOADER =====================================")
    train_size = int((1 - config['train_test_ratio']) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader

def train_func_per_worker(config):
    # 获取数据集和数据加载器
    data, labels = get_dataset()
    train_dataloader, test_dataloader = get_dataloader(data, labels, config['batch_size'])

    # 准备分布式数据加载器
    train_dataloader = prepare_data_loader(train_dataloader)
    test_dataloader = prepare_data_loader(test_dataloader)

    # 初始化模型
    model = WaveletModel(config)
    
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    

    # 检查是否有已保存的检查点
    start_epoch = 0
    best_loss = float('inf')
    latest_checkpoint_path = os.path.join(config['output_path'], "latest_checkpoint.pth")
    
    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path)
        # 移除 'module.' 前缀
        if 'module.' in next(iter(checkpoint['model_state_dict'])):
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:]  # 移除 'module.'
                new_state_dict[name] = v
            checkpoint['model_state_dict'] = new_state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['test_loss']
        print(f"Loaded checkpoint from {latest_checkpoint_path}. Resuming from epoch {start_epoch}.")
    else:
        print("No checkpoint found. Training from scratch...")

    # 准备分布式模型
    model = prepare_model(model)


    # 获取当前实验目录
    trial_dir = train.get_context().get_trial_dir()

    # 模型训练循环
    for epoch in range(start_epoch, config['n_epochs']):
        if train.get_context().get_world_size() > 1:
            # 分布式训练时，确保每个 epoch 的 shuffle 不同
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            output = model(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证逻辑
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                output = model(X)
                loss = loss_fn(output, y)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)


        # 保存模型检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss
        }
        torch.save(checkpoint, latest_checkpoint_path)
        epoch_checkpoint_path = os.path.join(config['output_path'], f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, epoch_checkpoint_path)

        # 报告指标和检查点
        train.report(
            metrics={"loss": test_loss},
            checkpoint=train.Checkpoint.from_directory(config['output_path'])
        )

def train_model(config):
    # 定义 ScalingConfig
    scaling_config = ScalingConfig(
        num_workers=config['num_workers'],  # 使用指定数量的 worker
        use_gpu=config['use_gpu'],  # 是否使用 GPU
    )

    # 使用 partial 绑定 config 参数
    train_func_with_config = partial(train_func_per_worker, config=config)

    # 初始化 TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_with_config,  # 使用绑定后的函数
        scaling_config=scaling_config
    )
    
    # 启动训练
    result = trainer.fit()
    print(f"Training result: {result}")

if __name__ == "__main__":
    config = get_config()
    DATA_FILE_PATH = config['home_path'] + "/" + "data.pt"
    LABELS_FILE_PATH = config['home_path'] + "/" + "labels.pt"
    # 调用 train_model 返回 TorchTrainer 对象
    train_model(config)