import datetime
import inspect
import os
import pickle
import random
import logging 

import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import torchvision
import psutil
import GPUtil

file_dir = os.path.dirname(inspect.getframeinfo(inspect.currentframe()).filename)
process = psutil.Process(os.getpid()) #获取当前进程



def load_pickle_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data


def _load_data(file_path):
    raw_data = load_pickle_data(file_path)
    labels = raw_data[b'labels']
    data = raw_data[b'data']
    filenames = raw_data[b'filenames']

    data = data.reshape(10000, 3, 32, 32) / 255
    return data, labels, filenames


def load_cifar_data(root_path):
    train_root_path = os.path.join(root_path, 'cifar-10-batches-py/data_batch_')
    train_data_record = []
    train_labels = []
    train_filenames = []
    for i in range(1, 6):
        train_file_path = train_root_path + str(i)
        data, labels, filenames = _load_data(train_file_path)
        train_data_record.append(data)
        train_labels += labels
        train_filenames += filenames
    train_data = np.concatenate(train_data_record, axis=0)
    train_labels = np.array(train_labels)

    val_file_path = os.path.join(root_path, 'cifar-10-batches-py/test_batch')
    val_data, val_labels, val_filenames = _load_data(val_file_path)
    val_labels = np.array(val_labels)

    tr_data = torch.from_numpy(train_data).float()
    tr_labels = torch.from_numpy(train_labels).long()
    val_data = torch.from_numpy(val_data).float()
    val_labels = torch.from_numpy(val_labels).long()
    return tr_data, tr_labels, val_data, val_labels


class TimedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(TimedCIFAR10, self).__init__(*args, **kwargs)
        self.preprocess_time = 0

    def __getitem__(self, index):
        start_time = time.time()
        data = super(TimedCIFAR10, self).__getitem__(index)
        end_time = time.time()
        self.preprocess_time += end_time - start_time
        return data

    def get_preprocess_time(self):
        preprocess_time = self.preprocess_time
        self.preprocess_time = 0
        return preprocess_time


def get_data(root_path, custom_data=False):
    if custom_data:
        train_samples, test_samples, img_size = 5000, 1000, 32
        tr_label = [1] * int(train_samples / 2) + [0] * int(train_samples / 2)
        val_label = [1] * int(test_samples / 2) + [0] * int(test_samples / 2)
        random.seed(2021)
        random.shuffle(tr_label)
        random.shuffle(val_label)
        tr_data, tr_labels = torch.randn((train_samples, 3, img_size, img_size)).float(), torch.tensor(tr_label).long()
        val_data, val_labels = torch.randn((test_samples, 3, img_size, img_size)).float(), torch.tensor(
            val_label).long()
        tr_set = TensorDataset(tr_data, tr_labels)
        val_set = TensorDataset(val_data, val_labels)
        return tr_set, val_set
    elif os.path.exists(os.path.join(root_path, 'cifar-10-batches-py')):
        tr_data, tr_labels, val_data, val_labels = load_cifar_data(root_path)
        tr_set = TensorDataset(tr_data, tr_labels)
        val_set = TensorDataset(val_data, val_labels)
        return tr_set, val_set
    else:
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),                 # 转换为Tensor
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
            ])
            tr_set = TimedCIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)
            val_set = TimedCIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
            '''
            tr_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)
            val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
                                                   '''
            return tr_set, val_set
        except Exception as e:
            raise Exception(
                f"{e}, you can download and unzip cifar-10 dataset manually, "
                "the data url is http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.residual_function(x) + self.shortcut(x)
        return nn.ReLU(inplace=True)(out)


class ResNet(nn.Module):

    def __init__(self, block, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = self.make_layer(block, 64, 64, 2, 1)
        self.conv3 = self.make_layer(block, 64, 128, 2, 2)
        self.conv4 = self.make_layer(block, 128, 256, 2, 2)
        self.conv5 = self.make_layer(block, 256, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_layer = nn.Linear(512, num_classes)

    def make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dense_layer(out)
        return out


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def obs_transfer(src_path, dst_path):
    import moxing as mox
    mox.file.copy_parallel(src_path, dst_path)
    logging.info(f"end copy data from {src_path} to {dst_path}")

# 获取系统状态的
def monitor_system_usage():
    cpu_usage = process.cpu_percent(interval=None)
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)
    disk_io_counters = psutil.disk_io_counters()
    disk_io = {
        'read_bytes': disk_io_counters.read_bytes,
        'write_bytes': disk_io_counters.write_bytes,
    }
    gpus = GPUtil.getGPUs()
    gpu_load = [gpu.load * 100 for gpu in gpus]
    gpu_load_avg = sum(gpu_load) / len(gpu_load) if gpu_load else 0

    return {
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'disk_io': disk_io,
        'gpu_load': gpu_load_avg
    }


def init_distributed_mode(args):
    master_addr = args.init_method.split('://')[1].split(':')[0]
    master_port = args.init_method.split(':')[-1]

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")

    try:
        dist.init_process_group(backend='nccl', init_method='tcp://223.109.239.7:13372', world_size=args.world_size, rank=args.rank)
        print("Process group initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize process group: {e}")

    print("Initialization complete")



def main():

    # -------------------------------------------------存储系统状态的数组----------------------------------------------
    train_cpu_usages = []
    train_memory_usages = []
    train_disk_read_bytes = []
    train_disk_write_bytes = []
    train_gpu_loads = []

    val_cpu_usages = []
    val_memory_usages = []
    val_disk_read_bytes = []
    val_disk_write_bytes = []
    val_gpu_loads = []

    # --------------------------------------------------------------------------------------------------------------
    seed = datetime.datetime.now().year
    setup_seed(seed)

    parser = argparse.ArgumentParser(description='Pytorch distribute training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--enable_gpu', default='true')
    parser.add_argument('--lr', default='0.01', help='learning rate')
    parser.add_argument('--epochs', default='5', help='training iteration')

    parser.add_argument('--init_method', default='None', help='tcp_port')
    parser.add_argument('--rank', type=int, default=0, help='index of current task')
    parser.add_argument('--world_size', type=int, default=1, help='total number of tasks')

    parser.add_argument('--custom_data', default='false')
    parser.add_argument('--data_url', type=str, default=os.path.join(file_dir, 'input_dir'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(file_dir, 'output_dir'))
    args, unknown = parser.parse_known_args()

    args.enable_gpu = args.enable_gpu == 'true'
    args.custom_data = args.custom_data == 'true'
    args.lr = float(args.lr)
    args.epochs = int(args.epochs)

    if args.custom_data:
        logging.warning('you are training on custom random dataset, '
              'validation accuracy may range from 0.4 to 0.6.')

    ### 分布式改造，DDP初始化进程，其中init_method, rank和world_size参数均由平台自动入参 ###
    # dist.init_process_group(init_method=args.init_method, backend="nccl", world_size=args.world_size, rank=args.rank)
    ### 分布式改造，DDP初始化进程，其中init_method, rank和world_size参数均由平台自动入参 ###

    init_distributed_mode(args)

    tr_set, val_set = get_data(args.data_url, custom_data=args.custom_data)

    batch_per_gpu = 128
    gpus_per_node = torch.cuda.device_count() if args.enable_gpu else 1
    batch = batch_per_gpu * gpus_per_node

    tr_loader = DataLoader(tr_set, batch_size=batch, shuffle=False)

    ### 分布式改造，构建DDP分布式数据sampler，确保不同进程加载到不同的数据 ###
    tr_sampler = DistributedSampler(tr_set, num_replicas=args.world_size, rank=args.rank)
    tr_loader = DataLoader(tr_set, batch_size=batch, sampler=tr_sampler, shuffle=False, drop_last=True)
    ### 分布式改造，构建DDP分布式数据sampler，确保不同进程加载到不同的数据 ###

    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False)

    lr = args.lr * gpus_per_node * args.world_size
    max_epoch = args.epochs
    model = ResNet(Block).cuda() if args.enable_gpu else ResNet(Block)

    ### 分布式改造，构建DDP分布式模型 ###
    model = nn.parallel.DistributedDataParallel(model)
    ### 分布式改造，构建DDP分布式模型 ###

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)

    train_time = 0
    val_time = 0
    start_disk_io_counters = psutil.disk_io_counters()
    cur_disk_io = {
        'read_bytes': start_disk_io_counters.read_bytes,
        'write_bytes': start_disk_io_counters.write_bytes,
    }

    for epoch in range(1, max_epoch + 1):
# -------------------------------------------------------------------------------训练计时开始-------------------------------------------------------------
        start_time = time.perf_counter()
        model.train()
        train_loss = 0

        ### 分布式改造，DDP sampler, 基于当前的epoch为其设置随机数，避免加载到重复数据 ###
        tr_sampler.set_epoch(epoch)
        ### 分布式改造，DDP sampler, 基于当前的epoch为其设置随机数，避免加载到重复数据 ###

        for step, (tr_x, tr_y) in enumerate(tr_loader):
            # ----------------------------------------检测系统状态------------------------------
            usage = monitor_system_usage()
            train_cpu_usages.append(usage['cpu_usage'])
            train_memory_usages.append(usage['memory_usage'])
            train_disk_read_bytes.append(usage['disk_io']['read_bytes'] - cur_disk_io['read_bytes'])
            train_disk_write_bytes.append(usage['disk_io']['write_bytes'] - cur_disk_io['write_bytes'])
            cur_disk_io = usage['disk_io']
            train_gpu_loads.append(usage['gpu_load'])
            # --------------------------------------------------------------------------------
            if args.enable_gpu:
                tr_x, tr_y = tr_x.cuda(), tr_y.cuda()
            out = model(tr_x)
            loss = loss_func(out, tr_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('train | epoch: %d | loss: %.4f' % (epoch, train_loss / len(tr_loader)))
    #-----------------------------------------------------------------------------训练计时结束，开始验证计时器-------------------------------------------------
        end_time = time.perf_counter()
        train_time += end_time-start_time
        start_time = time.perf_counter()
        val_loss = 0
        pred_record = []
        real_record = []
        model.eval()
        with torch.no_grad():
            for step, (val_x, val_y) in enumerate(val_loader):
                # ----------------------------------------检测系统状态------------------------------
                usage = monitor_system_usage()
                val_cpu_usages.append(usage['cpu_usage'])
                val_memory_usages.append(usage['memory_usage'])
                val_disk_read_bytes.append(usage['disk_io']['read_bytes']- cur_disk_io['read_bytes'])
                val_disk_write_bytes.append(usage['disk_io']['write_bytes']- cur_disk_io['write_bytes'])
                cur_disk_io = usage['disk_io']
                val_gpu_loads.append(usage['gpu_load'])
            # --------------------------------------------------------------------------------
                if args.enable_gpu:
                    val_x, val_y = val_x.cuda(), val_y.cuda()
                out = model(val_x)
                pred_record += list(np.argmax(out.cpu().numpy(), axis=1))
                real_record += list(val_y.cpu().numpy())
                val_loss += loss_func(out, val_y).item()
        val_accu = accuracy_score(real_record, pred_record)
        print('val | epoch: %d | loss: %.4f | accuracy: %.4f' % (epoch, val_loss / len(val_loader), val_accu), '\n')
            #-----------------------------------------------------------------------------验证计时器结束-------------------------------------------------
        end_time = time.perf_counter()
        val_time += end_time-start_time

        if args.rank == 0:
            # save ckpt every epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'epoch_{epoch}.pth'))

    
    print(f"Train time: {train_time:.6f} seconds")
    print(f"Validate time: {val_time:.6f} seconds")

    pre_time = tr_set.get_preprocess_time() + val_set.get_preprocess_time()
    print(f"data preprocessing time: {pre_time:.6f} seconds" )

     # 计算平均训练系统使用情况
    avg_train_cpu_usage = sum(train_cpu_usages) / len(train_cpu_usages)
    avg_train_memory_usage = sum(train_memory_usages) / len(train_memory_usages)
    total_train_disk_read_bytes = sum(train_disk_read_bytes)
    total_train_disk_write_bytes = sum(train_disk_write_bytes)
    avg_train_gpu_load = sum(train_gpu_loads) / len(train_gpu_loads)

    print(f"\nAverage CPU Usage during Training: {avg_train_cpu_usage:.2f}%")
    print(f"Average Memory Usage during Training: {avg_train_memory_usage:.2f} MB")
    print(f"Total Disk Read during Training: {total_train_disk_read_bytes / (1024 ** 2):.2f} MB")
    print(f"Total Disk Write during Training: {total_train_disk_write_bytes / (1024 ** 2):.2f} MB")
    print(f"Average GPU Load during Training: {avg_train_gpu_load:.2f}%")

    # 计算平均验证系统使用情况
    avg_val_cpu_usage = sum(val_cpu_usages) / len(val_cpu_usages)
    avg_val_memory_usage = sum(val_memory_usages) / len(val_memory_usages)
    total_val_disk_read_bytes = sum(val_disk_read_bytes)
    total_val_disk_write_bytes = sum(val_disk_write_bytes)
    avg_val_gpu_load = sum(val_gpu_loads) / len(val_gpu_loads)

    print(f"\nAverage CPU Usage during Validation: {avg_val_cpu_usage:.2f}%")
    print(f"Average Memory Usage during Validation: {avg_val_memory_usage:.2f} MB")
    print(f"Total Disk Read during Validation: {total_val_disk_read_bytes / (1024 ** 2):.2f} MB")
    print(f"Total Disk Write during Validation: {total_val_disk_write_bytes / (1024 ** 2):.2f} MB")
    print(f"Average GPU Load during Validation: {avg_val_gpu_load:.2f}%")
    


if __name__ == '__main__':
    main()
