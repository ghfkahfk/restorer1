import sys
sys.path.append("../")
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
from torch.autograd import Variable
from source import utils
from source.loss import SSIM
from source.net import Restormer
from source.dataset  import *
import os

def main():
    torch.backends.cudnn.benchmark = True

    random.seed(6)  # 随机种子
    torch.manual_seed(6)
    torch.cuda.manual_seed(6)
    torch.manual_seed(6)
    EPOCH = 500  # 训练次数, 30 万实在太狠, 可能我的理解有问题
    # 虽然原论文有一个 patch 逐渐变大的操作，但这种操作一般显卡远远不行，所以没加
    BATCH_SIZE = 16  # 每批的训练数量
    LEARNING_RATE = 1e-3  # 学习率
    lr_list = []  # 学习率存储数组
    loss_list = []  # 损失存储数组
    best_psnr = 0

    inputPathTrain = './data/inputTrain'  # 训练输入图片路径
    targetPathTrain = './data/targetTrain'  # 训练目标图片路径
    inputPathVal = './data/inputVal/'  # 测试输入图片路径
    targetPathVal = './data/targetVal/'  # 测试目标图片路径

    best_epoch = 0

    myNet = Restormer()
    myNet = myNet.cuda()  # 网络放入GPU中
    myNet = nn.DataParallel(myNet, device_ids=[0])

    criterion_mse = nn.MSELoss().cuda()
    criterion_ssim = SSIM().cuda()

    optimizer = optim.AdamW(myNet.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)  # 网络参数优化算法
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH, eta_min=1e-6)

    # 训练数据
    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain)  # 实例化训练数据集类
    # 可迭代数据加载器加载训练数据
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0,
                             pin_memory=True)

    # 评估数据
    datasetValue = MyValueDataSet(inputPathVal, targetPathVal)  # 实例化评估数据集类
    valueLoader = DataLoader(dataset=datasetValue, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0,
                             pin_memory=True)

    # 开始训练
    print('-------------------------------------------------------------------------------------------------------')


    for epoch in range(EPOCH):
        myNet.train()  # 指定网络模型训练状态
        iters = tqdm(trainLoader, file=sys.stdout)  # 实例化 tqdm，自定义
        epochLoss = 0  # 每次训练的损失
        timeStart = time.time()  # 每次训练开始时间
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()  # 模型参数梯度置0
            optimizer.zero_grad()  # 同上等效

            input_train, target = Variable(x).cuda(), Variable(y).cuda()  # 转为可求导变量并放入 GPU

            output_train = myNet(input_train)  # 输入网络，得到相应输出

            l_mse = criterion_mse(output_train, target)  # 计算网络输出与目标输出的损失
            l_ssim = criterion_ssim(output_train, target)

            loss = (1 - l_ssim) + l_mse

            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数
            epochLoss += loss.item()  # 累计一次训练的损失

            # 自定义进度条前缀
            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, EPOCH, loss.item()))

        # 评估
        myNet.eval()
        psnr_val_rgb = []
        for index, (x, y) in enumerate(valueLoader, 0):
            input_, target_value = x.cuda(), y.cuda()
            with torch.no_grad():
                output_value = myNet(input_)
            for output_value, target_value in zip(output_value, target_value):
                psnr_val_rgb.append(utils.torchPSNR(output_value, target_value))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(myNet.state_dict(), 'model_best.pth')

        loss_list.append(epochLoss)  # 插入每次训练的损失值
        lr_list.append(scheduler.get_lr())
        scheduler.step()  # 更新学习率
        torch.save(myNet.state_dict(), 'model.pth')  # 每次训练结束保存模型参数
        timeEnd = time.time()  # 每次训练结束时间
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch+1, timeEnd-timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

    # 绘制训练时损失曲线
    plt.figure(1)
    x = range(0, EPOCH)
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.plot(x, loss_list, 'r-')
    # 绘制学习率改变曲线
    plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.plot(x, lr_list, 'r-')

    plt.show()


if __name__ == '__main__':
    main()