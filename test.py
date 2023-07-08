import sys
import time
import torch.nn as nn
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from net import Restormer
from dataset import *


def main():

    inputPathTest = './data/inputTest/'  # 测试输入图片路径
    resultPathTest = './data/resultTest/'  # 测试结果图片路径

    ImageNames = os.listdir(inputPathTest)  # 测试图片的名字列表

    myNet = Restormer()
    myNet = myNet.cuda()  # 网络放入GPU中
    myNet = nn.DataParallel(myNet, [0, 1])

    # 测试数据
    datasetTest = MyTestDataSet(inputPathTest)  # 实例化测试数据集类
    # 可迭代数据加载器加载测试数据
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=6,
                            pin_memory=True)

    # 测试
    print('--------------------------------------------------------------')
    myNet.load_state_dict(torch.load('./model_best.pth'))  # 加载已经训练好的模型参数
    myNet.eval()  # 指定网络模型测试状态

    with torch.no_grad():  # 测试阶段不需要梯度
        timeStart = time.time()  # 测试开始时间
        for index, x in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()  # 释放显存
            input_test = x.cuda()  # 放入GPU
            output_test = myNet(input_test)  # 输入网络，得到输出
            save_image(output_test, resultPathTest + str(ImageNames[index]))  # 保存网络输出结果
        timeEnd = time.time()  # 测试结束时间
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))


if __name__ == '__main__':
    main()