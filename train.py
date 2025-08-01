import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from time import time


from framework import MyFrame
from loss import dice_bce_loss
# from abl import ABL
from data import ImageFolder
from evalution import IOU
from networks.Dlink11_7_9 import MSMD_AXIA
from networks.MFINEA2 import MSMD_AXIA
from modeling.HSN_Net import LinkNet34
import random


SHAPE = (512, 512)
ROOT = './dataset/CHN6/train/'
# ROOT='E://pycharm//workplace//mynetwork//DeepGlobe//deepglobe1//train//'
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))
NAME = 'CHN6 HSN_Net2'
BATCHSIZE_PER_CARD =1# 每个显卡给的batchsize=4

solver = MyFrame(LinkNet34, dice_bce_loss, IOU, 2e-4)  # 调用framework中的MyFrame
batchsize = 4 # torch.cuda.device_count()返回GPU的数量

dataset = ImageFolder(trainlist, ROOT)  # 调用data中的ImageFolder
data_loader = torch.utils.data.DataLoader(  # pytorch提供的数据加载类
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0
   ,drop_last=True  # 设置drop_last=True
)


mylog = open('log/' + NAME + '.log', 'w')
tic = time()
no_optim = 0
total_epoch = 300  # 训练的次数
train_epoch_best_loss = 100.  # 损失
recall_best = 0
precision_best = 0
f1_best = 0
iou_best = 0

b = 0
for epoch in range(1, total_epoch + 1):
    # 设置参数
    train_epoch_loss = 0
    recall_list = 0
    precision_list = 0
    f1_list = 0
    accuracy_list = 0
    iou_list = 0

    data_loader_iter = iter(data_loader)  # 返回图像列表，载入数据

    i = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)  # 训练
        train_loss, iou, recall, precision, f1, accuracy = solver.optimize()  # 计算损失函数，优化
        train_epoch_loss += train_loss
        recall_list += recall
        precision_list += precision
        f1_list += f1
        accuracy_list += accuracy
        iou_list += iou
        print('batch', i)
        i = i + 1

    train_epoch_loss /= len(data_loader_iter)
    recall_list /= len(data_loader_iter)
    precision_list /= len(data_loader_iter)
    f1_list /= len(data_loader_iter)
    accuracy_list /= len(data_loader_iter)
    iou_list /= len(data_loader_iter)
    mylog.write('********************' + '\n')
    mylog.write('--epoch:' + str(epoch) + '  --time:' + str(int(time() - tic)) +
                '  --train_loss:' + str(train_epoch_loss) +
                '  --recall:' + str(recall_list) +
                '  --precision:' + str(precision_list) +
                '  --f1:' + str(f1_list) +
                '  --accuracy:' + str(accuracy_list) +
                '  --iou:' + str(iou_list) + '\n')

    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('f1', f1_list)
    print('SHAPE:', SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weight/' + NAME + '.th')

    if recall_list > recall_best:
        recall_best = recall_list
        solver.save('weight/recall_%d' % b + NAME +'.th')

    if no_optim > 6:
        print(mylog, 'early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weight/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

mylog.write('Finish!')

print('Finish!')
mylog.close()