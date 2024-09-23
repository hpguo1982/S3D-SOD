# coding=utf-8
import numpy as np

import pytorch_iou
import pytorch_ssim
import torch
import torch.nn as nn
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
import albumentations as A
from model import SSDSeg
from torch.backends import cudnn
import utils1.func as func

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True
torch.manual_seed(2023)
torch.cuda.manual_seed_all(2023)


# load training default parameters
args = {
    'epoch': 600,
    'batch_size': 8,
    'lr': 0.0001,
    'workers': 5,
    'data': '_SD900',
    'tra_img_dir': '../data/SD-saliency-900/images/training/',              # path of training images
    'tra_lbl_dir': '../data/SD-saliency-900/annotations/training/',             # path of training labels
    'tst_img_dir': '../data/SD-saliency-900/images/training/',  # path of training images
    'tst_lbl_dir': '../data/SD-saliency-900/annotations/training/',  # path of training labels
    'image_ext': '.bmp',
    'label_ext': '.png',
    'checkpoint': './trained_models/',
    'Model': 'SSDSeg',
    'Scale': 256,
    'Crop': None,
    'Loss_fn': '_MultiLoss2'
}

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def multi_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + ssim_out + iou_out
    return loss


PVTSeg = args['Model']
Net = SSDSeg
b = '_b' + str(args['batch_size'])
scale = '_Scale' + str(args['Scale'])
crop = '' if args['Crop'] is None else '_Crop' + str(args['Crop'])
loss_fn_ = multi_loss


chkpt_dir = args['checkpoint']
func.check_mkdir(chkpt_dir)

def main():
    print(args)
    tra_img_name_list = glob.glob(args['tra_img_dir'] + '*' + args['image_ext'])
    tra_lbl_name_list = []

    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]          
        imgIdx = img_name.split(".")[0]
        tra_lbl_name_list.append(args['tra_lbl_dir'] + imgIdx + args['label_ext'])

    print('**********************************************')
    print('train images: ', len(tra_img_name_list))
    print('train labels: ', len(tra_lbl_name_list))
    print('**********************************************')
    train_num = len(tra_img_name_list)
    if crop == '':
        trans = transforms.Compose([RescaleT(args['Scale']), ToTensorLab(flag=0)])
    else:
        trans = transforms.Compose([RescaleT(args['Scale']),
                                    RandomCrop(args['Crop']), ToTensorLab(flag=0)])
    salobj_dataset = SalObjDataset(img_name_list=tra_img_name_list, lbl_name_list=tra_lbl_name_list,
                                   transform=trans)

    salobj_dataloader = DataLoader(salobj_dataset, batch_size=args['batch_size'], shuffle=True,
                                   num_workers=args['workers'])

    # ------- 3. define model --------
    # define the net
    net = Net()
    net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    #--------5. define loss-------------
    print("---define loss...")
    loss_fn = multi_loss

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    for epoch in range(0, args['epoch']):
        ite_num4val = 0.0
        running_loss = 0.0
        net.train()
        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs_v, labels_v = data['image'], data['label']
            inputs_v = inputs_v.type(torch.FloatTensor)
            labels_v = labels_v.type(torch.FloatTensor)

            inputs_v = inputs_v.cuda()
            labels_v = labels_v.cuda()

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = net(inputs_v)
            if isinstance(y_pred, tuple):
                loss = loss_fn(torch.sigmoid(y_pred[0]), labels_v)
                for y_ in y_pred[1:]:
                    loss += loss_fn(torch.sigmoid(y_), labels_v)
            else:
                loss = loss_fn(torch.sigmoid(y_pred), labels_v)
            loss.backward()
            optimizer.step()
            running_loss += loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f " % (
            epoch + 1, args['epoch'], (i + 1) * args['batch_size'],
            train_num, ite_num, running_loss / ite_num4val))

    torch.save(net.state_dict(), "./trained_models/SSDNet.pth")

    print('-------------Congratulations! Training Done!!!-------------')

if __name__ == "__main__":
    main()

