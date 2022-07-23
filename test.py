import os
import cv2
import math
import time
import torch
import random
import matplotlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import *
from option import args
from model import net_test
from pytorch_msssim import ssim
from dataset import MEFdataset, TestData
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

EPS = 1e-8
c = 3500

class Test(object):
    def __init__(self, ep=None):
        self.ep = ep
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
        self.batch_size = 1
        self.test_set = TestData(self.transform)
        self.test_loader = data.DataLoader(self.test_set, batch_size=1, shuffle=False,
                                           num_workers=0, pin_memory=False)
        # self.model = DenseNet().cuda()
        self.model = net_test().cuda()
        self.state = torch.load(args.model_path + args.model)
        # self.model.load_state_dict(self.state['model'])

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch, imgs in enumerate(self.test_loader):
                print('Processing picture No.{}'.format(batch + 1))
                imgs = torch.squeeze(imgs, dim=0)
                img1_y = imgs[0:1, 0:1, :, :].cuda()
                img2_y = imgs[1:2, 0:1, :, :].cuda()

                img_cr = imgs[:, 1:2, :, :].cuda()
                img_cb = imgs[:, 2:3, :, :].cuda()
                w_cr = (torch.abs(img_cr) + EPS) / torch.sum(torch.abs(img_cr) + EPS, dim=0)
                w_cb = (torch.abs(img_cb) + EPS) / torch.sum(torch.abs(img_cb) + EPS, dim=0)
                fused_img_cr = torch.sum(w_cr * img_cr, dim=0, keepdim=True).clamp(-1, 1)
                fused_img_cb = torch.sum(w_cb * img_cb, dim=0, keepdim=True).clamp(-1, 1)

                fused_img_y = self.model(img1_y, img2_y)
                fused_img = torch.cat((fused_img_y, fused_img_cr, fused_img_cb), dim=1)
                fused_img = (fused_img + 1) * 127.5
                fused_img = fused_img.squeeze(0)
                fused_img = fused_img.cpu().numpy()
                fused_img = np.transpose(fused_img, (1, 2, 0))
                fused_img = fused_img.astype(np.uint8)
                fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)

                if self.ep:
                    save_path = args.save_dir + str(self.ep) + '_epoch/'
                else:
                    save_path = args.save_dir

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite((save_path + str(batch + 1) + args.ext), fused_img)
            print('Finished testing!')
