import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
import os
from scipy import misc
import time
from utils.data import test_dataset
from Network import BIPGNet
model = BIPGNet()
model = nn.DataParallel(model.cuda(), device_ids=[0])
model.load_state_dict(torch.load('./model/BIPG.pth'))
model.cuda()
model.eval()

data_path = './dataset/'
valset = ['ECSSD', 'PASCAL-S', 'HKU-IS', 'DUT-OMRON', 'DUTS']
for dataset in valset:
    save_path = './saliency_BIPG/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = data_path + dataset + '/img/'
    gt_root = data_path + dataset + '/gt/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)
    all = 0

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.array(gt).astype('float')
            gt = gt / (gt.max() + 1e-8)
            image = Variable(image).cuda()
            start_time = time.time()
            pr1, pr2, pr3, pre, pe1_2, pe2_2, pe3_2, pe4_2 = model(image)
            all = all + (time.time() - start_time)


            res = F.interpolate(pre, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            misc.imsave(save_path + name + '.png', res)

    print(test_loader.size / all)

