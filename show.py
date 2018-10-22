import numpy as np
import tinyyolonet, func
import torch
from torch.autograd import Variable
import loss, data_set
import cv2
from PIL import Image
import torch.nn as nn
from torchvision import models

labels = {
        "0": 'car',
        "1": 'Van',
        "2": 'Truck',
        "3": 'Pedestrian',
        "4": 'Person_sitting',
        "5": 'Cyclist',
        }

net = tinyyolonet.Yolo_tiny()
#net = models.resnet18(pretrained=False)
#net.fc = nn.Linear(512, 784) 
net.cuda()
checkpoint_load = torch.load('model_best/Best_Models.pth')
#net.load_state_dict(checkpoint_load['state_dict'])
net.load_state_dict(checkpoint_load['state_dict'])
net.eval()

#hyper_parameter
i = 448
threshold = 0.005

x = func.get_image(i)
print(x)
ima_raw = func.get_image_raw(i)
x = x.unsqueeze(0)
x = Variable(x)
x = x.cuda()

#print(x)
pre = net(x)

probs = pre[0,:294].contiguous().view(49, 6)
confs = pre[0,294:392].contiguous().view(49, 2)
coords = pre[0,392:].contiguous().view(49, 2, 4)

confs_1 = confs[:, 0]
confs_2 = confs[:, 1]
confs_1 = confs_1.contiguous().view(-1,1)
confs_2 = confs_2.contiguous().view(-1,1)

test_pro_1 = probs * confs_1
test_pro_2 = probs * confs_2
#print(test_pro_1)

candi_bb_1 = test_pro_1.ge(threshold).float()
candi_bb_2 = test_pro_2.ge(threshold).float()

test_pro_1  = test_pro_1 * candi_bb_1
test_pro_2  = test_pro_2 * candi_bb_2
#
_, index_grid_1 = torch.max(test_pro_1, 1)
_, index_grid_2 = torch.max(test_pro_2, 1)

obj_1 = func.max_pro(index_grid_1)
obj_2 = func.max_pro(index_grid_2)

obj_num1 = len(obj_1)
obj_num2 = len(obj_2)

pre_coords1 = func.find_coord(coords[:,0,:], obj_1, obj_num1)
#print(pre_coords1)
pre_coords2 = func.find_coord(coords[:,1,:], obj_2, obj_num2)

font = cv2.FONT_HERSHEY_SIMPLEX
if len(pre_coords1) != 0:
    #change w h
    pre_coords1[:,4:6] = torch.pow(pre_coords1[:,4:6],2) * 224
    #change coords
    get_grid1_y1 = np.floor(pre_coords1[:,1] / 7)
    get_grid1_x1 = pre_coords1[:,1] - get_grid1_y1 *7
    pre_coords1[:,2] = (pre_coords1[:,2] + get_grid1_x1) * 32 - (pre_coords1[:,4] / 2)
    pre_coords1[:,3] = (pre_coords1[:,3] + get_grid1_y1) * 32 - (pre_coords1[:,5] / 2)
    pre_coords1[:,4] = pre_coords1[:,2] + pre_coords1[:,4]
    pre_coords1[:,5] = pre_coords1[:,3] + pre_coords1[:,5]
    for l in range(obj_num1):
        x1, y1, x2, y2 = int(pre_coords1[l,2]), int(pre_coords1[l,3]), int(pre_coords1[l,4]), int(pre_coords1[l,5])
        cv2.rectangle(ima_raw, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
#        cv2.rectangle(im, (x1, y1), (x1+, y2), (255, 0, 0), thickness=-1)
        cate = labels[str(int(pre_coords1[l,0]))]
        cv2.putText(ima_raw, cate, (x1,y1), font, 0.6, (0, 255, 0), 1)

if len(pre_coords2) != 0:
    pre_coords2[:,4:6] = torch.pow(pre_coords2[:,4:6],2) * 224
    get_grid1_y2 = np.floor(pre_coords2[:,1] / 7)
    get_grid1_x2 = pre_coords2[:,1] - get_grid1_y2 *7
    pre_coords2[:,2] = (pre_coords2[:,2] + get_grid1_x2) * 32 - (pre_coords2[:,4] / 2)
    pre_coords2[:,3] = (pre_coords2[:,3] + get_grid1_y2) * 32 - (pre_coords2[:,5] / 2)
    pre_coords2[:,4] = pre_coords2[:,2] + pre_coords2[:,4]
    pre_coords2[:,5] = pre_coords2[:,3] + pre_coords2[:,5]

subim_3 = Image.fromarray(ima_raw)
subim_3.show()

dd = func.get_label(i)
#print(dd)
_,f = data_set.create_label(dd)
ff = f['coord']
#loss_test = loss.ty_loss(pre, y)
