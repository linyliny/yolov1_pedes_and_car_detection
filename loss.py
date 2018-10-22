from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

#import matplotlib.pyplot as plt
#import numpy as np
import torch
#import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable
#import torch.optim as optim
#from torchvision import datasets, models, transforms
#import func
#from torch.autograd import Function

#class Myloss(Function):
#    def forward(self, output, label):
#        num_out = output.numpy()
#        num_la = label.numpy()
#        cha = num_out - num_la
#        result = abs(cha)
#        return torch.FloatTensor(result)
#
#    def backward(self, grad_output):
#        return grad_output

#def which_grid(label):
#    cate_num = len(label)
#    candi_grid = torch.FloatTensor(cate_num, 2).zero_()
#    for i in range(cate_num):
#        obj_str = label[i]
#        obj = obj_str[1:5]
#        centra_x = obj[0] + obj[2]/2
#        centra_y = obj[1] + obj[3]/2
#        centra_x = centra_x//32
#        centra_y = centra_y//32
#        candi_grid[i][0] = centra_x
#        candi_grid[i][1] = centra_y
#    return candi_grid
#
## return a tensor is probability label 6*7*7
#def gailv_label(cate_num, label, candi_grid):
#    gailv_01 = torch.FloatTensor(6, 7, 7).zero_()
#    for i in range(cate_num):
#        obj = label[i]
#        cla_num = func.categery(obj[0])
#        x_1 = int(candi_grid[i,0])
#        x_2 = int(candi_grid[i,1])
#        gailv_01[cla_num,x_1,x_2]
#    return gailv_01
#
#
#def ty_loss(pred, label):
#    #hyper-parameters
#    obj_para = 5
#    noobj_para = 0.5
#    
#    cla_pro_01 = torch.FloatTensor(16, 7, 7).zero_()
#    
#    cate_num = len(label)
#    candi_grid = which_grid(label)
#    
#    for i in range(cate_num):
#        x_1 = int(candi_grid[i,0])
#        x_2 = int(candi_grid[i,1])
#        cla_pro_01[:,x_1,x_2] = 1
#        
#    cla_pro_01 = Variable(cla_pro_01).cuda()
#    pre_obj = pred * cla_pro_01
#    pre_obj_pro = pre_obj[0:6,:,:]
#    
#    #                    gailv loss
#    pro_label = gailv_label(cate_num, label, candi_grid)
#    pro_loss = torch.sum(torch.pow(pro_label - pre_obj_pro, 2))
#    
#    #                    no_obj confidence loss
#    cla_pro_10 = torch.neg(cla_pro_01) + 1
#    confi_label = torch.FloatTensor(2, 7, 7).zero_()
#    confi_label[0,:,:] = pred[10,:,:]
#    confi_label[1,:,:] = pred[15,:,:]
#    cla_pro_10_2 = cla_pro_10[0:2,:,:]
#    pre_noobj = confi_label * cla_pro_10_2
#    confi_label_real = torch.FloatTensor(2, 7, 7).zero_()
#    confi_loss = torch.sum(torch.pow(pre_noobj - confi_label_real, 2))
#    confi_loss = confi_loss * noobj_para
#    print()
#
#    #                    obj congfidence loss
#    for i in range(cate_num):
#        x_1 = int(candi_grid[i,0])
#        x_2 = int(candi_grid[i,1])
#        label_1 = label[i]
#        label_xywh = label_1[1:5]
#        pre_xywh1 = pre[6:10,x_1,x_2]
#        pre_xywh2 = pre[11:15,x_1,x_2]
#        pre_xywh1 = pre_xywh1.view(1,-1)
#        pre_xywh2 = pre_xywh2.view(1,-1)
#    cla_pro_01_2 = cla_pro_01[0:2,:,:]
#    obj_confi_label = torch.FloatTensor(2, 7, 7).zero_()
#    obj_confi_label[0,:,:] = pred[10,:,:]
#    obj_confi_label[1,:,:] = pred[15,:,:]
#
##   test
#label = [['Pedestrian',
#   12.74169491525421,
#   81.10560000000001,
#   22.482033898305122,
#   32.60970666666665],
#  ['Pedestrian',
#   125.328813559322057,
#   140.8544,
#   30.774237288135602,
#   62.10474666666667]]
##bb = which_grid(label)

def ty_loss(y_pred, y_true, S=7, B=2, C=6, use_cuda=False):
    ''' Calculate the loss of YOLO model.
    args:
        y_pred: (Batch, 7 * 7 * 16)
        y_true: dict object that contains:
            class_probs,
            confs,
            coord,
            proid,
            areas,
            upleft,
            bottomright

    '''

    SS = S * S
    scale_class_prob = 1
    scale_object_conf = 1
    scale_noobject_conf = 0.5
    scale_coordinate = 5
    batch_size = y_pred.size(0)

    # ground truth
    _coord = y_true['coord']
    _coord = _coord.view(-1, SS, B, 4)
    _upleft = y_true['upleft']
    _bottomright = y_true['bottomright']
    _areas = y_true['areas']
    _confs = y_true['confs']
    _proid = y_true['proid']
    _probs = y_true['class_probs']

    # Extract the coordinate prediction from y_pred
    coords = y_pred[:, SS * (C + B):].contiguous().view(-1, SS, B, 4)
    wh = torch.pow(coords[:, :, :, 2:4], 2)
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2].contiguous()
    floor = centers - (wh * 0.5)
    ceil = centers + (wh * 0.5)

    # Calculate the intersection areas
    intersect_upleft = torch.max(floor, _upleft)
    intersect_bottomright = torch.min(ceil, _bottomright)
    intersect_wh = intersect_bottomright - intersect_upleft
    zeros = Variable(torch.zeros(batch_size, SS, B, 2)).cuda() if use_cuda else Variable(torch.zeros(batch_size, 49, B, 2))
    intersect_wh = torch.max(intersect_wh, zeros)
    intersect = intersect_wh[:, :, :, 0] * intersect_wh[:, :, :, 1]

    # Calculate the best IOU, set 0.0 confidence for worse boxes
    iou = intersect / (_areas + area_pred - intersect)
    best_box = torch.eq(iou, torch.max(iou, 2)[0].unsqueeze(2))
    confs = best_box.float() * _confs

    # Take care of the weight terms
    conid = scale_noobject_conf * (1. - confs) + scale_object_conf * confs
    weight_coo = torch.cat(4 * [confs.unsqueeze(-1)], 3)
    cooid = scale_coordinate * weight_coo
    proid = scale_class_prob * _proid

    # Flatten 'em all
    probs = flatten(_probs)
    proid = flatten(proid)
    confs = flatten(confs)
    conid = flatten(conid)
    coord = flatten(_coord)
    cooid = flatten(cooid)

    true = torch.cat([probs, confs, coord], 1)
    wght = torch.cat([proid, conid, cooid], 1)

    loss = torch.pow(y_pred - true, 2)
    loss = loss * wght
    loss = torch.sum(loss, 1)
    return .5 * torch.mean(loss)

def flatten(x):
    return x.view(x.size(0), -1)