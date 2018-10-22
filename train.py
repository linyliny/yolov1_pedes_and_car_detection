from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import torch
import random
import shutil
#import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np
#from torch.autograd import Variable
import torch.optim as optim
import data_set, loss, tinyyolonet
#from torch.utils.data import DataLoader

random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))

batch_size = 64
n_epoch    = 30

#ty_dataset = data_set.yolo_dataset(train=True)
#dataloader = torch.utils.data.DataLoader(ty_dataset, batch_size = batch_size, shuffle=True)

# define and initialize the net
net = tinyyolonet.Yolo_tiny()
#para = torch.load('yolov1-tiny.pth')
checkpoint_load = torch.load('model_best/Best_Models.pth')
net.load_state_dict(checkpoint_load['state_dict'])
#net.load_state_dict(para)
net.cuda()

optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9, weight_decay=0.00001)
if __name__ == '__main__':
    best = 1.
    for epoch in range(1, n_epoch+1):
        losses = 0.0
        for i, (x, y) in enumerate(data_set.train_batches(batch_size, use_cuda=True), 1):
            optimizer.zero_grad()
            y_pred = net(x)
#            y_pred = y_pred.view(7,7,-1)
            l = loss.ty_loss(y_pred, y, use_cuda=True)
            l.backward()
            optimizer.step()
            losses += l.data[0]
#            print("Epoch: {}, Batch: {}, Ave loss: {}".format(epoch, i*batch_size/14356*100, losses / 30))
            if i%30 == 29:
                print("Epoch: {}, Batch: {}, Ave loss: {}".format(epoch, i*batch_size/14356*100, losses / 30))
                losses = 0.0
        
        l_cpu = l
        l_cpu = float(l_cpu.cpu())
        is_best = l_cpu < best
        best = min(l_cpu, best)
        
        checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        if int(is_best) == 1:
            torch.save(checkpoint, 'model_best/Best_Models.pth')
            print('saved a best model so far')
            
        file_path = 'models/myModels' + str(epoch) + '.pth'
        torch.save(checkpoint, file_path)
        print('model saved')
