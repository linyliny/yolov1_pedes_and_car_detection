from __future__ import print_function
from torch.utils.data import Dataset
import numpy as np
import torch
#from torchvision import transforms
import func
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.autograd import Variable

def create_label(chunk):
    S, B = 7, 2
    C = 6
    labels = {
            "Car": 0,
            "Van": 1,
            "Truck": 2,
            "Pedestrian": 3,
            "Person_sitting": 4,
            "Cyclist": 5,
            }
    
    jpg = chunk[0]
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    img = plt.imread(jpg) / 255.
    img = np.transpose(img, (2, 0, 1))

    cell_x = 1. * w / S # width per cell
    cell_y = 1. * h / S # height per cell

    for obj in allobj:
        # center_x = 0.5 * (obj[1] + obj[3]) # (xmin + xmax) / 2
        # center_y = 0.5 * (obj[2] + obj[4]) # (ymin + ymax) / 2
        center_x = obj[1] + obj[3] / 2# left_x + width/ 2
        center_y = obj[2] + obj[4] / 2 # upper_y + height / 2

        cx = center_x / cell_x # rescale the center x to cell size
        cy = center_y / cell_y # rescale the center y to cell size

        # obj[3] = float(obj[3] - obj[1]) / w # calculate and normalize width
        # obj[4] = float(obj[4] - obj[2]) / h # calculate and normalize height
#        print(obj[3],obj[4],'\n')
        obj[3] = obj[3] / w # calculate and normalize width
        obj[4] = obj[4] / h # calculate and normalize height
#        if obj[3]<0:
#            print('\n\n\n\n\n\n\n\n\n\n\n')
#            print(chunk[0])
        obj[3] = np.sqrt(obj[3]) # sqrt w
        obj[4] = np.sqrt(obj[4]) # sqrt h
#        print(obj[3],obj[4],'\n\n\n')

        obj[1] = cx - np.floor(cx) # center x in each cell
        obj[2] = cy - np.floor(cy) # center x in each cell

        obj += [int(np.floor(cy) * S + np.floor(cx))] # indexing cell[0, 49)

    # each object: length: 6,
    # [label, center_x_in_cell, center_y_in_cell, w_in_image, h_in_image, cell_idx]

    class_probs = np.zeros([S*S, C]) # for one_hot vector per each cell
    confs = np.zeros([S*S, B]) # for 2 bounding box per each cell
    coord = np.zeros([S*S, B, 4]) # for 4 coordinates per bounding box per cell
    proid = np.zeros([S*S, C]) # for class_probs weight \mathbb{1}^{obj}
    prear = np.zeros([S*S, 4]) # for bounding box coordinates

    for obj in allobj:
        class_probs[obj[5], :] = [0.] * C # no need?
        if not obj[0] in labels: continue
        class_probs[obj[5], labels[obj[0]]] = 1.

        # for object confidence? -> the cell which contains object is 1 nor 0
        confs[obj[5], :] = [1.] * B 

        # assign [center_x_in_cell, center_y_in_cell, w_in_image, h_in_image]
        coord[obj[5], :, :] = [obj[1:5]] * B 

        # for 1_{i}^{obj} in paper eq.(3)
        proid[obj[5], :] = [1] * C

        # transform width and height to the scale of coordinates
        prear[obj[5], 0] = obj[1] - obj[3] ** 2 * 0.5 * S # x_left
        prear[obj[5], 1] = obj[2] - obj[4] ** 2 * 0.5 * S # y_top
        prear[obj[5], 2] = obj[1] + obj[3] ** 2 * 0.5 * S # x_right
        prear[obj[5], 3] = obj[2] + obj[4] ** 2 * 0.5 * S # y_bottom

    # for calculate upleft, bottomright and areas for 2 bounding box(not for 1 bounding box)
    upleft = np.expand_dims(prear[:, 0:2], 1)
    bottomright = np.expand_dims(prear[:, 2:4], 1)
    wh = bottomright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    bottomright = np.concatenate([bottomright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    y_true = {
            'class_probs': class_probs,
            'confs': confs,
            'coord': coord,
            'proid': proid,
            'areas': areas,
            'upleft': upleft,
            'bottomright': bottomright
            }
    return img, y_true
    
#class yolo_dataset(Dataset):
#    def __init__(self, train=True):
#        # TODO
#        # 1. Initialize file path or list of file names.
#        self.train = train
#    def __getitem__(self, index):
#        # TODO
#        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#        # 2. Preprocess the data (e.g. torchvision.Transform).
#        # 3. Return a data pair (e.g. image and label).
#        #这里需要注意的是，第一步：read one data，是一个data
#        if self.train:
#            image = func.get_image(i=index)
#            image = np.transpose(image, (2, 0, 1))
#            label_raw = func.get_label(i=index)
#            y_true = func.create_label(label_raw)
#            return image, y_true
#        else:
#            pass
#    def __len__(self):
#        # You should change 0 to the total size of your dataset.
#        if self.train:
#            return 14356
#        else:
#            pass

def get_datas(idx, use_cuda=True):
    x_batch = list()
    feed_batch = dict()
    for i in idx:
        chunk = func.get_label(i)
        img, new_feed = create_label(chunk)

        if img is None:
            continue
        x_batch += [np.expand_dims(img, 0)]

        for key in new_feed:
            new = new_feed[key]
            old_feed = feed_batch.get(key,
                    np.zeros((0,) + new.shape))

            feed_batch[key] = np.concatenate([
                old_feed, [new]])

    if use_cuda:
        x_batch = Variable(torch.from_numpy(np.concatenate(x_batch, 0)).float()).cuda()
        feed_batch = {key: Variable(torch.from_numpy(feed_batch[key]).float()).cuda()
                      for key in feed_batch}

    else:
        x_batch = torch.from_numpy(np.concatenate(x_batch, 0)).float()
        feed_batch = {key: Variable(torch.from_numpy(feed_batch[key]).float())
                      for key in feed_batch}

    return x_batch, feed_batch

def train_batches(batch_size=1, use_cuda=True):
    train_size = 14356
    shuffle_idx = np.random.permutation(list(range(1, train_size + 1)))
    for i in range(train_size // batch_size):
        yield get_datas(shuffle_idx[i*batch_size: (i+1)*batch_size], use_cuda)
#  test
#dd = yolo_dataset()
#print(len(dd))
