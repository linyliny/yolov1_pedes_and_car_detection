from PIL import Image
import numpy as np
import json
import torch

def IOU(label, pre):
#    pre = pred[0]
    cx1 = pre[0]
    cy1 = pre[1]
    cx2 = pre[2]
    cy2 = pre[3]

    gx1 = label[0]
    gy1 = label[1]
    gx2 = label[2]
    gy2 = label[3]

    carea = (cx2 - cx1) * (cy2 - cy1) #C的面积
    garea = (gx2 - gx1) * (gy2 - gy1) #G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h #C∩G的面积

    iou = area / (carea + garea - area)

    return iou

def categery(dd):
    if dd == 0: cate = 'Car'
    if dd == 1: cate = 'Van'
    if dd == 2: cate = 'Truck'
    if dd == 3: cate = 'Pedestrian'
    if dd == 4: cate = 'Person_sitting'
    if dd == 5: cate = 'Cyclist'
    print(cate)
    return cate

def get_image(i):
    fname = 'trainset/image_f/'
    filename = fname + str(i) + '.jpg'
    im = Image.open(filename)
    im_array = np.array(im) / 255
    im_array = np.transpose(im_array, (2, 0, 1))
    return torch.from_numpy(im_array).float()

def get_image_raw(i):
    fname = 'trainset/image_f/'
    filename = fname + str(i) + '.jpg'
    im = Image.open(filename)
    im_array = np.array(im)
    return im_array
    
def get_label(i):
#    print(i,type(i))
    filename = 'label_f/{}.json'.format(i)
#    filename = fname + str(i) + '.json'
    with open(filename, "r") as f:
        d1 = json.load(f)
#    la_num = len(d1)
    return d1

#def return_label(la):
    

    #test
    
#dfdf = get_image()
#subim_1 = Image.fromarray(dfdf)
#subim_1.show()
    
#x = [125, 140, 30.77423, 62]
#y = [125, 140, 30.77423, 62]
#kk = IOU(x,y)
#print(kk)

#d1 = get_label()
#print(d1)
    
def max_pro(tensor_):
    num = 0
    for i in range(49):
        if int(tensor_[i]) > 0:
            num += 1
    dd = 0
    out_ = torch.IntTensor(num, 2).zero_()
    for j in range(49):
        if int(tensor_[j]) > 0:
            out_[dd,0] = j
            out_[dd,1] = int(tensor_[j])
            dd += 1
    return out_

def find_coord(pre, index, num):
    out_ = torch.FloatTensor(num, 6).zero_()
    for i in range(num):
        a = index[i,0]
        b = index[i,1]
        pre_ = pre.data
        for j in range(4):
            out_[i,j + 2] = pre_[a,j]
        out_[i,0] = b
        out_[i,1] = a
    return out_

def gene_bb(im_arr, x, y, z, w):
    im = im_arr
    x = int(x)
    y = int(y)
    z = int(z)
    w = int(w)
    for i in range(x, z+1):
        for j in range(y, y+2):
            im[j, i, 0] = 0
            im[j, i, 1] = 255
            im[j, i, 2] = 0
    for i in range(x, x+2):
        for j in range(y+2, w-1):
            im[j, i, 0] = 0
            im[j, i, 1] = 255
            im[j, i, 2] = 0
    for i in range(z-1, z+1):
        for j in range(y+2, w-1):
            im[j, i, 0] = 0
            im[j, i, 1] = 255
            im[j, i, 2] = 0
    for i in range(x, z+1):
        for j in range(w-1,w+1):
            im[j, i, 0] = 0
            im[j, i, 1] = 255
            im[j, i, 2] = 0
    return(im)