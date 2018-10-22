#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:58:31 2018

@author: xh
"""
import json
import numpy as np

for i in range(14355,14356):
    just0 = '0'
    if i<10:just0 = just0 + '0000'
    if 10<=i<100:just0 = just0 + '000'
    if 100<=i<1000:just0 = just0 + '00'
    if 1000<=i<10000:just0 = '00'

    imnum = just0 + str(i)
    laset = 'label_test/'
    laset_test = 'label_f/'
    imset_test = 'image_test/'
    lafilename = 'trainset/' + laset + imnum + '.txt'
    c = i + 1
    lafilename_test = 'trainset/' + laset_test + str(c) + '.json'
    imfilename_test = 'trainset/' + imset_test + str(c) + '.jpg'
    
    final = []
    final.append(imfilename_test)
    
    
    b_b = np.loadtxt(lafilename, usecols = (1,2,3,4))
    label = np.loadtxt(lafilename, str, usecols = (0))
    kk = np.mat(b_b)
    kk[:,2] = kk[:,2] - kk[:,0]
    kk[:,3] = kk[:,3] - kk[:,1]
    final_2 = [224,224]
    
    
    ss = kk.size
    if ss == 4:
        label_li = []
        label_li.append(label.tostring())
        for u in range(0,4):
            label_li.append(kk[0,u])
        final_2.append(label_li)
        final.append(final_2)
        with open(lafilename_test,"w") as f:
            json.dump(final,f)
    else:
        cl_num = len(label)
        final_1 = []
        for i in range(0,cl_num):
            label_li = []
            label_li.append(label[i].tostring())
            for u in range(0,4):
                label_li.append(kk[i,u])
            final_1.append(label_li)
        final_2.append(final_1)
        final.append(final_2)
        with open(lafilename_test,"w") as f:
            json.dump(final,f)
    
    
#    fp = open(lafilename_test,'w+')
#    fp.write(json.loads(testDict))
#    fp.close()