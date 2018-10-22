from PIL import Image
import numpy as np
import os

#addre1 = 'training/image_2'
#addre2 = 'training/label_2'
out_i = 0
for i in range(0,1):

    just0 = '00'
    if i<10:just0 = just0 + '000'
    if 10<=i<100:just0 = just0 + '00'
    if 100<=i<1000:just0 = just0 + '0'
    imnum = just0 + str(i)
    imset = 'image_2/'
    laset = 'label_2/'
    imfilename = 'training/' + imset + imnum + '.png'
    lafilename = 'training/' + laset + imnum + '.txt'

    cl_list = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist']
    
    #output png and txt(not yet)
#    outnum_no0 = (i * 3, i * 3 + 1, i * 3 + 2)
    def outnum (j):
        if j<10:just00 = '00000'
        if 10<=j<100:just00 = '0000'
        if 100<=j<1000:just00 = '000'
        if 1000<=j<10000:just00 = '00'
        if j>=10000:just00 = '0'
        return (just00 + str(j))
    def outname (bo, jjj):
        if bo == 1:
            out_im_set = 'image/'
            im_name = 'trainset/' + out_im_set + outnum(jjj) + '.png'
            return(im_name)
        else:
            out_la_set = 'label/'
            la_name = 'trainset/' + out_la_set + outnum(jjj) + '.txt'
            return(la_name)

    #chansheng box-bounding in image.
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

    #read txt
    b_b = np.loadtxt(lafilename, usecols = (4,5,6,7))
    label = np.loadtxt(lafilename, str, usecols = (0))
#    label = label.tolist()
    kk = np.mat(b_b)
    ss = kk.size
    if ss == 4:
        h = np.zeros((1,4))
        for hh in range(0,4):
            h[0,hh] = b_b[hh]
        b_b = h
        cl_num = 1
        label = label.tolist()
    else:
        cl_num = len(label)
        
    #load and divide image into 3
    im = Image.open(imfilename)
    im_array = np.array(im)
    wid = len(im_array[0])/3
    high = len(im_array)
    subar_1 = im_array[:,0:wid+1,0:3]
    #subim_1 = Image.fromarray(subar_1)
    subar_2 = im_array[:,wid:wid*2+1,0:3]
    #subim_2 = Image.fromarray(subar_2)
    subar_3 = im_array[:,wid*2+1:wid*3+1,0:3]
    #subim_3 = Image.fromarray(subar_3)


#
#    cl_num = len(label)
    la_1 = np.zeros((cl_num,4))
    cl_1 = []
    la_2 = np.zeros((cl_num,4))
    cl_2 = []
    la_3 = np.zeros((cl_num,4))
    cl_3 = []




#gaibian g-t
    for l in range(0, cl_num):
        if ((b_b[l,0]<(wid+1)) and (b_b[l,2]>=(wid*2+1))):
            if label[l] in cl_list:
                cl_2.append(label[l])
                la_2[l,:] = b_b[l,:]
                la_2[l,0] = 1
                la_2[l,2] = wid-1
            if (b_b[l,2]-(wid*2+1))>=(wid/7):
                if label[l] in cl_list:
                    cl_3.append(label[l])
                    la_3[l,:] = b_b[l,:]
                    la_3[l,0] = 1
                    la_3[l,2] = b_b[l,2]-(wid*2+1)-1
            if ((wid+1)-b_b[l,0])>=(wid/7):
                if label[l] in cl_list:
                    cl_1.append(label[l])
                    la_1[l,:] = b_b[l,:]
                    la_1[l,0] = b_b[l,0]+1
                    la_1[l,2] = wid-1
        elif (b_b[l,0]<(wid+1)) and (b_b[l,2]>=(wid+1)):
            if (b_b[l,2]-(wid+1))>=(wid/7):
                if label[l] in cl_list:
                    cl_2.append(label[l])
                    la_2[l,:] = b_b[l,:]
                    la_2[l,0] = 1
                    la_2[l,2] = b_b[l,2]-(wid+1)-1
            if ((wid+1)-b_b[l,0])>=(wid/7):
                if label[l] in cl_list:
                    cl_1.append(label[l])
                    la_1[l,:] = b_b[l,:]
                    la_1[l,0] = b_b[l,0]+1
                    la_1[l,2] = wid-1
        elif (b_b[l,0]<(wid*2+1)) and (b_b[l,2]>=(wid*2+1)):
            if (b_b[l,2]-(wid*2+1))>=(wid/7):
                if label[l] in cl_list:
                    cl_3.append(label[l])
                    la_3[l,:] = b_b[l,:]
                    la_3[l,0] = 1
                    la_3[l,2] = b_b[l,2]-(wid*2+1)-1
            if ((wid*2+1)-b_b[l,0])>=(wid/7):
                if label[l] in cl_list:
                    cl_2.append(label[l])
                    la_2[l,:] = b_b[l,:]
                    la_2[l,0] = b_b[l,0] - wid+1
                    la_2[l,2] = wid-1
        elif (b_b[l,0]<wid+1) and (b_b[l,2]<wid+1):
            if label[l] in cl_list:
                cl_1.append(label[l])
                la_1[l,:] = b_b[l,:]
                la_1[l,0] = b_b[l,0]+1
                la_1[l,2] = b_b[l,2]-1
        elif (b_b[l,0]>=wid+1) and (b_b[l,2]<wid*2+1):
            if label[l] in cl_list:
                cl_2.append(label[l])
                la_2[l,:] = b_b[l,:]
                la_2[l,0] = b_b[l,0] - wid+1
                la_2[l,2] = b_b[l,2] - wid-1
        elif (b_b[l,0]>=wid*2+1) and (b_b[l,2]>wid*2+1):
            if label[l] in cl_list:
                cl_3.append(label[l])
                la_3[l,:] = b_b[l,:]
                la_3[l,0] = b_b[l,0] - wid*2+1
                la_3[l,2] = b_b[l,2] - (2*wid + 1)-1


        

    cl_num1 = len(cl_1)
    cl_num2 = len(cl_2)
    cl_num3 = len(cl_3)


    lla_1 = np.zeros((cl_num1,4))
    lla_2 = np.zeros((cl_num2,4))
    lla_3 = np.zeros((cl_num3,4))


#paixu
    kk = 0
    for u in range(0, cl_num):
        if la_1[u,0]:
            lla_1[kk,:] = la_1[u,:]
            kk = kk+1
#    if  len(lla_1):
#        for u in range(0, cl_num1):
#            ssubar_1 = gene_bb(subar_1,lla_1[u,0],lla_1[u,1],lla_1[u,2],lla_1[u,3])
#    else: ssubar_1 = subar_1
    kk = 0
    for u in range(0, cl_num):
        if la_2[u,0]:
            lla_2[kk,:] = la_2[u,:]
            kk = kk+1
#    if  len(lla_2):
#        for u in range(0, cl_num2):
#            ssubar_2 = gene_bb(subar_2,lla_2[u,0],lla_2[u,1],lla_2[u,2],lla_2[u,3])
#    else: ssubar_2 = subar_2
    kk = 0
    for u in range(0, cl_num):
        if la_3[u,0]:
            lla_3[kk,:] = la_3[u,:]
            kk = kk+1
#    if  len(lla_3):
#        for u in range(0, cl_num3):
#            ssubar_3 = gene_bb(subar_3,lla_3[u,0],lla_3[u,1],lla_3[u,2],lla_3[u,3])
#    else: ssubar_3 = subar_3


    cwd = os.getcwd()
    #subim_1.show()
    #subim_2.show()
    #subim_3.show()
    if cl_num1:
        subim_1 = Image.fromarray(subar_1)
        out_add1 = outname(1, out_i)
        out_addt1 = cwd + '/trainset/label/' + outnum(out_i) + '.txt'
        subim_1.save(out_add1)
        f=file(out_addt1, "a+")
        for i in range(0,cl_num1):
            new_c = cl_1[i] + ' '
            f.write(new_c)
            for j in range(0,4):
                b_b_1 = str(lla_1[i,j])
                new_c = b_b_1 + ' '
                f.write(new_c)
            new_c = '\n'
            f.write(new_c)
        f.close()
        out_i = out_i+1
    if cl_num2:
        subim_2 = Image.fromarray(subar_2)
        out_add2 = outname(1, out_i)
        out_addt2 = cwd + '/trainset/label/' + outnum(out_i) + '.txt'
        subim_2.save(out_add2)
        f=file(out_addt2, "a+")
        for i in range(0,cl_num2):
            new_c = cl_2[i] + ' '
            f.write(new_c)
            for j in range(0,4):
                b_b_2 = str(lla_2[i,j])
                new_c = b_b_2 + ' '
                f.write(new_c)
            new_c = '\n'
            f.write(new_c)
        f.close()
        out_i = out_i+1
    
    
    if cl_num3:
        subim_3 = Image.fromarray(subar_3)
        out_add3 = outname(1, out_i)
        out_addt3 = cwd + '/trainset/label/' + outnum(out_i) + '.txt'
        subim_3.save(out_add3)
        f=file(out_addt3, "a+")
        for i in range(0,cl_num3):
            new_c = cl_3[i] + ' '
            f.write(new_c)
            for j in range(0,4):
                b_b_3 = str(lla_3[i,j])
                new_c = b_b_3 + ' '
                f.write(new_c)
            new_c = '\n'
            f.write(new_c)
        f.close()
        out_i = out_i+1
#    subim_1 = Image.fromarray(ssubar_1)
#    subim_2 = Image.fromarray(ssubar_2)
#    subim_3 = Image.fromarray(ssubar_3)
#    out_add1 = outname(1, outnum_no0[0])
#    out_add2 = outname(1, outnum_no0[1])
#    out_add3 = outname(1, outnum_no0[2])
#    subim_1.save(out_add1)
#    subim_2.save(out_add2)
#    subim_3.save(out_add3)

#im_with_bb = gene_bb(im_array,b_b[0,0],b_b[0,1],b_b[0,2],b_b[0,3])
#imbb = Image.fromarray(im_with_bb)
#imbb.save('trainset/image/shiyanbb.png')