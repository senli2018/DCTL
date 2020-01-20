import cv2
import os
import random
import numpy as np

def get_type_image_names(data_path):
    """
    :param data_path:
    :return: 图片文件名的字符串列表
    """
    all_names = os.listdir(data_path)

    return all_names


def read_a_image_by_name(name, image_width, image_height, data_path):
    """
    :param name: 图片文件名
    :param image_width: 期望图片宽度
    :param image_height: 期望图片高度
    :param data_path:
    :return: 图片矩阵, 图片标签 /home/hit/codes/new_dataset/Y/0_malaria
    """
    # print(data_path)
    label = int(data_path[12])
    image = cv2.imread(os.path.join(data_path, name))
    if image is None:
        print("Error in build_data > read_fruits > main\n读取图片结果为空，请检查图片路径是否正确\n",
              os.path.join(data_path, name))
        exit()
    image = cv2.resize(image, (image_width, image_height))
    oimage =  image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image) / 127.5 - 1.
    return image, label,oimage


def get_batch_images(batch_size, image_width, image_height, data_path):
    """
    :param batch_size: batch大小
    :param image_width: 期望图片宽度
    :param image_height: 期望图片高度
    :param data_path:
    :return: 图片矩阵，文件名
    """
    images = []
    names = []
    all_names = get_type_image_names(data_path)

    index = 1
    while index <= batch_size:
        name = random.choice(all_names)
        image, _ ,o= read_a_image_by_name(name, image_width, image_height, data_path)
        images.append(image)
        names.append(name)
        index += 1

    return images, names
def get_train_batch(xy,batch_size,image_width,image_height, data_path):
    """
        :param batch_size: batch大小
        :param image_width: 期望图片宽度
        :param image_height: 期望图片高度
        :param data_path:
        :return: image, label
        """
    data_path=data_path+xy+'/'
    images = []
    labels = []
    x_folds=[]
    x_files=[]
    subFold="/train/"
    folds = os.listdir(data_path)
    #print(folds)
    for i in range(len(folds)):
        x_folds.append(data_path+folds[i]+subFold)
    for x_fold in x_folds:
        x_files.append(get_type_image_names(x_fold))
    #print('len(x_folds)',len(x_folds))
    #print('len(x_files)',len(x_files[0]))
    index=1
    #count=batch_size/len(x_folds)
    while index<=batch_size:
        #for i in range(len(x_folds)):
            i=random.randint(0,len(x_folds)-1)
            img_file=random.choice(x_files[i])
            #print(img_file)
            #print(x_folds[i])
            #print(img_file)
            image,label,o=read_a_image_by_name(img_file,image_width,image_height,x_folds[i])
            images.append(image)
            labels.append(label)
            #print(os.path.join(x_folds[i],img_file))
            #print(label)
            index=index+1

    return images,labels
def get_test_batch(xy,image_width,image_height,data_path):
    data_path = data_path + xy + '/'
    images = []
    labels = []
    x_folds = []
    x_files = []
    subFold = "/test/"
    folds = os.listdir(data_path)
    sum  =0
    #print(folds)
    for i in range(len(folds)):
        x_folds.append(data_path + folds[i] + subFold)
    for x_fold in x_folds:
        x_files.append(get_type_image_names(x_fold))
    for i in range(len(folds)):
        print('len(x_files)', len(x_files[i]))
    #print('len(x_folds)', len(x_folds))

    for i in range(len(x_folds)):
        for img_file in x_files[i]:
            image,label,o=read_a_image_by_name(img_file,image_width,image_height,x_folds[i])
            images.append(image)
            labels.append(label)
    return images,labels
def get_roc_batch(image_width,image_height,data_path):
    data_path = data_path + '/'
    images = []
    labels = []
    x_folds = []
    x_files = []
    subFold = "/test/"
    folds = os.listdir(data_path)
    #print(folds)
    for i in range(len(folds)):
        x_folds.append(data_path + folds[i] + subFold)
    for x_fold in x_folds:
        x_files.append(get_type_image_names(x_fold))
    #print('len(x_folds)', len(x_folds))
    #print('len(x_files)', len(x_files[0]))
    for i in range(len(x_folds)):
        for img_file in x_files[i]:
            image,label,o=read_a_image_by_name(img_file,image_width,image_height,x_folds[i])
            images.append(image)
            labels.append(label)
    return images,labels



def get_test_batch1(xy,batch_size,image_width,image_height, data_path):
    """
        :param batch_size: batch大小
        :param image_width: 期望图片宽度
        :param image_height: 期望图片高度
        :param data_path:
        :return: image, label
        """
    data_path=data_path+xy+'/'
    images = []
    labels = []
    path = []
    x_folds=[]
    x_files=[]
    oimage = []
    subFold="/test/"
    folds = os.listdir(data_path)
    #print(folds)
    for i in range(len(folds)):
        x_folds.append(data_path+folds[i]+subFold)
    for x_fold in x_folds:
        x_files.append(get_type_image_names(x_fold))
    #print('len(x_folds)',len(x_folds))
    #print('len(x_files)',len(x_files[0]))
    index=1
    #count=batch_size/len(x_folds)
    while index<=batch_size:
        for i in range(len(x_folds)):
            # print('x',i)
           # i=random.randint(0,len(x_folds)-1)
            img_file= random.choice(x_files[i])
            # print('img_file',img_file)
            # print('x_flods',x_folds[i])
            #print(img_file)
            image,label,o=read_a_image_by_name(img_file,image_width,image_height,x_folds[i])
            images.append(image)
            labels.append(label)
            oimage.append(o)
            path.append(os.path.join(x_folds[i],img_file))
            # print(os.path.join(x_folds[i],img_file))
            #print(label)
        index=index+1

    return images,labels,oimage
def get_test_batch2(xy,batch_size,image_width,image_height, data_path):
    """
        :param batch_size: batch大小
        :param image_width: 期望图片宽度
        :param image_height: 期望图片高度
        :param data_path:
        :return: image, label
        """
    data_path=data_path+xy+'/'
    images = []
    labels = []
    oimage = []
    path = []
    x_folds=[]
    x_files=[]
    subFold="/test/"
    folds = os.listdir(data_path)
    #print(folds)
    for i in range(len(folds)):
        x_folds.append(data_path+folds[i]+subFold)
    for x_fold in x_folds:
        x_files.append(get_type_image_names(x_fold))
    #print('len(x_folds)',len(x_folds))
    #print('len(x_files)',len(x_files[0]))
    index=1
    #count=batch_size/len(x_folds)
    while index<=batch_size:
        for i in range(len(x_folds)):
            # print('x',i)
           # i=random.randint(0,len(x_folds)-1)
            img_file= random.choice(x_files[i])
            # print('img_file',img_file)
            # print('x_flods',x_folds[i])
            #print(img_file)
            image,label,o =read_a_image_by_name(img_file,image_width,image_height,x_folds[i])
            images.append(image)
            labels.append(label)
            oimage.append(o)
            path.append(os.path.join(x_folds[i],img_file))
            # print(os.path.join(x_folds[i],img_file))
            #print(label)
        index=index+1

    return images,labels,oimage

# def get_test_batch3(batch_size,image_width,image_height, data_path):
#     """
#         :param batch_size: batch大小
#         :param image_width: 期望图片宽度
#         :param image_height: 期望图片高度
#         :param data_path:
#         :return: image, label
#         """
#     xdata_path = data_path+  'X/'
#     ydata_path = data_path + 'Y/'
#     ximages = []
#     xlabels = []
#     oximage = []
#
#     yimages = []
#     ylabels = []
#     oyimage = []
#
#     path = []
#     x_folds=[]
#     x_files=[]
#     y_folds = []
#     y_files = []
#     subFold="/test/"
#
#
#     xfolds = os.listdir(xdata_path)
#     yfolds = os.listdir(ydata_path)
#
#     #print(folds)
#     for i in range(len(xfolds)):
#         x_folds.append(xdata_path+xfolds[i]+subFold)
#         y_folds.append(ydata_path + yfolds[i] + subFold)
#     for x_fold in x_folds:
#         x_files.append(get_type_image_names(x_fold))
#     for y_fold in y_folds:
#         y_files.append(get_type_image_names(y_fold))
#
#
#     index=1
#     #count=batch_size/len(x_folds)
#     while index<=batch_size:
#         i = random.randint(0, len(x_folds) - 1)
#         for i in
#
#
#
#
#             ximg_file= random.choice(x_files[i])
#             ximage,xlabel,ox =read_a_image_by_name(ximg_file,image_width,image_height,x_folds[i])
#             ximages.append(ximage)
#             xlabels.append(xlabel)
#             oximage.append(ox)
#
#             yimg_file = random.choice(y_files[i])
#             yimage, ylabel, oy = read_a_image_by_name(yimg_file, image_width, image_height, y_folds[i])
#             yimages.append(yimage)
#             ylabels.append(ylabel)
#             oyimage.append(oy)
#
#
#         index=index+1
#
#     return ximages,xlabels,yimages,ylabels,  oximage,oyimage



if __name__ == '__main__':
    #i, l = get_batch_images(10, 224, 224, "/home/hit/paraProject/liaijia/CycleGAN_python/data/image/banana_train")
    # i, l=get_batch_images(1000,1000,224,224,"./dataset/")
    #print(len(i))
    #print(l

    i,l,c=get_test_batch1('Y', 100, 224, 225, "../dataset/")
    # print('X',len(i))
    print(l)
    print(len(l))
