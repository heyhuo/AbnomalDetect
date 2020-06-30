import pprint
import math
import os
import cv2
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
from glob import glob

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_varaibles():
    # 返回需要训练的变量列表
    model_vars = tf.trainable_variables()
    # 打印出所有与训练相关的变量信息
    slim.model_analyzer.analyze_vars(model_vars,print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
    return transform(imread(image_path),input_height,input_width,
                     resize_height,resize_width,crop)

def cut_image(image_path,save_path,imsize,size):
    h,w = imsize[0],imsize[1]
    cut_images = []
    for i in range(size[0]):
        for j in range(size[1]):
            img = image[j * h:j * h + h, i * w:i * w + w, :]
            path = 'b/{:d}.jpg'.format(int(i*size[0]+j))
            cut_images.append(img)
            scipy.misc.imsave(path,img)
    return cut_images


def merge_images(images,size):
    return inverse_transform(images)

def merge(images,size):
    h,w = images.shape[1],images.shape[2]
    # 如果图片的channel是3或4
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h*size[0],w*size[1],c))
        for idx,image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h,i * w:i * w + w,  :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[i * w:i * w + w,j * h:j * h + h] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def draw_box(images,size,image_path,labels):
    img = cv2.imread(image_path)
    h,w = 32,32
    for idx in range(size[0]*size[1]):
        i = idx % size[1]
        j = idx // size[1]
        if labels[idx] == 0:
            cv2.rectangle(img, (int(i * h), int(j * w)), (int((i + 1) * h), int((j + 1) * w)), (3, 97, 255), 2)
    cv2.imwrite(image_path, img)

def save_images(images,size,image_path):
    return imsave(inverse_transform(images),size,image_path)


def save_gen_images(images,size,image_path,labels):
    save_images(images, size, image_path)
    draw_box(images, size, image_path, labels)


def imread(path):
    return scipy.misc.imread(path,mode='RGB').astype(np.float)

def imsave(images,size,path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255 * img).astype(np.uint8))

def imsave_gen(images,size,path,labels):
    image = np.squeeze(merge_gen(images, size,labels))
    return scipy.misc.imsave(path, image)

def inverse_transform(images):
    return (images+1.) / 2.

def center_crop(x,crop_h,crop_w,resize_h=64,resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h,w = x.shape[:2]
    j = int(round(h - crop_h)/2.)
    i = int(round(w - crop_w) / 2.)
    return scipy.misc.imresize(
        x[j:j+crop_h,i:i+crop_w],[resize_h,resize_w])

def transform(image,input_height,input_width,
                     resize_height = 64,resize_width = 64,crop = True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image,[resize_height,resize_width])
    return np.array(cropped_image)/127.5 - 1

# 融合图片
def mixed_pics(rust_name,cell_name,hs,ws):
    rust_img = cv2.imread(rust_name)
    cell_img = cv2.imread(cell_name)
    # Create an all white mask
    mask = 255 * np.ones(rust_img.shape, rust_img.dtype)
    width, height, channels = cell_img.shape
    center = (int(height/2),int(width/2))
    print(center)
    mixed_clone = cv2.seamlessClone(rust_img, cell_img, mask, center, cv2.NORMAL_CLONE)
    return mixed_clone

# # 处理生锈贴图
# def deal_rust_img():
#     scale_img(32,32,8,'data/train/rust/','data/train/new_rust/')
