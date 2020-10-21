import os
import logging
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms

def load_img(input_path, img_resize):
    img_ori_list=[]
    img_tensor_list=[]
    img_name_list=[]

    raw_names = glob.glob(input_path+'*.raw')
    jpg_names = glob.glob(input_path+'*.jpg')
    png_names = glob.glob(input_path+'*.png')
    img_names = jpg_names + png_names

    eval_transform = transforms.Compose([
                transforms.Resize((img_resize[0], img_resize[1])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(raw_names)
    print(img_names)


    for raw_name in raw_names:
        name=raw_name.split('/')
        date, num, region, view, width, height, endpart = name[-1].split('_')
        width = int(width)
        height = int(height)
        bit, filetype = endpart.split('.')

        f = open(raw_name, 'rb')
        f_array = np.fromfile(f, dtype=np.uint16, count=width*height)
        f_8bit = np.uint8(f_array/256)
        f_8bit = np.reshape(f_8bit, (width, height))
        
        f_img = Image.fromarray(f_8bit).convert('RGB')

        f_tensor = eval_transform(f_img)
 
        img_ori_list.append(f_img.resize((img_resize[0], img_resize[1])))
        img_tensor_list.append(f_tensor)
        img_name_list.append(name[-1].split('.')[0] + '.png')

    for raw_name in img_names:
        name=raw_name.split('/')
        img_pil = Image.open(raw_name).convert('RGB')
       
        f_tensor = eval_transform(img_pil)

        img_ori_list.append(img_pil.resize((img_resize[0], img_resize[1])))
        img_tensor_list.append(f_tensor)
        img_name_list.append(name[-1])

    return img_ori_list, img_tensor_list, img_name_list

