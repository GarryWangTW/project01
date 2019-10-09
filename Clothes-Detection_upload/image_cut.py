from __future__ import division

from yolo.utils.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *
from yolo.utils.utils2 import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2
import sys
import copy
#from PIL import Image
import re
import glob
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {   
    "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
    "weights_path" : "yolo/weights/yolov3-df2_15000.weights",
    "class_path":"yolo/df2cfg/df2.names",
    "conf_thres" : 0.25,
    "nms_thres" :0.4,
    "img_size" : 416,
    "device" : device
}

classes = load_classes(params['class_path']) 
#print(classes)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
cmap = plt.get_cmap("tab20")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 20)])
np.random.shuffle(colors)

model = load_model(params)
print('Model loaded successfully.')

filenames = glob.glob("crawler_pics/*.jpg")
filenames.sort()
images = [cv2.imread(img) for img in filenames]

for index_img, img in enumerate(images):
    # following lists for cut_partial images:
    x_part = []
    y_part = []

    #img = cv2.imread(img_path)
    if img is None:
        print('Image not found...')
        continue
    
    img2= img.copy()     
    x , _ ,_= cv_img_to_tensor(img)
    
    x.to(device)   

            # Get detections
    with torch.no_grad():
        input_img= Variable(x.type(Tensor))  
        detections = model(input_img)
        detections = non_max_suppression(detections, params['conf_thres'], params['nms_thres'])

    if detections[0] is not None:

        detections_org = detections[0].clone()
        detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds , seed)
        bbox_colors = colors[:n_cls_preds]
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            
            print("\t+%05d Label: %s, Conf: %.5f" % (index_img, classes[int(cls_pred)], cls_conf.item()))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = tuple(c*255 for c in color)
            color = (color[2],color[1],color[0])
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf.item())

            cv2.rectangle(img2, (x1,y1), (x2,y2), color,3)
            cv2.rectangle(img2, (x1-2,y1-25), (x1 + 8.5*len(text),y1), color, -1)
            cv2.putText(img2, text, (x1,y1-5), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

            try:
                os.listdir('cut/part_cut/upper')
                os.listdir('cut/main_cut')
            except FileNotFoundError:
                os.makedirs('cut/part_cut/upper')
                os.makedirs('cut/part_cut/down')
                os.makedirs('cut/part_cut/entire')
                os.makedirs('cut/main_cut')
 
            if int(x1) <= 0:
                x1 == 0
            if int(y1) <= 0:
                y1 == 0
            if int(y2) >= img.shape[0]:
                y2 == img.shape[0]

            if int(x2) >= img.shape[1]:
                x2 == img.shape[1]

            if int(cls_pred) in range(0, 6):
                img_cut = img[abs(int(y1)):abs(int(y2)), abs(int(x1)):abs(int(x2))].copy()
                #img_cut_upper = cv2.resize(img_cut, (100, 100))
                x_part.extend([abs(int(x1)), abs(int(x2))])
                y_part.extend([abs(int(y1)), abs(int(y2))])
                cv2.imwrite('cut/part_cut/upper/%s_%d_%d.jpg'%(classes[int(cls_pred)], index_img, cls_pred), img_cut)

            elif int(cls_pred) in range(6, 9):
                img_cut = img[abs(int(y1)):abs(int(y2)), abs(int(x1)):abs(int(x2))].copy()
                #img_cut_down = cv2.resize(img_cut, (100, 150))
                x_part.extend([abs(int(x1)), abs(int(x2))])
                y_part.extend([abs(int(y1)), abs(int(y2))])
                cv2.imwrite('cut/part_cut/down/%s_%d_%d.jpg'%(classes[int(cls_pred)], index_img, cls_pred), img_cut)

            elif int(cls_pred) in range(9, 13):
                img_cut = img[abs(int(y1)):abs(int(y2)), abs(int(x1)):abs(int(x2))].copy()
                #img_cut_entire = cv2.resize(img_cut, (100, 250))
                x_part.extend([abs(int(x1)), abs(int(x2))])
                y_part.extend([abs(int(y1)), abs(int(y2))])
                cv2.imwrite('cut/part_cut/entire/%s_%d_%d.jpg'%(classes[int(cls_pred)], index_img, cls_pred), img_cut)
        img_sol = img[min(y_part):max(y_part), min(x_part):max(x_part)].copy()        
        cv2.imwrite('cut/main_cut/%d.jpg'%index_img, img_sol)
            
    else:
        print('No detections...') 
        with open('error_pics.txt', 'a') as txt_file:
            txt_file.write("%s\n"%(filenames[index_img])) 
