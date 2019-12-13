import cv2
import numpy as np
import os
import random

def point_in_box(x, y, box):
    y1, x1, y2, x2, y3, x3, y4, x4 = box
    a = (x2-x1)*(y-y1)-(y2-y1)*(x-x1)
    b = (x3-x2)*(y-y2)-(y3-y2)*(x-x2)
    c = (x4-x3)*(y-y3)-(y4-y3)*(x-x3)
    d = (x1-x4)*(y-y4)-(y1-y4)*(x-x4)

    return (a>=0 and b>=0 and c>=0 and d>=0) or (a<=0 and b<=0 and c<=0 and d<=0)

def load_data(img_path, label_path):
    img_path = img_path.decode()
    label_path = label_path.decode()
    img = cv2.imread(img_path)
    if img is None:
        print (img_path)
        return None, None
    min_l = float(min(img.shape[0], img.shape[1]))
    rate= 256.0/min_l
    img = cv2.resize(img, (0,0), fx = rate, fy = rate, interpolation=cv2.INTER_LINEAR)
    offsetx = random.randint(0, img.shape[0]-256)
    offsety = random.randint(0, img.shape[1]-256)
    img = img[offsetx:offsetx+256, offsety:offsety+256,:]

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            segs = line.strip().split(",")
            box = list(map(lambda x:float(x)*rate, segs[:8]))
            for i in range(len(box)):
                if i&1==0:
                    box[i]-=offsety
                else:
                    box[i]-=offsetx
            boxes.append(box)

    label_img = np.zeros([img.shape[0],img.shape[1],1])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for box in boxes:
                if point_in_box(x, y, box):
                    label_img[x, y, 0]=1
                    break
    return img, label_img.astype(np.uint8)

# def random_crop_resize(img, label_img, shape=[256, 256]):
#     l = min(img.shape[0], img.shape[1])
#     offsetx = random.randint(0, img.shape[0]-l)
#     offsety = random.randint(0, img.shape[1]-l)

#     img, label_img = img[offsetx:offsetx+l, offsety:offsety+l,:], label_img[offsetx:offsetx+l, offsety:offsety+l,:]

#     img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
#     label_img = cv2.resize(label_img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
#     return img, label_img