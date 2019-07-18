import os
import re
import cv2 as cv
import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util
import matplotlib.pyplot as plt
from glob import glob
from ast import literal_eval
import json
import sys

class grape_data:
    
    def __init__(self, filepath):

        #filenames
        self.filename = filepath.split('/')[-1]
        self.img_filename = self.filename + '.jpg'
        self.data_filename = self.filename + '.json'
        self.path = filepath.replace(self.filename,'')

        #get label data
        with open(self.path + self.data_filename) as json_file: 
            self.data = json.load(json_file)

        ##########################
        # I left this as is for now but the json file also stores
        # image height, width, and path for example - 
        # "imagePath": "18.jpg",
        # "imageData": null,
        # "imageHeight": 2412,
        # "imageWidth": 3016
        # it does not store channels but if choose to store imageData
        # channels can be extrapolated from that
        ##########################
        #image data
        img = cv.imread(self.path + self.img_filename)
        self.rows,self.cols,self.channels = img.shape        
        
        #get detections
        self.detections = self.data['shapes']

        #get class info
        self.classes = [d['label'] for d in self.detections]
        self.all_classes = ['zero_class', *set(self.classes)]
        self.num_classes = len(self.all_classes)
        class_to_ind = dict(zip(self.all_classes,range(self.num_classes)))

        #init bbox dimensions
        num_detections = len(self.detections)
        self.gt_classes_ind = np.zeros(num_detections, dtype=np.uint8)
        self.boxes = np.zeros(num_detections,dtype=np.uint16)
        self.xmins = np.zeros(num_detections,dtype=np.float32)
        self.xmaxs = np.zeros(num_detections,dtype=np.float32)
        self.ymins = np.zeros(num_detections,dtype=np.float32)
        self.ymaxs = np.zeros(num_detections,dtype=np.float32)

        for ix, ob in enumerate(self.detections):
            coor = ob['points']

            self.xmins[ix] = float(coor[0][0])
            self.ymins[ix] = float(coor[1][0])
            self.xmaxs[ix] = float(coor[0][0])
            self.ymaxs[ix] = float(coor[1][0])

            self.gt_classes_ind[ix] = class_to_ind[self.classes[ix]]   

class create_tf_example(grape_data):

    def __init__(self, filepath):

        grape_data.__init__(self,filepath)

        with tf.gfile.GFile(self.img_filename, 'rb') as fid:
            encoded_image = fid.read()

        for ix, _ in enumerate(self.xmins):
            self.xmins[ix] = self.xmins[ix] / self.cols
            self.xmaxs[ix] = self.xmaxs[ix] / self.cols
            self.ymins[ix] = self.ymins[ix] / self.rows
            self.ymaxs[ix] = self.ymaxs[ix] / self.rows

        self.img_format = b'jpg'
        self.img_filename = self.img_filename.encode('utf-8')

        for i in range(0,len(self.classes)):
           self.classes[i] = self.classes[i].encode('utf-8')

        self.tf_example = tf.train.Example(features = tf.train.Features(feature={
            'image/height':dataset_util.int64_feature(self.rows),
            'image/width':dataset_util.int64_feature(self.cols),
            'image/filename': dataset_util.bytes_feature(self.img_file),
            'image/source_id':dataset_util.bytes_feature(self.img_file),
            'image/encoded':dataset_util.bytes_feature(encoded_image),
            'image/format':dataset_util.bytes_feature(self.img_format),
            'image/object/bbox/xmin':dataset_util.float_list_feature(self.xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(self.xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(self.ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(self.ymaxs),
            'image/object/class/text':dataset_util.bytes_list_feature(self.classes),
            'image/object/class/label': dataset_util.int64_list_feature(self.gt_classes_ind),
        }))

def main(args):

    if(len(args)>1): data_path = args[1]
    else: print("no path given for images!")#data_path = '/home/nathaniel/bloomfield/labeled_images'
    if(len(args)>2): out_path = args[2]
    else: print("no path given for output!")#out_path = '/home/nathaniel/bloomfield/models/research/object_detection/data/grape.record'

    files = list(set(f.split(".")[0] for f in glob(data_path + '/*')))

    writer = tf.python_io.TFRecordWriter(out_path)
    for f in files:

        temp_check = f.split('/')[-1]
        if(temp_check =='gamma_temp'): continue
        elif(temp_check =='edge_temp'): continue
        else:
            img_gdata = grape_data(f)
            ex = create_tf_example(f)
            writer.write(ex.tf_example.SerializeToString())

    writer.close()

if __name__=='__main__':
    main(sys.argv)