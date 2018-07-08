# -*- coding: utf-8 -*-


import numpy as np
from numpy import linalg as LA
#from crow import run_feature_processing_pipeline, apply_crow_aggregation, apply_ucrow_aggregation
from keras.utils.conv_utils import convert_kernel
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
#from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from sklearn import preprocessing

class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = ResNet50(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        #self.model_add_64= Model(inputs=self.model.input, outputs=self.model.get_layer('res5c_branch2c').output)
        self.model.predict(np.zeros((1, 224, 224 , 3)))


    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        #feature_norm=preprocessing.normalize( feat[0])
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat


