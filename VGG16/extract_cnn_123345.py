# -*- coding: utf-8 -*-


import numpy as np
from numpy import linalg as LA
from keras import backend as K  
import platform
from sklearn import preprocessing
from keras.utils.conv_utils import convert_kernel
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from crow import run_feature_processing_pipeline, apply_crow_aggregation, apply_ucrow_aggregation, normalize
from keras.models import load_model,Model
import cv2


if platform.system() == "Windows":
    dataPath = r"J:/perfect_500K_datasets/acmmm/vgg_retrain"
else:
    dataPath = "/media/leiliang/新加卷//perfect_500K_datasets/acmmm/vgg_retrain"

class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.pre_model =load_model(dataPath+'/5.31_fcn_model.49-1.779945.hdf5')
        #self.pre_model =load_model(dataPath+'/vgg_all_datas_model.9-0.836696.hdf5')
        #plot_model(pre_model, to_file='model_vgg1.png')
        self.base_model_block5_conv3 = Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block5_conv3').output) 
        self.base_model_block5_conv1= Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block5_conv1').output) 
        self.base_model_block4_conv3 = Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block4_conv3').output)
        self.base_model_block4_conv1 = Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block4_conv1').output)        
        self.base_model_block3_conv3 = Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block3_conv3').output)
        self.base_model_block1_conv2= Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block1_conv2').output)
        self.base_model_block2_conv2= Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block2_conv2').output)
        self.base_model_block3_conv1 = Model(inputs=self.pre_model.input, outputs=self.pre_model.get_layer('block3_conv1').output)
		
        x_block5_conv3=self.base_model_block5_conv3.output
        x_block5_conv1=self.base_model_block5_conv1.output
        x_block4_conv3=self.base_model_block4_conv3.output
        x_block4_conv1=self.base_model_block4_conv1.output        
        x_block3_conv3=self.base_model_block3_conv3.output
        x_block3_conv1=self.base_model_block3_conv1.output
        x_block2_conv2=self.base_model_block2_conv2.output
        x_block1_conv2=self.base_model_block1_conv2.output
		
        self.model_vgg_block5_conv3=Model(inputs=self.pre_model.input,outputs=x_block5_conv3)
        self.model_vgg_block5_conv1=Model(inputs=self.pre_model.input,outputs=x_block5_conv1)        
        self.model_vgg_block4_conv3=Model(inputs=self.pre_model.input,outputs=x_block4_conv3)
        self.model_vgg_block4_conv1=Model(inputs=self.pre_model.input,outputs=x_block4_conv1)
        self.model_vgg_block3_conv3=Model(inputs=self.pre_model.input,outputs=x_block3_conv3)
        self.model_vgg_block3_conv1=Model(inputs=self.pre_model.input,outputs=x_block3_conv1)
        self.model_vgg_block2_conv2=Model(inputs=self.pre_model.input,outputs=x_block2_conv2)
        self.model_vgg_block1_conv2=Model(inputs=self.pre_model.input,outputs=x_block1_conv2)
		
        self.model_vgg_block5_conv3.predict(np.zeros((1, 224, 224 , 3)))
        self.model_vgg_block5_conv1.predict(np.zeros((1, 224, 224 , 3)))
        self.model_vgg_block4_conv3.predict(np.zeros((1, 224, 224 , 3)))
        self.model_vgg_block4_conv1.predict(np.zeros((1, 224, 224 , 3)))
        self.model_vgg_block3_conv3.predict(np.zeros((1, 224, 224 , 3)))
        self.model_vgg_block3_conv1.predict(np.zeros((1, 224, 224 , 3)))
        self.model_vgg_block2_conv2.predict(np.zeros((1, 224, 224 , 3)))
        self.model_vgg_block1_conv2.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        """
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        """
        img = image.load_img(img_path)
        img = image.img_to_array(img)
        h, w, c = img.shape
        resize_h=h
        resize_w=w
        minlength=min(h,w)
        if minlength>224:
            beta=minlength/224
            resize_h=int(h/beta)
            resize_w=int(w/beta)           
        img=cv2.resize(img,(resize_h,resize_w))

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        feat_0 = self.model_vgg_block1_conv2.predict(img)
        feat_0=convert_kernel(feat_0[0].T)
        feature_crow_0=apply_crow_aggregation(feat_0)
        feature_norm_0=normalize(feature_crow_0)
        feature_mean_norm_0=normalize(preprocessing.scale(feature_crow_0,axis=0, with_mean=True, with_std=False, copy=True))

        feat_1 = self.model_vgg_block2_conv2.predict(img)
        feat_1=convert_kernel(feat_1[0].T)
        feature_crow_1=apply_crow_aggregation(feat_1)
        feature_norm_1=normalize(feature_crow_1)
        feature_mean_norm_1=normalize(preprocessing.scale(feature_crow_1,axis=0, with_mean=True, with_std=False, copy=True))

        feat_2= self.model_vgg_block3_conv1.predict(img)
        feat_2=convert_kernel(feat_2[0].T)
        feature_crow_2=apply_crow_aggregation(feat_2)
        feature_norm_2=normalize(feature_crow_2)
        feature_mean_norm_2=normalize(preprocessing.scale(feature_crow_2,axis=0, with_mean=True, with_std=False, copy=True))

        feature_448=np.hstack((np.hstack((feature_crow_0.T,feature_crow_1.T)),feature_crow_2.T))
        feature_448_norm=np.hstack((np.hstack((feature_norm_0.T,feature_norm_1.T)),feature_norm_2.T))
        feature_448_mean_norm=np.hstack((np.hstack((feature_mean_norm_0.T,feature_mean_norm_1.T)),feature_mean_norm_2.T))
        
        feat_3 = self.model_vgg_block3_conv3.predict(img)
        feat_3=convert_kernel(feat_3[0].T)
        feature_crow_3=apply_crow_aggregation(feat_3)
        feature_norm_3=normalize(feature_crow_3)
        feature_mean_norm_3=normalize(preprocessing.scale(feature_crow_3,axis=0, with_mean=True, with_std=False, copy=True))

        feat_4 = self.model_vgg_block4_conv3.predict(img)
        feat_4=convert_kernel(feat_4[0].T)
        feature_crow_4=apply_crow_aggregation(feat_4)
        feature_norm_4=normalize(feature_crow_4)
        feature_mean_norm_4=normalize(preprocessing.scale(feature_crow_4,axis=0, with_mean=True, with_std=False, copy=True))

        feat_5= self.model_vgg_block5_conv3.predict(img)
        feat_5=convert_kernel(feat_5[0].T)
        feature_crow_5=apply_crow_aggregation(feat_5)
        feature_norm_5=normalize(feature_crow_5)
        feature_mean_norm_5=normalize(preprocessing.scale(feature_crow_5,axis=0, with_mean=True, with_std=False, copy=True))

        feature_1280=np.hstack((np.hstack((feature_crow_3.T,feature_crow_4.T)),feature_crow_5.T))
        feature_1280_norm=np.hstack((np.hstack((feature_norm_3.T,feature_norm_4.T)),feature_norm_5.T))
        feature_1280_mean_norm=np.hstack((np.hstack((feature_mean_norm_3.T,feature_mean_norm_4.T)),feature_mean_norm_5.T))
        #print(feature_norm.shape)
        #feature,pca_prams=run_feature_processing_pipeline(feature_norm)
        return np.hstack((feature_448.T,feature_1280.T)),np.hstack((feature_448_norm.T,feature_1280_norm.T)),np.hstack((feature_448_mean_norm.T,feature_1280_mean_norm.T))
    
def extract_feat_3_yalers(img_path):
    model =load_model(dataPath+'/f1cn_model.49-1.613893.hdf5')
    layer_1 = K.function([model.layers[0].input], [model.layers[7].output])    
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)    
    f1 = layer_1([img])[0]
    feat_5=convert_kernel(f1[0].T)
    feature_crow_5=apply_crow_aggregation(feat_5)
    feature_norm_5=normalize(feature_crow_5)
    return np.sqrt(feature_norm_5)
    
    
    
    
    
    
    
        