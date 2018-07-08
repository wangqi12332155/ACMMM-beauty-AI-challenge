# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:38:36 2018

@author: wangqi
"""
import os
import cv2
import keras.callbacks
from keras import backend as K    
from keras.utils import plot_model
import platform
from keras.utils import np_utils, generic_utils
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.models import load_model,Model 
import numpy as np

if platform.system() == "Windows":
    dataPath = r"G:/perfect_500K_datasets/"
else:
    dataPath = '/media/leiliang/my hard disk/perfect_500K_datasets/'
    
file_class=["0001_180889/","0002_112119/","0003_108112/","0004_57203/","0005_42338/","0006_8444/","0007_7572/","0008_4050/"]

def dataGen_mask(dirName, fileList,file_class, lSize=(224, 224), batchsize = 32):
    while True:
#        data_num=len(fileList)
#        loopcount=data_num//batchsize
        label=[]
        for fileName in fileList:
            if not fileName.strip()=="":
                path=dirName+file_class[int(fileName[1:5])-1]+ fileName
                srcImg = cv2.imread(path) 
                #cv2.imshow('src',srcImg)
                try:
                    if srcImg.shape[1] != lSize[0] or srcImg.shape[0] != lSize[1]:
                        srcImg = cv2.resize(srcImg, lSize)
                    srcImg = srcImg.astype(np.float32) / np.max(srcImg)
                    #label.append(int(fileName[4])-1)
                    label=int(fileName[4])-1
                    label = np_utils.to_categorical(label, 8)
                    yield (np.expand_dims(srcImg, axis=0), label)
                except:
                    print(fileName)
if __name__ == "__main__":
#模型构建
    classes=8
    #base_model = VGG16(include_top=False,weights='imagenet')
    pre_model =load_model('fcn_model.14-1.533272.hdf5')
    #plot_model(pre_model, to_file='model_vgg1.png')
    base_model = Model(inputs=pre_model.input, outputs=pre_model.get_layer('dense_2').output)
    x=base_model.output
    #model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
   # train_model= Model(inputs=base_model.input, outputs=base_model.get_layer(').output)
    # add a global spatial average pooling layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x=Dropout(0.5)(x)"""
    predictions = Dense(29, activation='softmax',name='dense_3')(x)
    train_model = Model(inputs=base_model.input, outputs=predictions)
    plot_model(train_model, to_file='model3.png')
    
    img_size = 224
    nb_class = 4
    epoch = 20
    valSize = 1
    batch_size=64
    
    ADAM=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    train_model.compile(loss='categorical_crossentropy', optimizer=ADAM,metrics=['accuracy'])
    
    
    tb_cb = keras.callbacks.TensorBoard(log_dir='./trainLog', write_graph=False, write_images=True)#, histogram_freq=0)
    mc_cb = keras.callbacks.ModelCheckpoint(
        'fcn_model.{epoch:d}-{val_loss:f}.hdf5',
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto', period=5)
    stop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    cbks = [tb_cb, mc_cb,stop]
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=360,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.2,
            zoom_range=0.5,
            horizontal_flip=True,
            fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        '/media/leiliang/文档/ACMMM2018_datasets/datasets/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        '/media/leiliang/文档/ACMMM2018_datasets/datasets/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    train_numbers = train_generator.samples
    val_numbers= validation_generator.samples
    train_model.fit_generator(
            train_generator,
            steps_per_epoch=train_numbers//batch_size,
            epochs=200,
            validation_data=validation_generator,
            validation_steps=val_numbers//batch_size,
            callbacks = cbks)
'''    
    train_model.fit_generator(dataGen_mask(dataPath, trainList,file_class),
                            steps_per_epoch = len(trainList),
                            validation_steps = len(testList),
                            validation_data= dataGen_mask(dataPath, testList,file_class),
                            initial_epoch = 0,
                            epochs=100,
                            callbacks = cbks)
    # train_model.save_weights("weights/temp", overwrite=True)
    print("Saved weights")

'''