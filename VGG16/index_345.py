# -*- coding: utf-8 -*-

import os
import numpy as np
from lshash.lshash import LSHash
from extract_cnn_123345 import VGGNet
#from extract_cnn_345 import VGGNet
#from extract_cnn_345 import extract_feat_3_yalers
import csv
import platform
import datetime,time

if platform.system() == "Windows":
    dataPath = r"J:/perfect_500K_datasets/"
else:
    dataPath = '/media/leiliang/新加卷/perfect_500K_datasets/'
file_class=["0001_180889/","0002_112119/","0003_108112/","0004_57203/","0005_42338/","0006_8444/","0007_7572/","0008_4050/"]

'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
if __name__ == "__main__":

    paths = dataPath+"acmmm/vgg_retrain/all_picture.txt"
    #paths = dataPath+"acmmm//top7_acc/name_pic.txt"
    datas = open(paths, 'r').read().split('\n')
    
    print ("--------------------------------------------------")
    print ("         feature extraction starts")
    print ("--------------------------------------------------")
    #feature = open('feature_52_1728','w+', newline='') # 设置newline，否则两行之间会空一行
    feature_norm= open('feature_1728_norm.csv','w+', newline='') # 设置newline，否则两行之间会空一行
    feature_mean_norm = open('feature_1728_mean_norm.csv','w+', newline='') # 设置newline，否则两行之间会空一行
    #feature_org = open('feature_org.csv','a+', newline='')
    query_csv=open('query_1728.csv','w+', newline='')
    
    #writer = csv.writer(feature)
    writer1 = csv.writer(feature_norm)
    writer2 = csv.writer(feature_mean_norm)
    #writer_org = csv.writer(feature_org)
    query_file = csv.writer(query_csv)    
	
    #model_org = VGGNet_org()
    model = VGGNet()
    file_order=[]
    lost=[]
    t=0
    starttime1 = datetime.datetime.now()
    for i, img_path in enumerate(datas):
        if not img_path.strip()=="":
            try:
                #img_path=img_path.split(",")[1]
                path_picture=dataPath+file_class[int(img_path[4])-1]+img_path
                #norm_feat=model.extract_feat(path_picture)
                feature_data,feature_norm_data,feature_mean_norm_data=model.extract_feat(path_picture)
                #print("start!")
                #norm_feat_org=model_org.extract_feat(path_picture)
                #norm_feat=extract_feat_3_yalers(path_picture)
                file_order=[t,img_path]
                print ("extracting feature num  from image No. %d , %d images in total" %((t), len(datas)),feature_norm_data.shape)
                t+=1
                #writer.writerow(feature_data)
                writer1.writerow(feature_norm_data)
                writer2.writerow(feature_mean_norm_data)
                #writer_org.writerow(norm_feat_org)
                query_file.writerow(file_order)
                if t%520==0:
                    starttime2 = datetime.datetime.now()
                    time_using=(starttime2 - starttime1).seconds
                    for k in range(10):
                        print(time_using)
                    starttime1=starttime2
            except:
                lost.append(img_path) 
    #feature.close()
    feature_norm.close()
    feature_mean_norm.close()
    #feature_org.close()
    query_csv.close()
    '''
    # directory for storing extracted features
    queryVec = model.extract_feat("./database/001_accordion_image_0001.jpg")
    output_query=lsh.query(queryVec,num_results=3, distance_func="euclidean")# ("hamming", "euclidean", "true_euclidean","centred_euclidean", "cosine", "l1norm","euclidean"  
    print(output_query[0][0][1])
    print(output_query[0][1])
    

m = len(data)
for i in range(m):
    writer.writerow(data[i])
'''
 
 
  
    
