# -*- coding: utf-8 -*-

#from extract_cnn_345 import VGGNet
import csv
import platform
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
import numpy as np
from numpy import *
from itertools import chain
import os
import datetime,time
from extract_cnn_123345 import VGGNet
from test_helper import readData,euclideanDistance

if platform.system() == "Windows":
    dataPath = r"J:/perfect_500K_datasets/"
else:
    dataPath =  '/media/leiliang/新加卷/perfect_500K_datasets/'
    
file_class=["0001_180889/","0002_112119/","0003_108112/","0004_57203/","0005_42338/","0006_8444/","0007_7572/","0008_4050/"]
#picture_name="n0001_0000001.jpg"
#class_index=int(picture_name[4])-1
"""
K_pram=[2.5,0.1,0.1,3.5,1.0,3.0] top7#ACC=0.24 ,map=0.218
K_pram=[2.0,0.1,2.0,3.0,1.0,3.0] top1000 #ACC=0.37
"""
k0=2
k1=0.1
k2=2
k3=3
k4=1
k5=3
top_n=5000
datasets_path="feature_1728_mean_norm.csv"#feature_1728_mean_norm
query_datasets_path="query_1728.csv"
if __name__ == "__main__":
    #C = pow(2, 32) - 5
    starttime1 = datetime.datetime.now()
    """
    dataSet_1 = []
    dataSet_2 = []
    with open(datasets_path, "r") as csvFile:
        reader = csv.reader(csvFile)
        for i,line in enumerate(reader):
            if i<=250000:
                dataSet_1.append([float(item) for item in line])
                if i==250000:
                    A_1=np.array(dataSet_1)
                    del(dataSet_1)
            else :
                dataSet_2.append([float(item) for item in line])
                if i==520638:
                    A_2=np.array(dataSet_2)
                    del(dataSet_2)
        A=np.concatenate((A_1,A_2),axis=0)
        del(A_1)
        del(A_2)
   
    dataSet = readData(datasets_path)
    A=np.array(dataSet)
    del(dataSet[:])
    """


    query_all=[]
    model = VGGNet()
    image_list=os.listdir(dataPath+"acmmm/Locality-sensitive-hashing-master/sample/testset_v1")#[1000:2000]#sample/sample_pic")
    image_list.sort()
    for image in image_list:
        #queryVec = model.extract_feat(dataPath+file_class[class_index]+picture_name)
        image_path=dataPath+"acmmm/Locality-sensitive-hashing-master/sample/testset_v1/"+image
        queryVec,queryVec_norm,queryVec_mean_norm = model.extract_feat(image_path)
        #queryVec=np.hstack((np.hstack((k1*queryVec[0:256],k2*queryVec[256:768])),k3*queryVec[768:1280]))
        query_all.append(queryVec_mean_norm)
    starttime3 = datetime.datetime.now()
    
    #A=np.array(dataSet)

    B=np.array(query_all)
    del(query_all[:])
    f0=np.dot(B[:,0:64],A.T[0:64])
    f1=np.dot(B[:,64:192],A.T[64:192])    
    score_01=k0*f0+k1*f1
    del(f0)
    del(f1)
    f2=np.dot(B[:,192:448],A.T[192:448])
    score_01=score_01+k2*f2
    del(f2)    
    f3=np.dot(B[:,448:704],A.T[448:704])
    score_01=score_01+k3*f3
    del(f3)    
    f4=np.dot(B[:,704:1216],A.T[704:1216])
    score_01=score_01+k4*f4    
    del(f4)
    f5=np.dot(B[:,1216:1728],A.T[1216:1728])  
    score_01=score_01+k5*f5
    del(f5)
    
    del(B)

    #scores = np.dot(B, A.T)
    k_1728=1
    k_2048=60
    scores_merge=k_1728*score_01+k_2048*scores_100
    rank_ID = np.argsort(-scores_merge,axis=1)
    #del(scores)   
    #rank_ID = np.argsort(scores)[::-1]
    #rank_score = scores[rank_ID]
    top_seven=[]   
    for rank in rank_ID:
        top_seven.append(rank[0:7])        
    del(rank_ID)   
    
    top_seven_name=[]
    all_index=list(chain.from_iterable(top_seven))
    starttime4 = datetime.datetime.now()
    print("top7 list is ok! using time is",(starttime4 - starttime3).seconds)
    
    name_query=[]  
    with open(query_datasets_path,'r') as csvfile:
       reader_list = csv.reader(csvfile)                    
       for csv_name in reader_list:
           name_query.append(csv_name)
    
    for s in all_index:
        top_seven_name.append(name_query[int(s)][1])        
    #del(name_query[:])                
    """
        for j,rows in enumerate(name_query):
            if(j==int(s)):
                top_seven_name.append(rows[1])
                break
    """   
    name_seven=[]
    result=[]               
    for i,name in enumerate(top_seven_name):
        name_seven.append(name)
        if (i+1)%7==0:
            result.append(name_seven)
            name_seven=[]
    starttime5 = datetime.datetime.now()
    print("result is ok! using time is",(starttime5 - starttime4).seconds)
    
    #test val datasets 100 pictures
    val_list=[]
    with open("val.csv",'r') as csv_file:
       reader = csv.reader(csv_file)                    
       for val in reader:
            val_list.append(val)

    ACC=0
    SCORE=0
    for k,val_name in enumerate(image_list):
        val_rows=val_list[int(val_name[1:7])]
        for j,pre_name in enumerate(result[k]):
                if val_rows[1]==pre_name[:-4]:
                    ACC=ACC+1
                    SCORE=SCORE+(7-j)
                else :
                    try:
                        if val_rows[2]==pre_name[:-4]:
                            ACC=ACC+1
                            SCORE=SCORE+(7-j)
                        else:
                            try:
                                if val_rows[3]==pre_name[:-4]:
                                    ACC=ACC+1
                                    SCORE=SCORE+(7-j)
                                else:
                                    ACC=ACC
                                    SCORE=SCORE
                            except:
                                ACC=ACC
                                SCORE=SCORE                                
                    except:                      
                        ACC=ACC
                        SCORE=SCORE
    print("acc_top7:",ACC/len(image_list),"map:",SCORE/7/len(image_list),"totle using time is",(starttime5 - starttime1).seconds)
#    pram=[k3,k4,k5,ACC/len(image_list),SCORE/7/len(image_list)]
#    list_pram_k.append(pram)
#    list_pram_k.sort(key=lambda x:x[4],reverse=True)
    #kkk=[str(kk) for kk in list_pram_k]
    #open("./pram_k.txt", 'w+').write('\n'.join(kkk))
"""       #test our datasets 10000 pictures        
                    name_List = open('name_pic.txt', 'r').read().split('\n')
                    val_list=[name1.split(',') for name1 in name_List]
                    ACC=0
                    SCORE=0 
                    for k,val_name in enumerate(image_list):
                        ture_num=int(val_name.split('_')[1])
                        ture_name=val_list[ture_num][1]
                        for j,pre_name in enumerate(result[k]):
                            if ture_name==pre_name:
                                ACC=ACC+1
                                SCORE=SCORE+(7-j)
                            else :
                                ACC=ACC
                                SCORE=SCORE
                    print("acc_top7:",ACC/len(image_list),"map:",SCORE/7/len(image_list),"totle using time is",(starttime5 - starttime1).seconds)
                    pram=[k3,k4,k5,ACC/len(image_list),SCORE/7/len(image_list)]
                    list_pram_k.append(pram)
    list_pram_k.sort(key=lambda x:x[4],reverse=False)
"""    
"""
    result_csv = open('result.csv','w+', newline='') # 设置newline，否则两行之间会空一行
    writer = csv.writer(result_csv)
    
    #result=[["n1.jpg","n2.jpg","n3.jpg","n4.jpg","n5.jpg","n6.jpg","n7.jpg"],["n11.jpg","n12.jpg","n13.jpg","n14.jpg","n15.jpg","n16.jpg","n17.jpg"]]
    result_images=""
    picture_list=os.listdir(dataPath+"acmmm/Locality-sensitive-hashing-master/sample/testset_v1")
    
    for i,name in enumerate(picture_list):
        test_image="<Test Image ID "+name+"#"+str(i+1)
        for x in range(7):
    	result_img=", <Training Image ID "+result[i][x]+"#"+str(i+1)+"_"+str(x+1)
    	result_images=result_images+result_img
        writer.writerow([test_image+result_images])
        result_images=""
    result_csv.close()
    print("result.csv is ok!")


    for k in range(10):
        indexes = e2LSH.nn_search(dataSet, query, k=20, L=5, r=1, tableSize=20)
        for i,index in enumerate(indexes):
            query_list_x=[euclideanDistance(dataSet[index], query),index]
            query_list.append(query_list_x)
            query_list.sort(key=lambda x:x[0],reverse=False)
           # print(i)
        query_totle.append(query_list[0])

    query_totle.sort(key=lambda x:x[0],reverse=False)        
    index_num=query_totle[0][1]        
    with open('query_csv.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for j,rows in enumerate(reader):
            if j== int(index_num):
                out_picture_name=rows[1]
                path=dataPath+file_class[int(out_picture_name[4])-1]+out_picture_name
                print(out_picture_name,query_totle[0])
                #srcImg=Image.open(path) 
               # srcImg.show()
               
picture_name="n0007_0000601.jpg"
class_index=int(picture_name[4])-1              
#queryVec = model.extract_feat(dataPath+file_class[class_index]+picture_name) 
queryVec = model.extract_feat(dataPath+"acmmm/Locality-sensitive-hashing-master/result/gen_0_2464.jpeg")
scores = np.dot(queryVec, A.T)
rank_ID = np.argsort(scores)[::-1]
       
"""
