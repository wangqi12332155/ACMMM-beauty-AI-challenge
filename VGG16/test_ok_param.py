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
K_pram=[1.0, 3.0, 5.0, 1.0, 7.0, 1.0]#calc top5000
K_pram_100=[1.0, 2.0, 0.1, 0.1, 1.0, 1.0] #calc top100
K_pram_7=[2, 0.1, 0.1, 4.5, 1.0, 3.0]  #calc top7
'''
#zuhe
K_pram=[1.0, 2.0, 0.1, 0.1, 1.0, 1.0] top100 #ACC=0.32,map=0.281
K_pram=[2, 0.1, 0.1, 4.5, 1.0, 3.0] top7#ACC=0.24 ,map=0.221
'''

"""
K_pram=[2.5, 0.1, 0.1, 3.5, 1.0, 3.0] top7#ACC=0.24 ,map=0.218
K_pram=[2.0, 2.0, 1.0, 3.0, 1.0, 1.0] top100 #ACC=0.3,map=0.270
K_pram=[1.0, 3.0, 1.0, 1.0, 1.0, 1.0] top500 #ACC=0.36,map=0.316
K_pram=[2.0, 0.1, 2.0, 3.0, 1.0, 3.0] top1000 #ACC=0.37
K_pram=[1.0, 3.0, 5.0, 1.0, 7.0, 1.0] top5000 #ACC=0.59 ,map=0.482
"""
k0=K_pram[0]
k1=K_pram[1]
k2=K_pram[2]
k3=K_pram[3]
k4=K_pram[4]
k5=K_pram[5]
k0_100=K_pram_100[0]
k1_100=K_pram_100[1]
k2_100=K_pram_100[2]
k3_100=K_pram_100[3]
k4_100=K_pram_100[4]
k5_100=K_pram_100[5]
k0_7=K_pram_7[0]
k1_7=K_pram_7[1]
k2_7=K_pram_7[2]
k3_7=K_pram_7[3]
k4_7=K_pram_7[4]
k5_7=K_pram_7[5]

top_n=5000
top_n_100=100
top_n_7=7
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
    name_query=[]  
    query_all=[]
    with open(query_datasets_path,'r') as csvfile:
       reader_list = csv.reader(csvfile)                    
       for csv_name in reader_list:
           name_query.append(csv_name)
    starttime2 = datetime.datetime.now()
    print("reader datas ok! using time is",(starttime2 - starttime1).seconds)
    
    model = VGGNet()
    image_list_1000=os.listdir(dataPath+"acmmm/Locality-sensitive-hashing-master/sample/testset_v1")#[1000:2000]#testset_v1/sample_pic")
    image_list_1000.sort()
    image_list=image_list_1000[500:]
    for image in image_list:
        #queryVec = model.extract_feat(dataPath+file_class[class_index]+picture_name)
        image_path=dataPath+"acmmm/Locality-sensitive-hashing-master/sample/testset_v1/"+image
        queryVec,queryVec_norm,queryVec_mean_norm = model.extract_feat(image_path)
        #queryVec=np.hstack((np.hstack((k1*queryVec[0:256],k2*queryVec[256:768])),k3*queryVec[768:1280]))
        query_all.append(queryVec_mean_norm)
    starttime3 = datetime.datetime.now()
    print("feature_extract is ok! using time is",(starttime3 - starttime2).seconds)
    #A=np.array(dataSet)
    B=np.array(query_all)
    del(query_all[:])
    f0=np.dot(B[:,0:64],A.T[0:64])
    f1=np.dot(B[:,64:192],A.T[64:192])
    f2=np.dot(B[:,192:448],A.T[192:448])
    f3=np.dot(B[:,448:704],A.T[448:704])
    f4=np.dot(B[:,704:1216],A.T[704:1216])
    f5=np.dot(B[:,1216:1728],A.T[1216:1728])                
    #del(query_all[:])
    scores=k0*f0+k1*f1+k2*f2+k3*f3+k4*f4+k5*f5
    del(f0)
    del(f1)
    del(f2)
    del(f3)
    del(f4)
    del(f5)
    #scores = np.dot(B, A.T)
    rank_ID = np.argsort(-scores,axis=1)
    del(scores)   
    #rank_ID = np.argsort(scores)[::-1]
    #rank_score = scores[rank_ID]
    top_seven=[]   
    for rank in rank_ID:
        top_seven.append(rank[0:top_n])        
    #del(rank_ID)   
    top_seven_name=[]
    all_index=list(chain.from_iterable(top_seven))
    starttime4 = datetime.datetime.now()
    print("top7 list is ok! using time is",(starttime4 - starttime3).seconds)
    
    for s in all_index:
        top_seven_name.append(name_query[int(s)][1])        
    #del(name_query[:])                
    name_seven=[]
    result=[]               
    for i,name in enumerate(top_seven_name):
        name_seven.append(name)
        if (i+1)%top_n==0:
            result.append(name_seven)
            name_seven=[]
    starttime5 = datetime.datetime.now()
    print("result is ok! using time is",(starttime5 - starttime4).seconds)
    
    result_csv_5000 = open('result_5000.csv','w+', newline='') # 设置newline，否则两行之间会空一行
    writer_5000 = csv.writer(result_csv_5000)
    result_images=[]
    for i,name in enumerate(image_list):
        result_images.append(name[:-4])
        for x in range(5000):
            result_img=result[i][x]
            result_images.append(result_img[:-4])
        writer_5000.writerow(result_images)
        result_images=[]
    result_csv_5000.close()
    print("result_5000.csv is ok!")
    """
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
                    SCORE=SCORE+(top_n-j)
                else :
                    try:
                        if val_rows[2]==pre_name[:-4]:
                            ACC=ACC+1
                            SCORE=SCORE+(top_n-j)
                        else:
                            try:
                                if val_rows[3]==pre_name[:-4]:
                                    ACC=ACC+1
                                    SCORE=SCORE+(top_n-j)
                                else:
                                    ACC=ACC
                                    SCORE=SCORE
                            except:
                                ACC=ACC
                                SCORE=SCORE                                
                    except:                      
                        ACC=ACC
                        SCORE=SCORE
    print("acc_top7:",ACC/len(image_list),"map:",SCORE/top_n/len(image_list),"totle using time is",(starttime5 - starttime1).seconds)
    """
    #A_5000=np.array([])
    top_seven_100_totle=[]
    for i,indexes in enumerate(top_seven):
        A_5000_list=[]
        for k,index in enumerate(indexes):
            A_5000_list.append(A[index])
        A_5000=np.array(A_5000_list)
        f0_5000=np.dot(B[i][0:64],A_5000.T[0:64])
        f1_5000=np.dot(B[i][64:192],A_5000.T[64:192])
        f2_5000=np.dot(B[i][192:448],A_5000.T[192:448])
        f3_5000=np.dot(B[i][448:704],A_5000.T[448:704])
        f4_5000=np.dot(B[i][704:1216],A_5000.T[704:1216])
        f5_5000=np.dot(B[i][1216:1728],A_5000.T[1216:1728])                
        #del(query_all[:])
        scores_5000=k0_100*f0_5000+k1_100*f1_5000+k2_100*f2_5000+k3_100*f3_5000+k4_100*f4_5000+k5_100*f5_5000
        del(f0_5000)
        del(f1_5000)
        del(f2_5000)
        del(f3_5000)
        del(f4_5000)
        del(f5_5000)
        #scores = np.dot(B, A.T)
        rank_ID_5000 = np.argsort(-scores_5000,axis=0) 
        top_seven_100_totle.append(rank_ID_5000[0:top_n_100])
     
    top_seven_100=[]
    for i,indexes in enumerate(top_seven_100_totle):
        top_seven_a=[]
        for index in indexes:
            top_seven_a.append(top_seven[i][index])
        top_seven_100.append(top_seven_a)
            
    top_seven_name=[]
    all_index=list(chain.from_iterable(top_seven_100))
    starttime4 = datetime.datetime.now()
    print("top7 list is ok! using time is",(starttime4 - starttime3).seconds)
    
    for s in all_index:
        top_seven_name.append(name_query[int(s)][1])        
    #del(name_query[:])                
    name_seven=[]
    result=[]               
    for i,name in enumerate(top_seven_name):
        name_seven.append(name)
        if (i+1)%top_n_100==0:
            result.append(name_seven)
            name_seven=[]
    starttime5 = datetime.datetime.now()
    print("result is ok! using time is",(starttime5 - starttime4).seconds)
    
    result_csv_100 = open('result_best.csv','w+', newline='') # 设置newline，否则两行之间会空一行
    writer_100 = csv.writer(result_csv_100)
    result_images=[]
    for i,name in enumerate(image_list):
        result_images.append(name[:-4])
        for x in range(100):
            result_img=result[i][x]
            result_images.append(result_img[:-4])
        writer_100.writerow(result_images)
        result_images=[]
    result_csv_100.close()
    print("result_100.csv is ok!")
    """    
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
                    SCORE=SCORE+(top_n_100-j)
                else :
                    try:
                        if val_rows[2]==pre_name[:-4]:
                            ACC=ACC+1
                            SCORE=SCORE+(top_n_100-j)
                        else:
                            try:
                                if val_rows[3]==pre_name[:-4]:
                                    ACC=ACC+1
                                    SCORE=SCORE+(top_n_100-j)
                                else:
                                    ACC=ACC
                                    SCORE=SCORE
                            except:
                                ACC=ACC
                                SCORE=SCORE                                
                    except:                      
                        ACC=ACC
                        SCORE=SCORE
    print("acc_top7:",ACC/len(image_list),"map:",SCORE/top_n_100/len(image_list),"totle using time is",(starttime5 - starttime1).seconds)
    """
    top_seven_7_totle=[]
    for i,indexes in enumerate(top_seven_100):
        A_100_list=[]
        for k,index in enumerate(indexes):
            A_100_list.append(A[index])
        A_100=np.array(A_100_list)
        f0_100=np.dot(B[i][0:64],A_100.T[0:64])
        f1_100=np.dot(B[i][64:192],A_100.T[64:192])
        f2_100=np.dot(B[i][192:448],A_100.T[192:448])
        f3_100=np.dot(B[i][448:704],A_100.T[448:704])
        f4_100=np.dot(B[i][704:1216],A_100.T[704:1216])
        f5_100=np.dot(B[i][1216:1728],A_100.T[1216:1728])                
        #del(query_all[:])
        scores_100=k0_7*f0_100+k1_7*f1_100+k2_7*f2_100+k3_7*f3_100+k4_7*f4_100+k5_7*f5_100
        del(f0_100)
        del(f1_100)
        del(f2_100)
        del(f3_100)
        del(f4_100)
        del(f5_100)
        #scores = np.dot(B, A.T)
        rank_ID_100 = np.argsort(-scores_100,axis=0) 
        top_seven_7_totle.append(rank_ID_100[0:top_n_7])
     
    top_seven_7=[]
    for i,indexes in enumerate(top_seven_7_totle):
        top_seven_b=[]
        for index in indexes:
            top_seven_b.append(top_seven_100[i][index])
        top_seven_7.append(top_seven_b)
            
    top_seven_name=[]
    all_index=list(chain.from_iterable(top_seven_7))
    starttime4 = datetime.datetime.now()
    print("top7 list is ok! using time is",(starttime4 - starttime3).seconds)
    
    for s in all_index:
        top_seven_name.append(name_query[int(s)][1])        
    #del(name_query[:])                
    name_seven=[]
    result=[]               
    for i,name in enumerate(top_seven_name):
        name_seven.append(name)
        if (i+1)%top_n_7==0:
            result.append(name_seven)
            name_seven=[]
    starttime5 = datetime.datetime.now()
    print("result is ok! using time is",(starttime5 - starttime4).seconds)
    
    #test val datasets 100 pictures
    """
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
                    SCORE=SCORE+(top_n_7-j)
                else :
                    try:
                        if val_rows[2]==pre_name[:-4]:
                            ACC=ACC+1
                            SCORE=SCORE+(top_n_7-j)
                        else:
                            try:
                                if val_rows[3]==pre_name[:-4]:
                                    ACC=ACC+1
                                    SCORE=SCORE+(top_n_7-j)
                                else:
                                    ACC=ACC
                                    SCORE=SCORE
                            except:
                                ACC=ACC
                                SCORE=SCORE                                
                    except:                      
                        ACC=ACC
                        SCORE=SCORE
    print("acc_top7:",ACC/len(image_list),"map:",SCORE/top_n_7/len(image_list),"totle using time is",(starttime5 - starttime1).seconds)
    """
    result_csv = open('result.csv','w+', newline='') # 设置newline，否则两行之间会空一行
    writer = csv.writer(result_csv)
    result_images=[]
    for i,name in enumerate(image_list):
        result_images.append(name[:-4])
        for x in range(7):
            result_img=result[i][x]
            result_images.append(result_img[:-4])
        writer.writerow(result_images)
        result_images=[]
    result_csv.close()
    print("result.csv is ok!")
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
