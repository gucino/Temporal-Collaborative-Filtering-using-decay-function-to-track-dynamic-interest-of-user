# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:20:54 2020

@author: Tisana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##import data
data_set=pd.read_csv("u_data.csv",header=None).values


##clean data
########################################################################
########################################################################
########################################################################
#compute user rating matrix and timestamp matrix
num_user=943 #user id 1 to 943
num_movie=1682 #movie id 1 to 1682

user_rating_dict={}
#key is user id : value are rating of all movie 
for user_id in range(1,num_user+1):
    user_rating_dict[user_id]=np.array([0]*num_movie)
    
user_timestamp_dict={}
for user_id in range(1,num_user+1):
    user_timestamp_dict[user_id]=np.array([0]*num_movie)

#append rating data set to user rating dict
data_set_list=data_set.tolist()
for each_row in data_set_list:
    user_id=each_row[0]
    mmovie_id=each_row[1]
    rating=each_row[2]
    movie_index=mmovie_id-1
    timestamp=each_row[3]
    #append to dictionary
    user_rating_dict[user_id][movie_index]=rating
    user_timestamp_dict[user_id][movie_index]=timestamp

user_rating_array=[]
for each in user_rating_dict.values():
    user_rating_array.append(each)
user_rating_array=np.array(user_rating_array) #index by user index (user id -1)

#convert rating matrix to user-like matrix
user_like_matrix=[]
for i in range(0,num_user):
    row_list=[]
    for j in range(0,num_movie):
        rating=user_rating_array[i,j]
        if rating>=3:
            row_list.append(1)
        else:
            row_list.append(0)
    user_like_matrix.append(np.array(row_list))
user_like_matrix=np.array(user_like_matrix)

#convert user-like matrix to user-user network
user_user_network=[]
for i in range(0,num_user):
    if i%10==0:
        print(i)
    row_list=[]
    for j in range(0,num_user):
        common_prefered_item=user_like_matrix[i,:]*user_like_matrix[j,:]
        row_list.append(common_prefered_item)
    row_list=np.array(row_list).sum(axis=1)
    user_user_network.append(row_list)
user_user_network=np.array(user_user_network)
#normalization
row_mean=np.mean(user_rating_array,axis=1)
row_mean=row_mean[:,np.newaxis]
user_rating_array=(user_rating_array-row_mean)*(user_rating_array)/(user_rating_array)

for each_row in range(0,num_user):
    for each_column in range(0,num_movie):
        if np.isnan(user_rating_array[each_row,each_column])==True:
            user_rating_array[each_row,each_column]=0

########################################################################
########################################################################
########################################################################
#get timestamp matrix
user_timestamp_array=[]
for each in user_timestamp_dict.values():
    user_timestamp_array.append(each)
user_timestamp_array=np.array(user_timestamp_array) 

########################################################################
########################################################################
########################################################################
#compute user similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


user_similarity_matrix=[]
for i in range(0,num_user):
    if i%10==0:
        print(i," out of ",num_user)
    user_1_id=i+1
    row=[]
    for j in range(0,num_user):
        user_2_id=j+1
        similarity=pearsonr(user_rating_array[user_1_id-1],user_rating_array[user_2_id-1])[0]
        #similarity=cosine_similarity([user_rating_array[user_1_id-1]],[user_rating_array[user_2_id-1]])[0][0]
        row.append(similarity)
    user_similarity_matrix.append(np.array(row))    
user_similarity_matrix=np.array(user_similarity_matrix)    


########################################################################
########################################################################
########################################################################   
#prediction function

def faster_rating_prediction(k,user_similarity_matrix,time,alpha,user_timestamp_array):
    #avg rating of all user matrix
    avg_rating_user_matrix=np.mean(user_rating_array,axis=1)
    avg_rating_user_matrix=avg_rating_user_matrix[:,np.newaxis]
    avg_rating_user_matrix=np.repeat(avg_rating_user_matrix,num_movie,axis=1)
    
    predicted_rating_array=[]
    for target_user_index in range(0,num_user):
        
        #avg rating of target user
        avg_rating_of_target_user=avg_rating_user_matrix[target_user_index,:]
        
        #find k similar user
        lst=pd.Series(list(user_similarity_matrix[target_user_index,:]))
        i=lst.nlargest(k+1)
        similar_user_index_list=i.index.values.tolist()
        similar_user_index_list=similar_user_index_list[1:] #exclude yourself
        
        #avg rating of similar user
        avg_rating_of_similar_user=avg_rating_user_matrix[similar_user_index_list,:]
        rating_of_similar_user=user_rating_array[similar_user_index_list,:]
        diff_of_similar_user=rating_of_similar_user-avg_rating_of_similar_user
        
        
        
        time_diff=weighted_time(target_user_index,similar_user_index_list,alpha,user_timestamp_array)
        
        #check for time
        if time==True:
            diff_of_similar_user=diff_of_similar_user*time_diff
        
        #second term
        similarity_to_target_user=user_similarity_matrix[target_user_index,similar_user_index_list]
        similarity_to_target_user=similarity_to_target_user[:,np.newaxis]
        numerator=sum(diff_of_similar_user*similarity_to_target_user)
        
        if time==True:
            denominator=sum(similarity_to_target_user*time_diff)
        else:
            denominator=sum(similarity_to_target_user)
        
        
        second_term=numerator/denominator
        
        #prediction
        predicted_rating_of_target_user=avg_rating_of_target_user+second_term
        predicted_rating_array.append(predicted_rating_of_target_user)

    predicted_rating_array=np.array(predicted_rating_array)
    return predicted_rating_array

########################################################################
########################################################################
######################################################################## 
#MAE function
def MAE_calculator(predicted_user_rating_array,user_rating_array):
    #change predict matrix to have only known value
    filter_matrix=np.copy(user_rating_array)
    filter_matrix[filter_matrix>0]=1
    predicted_user_rating_array=predicted_user_rating_array*filter_matrix
    
    num_predict=np.count_nonzero(predicted_user_rating_array)
    MAE=(abs(predicted_user_rating_array-user_rating_array).sum())/num_predict
    return MAE


########################################################################
########################################################################
########################################################################     

#generate abs time diff matrix
def weighted_time(target_user_index,similar_user_index_list,alpha,user_timestamp_array):


    a=user_timestamp_array[target_user_index,:]
    b=user_timestamp_array[similar_user_index_list,:]
    time_diff_matrix=abs(a-b)
    
    #standardization
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    time_diff_matrix = scaler.fit_transform(time_diff_matrix)
    lam_matrix=np.exp(-1*time_diff_matrix*alpha)
    return lam_matrix

########################################################################
########################################################################
######################################################################## 
#find best value of alpha (1.7)
k=3
MAE_list=[]
alpha_list=[]
alpha=0
for i in range(0,100):
    predicted_rating=faster_rating_prediction(k,user_similarity_matrix,True,alpha,user_timestamp_array)
    MAE=MAE_calculator(predicted_rating,user_rating_array)
    MAE_list.append(MAE)
    alpha_list.append(alpha)
    print(" MAE : ",MAE)
    alpha+=0.1
plt.title("find best value of alpha")
plt.ylabel("MAE")
plt.xlabel("alpha")
plt.plot(alpha_list,MAE_list)
best_alpha_index=MAE_list.index(min(MAE_list))
best_alpha=alpha_list[best_alpha_index]

########################################################################
########################################################################
######################################################################## 
#compare performance of no time and time
alpha=best_alpha
MAE_time_list=[]
MAE_no_time_list=[]
k_list=[]
for k in range(1,100,10):
    print("k : ",k)
    time=faster_rating_prediction(k,user_similarity_matrix,True,alpha,user_timestamp_array)
    no_time=faster_rating_prediction(k,user_similarity_matrix,False,alpha,user_timestamp_array)
    
    MAE_time=MAE_calculator(time,user_rating_array)
    MAE_no_time=MAE_calculator(no_time,user_rating_array)
    
    MAE_time_list.append(MAE_time)
    MAE_no_time_list.append(MAE_no_time)
    k_list.append(k)
plt.figure()
plt.xlabel("number of neighbourhood")
plt.ylabel("MAE")
plt.plot(k_list,MAE_time_list,c="green",label="consider dynamic user interest")
plt.plot(k_list,MAE_no_time_list,c="red",label="do not consider dynamic user interest")
plt.legend()
plt.show()