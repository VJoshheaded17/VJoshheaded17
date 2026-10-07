import numpy as np

import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split

X=[]
y=[]

with open("datasets/wdbc.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        features=[]
        for x in row[:30]:
            features.append(float(x))
        
        X.append(features)  
        y.append(int(row[30]))
#print(np.array(X).shape)
X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True) #remove 


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

print(X_test[0])


def normalize_data(X_train):
    min = np.min(X_train,axis=0)
    max = np.max(X_train,axis=0)
    
    difference = max-min
    # for i in range(len(difference)):
    #     if difference[i]==0:
    #         difference[i]=1
    X_train_norm = (X_train-min)/(difference) 
    #X_test_norm = (X_test-min)/(difference) #shouldnt normalize testing params
    
    return X_train_norm

X_train=normalize_data(X_train)

# print(X_train)
# print(X_test)

def euclidean_dist(a,b):
    diff=(a-b)**2
    return np.sqrt(np.sum(diff))

def knn(X_train, y_train, x, k):
    distances=[]
    for i in range(len(X_train)):
        distance= euclidean_dist(X_train[i],x) 
        distances.append((distance, y_train[i]))
    distances.sort()
    # print(distances)
    # print(distances[0:20])
    return distances[:k]

def voting(X_train, y_train,x,k):
    knns = knn(X_train, y_train, x, k)
    
    ones=0
    zeros=0
    
    for distance, label in knns:
        if label==0:
            zeros+=1
        else:
            ones+=1
            
    if zeros>ones:
        return 0
    else:
        return 1

# print(voting(X_train, y_train, X_test[0], 2))


def accuracy(X,Y, x,y,k):
    correct=0
    for j in range(len(x)):
        pred=voting(X,Y,x[j],k)
        if pred==y[j]:
            correct+=1
            
    accuracy = correct/len(x)
    return accuracy


dict_train={}
dict_test={}

for k in range(1, 52, 2):
    accuracies_train=[]
    accuracies_test=[]
    for j in range(20):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
        X_train= normalize_data(X_train)
        X_test=normalize_data(X_test)
        
        
        # voting_train = voting(X_train, y_train, X_train, k)
        # voting_test = voting(X_test, y_test, X_test, k) 
        
        
        each_acc_training=accuracy(X_train, y_train, X_train, y_train, k)
        accuracies_train.append(each_acc_training)
        
        each_acc_test=accuracy(X_train, y_train, X_test, y_test,  k)
        accuracies_test.append(each_acc_test)
     
    dict_train[k]=accuracies_train
    dict_test[k]=accuracies_test
    
print("training",dict_train)
print("testing", dict_test)
    
    
k_values=[]
train_means=[]
train_stds=[]
test_means=[]
test_stds=[]   

for k in range(1,52,2): 
    k_values.append(k)
    train_means.append(np.mean(dict_train[k]))
    train_stds.append(np.std(dict_train[k]))
    test_means.append(np.mean(dict_test[k]))
    test_stds.append(np.std(dict_test[k]))

best_index=np.argmax(test_means)
print(k_values[best_index])



        
#1.6 without normalization run
#dict_train_nonorm={}
dict_test_nonorm={}

for k in range(1, 52, 2): 
    accuracies_train=[]
    accuracies_test=[]
    for j in range(20):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
        
        
        # voting_train = voting(X_train, y_train, X_train, k)
        # voting_test = voting(X_test, y_test, X_test, k)
        
        
        each_acc_test=accuracy(X_train, y_train, X_test, y_test,  k)
        accuracies_test.append(each_acc_test)
     
    dict_test_nonorm[k]=accuracies_test
    
print("testing", dict_test_nonorm)
    

k_values_nonorm=[]
test_means_nonorm=[]
test_stds_nonorm=[]   

import matplotlib.pyplot as plt
for k in range(1,52,2):
    k_values_nonorm.append(k)
    test_means_nonorm.append(np.mean(dict_test_nonorm[k]))
    test_stds_nonorm.append(np.std(dict_test_nonorm[k]))
    
    
best_index_nonorm=np.argmax(test_means_nonorm)
print(k_values_nonorm[best_index_nonorm])



#All plots
#1.1
plt.errorbar(k_values,train_means,yerr=train_stds)
plt.xlabel('Values of k')
plt.ylabel('Accuracy over training data')
plt.show()
     
#1.2    
plt.errorbar(k_values,test_means,yerr=test_stds)
plt.xlabel('Values of k')
plt.ylabel('Accuracy over test data')
plt.show()

#1.6
plt.errorbar(k_values_nonorm,test_means_nonorm,yerr=test_stds_nonorm)
plt.xlabel('Values of k')
plt.ylabel('Accuracy over test data')
plt.show()
    
    
    
#testing runs

# for k in range(1, 11, 2):
#     accuracies_train=[]
#     accuracies_test=[]
#     for j in range(20):
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
#         X_train= normalize_data(X_train)
#         X_test=normalize_data(X_test)
        
        
#         # voting_train = voting(X_train, y_train, X_train, k)
#         # voting_test = voting(X_test, y_test, X_test, k)
        
        
#         each_acc_training=accuracy(X_train, y_train, X_train, y_train, k)
#         accuracies_train.append(each_acc_training)
        
#         each_acc_test=accuracy(X_train, y_train, X_test, y_test,  k)
#         accuracies_test.append(each_acc_test)
     
#     dict_train[k]=accuracies_train
#     dict_test[k]=accuracies_test
    
# print("training",dict_train)
# print("testing", dict_test)
    
    
# k_values=[]
# train_means=[]
# train_stds=[]
# test_means=[]
# test_stds=[]   

# for k in range(1,11,2):
#     k_values.append(k)
#     train_means.append(np.mean(dict_train[k]))
#     train_stds.append(np.std(dict_train[k]))
#     test_means.append(np.mean(dict_test[k]))
#     test_stds.append(np.std(dict_test[k]))
    

# import matplotlib.pyplot as plt
# #1.1
# plt.errorbar(k_values,train_means,yerr=train_stds)
# plt.xlabel('Values of k')
# plt.ylabel('Accuracy over training data')
# plt.show()
     
# #1.2    
# plt.errorbar(k_values,test_means,yerr=test_stds)
# plt.xlabel('Values of k')
# plt.ylabel('Accuracy over test data')
# plt.show()
     
     
     
# dict_test_nonorm={}

# for k in range(1, 7, 2):
#     accuracies_train=[]
#     accuracies_test=[]
#     for j in range(20):
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
        
        
#         # voting_train = voting(X_train, y_train, X_train, k)
#         # voting_test = voting(X_test, y_test, X_test, k)
        
        
#         each_acc_test=accuracy(X_train, y_train, X_test, y_test,  k)
#         accuracies_test.append(each_acc_test)
     
#     dict_test_nonorm[k]=accuracies_test
    
# print("testing", dict_test_nonorm)
    

# k_values_nonorm=[]
# test_means_nonorm=[]
# test_stds_nonorm=[]   

# import matplotlib.pyplot as plt
# for k in range(1,7,2):
#     k_values_nonorm.append(k)
#     test_means_nonorm.append(np.mean(dict_test_nonorm[k]))
#     test_stds_nonorm.append(np.std(dict_test_nonorm[k]))

# plt.errorbar(k_values_nonorm,test_means_nonorm,yerr=test_stds_nonorm)
# plt.xlabel('Values of k')
# plt.ylabel('Accuracy over test data')
# plt.show()
    