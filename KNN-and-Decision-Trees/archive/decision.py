import csv
from sklearn.model_selection import train_test_split
import numpy as np
import math

X=[]
y=[]

with open("datasets/car.csv") as file:
    reader = csv.reader(file)
    head = next(reader)
    #print(head)
    for row in reader:
        features=[]
        for x in row[:6]:
            features.append(x)
        X.append(features)  
        y.append(row[6])    
        
#print(np.array(X).shape)
X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True) #remove after done (experiment code splits it already)


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

print( X[2][4])
print(y)


def entropy(y):
    I=0
    labels_visited=[] #track unqiues
    for i in y:
        if i not in labels_visited:
            count=0
            for j in y:
                if j==i: 
                    count+=1 
            probability=count/len(y)
            # print(probability)
            I-=probability*math.log(probability,2) #entropy formula
            labels_visited.append(i) 
    # print(labels_visited)
    
    return I


def information_gain(attribute, y, data):
    Id=entropy(y)
    
    values=[] #track unique values
    for i in data:
        if i[attribute] not in values:
            values.append(i[attribute])
    info=0
    for i in values:
        y_partition=[]
        for j in range(len(data)):
            if data[j][attribute]==i:
                y_partition.append(y[j])
        
        I_dj=(len(y_partition)/len(y))*entropy(y_partition) #info gain formula
        info+=I_dj
        
    return Id - info









# print(information_gain("luggage", y, X))

# def majority_class(y):
#     l=[]
#     for i in y:
#         if i not in l:
#             l.append(i)
    
#     counts=[]
#     for i in l:
#         count=0
#         for j in y:
#             if j==i:
#                 count+=1
#         counts.append(count)
        
#     id = np.argmax(counts)
    
#     return l[id]
        
    


def decision_tree(D,L,y, vals):
    
    Node={} 
    
    labels_unique=[]
    
    for i in y:
        if i not in labels_unique:
            labels_unique.append(i)
    
    if len(labels_unique)==1: #stop 1
        return {"key":"leaf","value":labels_unique[0]}
    
    counts=[] 
    for i in labels_unique: #loop for majority 
        count=0
        for j in y:
            if j==i:
                count+=1
        counts.append(count)
    majority=labels_unique[np.argmax(counts)] #directly getting the majority one instead of getting the id of it (ref 2)
        
    id = np.argmax(counts)  #ref 2
    
        
    if len(L)==0: #stop 2
        
       return {"key":"leaf", "value": majority}
   
    l=[] #needed?
    for i in y:
        if i not in l:
            l.append(i) #tentatively using for keeping track of labels
    
   
    
    best_attribute_list=[]
    for i in L:
        best_attribute_list.append(information_gain(i,y,D)) 
        
    best_attribute= L[np.argmax(best_attribute_list)] #dropping separate vars for indices altogether and using list indexing 
    
    Node["key"]="node"
    Node["attribute"]=best_attribute
    Node["branches"]={}
    
    L2=L.copy()
    L2.remove(best_attribute) 
    
    for v in vals[best_attribute]: #main loop for recursion
        # Dv_id=[] #remove for now (1)
        Xv=[]
        Yv=[]
       
        for i in range(len(D)):
            if D[i][best_attribute]==v:
                Xv.append(D[i])
                Yv.append(y[i]) #directly appending into partitions instead of indices (1)
        
     
        # print(Dv_id)
        
        # for i in Dv_id:
        #     Xv.append(D[i])
        #     Yv.append(y[i])  #removed for now (1)
            
            
        
        if len(Xv)==0:
            Tv={"key":"leaf", "value": "majority"}
        else:
            Tv=decision_tree(Xv,L2, Yv,vals)
            
        Node["branches"][v]=Tv
    
    return Node
            
    
    
def predict(test,tree):
    if tree["key"]=="leaf":
        return tree["value"] #leaf node logic
    
    while tree["key"]!="leaf": #main tree logic
        attribute=tree["attribute"] 
        value=test[attribute]
        tree=tree["branches"][value]
    return tree["value"] 


def accuracy(X, y, tree):
    correct=0
    for j in range(len(X)):
        pred = predict(X[j], tree)
        if pred == y[j]:
            correct+=1
            
    accuracy = correct/len(X)
    return accuracy





accuracies_train=[]
accuracies_test=[]
for j in range(100):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
    
    L=[0,1,2,3,4,5]
    
    vals=[]
    
    for i in range(len(X_train[0])):
        each_attribute_values=[]
        for j in X_train:
            if j[i] not in each_attribute_values:
                each_attribute_values.append(j[i])
        vals.append(each_attribute_values)


    tree=decision_tree(X_train,L,y_train,vals)    
    # voting_train = voting(X_train, y_train, X_train, k)
    # voting_test = voting(X_test, y_test, X_test, k)
    
    each_acc_training=accuracy(X_train, y_train, tree)
    accuracies_train.append(each_acc_training)
    
    each_acc_test=accuracy(X_test, y_test, tree)
    accuracies_test.append(each_acc_test)
    

    
print(accuracies_train)
print(accuracies_test)


mean_train = np.mean(accuracies_train)
std_train = np.std(accuracies_train)
mean_test = np.mean(accuracies_test)
std_test = np.std(accuracies_test)

print("Training Mean", mean_train)
print("Training Standard Deviation", std_train)
print("Testing Mean", mean_test)
print("Testing Standard Deviation", std_test)

#2.1
import matplotlib.pyplot as plt









#EC 

def gini_criteria(y):
    I=1
    labels_visited=[]
    for i in y:
        if i not in labels_visited:
            count=0
            for j in y:
                if j==i:
                    count+=1
            probability=count/len(y)
            # print(probability)
            I-=(probability**2)
            labels_visited.append(i)
    # print(labels_visited)
    
    return I

p_test=['yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
# print(entropy(p_test))

print("gini",gini_criteria(p_test))

def gini_gain(attribute, y, data):
    # Id=gini_criteria(y) #nope not needed
    
    values=[]
    for i in data:
        if i[attribute] not in values:
            values.append(i[attribute])
    info=0
    for i in values:
        y_partition=[]
        for j in range(len(data)):
            if data[j][attribute]==i:
                y_partition.append(y[j])
        
        I_dj=(len(y_partition)/len(y))*gini_criteria(y_partition)
        info+=I_dj
        
    return info

def decision_tree_gini(D,L,y, vals):
    
    Node={}
    
    labels_unique=[]
    
    for i in y:
        if i not in labels_unique:
            labels_unique.append(i)
    
    if len(labels_unique)==1: #stop 1
        return {"key":"leaf","value":labels_unique[0]}
    
    counts=[]
    for i in labels_unique:
        count=0
        for j in y:
            if j==i:
                count+=1
        counts.append(count)
    majority=labels_unique[np.argmax(counts)]
        
    # id = np.argmax(counts) 
    
        
    if len(L)==0: #stop 2
        
       return {"key":"leaf", "value": "majority"} #directly assigning keys and values 
   
    l=[]
    for i in y:
        if i not in l:
            l.append(i)  
    
   
    
    best_attribute_list=[]
    for i in L:
        best_attribute_list.append(gini_gain(i,y,D))
        
    best_attribute= L[np.argmin(best_attribute_list)]
    
    Node["key"]="node"
    Node["attribute"]=best_attribute
    Node["branches"]={}
    
    L2=L.copy()
    L2.remove(best_attribute)
    
    for v in vals[best_attribute]:
        # Dv_id=[] #remove for now (1)
        Xv=[]
        Yv=[]
       
        for i in range(len(D)):
            if D[i][best_attribute]==v:
                Xv.append(D[i])
                Yv.append(y[i]) #directly appending into partitions instead of indices
        
     
        # print(Dv_id)
        
        # for i in Dv_id:
        #     Xv.append(D[i])
        #     Yv.append(y[i])  #removed for now (1)
            
            
        
        if len(Xv)==0:
            Tv={"key":"leaf", "value": "majority"}
        else:
            Tv=decision_tree_gini(Xv,L2, Yv,vals)
            
        Node["branches"][v]=Tv
    
    return Node
            

accuracies_train_gini=[]
accuracies_test_gini=[]
for j in range(100):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
    
    L=[0,1,2,3,4,5] #indices for attr
    
    vals=[]
    
    for i in range(len(X_train[0])):
        each_attribute_values=[]
        for j in X_train:
            if j[i] not in each_attribute_values:
                each_attribute_values.append(j[i])
        vals.append(each_attribute_values)


    tree=decision_tree_gini(X_train,L,y_train,vals)    
    # voting_train = voting(X_train, y_train, X_train, k)
    # voting_test = voting(X_test, y_test, X_test, k)
    
    each_acc_training=accuracy(X_train, y_train, tree)
    accuracies_train_gini.append(each_acc_training)
    
    each_acc_test=accuracy(X_test, y_test, tree)
    accuracies_test_gini.append(each_acc_test)




mean_train_gini = np.mean(accuracies_train_gini)
std_train_gini = np.std(accuracies_train_gini)
mean_test_gini = np.mean(accuracies_test_gini)
std_test_gini = np.std(accuracies_test_gini)

print("Training Mean gini", mean_train_gini)
print("Training Standard Deviation gini", std_train_gini)
print("Testing Mean gini", mean_test_gini)
print("Testing Standard Deviation gini", std_test_gini)


#All plots


#2.1 and #2.2
plt.hist(accuracies_train,bins=10)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Hist for training accuracy')
plt.show()

plt.hist(accuracies_test,bins=10)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Hist for testing accuracy')
plt.show()

#EC
plt.hist(accuracies_train_gini,bins=10)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Hist for training accuracy gini')
plt.show()

plt.hist(accuracies_test_gini,bins=10)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Hist for testing accuracy gini')

plt.show()


#EC2

# def decision_tree_heuristic(D,L,y, vals):
    
#     Node={} 
    
#     labels_unique=[]
    
#     for i in y: 
#         if i not in labels_unique:
#             labels_unique.append(i)
    
#     if len(labels_unique)==1: 
#         return {"key":"leaf","value":labels_unique[0]}
    
#     if 
    
#     counts=[] 
#     for i in labels_unique: #loop for majority 
#         count=0
#         for j in y:
#             if j==i:
#                 count+=1
#         counts.append(count)
#     majority=labels_unique[np.argmax(counts)] #directly getting the majority one instead of getting the id of it (ref 2)
        
#     id = np.argmax(counts)  #ref 2
    
        
#     if len(L)==0:
        
#        return {"key":"leaf", "value": "majority"}
   
#     l=[] #needed?
#     for i in y:
#         if i not in l:
#             l.append(i) #tentatively using for keeping track of labels
    
    
    
   
    
#     best_attribute_list=[]
#     for i in L:
#         best_attribute_list.append(information_gain(i,y,D)) 
        
#     best_attribute= L[np.argmax(best_attribute_list)] #dropping separate vars for indices altogether and using list indexing 
    
#     Node["key"]="node"
#     Node["attribute"]=best_attribute
#     Node["branches"]={}
    
#     L2=L.copy()
#     L2.remove(best_attribute) 
    
#     for v in vals[best_attribute]: #main loop for recursion
#         # Dv_id=[] #remove for now (1)
#         Xv=[]
#         Yv=[]
       
#         for i in range(len(D)):
#             if D[i][best_attribute]==v:
#                 Xv.append(D[i])
#                 Yv.append(y[i]) #directly appending into partitions instead of indices (1)
        
     
#         # print(Dv_id)
        
#         # for i in Dv_id:
#         #     Xv.append(D[i])
#         #     Yv.append(y[i])  #removed for now (1)
            
            
        
#         if len(Xv)==0:
#             Tv={"key":"leaf", "value": "majority"}
#         else:
#             Tv=decision_tree(Xv,L2, Yv,vals)
            
#         Node["branches"][v]=Tv
    
#     return Node
            
    




# accuracies_train_heuristic=[]
# accuracies_test_heuristic=[]
# for j in range(100):
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
    
#     L=[0,1,2,3,4,5]
    
#     vals=[]
    
#     for i in range(len(X_train[0])):
#         each_attribute_values=[]
#         for j in X_train:
#             if j[i] not in each_attribute_values:
#                 each_attribute_values.append(j[i])
#         vals.append(each_attribute_values)


#     tree=decision_tree(X_train,L,y_train,vals)    
#     # voting_train = voting(X_train, y_train, X_train, k)
#     # voting_test = voting(X_test, y_test, X_test, k)
    
#     each_acc_training=accuracy(X_train, y_train, tree)
#     accuracies_train.append(each_acc_training)
    
#     each_acc_test=accuracy(X_test, y_test, tree)
#     accuracies_test.append(each_acc_test)
    

    
# print(accuracies_train_heuristic)
# print(accuracies_test_heuristic)


# mean_train_heuristic = np.mean(accuracies_train_heuristic)
# std_train_heuristsic = np.std(accuracies_train_heuristic)
# mean_test_heuristic = np.mean(accuracies_test_heuristic)
# std_test_heuristic = np.std(accuracies_test_heuristic)

# print("Training Mean", mean_train_heuristic)
# print("Training Standard Deviation", std_train_heuristsic)
# print("Testing Mean", mean_test_heuristic)
# print("Testing Standard Deviation", std_test_heuristic)