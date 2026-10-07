import numpy as np
import random
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

selected_features = None


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    z = np.clip(z, -500, 500)
    x = 1 / (1 + np.exp(-z))
    return x # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains          50,000
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data         10,000
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains            10,000
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    global selected_features

    total_data_ordered = []
    total_label_ordered = []
    for idx in range(10):
        stridx = "train"+str(idx)
        digit_data = mat[stridx]
        for val in digit_data:
            total_data_ordered.append(val)
        for i in range(len(digit_data)):
            total_label_ordered.append(idx)
    zipped = list(zip(total_data_ordered, total_label_ordered))

    random.shuffle(zipped)

    total_data, total_label = zip(*zipped)

    total_data = list(total_data)
    total_label = list(total_label)

    train_data = total_data[:50000]
    train_label = total_label[:50000]

    validation_data = total_data[50000:]
    validation_label = total_label[50000:]


    test_data_ordered = []
    test_label_ordered = []
    for idx in range(10):
        stridx = "test"+str(idx)
        digit_data = mat[stridx]
        for val in digit_data:
            test_data_ordered.append(val)
        for i in range(len(digit_data)):
            test_label_ordered.append(idx)
    zipped = list(zip(test_data_ordered, test_label_ordered))

    random.shuffle(zipped)

    test_data, test_label = zip(*zipped)

    test_data = list(test_data)
    test_label = list(test_label)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    validation_data = np.array(validation_data)
    validation_label = np.array(validation_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    

    


    # Feature selection
    # Your code here.

    threshold = 1e-5
    feature_std = np.std(train_data, axis=0)
    selected_features = feature_std > threshold  # Boolean mask

    train_data = train_data[:, selected_features]
    validation_data = validation_data[:, selected_features]
    test_data = test_data[:, selected_features]

    selected_features = feature_std > threshold

    print(f'preprocess done. Selected {np.sum(selected_features)} of {len(selected_features)} features.')
    print(train_data.size)


    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    ones = np.ones((training_data.shape[0], 1))
    ones_data_with_bias = np.concatenate((training_data, ones), axis=1)  # Shape: (n_samples, n_input + 1)

    #forward propogation (equations 1, 2, 3, 4 in the handout)
    aj = np.dot(ones_data_with_bias,np.transpose(w1)) #hidden layer computation 
    zj = sigmoid(aj) #
    zj_bias = np.concatenate((zj, np.ones((zj.shape[0], 1))), axis=1)  #bias added to hidden layer. zj_bias = hidden activation

    bl = np.dot(zj_bias, np.transpose(w2))  #output layer calculation
    ol = sigmoid(bl) 

   #convert labels in bin vectors to match with ol (10 classes)
    yl = np.eye(n_class)[training_label.astype(int)]

    w1_no_bias = np.delete(w1, -1, axis=1) #bias not taken for reg
    w2_no_bias = np.delete(w2, -1, axis=1)    

    
    obj_val = -np.sum(yl * np.log(ol) + (1 - yl) * np.log(1 - ol )) / training_data.shape[0]  #equation 5 
    reg_term = (lambdaval / (2 * training_data.shape[0])) * (np.sum(w1_no_bias**2) + np.sum(w2_no_bias**2)) #equation 15 regularization
    obj_val += reg_term

    
    deltal = ol - yl  
    delta2 = zj * (1 - zj) * np.dot(deltal, w2_no_bias)  #observation 3 and equation 12
    w1_grad = np.dot(np.transpose(delta2), ones_data_with_bias) / training_data.shape[0]  
    w2_grad = np.dot(np.transpose(deltal), zj_bias) / training_data.shape[0]  

    
    obj_grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten()))

  


   

    # Your code here
    #
    #
    #
    #
    #



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    #labels = np.array([])

    #summation = np.dot(data, w1)

    #sigmoid_a = sigmoid(summation)

    #summation2 = np.dot(sigmoid_a, w2)

    #sigmoid_b = sigmoid(summation2)  

    #labels = np.argmax(sigmoid_b,1)

    # num_samples = data.shape[0]
    ones = np.ones((data.shape[0], 1))
    data_with_bias = np.concatenate((data, ones), axis=1)
    
    z_hidden = np.dot(data_with_bias, np.transpose(w1))
    a_hidden = sigmoid(z_hidden)

    hidden_layer_bias_ones = np.ones((a_hidden.shape[0], 1))
    a_hidden_with_bias = np.concatenate((a_hidden, hidden_layer_bias_ones), axis=1)
  
    z_output = np.dot(a_hidden_with_bias, np.transpose(w2))
    ol = sigmoid(z_output)
    
    labels = np.argmax(ol, axis=1)
    return labels



    # Your code here

"""*Neural Network Script Starts here*"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 200

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


