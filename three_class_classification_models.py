import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from processing_functions import create_numeric_labels, process_and_save
from classification_models import partial_flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier


def kNN_cross_3class(rbp_array,num_epochs,target_labels, n_neighbors = 3):
    '''
    This function runs a three-class kNN modeling routine with leave-one-subject-out cross-validation. It returns a list of cross-validation accuracies for when each subject is left out, and prints out the mean accuracy. 
    '''
    cross_valid_acc_3class = []
    for i in range(len(target_labels)):
        train_X, train_y = partial_flatten(rbp_array, num_epochs, target_labels, exclude=i,flatten_final=True)
        test_X = rbp_array[i, 0: num_epochs[i],:,:].reshape(num_epochs[i],-1)
        test_y = target_labels[i]*np.ones(num_epochs[i])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        
        ThreeNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        ThreeNN.fit(train_X, train_y)
        
        test_X = scaler.transform(test_X)
        
        cross_valid_acc_3class += [ThreeNN.score(test_X, test_y)]
    print('The accuracy is: ', np.mean(cross_valid_acc_3class))
    return cross_valid_acc_3class
    
def RF_cross_3class(rbp_array,num_epochs,target_labels):
    '''
    This function runs a three-class Random Forest modeling routine with leave-one-subject-out cross-validation. It returns a list of cross-validation accuracies for when each subject is left out, and prints out the mean accuracy. 
    '''
    cross_valid_acc_3class = []
    for i in range(len(target_labels)):
        train_X, train_y = partial_flatten(rbp_array, num_epochs, target_labels, exclude=i,flatten_final=True)
        test_X = rbp_array[i, 0: num_epochs[i],:,:].reshape(num_epochs[i],-1)
        test_y = target_labels[i]*np.ones(num_epochs[i])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        
        RF = RandomForestClassifier()
        RF.fit(train_X, train_y)
        
        test_X = scaler.transform(test_X)
        
        cross_valid_acc_3class += [RF.score(test_X, test_y)]
    print('The accuracy is: ', np.mean(cross_valid_acc_3class))
    return cross_valid_acc_3class
    

def MLP_cross_3class(rbp_array,num_epochs,target_labels, hidden_layer_sizes = (3,1)):
    '''
    This function runs a three-class MultiLayer Perceptron modeling routine with leave-one-subject-out cross-validation. It returns a list of cross-validation accuracies for when each subject is left out, and prints out the mean accuracy. 
    '''
    cross_valid_acc_3class = []
    for i in range(len(target_labels)):
        train_X, train_y = partial_flatten(rbp_array, num_epochs, target_labels, exclude=i,flatten_final=True)
        test_X = rbp_array[i, 0: num_epochs[i],:,:].reshape(num_epochs[i],-1)
        test_y = target_labels[i]*np.ones(num_epochs[i])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        
        MLP = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes)
        MLP.fit(train_X, train_y)
        
        test_X = scaler.transform(test_X)
        
        cross_valid_acc_3class += [MLP.score(test_X, test_y)]
    print('The accuracy is: ', np.mean(cross_valid_acc_3class))
    return cross_valid_acc_3class
