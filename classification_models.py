import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from processing_functions import create_numeric_labels, process_and_save
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

def remove_class(rbp_array,num_epochs,target_labels,class_):
    if class_ == 'F':
        return rbp_array[:65].copy(),num_epochs[:65].copy(),target_labels[:65].copy()
    if class_ == 'A':
        return rbp_array[36:].copy(),num_epochs[36:].copy(),target_labels[36:].copy()
    if class_ == 'C':
        return np.concatenate((rbp_array[:36],rbp_array[65:])), np.concatenate((num_epochs[:36],num_epochs[65:])), np.concatenate((target_labels[:36],target_labels[65:]))

def partial_flatten(rbp_array,num_epochs,target_labels,exclude=None,flatten_final=True):
    
    '''The sklearn models expect the input to take the form (number of examples) x (number of features). This requires flattening some of the dimensions of our arrays, which is done using the following helper function. The partial_flatten function reshapes the relative band power array so that the first dimension is epochs and the second dimension covers the relative band power for each of the five bands across all 19 channels. The num_epochs array we loaded in is passed as the second argument in order to easily exclude the zero-padded parts of the array. The function also returns a 1-d array of dimension (number of examples,) containing the corresponding class labels for each of these epochs. 

The function assumes that classes that aren't used in the classification have already been removed from the rbp_array, i.e., if you are doing Alzheimer's/healthy classification (for instance) then all FTD examples have been removed. This can be done by feeding the rbp_array, num_epochs, and target_labels into the remove_class function first and then feeding the results into the partial_flatten function. 

If the "exclude" argument is not None then the subject corresponding to that index is left out in the returned arrays. Note that this exclude index corresponds to the index of the subject in the rbp_array being fed into the function, which may not necessarily be the same as the subject's index in the original rbp array that we loaded. The flatten_final argument is a boolean that specifies whether or not the bands x channels part of the array should be flattened into a single dimension. This defaults to True, which is used for the sklearn models, but the non-flattened version will likely later be used in some other models (it's used in the CNN paper, for instance
    
    '''
    total_subjects = len(target_labels)
    feature_arrays = []
    target_arrays = []
    for i in range(total_subjects):
        feature_arrays.append(rbp_array[i,0:num_epochs[i],:,:])
        target_arrays.append(target_labels[i]*np.ones(num_epochs[i]))
    if exclude==None: 
        features= np.concatenate(feature_arrays)
        targets = np.concatenate(target_arrays)
    else:
        features= np.concatenate(feature_arrays[:exclude] + feature_arrays[exclude+1:])
        targets = np.concatenate(target_arrays[:exclude] + target_arrays[exclude+1:])
    if flatten_final:
        features = features.reshape((features.shape[0],-1))
    return features, targets

'''The next set of functions take in a 2 x 2 (total) confusion matrix presented as a numpy array and compute the metrics we are interested in. These functions assume that the 0-index of the confusion matrix corresponds to negative examples and the 1-index corresponds to positive examples. 

In the comments TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative. 
'''
def accuracy(confusion):
    # (TN + TP)/total
    return (confusion[0,0]+confusion[1,1])/np.sum(confusion)
def sensitivity(confusion):
    # TP/(TP+FN)
    return confusion[1,1]/(confusion[1,1]+confusion[1,0])
def specificity(confusion):
    # TN/(TN+FP)
    return confusion[0,0]/(confusion[0,0]+confusion[0,1])
def precision(confusion):
    # TP/(TP+FP)
    return confusion[1,1]/(confusion[1,1]+confusion[0,1])
def f1(confusion):
    # harmonic mean of precision and sensitivity
    return 2*(precision(confusion)*sensitivity(confusion))/(precision(confusion)+sensitivity(confusion))

def kNN_cross(rbp_array,num_epochs,target_labels,removed_class,n_neighbors):
    '''
    The function below runs our two-class kNN modeling routine with leave-one-subject-out cross-validation. It returns a dictionary of cross-validation accuracy, sensitivity, specificity, and F1 scores computed from the total confusion matrix. removed_class indicates the class to be excluded to create a two-class classification problem (same labels as the remove_class function) and n_neighbors indicates the number of neighbors k to use in kNN. 
    '''
    
    if removed_class == 'F':
        labels = [0,1]
    if removed_class == 'A':
        labels = [0,2]
    if removed_class == 'C':
        labels = [1,2]
    confusion_matricesTest = []
    confusion_matricesTrain = []
    mod_rbp, mod_num_epochs, mod_target_labels = remove_class(rbp_array,num_epochs,target_labels,removed_class)
    for i in range(len(mod_target_labels)):
        train_X, train_y = partial_flatten(mod_rbp,mod_num_epochs,mod_target_labels,exclude=i,flatten_final=True)
        test_X = mod_rbp[i,0:mod_num_epochs[i],:,:].reshape(mod_num_epochs[i],-1)
        test_y = mod_target_labels[i]*np.ones(mod_num_epochs[i])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        
        ThreeNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        ThreeNN.fit(train_X, train_y)
        
        test_X = scaler.transform(test_X)
        confusion_matricesTrain += [confusion_matrix(train_y,ThreeNN.predict(train_X),labels=labels)]
        confusion_matricesTest += [confusion_matrix(test_y,ThreeNN.predict(test_X),labels=labels)]
    confusion_matricesTrain = np.array(confusion_matricesTrain)
    confusion_matricesTest = np.array(confusion_matricesTest)
    total_confusion_Train = np.sum(confusion_matricesTrain, axis= 0)
    total_confusion_Test = np.sum(confusion_matricesTest, axis= 0)
    return {'Test_acc':accuracy(total_confusion_Test), 'Test_sens':sensitivity(total_confusion_Test), 'Test_spec':specificity(total_confusion_Test), 'Test_f1':f1(total_confusion_Test), 'Train_acc':accuracy(total_confusion_Train), 'Train_sens':sensitivity(total_confusion_Train), 'Train_spec':specificity(total_confusion_Train), 'Train_f1':f1(total_confusion_Train) }


def RF_cross(rbp_array,num_epochs,target_labels,removed_class, min_samples_split = 0.01, PCA_components = 95):
    
    '''
    The function below runs our two-class Ranfom Forest modeling routine with leave-one-subject-out cross-validation. It returns a dictionary of cross-validation accuracy, sensitivity, specificity, and F1 scores computed from the total confusion matrix. removed_class indicates the class to be excluded to create a two-class classification problem (same labels as the remove_class function).
    '''
    
    if removed_class == 'F':
        labels = [0,1]
    if removed_class == 'A':
        labels = [0,2]
    if removed_class == 'C':
        labels = [1,2]
    confusion_matricesTest = []
    confusion_matricesTrain = []
    mod_rbp, mod_num_epochs, mod_target_labels = remove_class(rbp_array,num_epochs,target_labels,removed_class)
    for i in range(len(mod_target_labels)):
        train_X, train_y = partial_flatten(mod_rbp,mod_num_epochs,mod_target_labels,exclude=i,flatten_final=True)
        test_X = mod_rbp[i,0:mod_num_epochs[i],:,:].reshape(mod_num_epochs[i],-1)
        test_y = mod_target_labels[i]*np.ones(mod_num_epochs[i])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        
        pca = PCA(n_components = PCA_components)
        train_X = pca.fit_transform(train_X)
        
        RF = RandomForestClassifier(min_samples_split = min_samples_split)
        RF.fit(train_X, train_y)
        
        test_X = scaler.transform(test_X)
        test_X = pca.transform(test_X)
        
        confusion_matricesTrain += [confusion_matrix(train_y, RF.predict(train_X),labels=labels)]
        confusion_matricesTest += [confusion_matrix(test_y, RF.predict(test_X),labels=labels)]
    confusion_matricesTrain = np.array(confusion_matricesTrain)
    confusion_matricesTest = np.array(confusion_matricesTest)
    total_confusion_Train = np.sum(confusion_matricesTrain, axis= 0)
    total_confusion_Test = np.sum(confusion_matricesTest, axis= 0)
    return {'Test_acc':accuracy(total_confusion_Test), 'Test_sens':sensitivity(total_confusion_Test), 'Test_spec':specificity(total_confusion_Test), 'Test_f1':f1(total_confusion_Test), 'Train_acc':accuracy(total_confusion_Train), 'Train_sens':sensitivity(total_confusion_Train), 'Train_spec':specificity(total_confusion_Train), 'Train_f1':f1(total_confusion_Train) }




def MLP_cross(rbp_array,num_epochs,target_labels,removed_class, hidden_layer_sizes = (3,1)):
    if removed_class == 'F':
        labels = [0,1]
    if removed_class == 'A':
        labels = [0,2]
    if removed_class == 'C':
        labels = [1,2]
    confusion_matricesTest = []
    confusion_matricesTrain = []
    mod_rbp, mod_num_epochs, mod_target_labels = remove_class(rbp_array,num_epochs,target_labels,removed_class)
    for i in range(len(mod_target_labels)):
        train_X, train_y = partial_flatten(mod_rbp,mod_num_epochs,mod_target_labels,exclude=i,flatten_final=True)
        test_X = mod_rbp[i,0:mod_num_epochs[i],:,:].reshape(mod_num_epochs[i],-1)
        test_y = mod_target_labels[i]*np.ones(mod_num_epochs[i])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        
        MLP = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes)
        MLP.fit(train_X, train_y)
        
        test_X = scaler.transform(test_X)
        
        confusion_matricesTrain += [confusion_matrix(train_y, MLP.predict(train_X),labels=labels)]
        confusion_matricesTest += [confusion_matrix(test_y, MLP.predict(test_X),labels=labels)]
    confusion_matricesTrain = np.array(confusion_matricesTrain)
    confusion_matricesTest = np.array(confusion_matricesTest)
    total_confusion_Train = np.sum(confusion_matricesTrain, axis= 0)
    total_confusion_Test = np.sum(confusion_matricesTest, axis= 0)
    return {'Test_acc':accuracy(total_confusion_Test), 'Test_sens':sensitivity(total_confusion_Test), 'Test_spec':specificity(total_confusion_Test), 'Test_f1':f1(total_confusion_Test), 'Train_acc':accuracy(total_confusion_Train), 'Train_sens':sensitivity(total_confusion_Train), 'Train_spec':specificity(total_confusion_Train), 'Train_f1':f1(total_confusion_Train) }