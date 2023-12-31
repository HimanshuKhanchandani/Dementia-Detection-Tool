from model_functions import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
#import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

'''
This file defines the necessary functions to perform leave-one-subject-out cross validation using different classification 
algorithms. To prepare the data ready for training, it uses functions from the model_functions file. 
'''


def log_reg_cross(rbps,targets, PCA_components = 0,reg_parameter=1.0,max_iter=500):
    
    '''
    Performs training on features in rbps and targets using k nearest neighbor classification and performs cross-validation 
    using leave one subject out method. 
    
    Parameters
    ----------
    rbps : list[ndarray]
        List of feature arrays corresponding to each subject. 
    targets : ndarray
        Array of numeric class labels for each feature array.
    n_neighbors: int
        number of nearest neighbors to use
    PCA_components: int
        Number of components to keep for principal components analysis. Defaults to 0 which is the case when we don't do PCA.
    reg_parameter: float
        Parameter inversely proportional to strength of regularization
    max_iter: int
        Number of iterations taken for solvers to converge. 
           
    Returns
    -------
    train_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the training data.
    test_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the validation  data.
    
    '''
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        log_reg = LogisticRegression(max_iter = max_iter,C=reg_parameter)
        log_reg.fit(train_X, train_y)
        
        confusion_matrices_train += [confusion_matrix(train_y, log_reg.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y, log_reg.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict








def kNN_cross(rbps,targets,n_neighbors, PCA_components = 0):
    
    '''
    Performs training on features in rbps and targets using k nearest neighbor classification and performs cross-validation 
    using leave one subject out method. 
    
    Parameters
    ----------
    rbps : list[ndarray]
        List of feature arrays corresponding to each subject. 
    targets : ndarray
        Array of numeric class labels for each feature array.
    n_neighbors: int
        number of nearest neighbors to use
    PCA_components: int
        Number of components to keep for principal components analysis. Defaults to 0 which is the case when we don't do PCA.
           
    Returns
    -------
    train_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the training data.
    test_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the validation  data.
    
    '''
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        ThreeNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        ThreeNN.fit(train_X, train_y)
        
        confusion_matrices_train += [confusion_matrix(train_y, ThreeNN.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y,ThreeNN.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict




def RF_cross(rbps, targets, n_estimators = 100, min_samples_split = 16, PCA_components = 0):
    
    '''
    Performs training on features in rbps and targets using random forest classification and performs cross-validation 
    using leave one subject out method. 
    
    Parameters
    ----------
    rbps : list[ndarray]
            List of feature arrays corresponding to each subject. 
    targets : ndarray
            Array of numeric class labels for each feature array.
    n_estimators: int
            number of trees in the forest
    min_samples_split: int or floar
           The minimum number of samples required to split an internal node in random forest classifier. If int, then consider 
           min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * 
           n_samples) are the minimum number of samples for each split. Defaults to 16 which was found to maximize the 
           cross-validation accuracy.  
    PCA_components: int
           Number of components to keep for principal components analysis. Defaults to 0 which is the case when we don't do PCA.
    Returns
    -------
    train_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the training data.
    test_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the validation  data.
    
    '''
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
       
        
        RF = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split)
        RF.fit(train_X, train_y)
    
        
        confusion_matrices_train += [confusion_matrix(train_y, RF.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y,RF.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict
    
def svm_cross(rbps,targets,kernel,reg_parameter=1.0, degree=2, PCA_components = 0):
    """
    Performs training on features in rbps and targets using sklearn's svm C-Support vector classificaton and performs 
    cross-validation using leave one subject out method. 

    Parameters
    ----------
    rbps : list[ndarray]
            List of feature arrays corresponding to each subject. 
    targets : ndarray
            Array of numeric class labels for each feature array.
    kernel : 
        Kernel used for SVM. See sklearn documentation for options. 
    reg_parameter : float, optional
        Parameter inversely proportional to strength of regularization in SVM, by default 1.0
    degree : int, optional
        degree of polynomial used for kernel when kernel = 'poly', by default 2. Ignored when kernel != 'poly'.
    PCA_components: int
           Number of components to keep for principal components analysis. Defaults to 0 which is the case when we don't do PCA.

    Returns
    -------
    train_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the training data.
    test_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the validation  data.
    """    
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        
        svc = SVC(C = reg_parameter, kernel=kernel, degree=degree)
        svc.fit(train_X, train_y)

        
        confusion_matrices_train += [confusion_matrix(train_y, svc.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y,svc.predict(test_X),labels=labels)]

    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)

    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}


    return train_metrics_dict, test_metrics_dict

def mlp_cross(rbps,targets, PCA_components = 0, alpha = 0.0001, learning_rate_init = 0.001, hidden_layer_sizes = (3,1), max_iter = 200, solver = 'adam', random_state = None):
    
    '''
    Performs training on features in rbps and targets using multi-layer perceptron classification and performs cross-validation 
    using leave one subject out method. 
    
    Parameters
    ----------
    rbps : list[ndarray]
        List of feature arrays corresponding to each subject. 
    targets : ndarray
        Array of numeric class labels for each feature array.
    PCA_components: int
        Number of components to keep for principal components analysis. Defaults to 0 which is the case when we don't do PCA.
    alpha: float
        Parameter describing strength of L2 regularization term. 
    max_iter: int
        Maximum number of iterations for solver. 
    learning_rate_init: float
        Initial learning rate used.
    hidden_layer_sizes: 
        tuple describing the number of neurons in each hidden layer.
    solver: {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
        Solver to use for weight optimization.
    random_state: int, default=None
        Determines random number generation for various parts of the training process. 
           
    Returns
    -------
    train_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the training data.
    test_metrics_dict : dictionary
        Dictionary containing accuracy, sensitivity, specificity and f1 scores on the validation  data.
    
    '''
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        mlp = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, max_iter = max_iter,alpha = alpha, learning_rate_init = learning_rate_init,random_state=random_state)
        mlp.fit(train_X, train_y)
        
        confusion_matrices_train += [confusion_matrix(train_y, mlp.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y, mlp.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict
    
    
   

  


