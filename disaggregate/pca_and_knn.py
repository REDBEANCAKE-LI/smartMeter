#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#constants
color_list = ['r', 'y', 'g', 'c', 'b', 'm', 'k', 'w']
dp_per = 1 #display_percentage
dpa_per = 1 #display_all_percentage

def display(pos, lab, color):
    
    '''
    Display classification result.
    
    parameters
    ---------------------------
    pos:
        dot positions
    lab:
        color index
    color:
        a list about colors
        
    '''
    
    plt.figure(figsize = [10, 20])
    for index, loc in enumerate(pos):
        plt.scatter(loc[0], loc[1], c = color[lab[index]])
    plt.show()
    return
    

def train(x_train, y_train, n_clusters):
    
    '''
    Use PCA to reduce training data dimension into 2 first. Then use naive_bayes to classify training data into
    'n_clusters' clusters.
    
    parameters
    -------------------------
    x_train:
        training data of power info
    y_train:
        true labels of x_train
    n_clustesrs:
        number of clusters
    
    returns
    -------------------------
    model:
        a tuple, contains a trained pca model and a trained naive_bayes model
    
    '''
    #initialize
    model_pca = PCA(n_components = 2)
    model_KNN = KNeighborsClassifier()
    
    #reduce dimension
    x_train_pca = model_pca.fit_transform(x_train)
    
    #KMeans
    model_KNN.fit(x_train_pca, y_train)
    
    #display trained model
    print('Trained: displaying', dp_per*100, '% data...\n')
    length = round(dp_per*x_train_pca.shape[0])
    display(x_train_pca[0:length,:], model_KNN.predict(x_train_pca)[0:length], color_list[0:n_clusters])
    #display true labels
    print('True: displaying', dp_per*100, '% data...\n')
    display(x_train_pca[0:length,:], y_train[0:length], color_list[0:n_clusters])
    
    return (model_pca, model_KNN)


def predict(x_predict, model):
    
    '''
    Use trained model to predict label of x_predict.
    
    parameters
    ---------------------------
    x_predict:
        data used to fit the model, should be a numpy array with eight columns
    model:
        contains a trained pca model and a trained naive_bayes model
        
    returns
    ---------------------------
    y_predict:
        the predicted result
        
    '''
    
    #check dimension
    if len(x_predict.shape) != 1:
        print('Dimension Error: Expected 1D array.\n')
        return
    
    #reduce dimension
    x_predict_pca = model[0].transform(x_predict.reshape(1, -1))
    
    #predict
    y_predict = model[1].predict(x_predict_pca)
    
    return y_predict


def predict_all(x_predict, y_true, n_clusters, model):
    
    '''
    Use trained model to predict labels of x_predict array.
    
    parameters
    -----------------------------
    x_predict:
        data used to fit the model, should be a numpy array with multiple rows and eight columns
    y_true:
        true labels of x_predict
    n_clusters:
        number of clusters
    model:
        contains a trained pca model and a trained naive_bayes model
    
    returns
    -----------------------------
    y_predict:
        the predicted result, a multiple-row array
        
    '''
    
    #reduce dimension
    x_predict_pca = model[0].transform(x_predict)
    
    #predict
    y_predict = model[1].predict(x_predict_pca)
    
    #display predicted result
    print('Predict: displaying', dpa_per*100, '% data...\n')
    length = round(dpa_per*x_predict_pca.shape[0])
    display(x_predict_pca[0:length,:], y_predict[0:length], color_list[0:n_clusters])
    #display true labels
    print('True: displaying', dpa_per*100, '% data...\n')
    display(x_predict_pca[0:length,:], y_true[0:length], color_list[0:n_clusters])
    
    return y_predict