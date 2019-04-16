#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
from scipy import stats

def datapack_extender(pack):
    '''
    Convert a one-dimensional array which contains power information into eight-dimensional array.
    The eight dimensions include mean， median， mode, variance， ptp, kurtosis, skewness and quartile deviation.
    
    parameters
    ---------------------------
    pack:
        numpy array of one dimension, contains a set of raw data
    
    returns
    ---------------------------
    result:
        numpy array of one row and eight columns
        
    '''
    
    #calculate components
    mean = np.mean(pack)
    median = np.median(pack)
    mode = stats.mode(pack)[0][0]
    var = np.var(pack)
    ptp = np.ptp(pack)
    kurtosis = stats.kurtosis(pack)
    skew = stats.skew(pack)
    quart = np.percentile(pack, 75) - np.percentile(pack, 25)
    
    #combine eight components into a one-row array
    result = np.array((mean, median, mode, var, ptp, kurtosis, skew, quart), dtype = np.float32)
    return result


def data_extender(raw, gap):
    '''
    Divide the 'raw' array into sections. Each section contains 'gap' rows. Then call 'datapack_extender' to 
    convert each section into a one-row-wide and eight-columns-long array.
    
    parameters
    -----------------------------
    raw:
        raw data, numpy array of one dimension, contains power information
    gap:
        division scale
    
    returns
    -----------------------------
    result:
        numpy array of one row and eight columns
        
    '''
    
    #check whether 'raw' is a one-dimensional array
    if len(raw.shape) != 1:
        print('Dimension Error: the array provided is not a one-dimensional array.\n')
        return
    
    #divide and initialize array
    result = np.zeros((math.floor(len(raw)/gap),8), dtype = np.float32)
    
    #call datapack_extender and fulfill the array
    index = 0
    while (index+1)*gap < len(raw):
        result[index,:] = datapack_extender(raw[index*gap:(index+1)*gap])
        index = index + 1
        
    return result


def extend_and_combine(gap, *raws):
    
    '''
    Extend raw datasets of different meters and combine them into one dataset.

    parameters
    -----------------------------
    gap:
        division scale
    *raws:
        raw datasets, contain uncertain numbers of numpy arrays

    returns
    -----------------------------
    dataset:
        a numpy array of eight columns, a combination of power info of different meters
    labels:
        a numpy array of one dimension, a combination of types of different meters

    '''
    #initialize 
    dataset = -1*np.ones((1,8), dtype = np.float32)
    labels = -1*np.ones((1,), dtype = np.int32)
    
    #add meters
    for index, data in enumerate(raws):
        #extend new dataset
        print('extending dataset', index, ' size:', data.shape, '\n')
        data_cur = data_extender(data, gap)
        labels_cur = index*np.ones((data_cur.shape[0],), dtype = np.int32)
        #combine
        print('combining dataset', index, ' size:', data_cur.shape, '\n')
        dataset = np.vstack((dataset, data_cur))
        labels = np.hstack((labels, labels_cur))
    
    #delete first rows
    dataset = np.delete(dataset, 0, 0)
    labels = np.delete(labels, 0, 0)
    
    print('done!')
    return dataset, labels
        

