# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 23:08:26 2023

@author: PC
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import stft
from db1_preprocess_utils import *



def get_features(data, fs=2048, n_samples=40):
   
    f, t, cstft = stft(data, fs, nperseg=n_samples)
    return np.abs(cstft)


def stft_image(data, samples=40):
   
    cstft_tr = []
   
    for i, idx in enumerate(data):
        a = get_features(data=data[i], fs=2048, n_samples=samples)
        cstft_tr.append(a)
   
    cstft_x_train = np.array(cstft_tr)
    return cstft_x_train



def get_sawt_features(data, depth='db4', axis=1):
    
    wavelet = pywt.Wavelet(depth)
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', axis=-1)

    sawt = [np.mean(c, axis=axis) / np.sqrt(data.shape[1]) for c in coeffs]   
    
    return sawt

def sawt_image(x, depth='db10', axis=1):
    
    data = []
    
    for i, j in enumerate(x):
        datum = get_sawt_features(x[i], depth='db4', axis=1)
        data.append(datum)
    
    return np.array(data)


def get_tkeo_features(data):
    
    tkeo = (data[:, 2:] - data[:, :-2])**2 - data[:, 2:] * data[:, :-2]
    return tkeo



def tkeo_image(x):
    
    data = []
    
    for i, j in enumerate(x):
        datum = get_tkeo_features(x[i])
        data.append(datum)
    
    return np.array(data)
        