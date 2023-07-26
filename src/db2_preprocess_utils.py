# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:16:27 2023

@author: PC
"""

from scipy.io import loadmat
import pickle
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, iirnotch, filtfilt
from scipy.signal import stft
import mat73
import os

from emgdecompy.decomposition import *
from emgdecompy.contrast import *
from emgdecompy.viz import *
from emgdecompy.preprocessing import *
from sklearn.model_selection import train_test_split


from db1_preprocess_utils import *


def load_db2_data(path, subject):
    
    files = os.listdir(path)
    subject_file = list(filter(lambda file: file[2] == str(subject), files))
    
    
    for i, file in enumerate(subject_file):
        data_path = os.path.join(path, file)
        data = loadmat(data_path)['data']
        data = data[1500:-1500]
        label = loadmat(data_path)['label'].reshape(-1)
        label = label[1500:-1500]
        
        if label[0] <= 150:
            label = label - 100
        else:
            label = label - 188
            
        
        if i == 0:
            all_data =  data
            all_label = label
        else:
            all_data = np.row_stack((all_data, data))
            all_label = np.hstack((all_label, label))
    
    return all_data.transpose(), all_label


def window_data_db2(data, label, sampling_frequency=2048, window_time=250, overlap=50):
    
    

    label = np.array(label)
    
    window_samples = int(sampling_frequency * (window_time / 1000))
    
    # Calculate the number of overlapping samples
    overlap_samples = int(window_samples * (overlap/100))
    
    # Calculate the number of non-overlapping samples
    nonoverlap_samples = window_samples - overlap_samples
    
    # number_of_samples = int(data.shape[1] / nonoverlap_samples)
    
    # Initialize a list to store the windowed data
    #windowed_data = np.zeros([number_of_samples, data.shape[0], window_samples])
    windowed_data = []
    windowed_label = []
    
    # Initialize a variable to keep track of the current position in the data
    current_position = 0
    #counter = 0
 
  
     # Loop through the data, windowing it into overlapping segments
    while current_position + window_samples <= data.shape[1]:
        # Extract a window of data
        window = data[: , current_position:current_position + window_samples]
        label_win = int(np.mean(label[current_position:current_position + window_samples]))
        # Add the window to the list
        #windowed_data[counter] = window
        windowed_data.append(window)
        windowed_label.append(label_win)
        
        # Advance the current position by the number of non-overlapping samples
        current_position += nonoverlap_samples
        #counter += 1

# Return the windowed data as a N
    return np.array(windowed_data), np.array(windowed_label)



def aggregate_db2_data_label(data, label, no_channels=256, signal_type='raw', l_cutoff=20, h_cutoff=500, order=6,
                         window_size=250, overlap_per=50, fs=1000, extend_size=0, center=True, 
                         extend=True, white=True, normalize=True, mu=0.5):

    #Filtering 
    return_data  = butterworth_bandpass_filter(data , axis=1, lowcut=l_cutoff, highcut=h_cutoff,
                                                               fs=fs, order=order)

    # min-max normalization 
    #return_data = min_max_normalize(return_data)
    #print("Min-Max Normalize.")
    
    if normalize:
        return_data = mu_law_normalization(return_data, mu)
        print("Normalize.")
    
    # Center
    if center:
        return_data = center_matrix(return_data)
        print("Center.")       
            
    # Extend
    if extend:
        return_data = extend_all_channels(return_data , extend_size) 
        print("Extended.")
        
    # Whiten
    if white:
        return_data = whiten(return_data)
        print("Whiten.")
        
    windowed_data, windowed_label = window_data_db2(return_data, label, sampling_frequency=fs, window_time=window_size, overlap=overlap_per)


    return windowed_data, windowed_label   