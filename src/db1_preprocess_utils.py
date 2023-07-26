# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:11:59 2022

@author: PC
"""

from scipy.io import loadmat
import pickle
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, iirnotch, filtfilt
from scipy.signal import stft

from emgdecompy.decomposition import *
from emgdecompy.contrast import *
from emgdecompy.viz import *
from emgdecompy.preprocessing import *
from sklearn.model_selection import train_test_split

##### Preprocessing of the data ################

# Discard unwanted channels
def delete_channel(x, discard, axis):
    
    # discard --> Indicies of channels to discard 
    if discard != None:
        x = np.delete(x, discard, axis=axis)
    return x


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos


def butterworth_bandpass_filter(data, axis=0, lowcut=10, highcut=500, fs=2048, order=6):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    
    y = sosfilt(sos, data, axis=axis)
    return y



def window_data(data, label, sampling_frequency=2048, window_time=250, overlap=50):
    # Calculate the number of samples in each window
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
        window = data[:, current_position:current_position+window_samples]
      
        # Add the window to the list
        #windowed_data[counter] = window
        windowed_data.append(window)
        windowed_label.append(label)
        
        # Advance the current position by the number of non-overlapping samples
        current_position += nonoverlap_samples
      #counter += 1
    
    # Return the windowed data as a N
    return np.array(windowed_data), np.array(windowed_label)
  

def load_single_subject_session(path, subject=1, session=1):
    
    if subject < 10:
        data_path = path + "\\subject_0" + str(subject) + "\\session_" + str(session) + "\\data.mat"
        label_path = path + "\\subject_0" + str(subject) + "\\session_" + str(session) + "\\label.mat"
    
    else:
        data_path = path + "\\subject_" + str(subject) + "\\session_" + str(session) + "\\data.mat"
        label_path = path + "\\subject_" + str(subject) + "\\session_" + str(session) + "\\label.mat"
        
    raw = loadmat(data_path)['data']
    raw_label = loadmat(label_path)['label']
    
    
    after_flatten = flatten_signal(raw)  # shape is (trial gesture, channels, datapoints)
    label = flatten_signal(raw_label)
    label = label.reshape(label.shape[0], 1)
    after_deleting = delete_channel(after_flatten, discard=None, axis=1) # Delete bad channel
    
    return after_deleting, label

def load_all_subjects(path, subjects=20, sessions=1):
    
    #all_label = []
    #all_data = []
    
    for su in range(1,(subjects+1)):
        print(f"Loading subject {su}")
        data, label = load_single_subject_session(path, subject=su, session=sessions)
        #label = label.reshape(label.shape[0], 1)
        #all_data.append(data)
        #all_label.append(label)

        if su == 1:
            all_data =  data
            all_label = label
            #print(all_label.shape)

        else:
            all_data = np.row_stack((all_data, data))
            all_label = np.row_stack((all_label, label))

    
    return all_data, all_label
        
        

def aggregate_data_label(data, label, no_channels=256, signal_type='raw', l_cutoff=20, h_cutoff=500, order=6,
                         window_size=250, overlap_per=50, fs=2048, extend_size=0, center=True, 
                         extend=True, white=True, normalize=True, mu=0.05):

    count = len(label)  # looping throush label
    cnt = 0
    start = 0
    while cnt < count:
        #print(cnt)
        
        if signal_type == 'raw':
            
            
            try:
                if label[cnt]==label[(cnt + 1)]:
                    return_data = np.concatenate((data[cnt,:no_channels,:], data[(cnt + 1), :no_channels,:]), axis=1) # concat after deleting
                    return_data  = butterworth_bandpass_filter(return_data , axis=1, lowcut=l_cutoff, highcut=h_cutoff,
                                                               fs=fs, order=order)
                    reindex_label = label[cnt] - 1
                    #print(f"Label is {reindex_label}")
                    cnt = cnt + 2
                    # print('Filter is low {} and {}'.format(l_cutoff, h_cutoff))
                else:
                    return_data = data[cnt, :no_channels,:]
                    return_data  = butterworth_bandpass_filter(return_data, axis=1, lowcut=l_cutoff, highcut=h_cutoff,
                                                               fs=fs, order=order)
                    reindex_label = label[cnt] - 1
                    #print(f"Label is {reindex_label}")
                    cnt = cnt + 1
                    # print('Filter is low {} and {}'.format(l_cutoff, h_cutoff))
            
            except IndexError:
                return_data = data[cnt, :no_channels,:]
                return_data  = butterworth_bandpass_filter(return_data, axis=1, lowcut=l_cutoff, highcut=h_cutoff,
                                                           fs=fs, order=order)
                reindex_label = label[cnt] - 1
                #print(f"Label is {reindex_label}")
                cnt = cnt + 1
                    
                
        
        elif signal_type == 'preprocess':
            
            if label[cnt] == label[(cnt + 1)]:
                return_data = np.concatenate((data[cnt,:no_channels,:], data[(cnt + 1), :no_channels, :]), axis=1) # concat after deleting
                #return_data  = butterworth_bandpass_filter(return_data , axis=1, lowcut=l_cutoff, highcut=h_cutoff, fs=fs, order=6)
                reindex_label = label[cnt] - 1
                #print(f"Label is {reindex_label}")
                cnt = cnt + 2
            else:
                return_data = data[cnt, :no_channels,:]
                #return_data  = butterworth_bandpass_filter(return_data, axis=1, lowcut=l_cutoff, highcut=h_cutoff, fs=fs, order=6)
                reindex_label = label[cnt] - 1
                #print(f"Label is {reindex_label}")
                cnt = cnt + 1
            
        
        if normalize:
            return_data = mu_law_normalization(return_data, mu)
            print("Normalize. ")
        
        # Center
        if center:
            return_data = center_matrix(return_data)  # make the channel datapoint centered around zero # Experiment with this
            #return_data = centered_data
            #center = False
            if start == 0:
                print("Centred.")
                
        # Extend
        if extend:
            return_data = extend_all_channels(return_data , extend_size)  # returns in shape ((channels * (extend + 1)),  datapoints)
            # no_channels = return_data.shape[0]
            # return_data = extended_data
            # extend = False
            if start == 0:
                print("Extended.")
        
        
        
        # Whiten
        if white:
            return_data = whiten(return_data)
            #return_data = whiten_data
            #white = False
            if start == 0:
                print("Whiten.")
            
        #print('####################################################################################') 
        #print(f'Concatenated label {reindex_label}')
        #print('####################################################################################') 
        windowed_data, windowed_label = window_data(return_data, reindex_label, sampling_frequency=fs, window_time=window_size, overlap=overlap_per)
        # print("Windowed data at point {} and point {}".format(i, (i+1)))
        # print("Correpsonding label is {}".format(label[i]))
       
        if  start == 0:
            data_stack = windowed_data
            label_stack = windowed_label
            # print("Segementing from first block")

        else:
            data_stack = np.row_stack((data_stack, windowed_data))
            label_stack = np.row_stack((label_stack, windowed_label))
        
        start += 1
        
    
    return data_stack, label_stack.flatten()


def seperate_data(data, label, no_channels=256, fs=2048, window_time=250, overlap=50):
    
    
    count = len(label)  # looping throush label
    cnt = 0
    end = 8192
    increment = 8192
    start = 0
    start_seg = 0
    
    while cnt < count:
        
        if label[cnt] == label[(cnt + 1)]:
            
            #print("Inside first block")
            end = end + increment
            #print(f"End block is {end}")
            return_data = data[:, start_seg:end]
            reindex_label = label[cnt] - 1
            
            cnt = cnt + 2
            start_seg = end
            end = end + increment
            
        
        else:
            #print("Inside second block")
            return_data = data[:, start_seg:end]
            reindex_label = label[cnt] - 1
            cnt = cnt + 1
            
            start_seg = end
            end = end + increment
        
        windowed_data, windowed_label = window_data(return_data, reindex_label, channels=no_channels, sampling_frequency=fs, window_time=window_time, overlap=overlap)
        #print("Windowed data shape is  {}".format(g))
        #print("Correpsonding label is {}".format(label[i]))
       
        if  start == 0:
            data_stack = windowed_data
            label_stack = windowed_label
            #print("Segementing from first block")

        else:
            data_stack = np.row_stack((data_stack, windowed_data))
            label_stack = np.row_stack((label_stack, windowed_label))
        
        start += 1
    
    return data_stack, label_stack.flatten()
            
            
        
    
def stack_all_data(data):
    
    length = len(data)
    #stack_len = data.shape[2] * data.shape[0]
    
    #all_data = np.zeros((data.shape[1], stack_len))
    
    for i in range(length):
        
        idx = i + 1 
        ds = data[i].transpose()
        if idx == 1:
           all_data = ds #data[i]
        
        else:
           all_data = np.row_stack((all_data, ds))
    
    return all_data.transpose()


def bss_preprocess(data, l_cut=10, h_cut=900, fs=2048, order=6, extension=4):
    
    data  = butterworth_bandpass_filter(data , axis=1, lowcut=l_cut, highcut=h_cut, fs=fs, order=order)
    data = extend_all_channels(data , extension)
    data = center_matrix(data)
    data = whiten(data)
    
    return (data)



def shuffle_data(data, label):
    
    idx = np.random.permutation(len(data))
    x,y = data[idx], label[idx]
    
    return x, y



def mu_law_normalization(x, mu):
    
    y = np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
    
    
    return y


def normalize_data(data, mu):
    
    normalized_data = np.empty_like(data)
    
    for i in range(data.shape[0]):
        
        normalized_data[i] = mu_law_normalization(data[i], mu)
    
    return normalized_data


def add_noise_in_db(data, noise_level_db, std):
    
    
    linear_scale = 10 ** (noise_level_db / 10)

   
    noise = np.random.normal(0, std, len(data))

    # Scale noise signal to desired level
    scaled_noise = np.multiply(noise, linear_scale)

    # Add noise to EMG signal
    data = data.transpose()
    noisy_emg = np.add(data, scaled_noise)

    return noisy_emg.transpose()


def add_noise_all_channel(data, noise_db, std):
    
    noisy_data = np.empty_like(data)
    
    for i in range(data.shape[0]):
        
        noisy_data[i] = add_noise_in_db(data[i], noise_db, std)
    
    return noisy_data


def spilt_data(data, label, ratio):
    
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=ratio, random_state=42) 
    
    return X_train, y_train, X_test, y_test


def white_noise(data):
    
    # Add white noise with amplitude 0.5 and mean 0
    noise_amplitude = 0.5
    noise = noise_amplitude * np.random.randn(len(data))
    data = data.transpose()
    noisy_signal = np.add(data, noise)
    
    
    return noisy_signal.transpose()

def add_white_noise_all_channel(data):
    
    noisy_data = np.empty_like(data)
    
    for i in range(data.shape[0]):
        
        noisy_data[i] = white_noise(data[i])
    
    return noisy_data


def min_max_normalize(data):
    min_val = np.min(data, axis=1)
    max_val = np.max(data, axis=1)
    
    normalized_data = (data.transpose() - min_val) / (max_val - min_val)
    return normalized_data.transpose()
    


###################################################









