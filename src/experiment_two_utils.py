# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:18:18 2023

@author: PC
"""


from emgdecompy.decomposition import *
from emgdecompy.contrast import *
from emgdecompy.viz import *
from emgdecompy.preprocessing import *
from db1_preprocess_utils import *
from db2_preprocess_utils import *
from feature_extraction import *

import numpy as np
from sklearn.model_selection import KFold
from scipy.signal import stft
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os



def get_db2_experiment_data(path, subjects, sessions, signal_type, input_type, channels, l_cutoff, h_cutoff,
                        order, window_size, overlap, fs, extend, extend_size, center, whiten,
                        normalize, mu, ratio):
    
    print('####################################################################################') 
    print(f'Loading subject {subjects}')
    print('####################################################################################')
    

    data, label = load_db2_data(path, subjects)
    
    
    win_data, win_label =   aggregate_db2_data_label(data, label, no_channels=channels, signal_type=signal_type,
                        l_cutoff=l_cutoff, h_cutoff=h_cutoff, order=order, window_size=window_size, overlap_per=overlap, 
                        fs=fs, extend_size=extend_size, center=center, extend=extend, 
                        white=whiten, normalize=normalize, mu=mu)
    
    
    if input_type == 'raw':
        
        win_data, win_label = shuffle_data(win_data, win_label)
        win_data, win_label, X_test, y_test = spilt_data(win_data, win_label, ratio)

        win_data = np.expand_dims(win_data, axis=3)
        print("Size of the input data is {}".format(win_data.shape))
        print("The input label shape is {}".format(win_label.shape))
        
        no_classes = len(np.unique(win_label))
        print(f'The total number of classes is {no_classes}')
        
        print('************************************************************************************') 
        print(f'Loaded RAW input data')
        print('************************************************************************************') 
        
        
        return win_data, win_label, X_test, y_test
        
    
    elif input_type == 'stft':
        
        win_data, win_label = shuffle_data(win_data, win_label)
        win_data, win_label, X_test, y_test = spilt_data(win_data, win_label, ratio)
        
        win_data = stft_image(win_data, samples=win_data.shape[2])
        
        print("Size of the input data is {}".format(win_data.shape))
        print("The input label shape is {}".format(win_label.shape))
        no_classes = len(np.unique(win_label))
        print(f'The total number of classes is {no_classes}')
        
        print('************************************************************************************') 
        print(f'Loaded STFT input data')
        print('************************************************************************************')
        
        return win_data, win_label, X_test, y_test #no_classes
    
    
    elif input_type == 'tkeo':
        
        win_data, win_label = shuffle_data(win_data, win_label)
        win_data, win_label, X_test, y_test = spilt_data(win_data, win_label, ratio)
        
        win_data = tkeo_image(win_data)
        win_data = np.expand_dims(win_data, axis=3)
        print("Size of the input data is {}".format(win_data.shape))
        print("The input label shape is {}".format(win_label.shape))
      
        no_classes = len(np.unique(win_label))
        print(f'The total number of classes is {no_classes}')
        
        print('************************************************************************************') 
        print(f'Loaded TKEO input data')
        print('************************************************************************************') 
        return win_data, win_label, X_test, y_test 
    
    else:
        print('Use the right input TYPE .................')
        
        


class Generate_Patches(tf.keras.layers.Layer):
    
    def __init__(self, patch_size):
        super(Generate_Patches, self).__init__()
        self.patch_size = patch_size
    
    
    def call(self, images):
        
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                           sizes=[1, self.patch_size, self.patch_size, 1],
                           strides=[1, self.patch_size, self.patch_size, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
        h_w_c = patches.shape[-1]
        patch_reshape = tf.reshape(patches, [batch_size, -1, h_w_c]) # (batch_size, no_of_patches, h*w*c)
        
        return patch_reshape

    
class Embed_Position(tf.keras.layers.Layer):
    
    def __init__(self, num_patches, projection_dims):
        
        super(Embed_Position, self).__init__()
        
        self.num_patches = num_patches
        self.project = tf.keras.layers.Dense(units=projection_dims)
        self.embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dims)
    
    def call(self, patch):
        
        position = tf.range(start=0, limit=self.num_patches, delta=1)
        encode = self.project(patch) + self.embed(position)
        
        return encode


def mlp_mixer(inputs, no_units, drop_out):
    
    for units in no_units:
        x = tf.keras.layers.Dense(units=units, activation = tf.nn.gelu)(inputs)
        x = tf.keras.layers.Dropout(drop_out)(x)
        # x = tf.keras.layers.Dense(inputs.shape[-1], activation=tf.nn.gelu)(x)
    
    return x


# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(x).shape[0] - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    


def vit_db2_gesture_classification(input_shape, dims, patch_size, no_patches, no_transformer_layers,
                               no_heads, transformer_units, mlp_head_units, no_classes,
                               stochastic_depth,stochastic_depth_rate):
    
    inputs = layers.Input(shape=input_shape)
    # Token embedding.
    
    ##########################################################
    conv_layer = keras.Sequential(
        [
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(dims, (1, 1), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
        ],
        name="covolution_layer",
    )
    
    ##########################################################
    con_emb = conv_layer(inputs)
    # Create patches.
    #patches = Patches(patch_size)(tokenemb)
    # Encode patches.
    #encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    
    #inputs = tf.keras.Input(shape=input_shape)
    #x = tf.keras.layers.BatchNormalization()(inputs)
    #returned_patches = Generate_Patches(patch_size)(inputs)
    returned_patches = Generate_Patches(patch_size)(con_emb)
    encoded_patches = Embed_Position(no_patches, dims)(returned_patches)
    encoded_patches = tf.keras.layers.BatchNormalization()(encoded_patches)
    # Create multiple layers of the Transformer block.
    
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, no_transformer_layers)]
    
    for i in range(no_transformer_layers):
        # Layer normalization 1.
        input_two_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=no_heads, key_dim=dims, dropout=0.1)(input_two_attention, input_two_attention)
        # Skip connection 1.
        if  stochastic_depth:
            input_two_attention_2 = StochasticDepth(dpr[i])(attention_output)
            input_two_attention_2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        else:
            input_two_attention_2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        input_two_attention_3  = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_two_attention_2)
        # MLP.
        input_two_attention_3 = mlp_mixer(input_two_attention_3, no_units=transformer_units, drop_out=0.1)
        
        if stochastic_depth:
            input_two_attention_3 = StochasticDepth(dpr[i])(input_two_attention_3)
            encoded_patches = tf.keras.layers.Add()([input_two_attention_3, input_two_attention_2])
        # Skip connection 2.
        else:
            encoded_patches = tf.keras.layers.Add()([input_two_attention_3, input_two_attention_2])
        
        
# Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp_mixer(representation, no_units=mlp_head_units, drop_out=0.5)
    # Classify outputs.
    
    features1 = tf.keras.layers.Dense(576, activation='relu')(features)
    output = tf.keras.layers.Dense(no_classes, activation='softmax')(features1)

    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=output, name="sEMG-Decomposition")
    
    return model


def CNN_db2_gesture_classification(input_shape, dims, patch_size, no_patches, no_transformer_layers,
                               no_heads, transformer_units, mlp_head_units, no_classes,
                               stochastic_depth,stochastic_depth_rate):
    
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(576, activation='relu')(x)
    outputs = tf.keras.layers.Dense(no_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="HD-sEMG-CNN")
    
    return model


def CNN_LSTM_db2_gesture_classification(input_shape, dims, patch_size, no_patches, no_transformer_layers,
                               no_heads, transformer_units, mlp_head_units, no_classes,
                               stochastic_depth,stochastic_depth_rate):
    
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.LSTM(10, return_sequences=False, activation="tanh")(x)
    outputs = tf.keras.layers.Dense(no_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="HD-sEMG-CNN-lstm")
    
    return model

def LSTM_db2_gesture_classification(input_shape, dims, patch_size, no_patches, no_transformer_layers,
                               no_heads, transformer_units, mlp_head_units, no_classes,
                               stochastic_depth,stochastic_depth_rate):
    
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(150, activation='tanh', input_shape=input_shape)(inputs)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    #x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    #x = tf.keras.layers.TimeDistributed(x)(x)
    #x = tf.keras.layers.ConvLSTM2D(32, 3)
    x =  tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Flatten()(lstm)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(no_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="HD-sEMG-LSTM")
    
    return model

def run_DB2_experiment(n_folds, n_batches, n_epochs, start_subject, total_subject, session, path, 
                       input_type, n_channels, window_size, overlap, extend, extend_size, center, whiten, ratio,
                       noise_db, std, l_cutoff, h_cutoff, fs, order,
                       input_shape, dims, patch_size, n_patches,
                       n_transformer_layers, n_heads, transformer_units,
                       mlp_head_units, n_classes,stochastic_depth,stochastic_depth_rate, type_of_experiment):
   
    
    if  type_of_experiment == 1:
        print('************************************************')
        print('Experiment One DB2')
        print('************************************************')
    elif type_of_experiment == 2:
        print('************************************************')
        print('Experiment Two DB2')
        print('************************************************')
    else:
        print('************************************************')
        print('Enter Valid Experiment')
        print('************************************************')
        
        
    result = pd.DataFrame({
    'Subject': [0],
    'Validation_accuracy': [0],
    'No_noise': [0],
    'precision':[0],
    'recall':[0],
    'f1_score':[0],
    '5_dB': [0],
    'precision_5dB':[0],
    'recall_5dB':[0],
    'f1_score_5dB':[0],
    '10_dB': [0],
    'precision_10dB':[0],
    'recall_10dB':[0],
    'f1_score_10dB':[0],
    '15_dB': [0],
    'precision_15dB':[0],
    'recall_15dB':[0],
    'f1_score_15dB':[0],
    'white_noise':[0],
    'Fold_1': [0],
    'Fold_2': [0],
    'Fold_3': [0]
    })
    
    
    
    for s in range(start_subject, (total_subject+1)):
        
            
        X_train, y_train, X_test, y_test = get_db2_experiment_data(path, subjects=s, sessions=session,
                                                       signal_type='raw', input_type=input_type, 
                                                       channels=n_channels, l_cutoff=l_cutoff, h_cutoff=h_cutoff,
                                                      order=order, window_size=window_size, overlap=overlap, fs=fs, 
                                                      extend=extend,  extend_size=extend_size, 
                                                      center=center, whiten=whiten,
                                                      normalize=False, mu=0, ratio=ratio)

        if input_type == 'raw':
            print('Adding noise to RAW input test data')
            X_test = np.expand_dims(X_test, axis=3)
            X_test_1 = add_noise_all_channel(X_test, noise_db[0], std)
            X_test_2 = add_noise_all_channel(X_test, noise_db[1], std)
            X_test_3 = add_noise_all_channel(X_test, noise_db[2], std)
            print('Adding white noise to RAW input test data')
            X_white = add_white_noise_all_channel(X_test)

            
        elif input_type == 'tkeo':
            print('Adding noise to  TKEO input test data')
            X_test = tkeo_image(X_test)
            X_test = np.expand_dims(X_test, axis=3)
            
            X_test_1 = add_noise_all_channel(X_test, noise_db[0], std)
            X_test_2 = add_noise_all_channel(X_test, noise_db[1], std)
            X_test_3 = add_noise_all_channel(X_test, noise_db[2], std)
            print('Adding white noise to  TKEO input test data')
            X_white = add_white_noise_all_channel(X_test)
            
        elif input_type == 'stft':
            print('Adding noise to STFT input test data')
            
            
            X_test_1 = add_noise_all_channel(X_test, noise_db[0], std)
            X_test_1 = stft_image(X_test_1, samples=X_test_1.shape[2])
            
            X_test_2 = add_noise_all_channel(X_test, noise_db[1], std)
            X_test_2 = stft_image(X_test_2, samples=X_test_2.shape[2])
            
            X_test_3 = add_noise_all_channel(X_test, noise_db[2], std)
            X_test_3 = stft_image(X_test_3, samples=X_test_3.shape[2])
            
            print('Adding white noise to STFT input test data')
            X_white = add_white_noise_all_channel(X_test)
            X_white = stft_image( X_white, samples=X_test.shape[2])
            X_test = stft_image(X_test, samples=X_test.shape[2])
        else:
            print('Use correct input type')
        
        
        ls =  'sparse_categorical_crossentropy'
        mtr = 'accuracy'
        opt = 'adam'
        
        kfold = KFold(n_splits=n_folds, shuffle=False)
        accuracy_per_fold = []
        loss_per_fold = []
        fold_no = 1
        
        convergence_speed = []
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
        
        for train, test in kfold.split(X_train, y_train):
        
            model = vit_db2_gesture_classification(input_shape, dims, patch_size, n_patches,
                                  n_transformer_layers, n_heads, transformer_units,
                                  mlp_head_units, n_classes,stochastic_depth,stochastic_depth_rate)
            
            #model = CNN_db2_gesture_classification(input_shape, dims, patch_size, n_patches,
            #                     n_transformer_layers, n_heads, transformer_units,
            #                   mlp_head_units, n_classes,stochastic_depth,stochastic_depth_rate)
            #cnn_gesture_classification
            #CNN_LSTM_gesture_classification
            
            #model = CNN_LSTM_db2_gesture_classification(input_shape, dims, patch_size, n_patches,
            #                      n_transformer_layers, n_heads, transformer_units,
            #                      mlp_head_units, n_classes,stochastic_depth,stochastic_depth_rate)
            
            
            #model = LSTM_db2_gesture_classification(input_shape, dims, patch_size, n_patches,
            #                      n_transformer_layers, n_heads, transformer_units,
            #                      mlp_head_units, n_classes,stochastic_depth,stochastic_depth_rate)
            
            
            model.compile(optimizer=opt, loss=ls, metrics=mtr)
            
            print('---------------------------------------------------')
            print(f'Training for fold {fold_no} -------')
            
            history = model.fit(X_train[train], y_train[train], batch_size=n_batches, 
                                epochs= n_epochs, verbose=1, 
                                callbacks=callback)
            
            conv_speed = len(history.history['accuracy'])
            convergence_speed.append(conv_speed)
            
            scores = model.evaluate(X_train[test], y_train[test], verbose=0)
            print(f'Score for fold  {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
            accuracy_per_fold.append(scores[1] *100)
            loss_per_fold.append(scores[0])
                  
            fold_no = fold_no + 1
        
        print("Average Score per fold ")
    
        for i in range(0, len(accuracy_per_fold)):
            print('-----------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {accuracy_per_fold[i]}%')
        print('-----------------------------------------------')
        print('Average Metrics for all folds: ')
        print(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('-----------------------------------------------')
        
        print('************************************************')
        print(f'For subject {s} without noise')
        scores_0 = model.evaluate(X_test, y_test, verbose=0)
        print('The loss is {} and accuracy is {}%'.format(scores_0[0], (scores_0[1]*100)))
        print(f'For subject {s} and noise 5 dB')
        scores_1 = model.evaluate(X_test_1, y_test, verbose=0)
        print('The loss is {} and accuracy is {}%'.format(scores_1[0], (scores_1[1]*100)))
        print(f'For subject {s} and noise 10 dB')
        scores_2 = model.evaluate(X_test_2, y_test, verbose=0)
        print('The loss is {} and accuracy is {}%'.format(scores_2[0], (scores_2[1]*100)))
        print(f'For subject {s} and noise 15 dB')  
        scores_3 = model.evaluate(X_test_3, y_test, verbose=0) 
        print('The loss is {} and accuracy is {}%'.format(scores_3[0], (scores_3[1]*100)))
        print('************************************************') 
        

        label_pred = model.predict(X_test)
        label_pred = np.argmax(label_pred, axis=-1)
        prec = precision_score(y_test, label_pred, average='weighted')
        recal = recall_score(y_test, label_pred, average='weighted')
        f_score = f1_score(y_test, label_pred, average='weighted')
        
        
        label_pred_5dB = model.predict(X_test_1)
        label_pred_5dB = np.argmax(label_pred_5dB, axis=-1)
        prec_5dB = precision_score(y_test, label_pred_5dB, average='weighted')
        recal_5dB = recall_score(y_test, label_pred_5dB, average='weighted')
        f_score_5dB = f1_score(y_test, label_pred_5dB, average='weighted')
        
        label_pred_10dB = model.predict(X_test_2)
        label_pred_10dB = np.argmax(label_pred_10dB, axis=-1)
        prec_10dB = precision_score(y_test, label_pred_10dB, average='weighted')
        recal_10dB = recall_score(y_test, label_pred_10dB, average='weighted')
        f_score_10dB = f1_score(y_test, label_pred_10dB, average='weighted')
        
        
        label_pred_15dB = model.predict(X_test_3)
        label_pred_15dB = np.argmax(label_pred_15dB, axis=-1)
        prec_15dB = precision_score(y_test, label_pred_15dB, average='weighted')
        recal_15dB = recall_score(y_test, label_pred_15dB, average='weighted')
        f_score_15dB = f1_score(y_test, label_pred_15dB, average='weighted')
        
        scores_4 = model.evaluate(X_white, y_test, verbose=0) 
        label_pred_white = model.predict(X_white)
        label_pred_white = np.argmax(label_pred_white, axis=-1)
        prec_white = precision_score(y_test, label_pred_white, average='weighted')
        recal_white = recall_score(y_test, label_pred_white, average='weighted')
        f_score_white = f1_score(y_test, label_pred_white, average='weighted')
        
        result.at[s-1, 'Subject'] =  s
        result.at[s-1, 'Validation_accuracy'] =  np.mean(accuracy_per_fold)       
        result.at[s-1, 'No_noise'] =  scores_0[1]*100     
        
        result.at[s-1, 'precision'] =  prec
        result.at[s-1, 'recall'] =  recal
        result.at[s-1, 'f1_score'] =  f_score

         
        result.at[s-1, '5_dB'] =  scores_1[1]*100  
        result.at[s-1, 'precision_5dB'] =  prec_5dB
        result.at[s-1, 'recall_5dB'] =  recal_5dB
        result.at[s-1, 'f1_score_5dB'] =  f_score_5dB
        result.at[s-1, '10_dB'] =  scores_2[1]*100
        result.at[s-1, 'precision_10dB'] =  prec_10dB
        result.at[s-1, 'recall_10dB'] =  recal_10dB
        result.at[s-1, 'f1_score_10dB'] =  f_score_10dB
        result.at[s-1, '15_dB'] =  scores_3[1]*100   
        result.at[s-1, 'precision_15dB'] =  prec_15dB
        result.at[s-1, 'recall_15dB'] =  recal_15dB
        result.at[s-1, 'f1_score_15dB'] =  f_score_15dB
        result.at[s-1, 'white_noise'] =  scores_4[1]*100
        result.at[s-1, 'Fold_1'] =  convergence_speed[0]
        result.at[s-1, 'Fold_2'] =  convergence_speed[1]
        #result.at[s-1, 'Fold_3'] =  convergence_speed[2] 
        
        save_path =  'DB2_Experiment_'+ str(type_of_experiment)+'_'+input_type+'_'+str(start_subject)+ '_to_'+str(total_subject)+'_CViT_TKEO.csv'
        save_dir = os.path.join('../results', save_path)
        result.to_csv(save_dir, index=False)