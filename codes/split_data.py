import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
# import shap
from tensorflow.keras.layers import LayerNormalization, Attention
from scipy.stats import zscore
import argparse
from tensorflow.keras.callbacks import Callback
# from tf_explain.core.integrated_gradients import IntegratedGradients

# Split data into training(60%)/val(10%)/testing(30%). Indices for data points within a file. Used for features/where models...


class splitting:
    def __init__(self, timing_ratio, split, target, shape, rep):

        # self.comp = comp
        self.timing_ratio = timing_ratio
        self.split = split
        self.target = target
        self.shape = shape
        self.rep = rep

        with open('../features/corrected_features_'+str(int(100*self.timing_ratio))+'.pkl', 'rb') as file:
            corrected_data_pre = pickle.load(file)
        corrected_data = self.filter_data(corrected_data_pre)

        # self.split_data(corrected_data)
        self.split_indices(data = corrected_data, save_file= '../splits/indices_'+str(int(100*self.timing_ratio))+
                  '_'+str(int(100*self.split))+'_shape_'+self.shape+'_target_'+str(self.target)+'_'+str(int(self.rep))+'.pkl')

    def filter_data(self, data):

        selected_data = []
        for i in range(len(data)):
            # if data[i]["comp"].strip() == str(self.comp).strip() and data[i]["features"] != False:
            # if data[i]["features"] != False and int(data[i]["target"]) == self.target: # features != false for indices!
            if data[i]["features"] != False:
                selected_data.append(data[i])
        print(len(selected_data))
        # input()

        return selected_data
    
  

    def split_indices(self, data, save_file=None):
        """
        Randomly splits indices into train, validation, and test sets, but only for items with self.target.
        Indices refer to the original `data` list.
        Ratios: train 0.6, val 0.1, test 0.3
        """

        seed = self.rep  # repeatable

        # --- Step 1: Find indices of items that match the target ---
        target_indices = [i for i, dp in enumerate(data) if int(dp["target"]) == self.target and dp["shape"] == self.shape]

        print(f"Found {len(target_indices)} items for target {self.target} and shape {self.shape} out of {len(data)} total")

        # --- Step 2: Shuffle indices ---
        np.random.seed(seed)
        target_indices = np.array(target_indices)
        np.random.shuffle(target_indices)

        # --- Step 3: Split into train, val, test ---
        num_total = len(target_indices)
        num_train = int(num_total * self.split)
        num_val = int(num_total * 0.1)
        if num_val < 1:
            num_val = 1 # to at least add one val data point for the example data (less data)
        # test gets the rest
        num_test = num_total - num_train - num_val

        train_indices = target_indices[:num_train].tolist()
        val_indices = target_indices[num_train:num_train+num_val].tolist()
        test_indices = target_indices[num_train+num_val:].tolist()

        print(f"train = {len(train_indices)}, val = {len(val_indices)}, test = {len(test_indices)}")

        # --- Step 4: Save if requested ---
        if save_file is not None:
            with open(save_file, 'wb') as f:
                pickle.dump({
                    "train": train_indices,
                    "val": val_indices,
                    "test": test_indices
                }, f)
            print(f"Saved split indices to {save_file}")

        return train_indices, val_indices, test_indices




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument('-comp','--competency', type = str) # all anyways doesn't matter here
    parser.add_argument('-tr','--timing_ratio', type = float)
    parser.add_argument('-s','--split', type = float) # percentage for training
    parser.add_argument('-tg','--target', type = int)
    parser.add_argument('-sh','--shape', type = str)
    parser.add_argument('-r','--repeat', type = int)

    args = parser.parse_args()

    # for i in range(150):
    #     sp = splitting(timing_ratio=args.timing_ratio, split = args.split, target = args.target, shape = args.shape, rep = i + 50)
    # sp = splitting(timing_ratio=args.timing_ratio, split = args.split, target = args.target, shape = args.shape, rep = args.repeat + 50) # to generate more data 
    sp = splitting(timing_ratio=args.timing_ratio, split = args.split, target = args.target, shape = args.shape, rep = args.repeat)