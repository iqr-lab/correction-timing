import torch
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import random

# pick the simulatipr that has the most similar block dropping results compared to real life.

class PickSim:
    def __init__(self, shape):

        self.shape = shape

        # sim data
        if self.shape == "square":
            directory = '../dropping_blocks/simulator/square_0.03/'
        elif self.shape == "circle":
            directory = '../dropping_blocks/simulator/circle_0.5/'
        else:
            directory = '../dropping_blocks/simulator/'

        total_labels = []
        total_poses = []
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            # Construct the full file path
            filepath = os.path.join(directory, filename)

            # Check if the file has a .pkl extension
            if filename.endswith('.pkl'):
                try:
                    # Open and load the pickle file
                    with open(filepath, 'rb') as file:
                        data = pickle.load(file)
                        print(f"Successfully loaded {filename}")
                        # print("Data:", len(data["triangle"]))  # Print or process the data as needed
                        labels = []
                        poses = []
                        for i in range(len(data[shape])):
                            if data[shape][i]["reachable"] == True:
                                labels.append(data[shape][i]['success'])
                                poses.append(data[shape][i]['pose'])
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                total_labels.append(labels) # append at filename level when it is .pkl!!
                total_poses.append(poses)
        
        self.labels = np.array(total_labels).astype(int) # n_datasets, n_datapoints, turn true and false into 1 and 0
        print(len(self.labels))
        # print(self.labels[0, 0])
        # input()
        self.poses = np.array(total_poses)
        self.num_datasets = self.labels.shape[0]
        self.N = self.labels.shape[1]
        print(self.num_datasets, self.N)
        # input()


        # real life data 

        real_labels = []
        real_poses = []
        real_indices = []
        # Directory containing the real life dropping block pickle files
        filepath = '../dropping_blocks/validated/'+str(self.shape)+'.pkl'
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        for i in range(len(data)):
            label = data[i]["actual"]

            # Check if label is exactly "0" or "1" to exclude corrupted data
            if label in ['0', '1', ' 0']:
                real_labels.append(int(label))  # Convert to integer 0 or 1
                real_poses.append(data[i]["pose"])
                real_indices.append(data[i]["index"])
            elif label == 'c1':
                real_labels.append(int(0))  # Convert to integer 0 or 1
                real_poses.append(data[i]["pose"])
                real_indices.append(data[i]["index"])
            else: # '\x01\x180' case
                # print(f"Skipping invalid label at index {i}: {label!r}")
                print(label)
                real_labels.append(int(1))  # Convert to integer 0 or 1
                real_poses.append(data[i]["pose"])
                real_indices.append(data[i]["index"])
        
        # self.real_labels = np.array(real_labels).astype(int) # string to num
        self.real_labels = np.array(real_labels)
        print(self.real_labels)
        # input()
        self.real_poses = np.array(real_poses)
        self.real_indices = np.array(real_indices)


        top_indices = self.find_best_sim(top_k=10)

        #find file

        all_data = []
        all_filenames = []

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            if filename.endswith('.pkl'):
                try:
                    with open(filepath, 'rb') as file:
                        data = pickle.load(file)
                        all_data.append(data)
                        all_filenames.append(filename)  # Track filename in the same order
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        # best_filename = all_filenames[best_index]
        # print(f"The best matching dataset for {self.shape} is from file: {best_filename}")
        for rank, idx in enumerate(top_indices, 1):
            print(idx, len(all_filenames))
            best_filename = all_filenames[idx]
            print(f"{rank}. Dataset index: {idx}, File: {best_filename}")


    def find_best_sim(self, top_k=5):
        # Step 1: Extract labels at real_indices from all datasets
        # Resulting shape: (n_datasets, sample_num)
        sampled_labels = self.labels[:, self.real_indices]
        print("Sampled labels (first dataset):", sampled_labels[0])
        print("Real labels:", self.real_labels)

        # Step 2: Compare with real_labels using accuracy (number of matches)
        matches = (sampled_labels == self.real_labels)  # shape (n_datasets, sample_num)
        accuracies = matches.mean(axis=1)  # accuracy for each dataset

        # Step 3: Find the top k datasets
        top_indices = np.argsort(accuracies)[-top_k:][::-1]  # indices of top k, sorted descending
        top_accuracies = accuracies[top_indices]

        print(f"Top {top_k} matching datasets for {self.shape}:")
        for rank, (idx, acc) in enumerate(zip(top_indices, top_accuracies), 1):
            print(f"{rank}. Dataset index: {idx}, Accuracy: {acc:.4f}")

        return top_indices



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-sh','--shape', type = str)

    args = parser.parse_args()

    ps = PickSim(shape = args.shape)