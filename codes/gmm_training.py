import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import argparse
import random
from scipy.stats import pearsonr, spearmanr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from tqdm import tqdm
# from tqdm.contrib import tqdm_joblib 
from contextlib import contextmanager
from joblib import parallel
import random
import os
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from scipy.stats import lognorm
from scipy.stats import gamma
from scipy.stats import weibull_min
from scipy.stats import genextreme
from scipy.stats import skewnorm
from scipy.stats import beta
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import joblib

# for generating gmm for where people leave the gripper.

class gmm_training():
    def __init__(self, timing_ratio, split, rep):
        """
        Initialize the Metropolis-Hastings algorithm with the given parameters.
        
        Args:
        - n_timesteps (int): Number of time steps in the trajectory.
        - n_goals (int): Number of goals to manage per time step.
        - n_neighbors (int): Number of neighbors to generate for each goal.
        - boself.x_min (tuple): The minimum (x, y, z) coordinates for sampling initial goal positions.
        - box_max (tuple): The maximum (x, y, z) coordinates for sampling initial goal positions.
        - actual_goal_index (int): Index of the actual goal that should be inserted into the goal positions.
        - neighbor_range (float): Range within which to generate neighboring states (3D positions).
        - betap (float): A parameter for external probability calculations.
        """

        # goal positions
        with open('../config/target_position.yaml','r') as file:

            self.target_position = yaml.safe_load(file)
            
        c_gs = np.array([self.target_position['circle']['size_1'][:3],self.target_position['circle']['size_2'][:3],
                        self.target_position['circle']['size_3'][:3],self.target_position['circle']['size_4'][:3]])
        s_gs = np.array([self.target_position['square']['size_1'][:3],self.target_position['square']['size_2'][:3],
                        self.target_position['square']['size_3'][:3],self.target_position['square']['size_4'][:3]])
        t_gs = np.array([self.target_position['triangle']['size_1'][:3],self.target_position['triangle']['size_2'][:3],
                        self.target_position['triangle']['size_3'][:3],self.target_position['triangle']['size_4'][:3]])
        r_gs = np.array([self.target_position['rectangle']['size_1'][:3],self.target_position['rectangle']['size_2'][:3],
                        self.target_position['rectangle']['size_3'][:3],self.target_position['rectangle']['size_4'][:3]])
        
        self.gs = {'circle':c_gs,'square':s_gs,'triangle':t_gs,'rectangle':r_gs}

        self.timing_ratio = timing_ratio
        self.split = split
        self.rep = rep


        # List of target numbers you have in the file names
        target_nums = [0, 1, 2, 3]  # adjust based on your files
        shapes = ["circle", "rectangle", "triangle", "square"]

        train_idx_all = []
        test_idx_all = []

        for target in target_nums:
            for shape in shapes:
                filename = f'../splits/indices_{int(100*self.timing_ratio)}_{int(100*self.split)}_shape_{shape}_target_{int(target)}_{str(int(self.rep))}.pkl'
                
                with open(filename, 'rb') as f:
                    idx_dict = pickle.load(f)
                
                train_idx_all.extend(idx_dict["train"])
                test_idx_all.extend(idx_dict["test"])


        with open('../features/corrected_features_'+str(int(100*self.timing_ratio))+
                  '.pkl', 'rb') as file:
            all_data_pre = pickle.load(file)
        all_data = self.filter_data(all_data_pre)  # corrected data has features = False, needs to be filtered before indexing

        corrected_data = [all_data[i] for i in train_idx_all]
        test_data = [all_data[i] for i in test_idx_all]
        print("corrected length =", len(corrected_data), " test length =",len(test_data))
        self.filtered_data = corrected_data


    def filter_data(self, data):

        selected_data = []
        for i in range(len(data)):
            if data[i]["features"] != False: # important for training
                selected_data.append(data[i])
        print(len(selected_data))

        return selected_data

    def uwhere_dist_leaving(self):
        u_wheres = []

        for j in range(len(self.filtered_data)):
            actual_goal_index = int(self.filtered_data[j]["target"])
            shape = self.filtered_data[j]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

            correction_traj = np.array(self.filtered_data[j]["correction_pose_list"]).copy()
            u_where = correction_traj[-1][:3] - target_pose  # relative pose to goal
            u_wheres.append(u_where)

        u_wheres = np.array(u_wheres)  # Shape: (N, 3)

        gmm_3d, best_k = self.select_gmm_components(u_wheres, max_components=10)
        print(f"Selected number of components: {best_k}")
        # input()

        # --- Fit 3D GMM ---
        gmm_3d = GaussianMixture(n_components=best_k, covariance_type='full')
        gmm_3d.fit(u_wheres)

        # --- Compute joint densities for each data point ---
        densities = np.exp(gmm_3d.score_samples(u_wheres))  # shape: (N,)

        joblib.dump(gmm_3d, '../goal_infer_files/gmms/leaving/gmm_uwhere_leaving_'+str(int(100*self.timing_ratio))+'_'+str(int(100*self.split))+'_'+str(int(self.rep))+'.pkl')

    # def uwhere_dist_grabbing(self): # where people grab the gripper + future projection
    #     u_wheres = []
    #     cor_dirs = []

    #     for j in range(len(self.filtered_data)):
    #         actual_goal_index = int(self.filtered_data[j]["target"])
    #         shape = self.filtered_data[j]["shape"]
    #         target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

    #         correction_traj = np.array(self.filtered_data[j]["correction_pose_list"]).copy()
    #         u_where = correction_traj[0][:3] - target_pose  # relative pose to goal; where people grab the gripper
    #         u_wheres.append(u_where)
    #         cor_dir = correction_traj[1][:3] - correction_traj[0][:3]
    #         cor_dirs.append(cor_dir)

    #     u_wheres = np.array(u_wheres)  # Shape: (N, 3)
    #     cor_dirs = np.array(cor_dirs)  # Shape: (N, 3)

    #     alpha = 20
    #     u_future = u_wheres + alpha * cor_dirs


    #     gmm_3d, best_k = self.select_gmm_components(u_future, max_components=10)
    #     print(f"Selected number of components: {best_k}")
    #     input()

    #     # --- Fit 3D GMM ---
    #     gmm_3d = GaussianMixture(n_components=best_k, covariance_type='full')
    #     gmm_3d.fit(u_future)

    #     # --- Compute joint densities for each data point ---
    #     densities = np.exp(gmm_3d.score_samples(u_future))  # shape: (N,)

    #     # --- 3D scatter plot with arrows ---
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Scatter points colored by density
    #     sc = ax.scatter(u_future[:, 0], u_future[:, 1], u_future[:, 2],
    #                     c=densities, cmap='viridis', s=20, alpha=0.5)
    #     fig.colorbar(sc, label='3D GMM PDF')

    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("Data Points with 3D GMM Density and Correction Directions")
    #     plt.show()
    #     joblib.dump(gmm_3d, './goal_infer_files/gmm_uwhere_grabbing_'+str(int(100*self.timing_ratio))+'_'+str(int(100*self.split))+'.pkl')



    def select_gmm_components(self, data, max_components=10):
        aics, bics, models = [], [], []

        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
            gmm.fit(data)
            models.append(gmm)
            aics.append(gmm.aic(data))
            bics.append(gmm.bic(data))

        best_k = np.argmin(bics)  # or use aics
        best_gmm = models[best_k]

        # # Plot AIC/BIC
        # plt.figure(figsize=(8, 4))
        # plt.plot(range(1, max_components + 1), aics, label='AIC', marker='o')
        # plt.plot(range(1, max_components + 1), bics, label='BIC', marker='o')
        # plt.axvline(best_k + 1, color='gray', linestyle='--', label=f'Best k = {best_k + 1}')
        # plt.xlabel("Number of GMM Components")
        # plt.ylabel("Information Criterion")
        # plt.title("GMM Model Selection")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        return best_gmm, best_k + 1
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument('-comp','--competency', type = str) # all anyways doesn't matter here
    parser.add_argument('-tr','--timing_ratio', type = float)
    parser.add_argument('-s','--split', type = float) # percentage for training
    parser.add_argument('-r','--repeat', type = int)

    args = parser.parse_args()

    gt = gmm_training(timing_ratio=args.timing_ratio, split = args.split, rep = args.repeat)
    gt.uwhere_dist_leaving()
    # gt.uwhere_dist_grabbing()