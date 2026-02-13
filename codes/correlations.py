# from pid_cotroller_poses import PIDController
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
from sklearn.cross_decomposition import CCA
import statsmodels.api as sm

# correlation between correction timing and spatial information
# correlation between grasp position and trajectory similarities

class correlation():
    def __init__(self, corrected, timing_ratio):
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

        # load the data
        with open('../config/example_data_rescaled.pkl', 'rb') as file:
            self.training_data = pickle.load(file)
        print(len(self.training_data))

        self.corrected = bool(corrected)
        print(self.corrected)

        self.timing_ratio = timing_ratio
        
        self.selected_data = self.get_training_data(self.corrected)

        if self.corrected == True:
            self.filter_data_corrected()
        else:
            self.filter_data_uncorrected()




    def get_training_data(self, corrected):

        print(corrected)
        # input()

        selected_data = []
        for i in range(len(self.training_data)):
            if self.training_data[i]["corrected"] == corrected:
                selected_data.append(self.training_data[i])
        print(len(selected_data))

        return selected_data
    
    def filter_data_corrected(self):
        self.filtered_data = []
        for ind in range(len(self.selected_data)):
            pre_traj = np.array(self.selected_data[ind]["pre_pose_list"]).copy()
            correction_traj = np.array(self.selected_data[ind]["correction_pose_list"]).copy()
            post_traj = np.array(self.selected_data[ind]["post_pose_list"]).copy()
            if post_traj.shape[0] == 0: # some data points are emppty
                waypoints = np.concatenate((pre_traj, correction_traj), axis=0)
            else:
                waypoints = np.concatenate((pre_traj, correction_traj, post_traj), axis=0)
            planned_traj = np.array(self.selected_data[ind]["entire_pose_list"])
            planned_correction = np.array(self.selected_data[ind]["rescaled_correction_timing"])

            if self.selected_data[ind]["fake_correction"] == False and planned_correction <= self.timing_ratio * len(planned_traj): # even dt
                self.filtered_data.append(self.selected_data[ind])


        print("total corrected data ", len(self.selected_data))
        print("filtered data ", len(self.filtered_data))

    def filter_data_uncorrected(self):

        self.filtered_data = self.selected_data
        print(len(self.filtered_data))
        input()


    def correlation_when_where(self): # where people leave the gipper

        u_wheres = []
        u_whens = []

        for j in range(len(self.filtered_data)):

            actual_goal_index = int(self.filtered_data[j]["target"])
            shape = self.filtered_data[j]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

            correction_traj = np.array(self.filtered_data[j]["correction_pose_list"]).copy()
            u_where = correction_traj[-1][:3] - target_pose # relative pose to goal
            u_wheres.append(u_where)
            planned_correction = np.array(self.filtered_data[j]["rescaled_correction_timing"])
            planned_traj = np.array(self.filtered_data[j]["entire_pose_list"])
            timing_perc = planned_correction/len(planned_traj)
            u_whens.append(timing_perc) # time percentage

        u_wheres = np.array(u_wheres)
        u_whens = np.array(u_whens)

        # Correlation with each coordinate
        corr_x, _ = pearsonr(u_whens, u_wheres[:, 0])
        corr_y, _ = pearsonr(u_whens, u_wheres[:, 1])
        corr_z, _ = pearsonr(u_whens, u_wheres[:, 2])

        # Correlation with overall distance (magnitude)
        pose_magnitudes = np.linalg.norm(u_wheres, axis=1)
        corr_mag, _ = pearsonr(u_whens, pose_magnitudes)

        print(f"Correlation with x: {corr_x:.3f}")
        print(f"Correlation with y: {corr_y:.3f}")
        print(f"Correlation with z: {corr_z:.3f}")
        print(f"Correlation with pose magnitude: {corr_mag:.3f}")


        # Scatter plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].scatter(u_whens, u_wheres[:, 0])
        axs[0].set_title("Timing vs X")
        axs[0].set_xlabel("Correction Time")
        axs[0].set_ylabel("X")

        axs[1].scatter(u_whens, u_wheres[:, 1])
        axs[1].set_title("Timing vs Y")
        axs[1].set_xlabel("Correction Time")
        axs[1].set_ylabel("Y")

        axs[2].scatter(u_whens, u_wheres[:, 2])
        axs[2].set_title("Timing vs Z")
        axs[2].set_xlabel("Correction Time")
        axs[2].set_ylabel("Z")

        plt.tight_layout()
        plt.show()

    def correlation_when_where_grasp(self):

        u_wheres = []
        u_whens = []

        for j in range(len(self.filtered_data)):

            actual_goal_index = int(self.filtered_data[j]["target"])
            shape = self.filtered_data[j]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

            correction_traj = np.array(self.filtered_data[j]["correction_pose_list"]).copy()
            # u_where = correction_traj[0][:3] - target_pose # relative pose to goal
            u_where = correction_traj[0][:3] # relative pose to goal
            u_wheres.append(u_where)
            planned_correction = np.array(self.filtered_data[j]["rescaled_correction_timing"])
            planned_traj = np.array(self.filtered_data[j]["entire_pose_list"])
            # timing_perc = planned_correction/len(planned_traj)
            timing_perc = planned_correction * 0.3
            u_whens.append(timing_perc) # time percentage

        u_wheres = np.array(u_wheres)
        u_whens = np.array(u_whens)

        # Correlation with each coordinate
        corr_x, _ = pearsonr(u_whens, u_wheres[:, 0])
        corr_y, _ = pearsonr(u_whens, u_wheres[:, 1])
        corr_z, _ = pearsonr(u_whens, u_wheres[:, 2])

        # Correlation with overall distance (magnitude)
        pose_magnitudes = np.linalg.norm(u_wheres, axis=1)
        corr_mag, _ = pearsonr(u_whens, pose_magnitudes)

        print(f"Correlation with x: {corr_x:.3f}")
        print(f"Correlation with y: {corr_y:.3f}")
        print(f"Correlation with z: {corr_z:.3f}")
        print(f"Correlation with pose magnitude: {corr_mag:.3f}")


        # Scatter plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].scatter(u_whens, u_wheres[:, 0])
        axs[0].set_title("Timing vs X")
        axs[0].set_xlabel("Correction Time")
        axs[0].set_ylabel("X")

        axs[1].scatter(u_whens, u_wheres[:, 1])
        axs[1].set_title("Timing vs Y")
        axs[1].set_xlabel("Correction Time")
        axs[1].set_ylabel("Y")

        axs[2].scatter(u_whens, u_wheres[:, 2])
        axs[2].set_title("Timing vs Z")
        axs[2].set_xlabel("Correction Time")
        axs[2].set_ylabel("Z")

        plt.tight_layout()
        plt.show()


    def correlation_when_v_grasp(self):

        v_grasps = []
        u_whens = []

        for j in range(len(self.filtered_data)):

            actual_goal_index = int(self.filtered_data[j]["target"])
            shape = self.filtered_data[j]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

            correction_traj = np.array(self.filtered_data[j]["correction_pose_list"]).copy()
            v_grasp = correction_traj[1][:3] - correction_traj[0][:3]
            v_grasps.append(v_grasp)
            planned_correction = np.array(self.filtered_data[j]["rescaled_correction_timing"])
            planned_traj = np.array(self.filtered_data[j]["entire_pose_list"])
            timing_perc = planned_correction * 0.3
            u_whens.append(timing_perc) # time percentage

        # u_wheres = np.array(u_wheres)
        v_grasps = np.array(v_grasps)
        u_whens = np.array(u_whens)

        # Correlation with each coordinate
        corr_x, _ = pearsonr(u_whens, v_grasps[:, 0])
        corr_y, _ = pearsonr(u_whens, v_grasps[:, 1])
        corr_z, _ = pearsonr(u_whens, v_grasps[:, 2])

        # Correlation with overall distance (magnitude)
        pose_magnitudes = np.linalg.norm(v_grasps, axis=1)
        corr_mag, _ = pearsonr(u_whens, pose_magnitudes)

        print(f"Correlation with x: {corr_x:.3f}")
        print(f"Correlation with y: {corr_y:.3f}")
        print(f"Correlation with z: {corr_z:.3f}")
        print(f"Correlation with pose magnitude: {corr_mag:.3f}")


        # Scatter plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].scatter(u_whens, v_grasps[:, 0])
        axs[0].set_title("Timing vs X")
        axs[0].set_xlabel("Correction Time")
        axs[0].set_ylabel("X")

        axs[1].scatter(u_whens, v_grasps[:, 1])
        axs[1].set_title("Timing vs Y")
        axs[1].set_xlabel("Correction Time")
        axs[1].set_ylabel("Y")

        axs[2].scatter(u_whens, v_grasps[:, 2])
        axs[2].set_title("Timing vs Z")
        axs[2].set_xlabel("Correction Time")
        axs[2].set_ylabel("Z")

        plt.tight_layout()
        plt.show()



    def correlation_similarity(self):  # need to be ran on the cluster bc of the data size

        trajs = []
        u_wheres = []

        for j in range(len(self.filtered_data)):

            actual_goal_index = int(self.filtered_data[j]["target"])
            shape = self.filtered_data[j]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

            correction_traj = np.array(self.filtered_data[j]["correction_pose_list"]).copy()
            u_where = correction_traj[-1][:3] - target_pose # relative pose to goal
            u_wheres.append(u_where)
            planned_traj = np.array(self.filtered_data[j]["entire_pose_list"])[:, :3]
            trajs.append(planned_traj)



        # List of all index pairs (i < j)
        n = len(trajs)
        index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]



        # -- Setup --
        results_file = "../results/dtw_endpoint_results.csv"
        done_pairs = set()

        # Load already computed results (if any)
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    done_pairs.add((int(row[0]), int(row[1])))

        # Build list of pairs that haven't been done
        n = len(trajs)
        pending_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in done_pairs]

        # Create file and write header if needed
        if not os.path.exists(results_file) or os.stat(results_file).st_size == 0:
            with open(results_file, "w", newline="") as f_out:
                writer = csv.writer(f_out)
                writer.writerow(["i", "j", "dtw_dist", "endpoint_dist"])

        # Process in chunks of 100
        chunk_size = 100
        for chunk_start in tqdm(range(0, len(pending_pairs), chunk_size), desc="Processing chunks"):
            chunk = pending_pairs[chunk_start : chunk_start + chunk_size]

            results = Parallel(n_jobs=-1)(
                delayed(self.compute_pair)(i, j, trajs, u_wheres) for i, j in chunk
            )

            with open(results_file, "a", newline="") as f_out:
                writer = csv.writer(f_out)
                for result in results:
                    writer.writerow(result)





    def compute_pair(self, i, j, trajs, u_wheres):

        traj1 = trajs[i]
        traj2 = trajs[j]
        dtw_dist, _ = fastdtw(traj1, traj2, dist=euclidean)

        end1 = u_wheres[i]
        end2 = u_wheres[j]
        end_dist = np.linalg.norm(end1 - end2)

        return (i, j, dtw_dist, end_dist)


    def read_csvfile(self):
        # Load the CSV
        df = pd.read_csv("../results/dtw_endpoint_results.csv")

        # Access individual columns
        i_values = df["i"]
        j_values = df["j"]
        dtw_dists = df["dtw_dist"]
        endpoint_dists = df["endpoint_dist"]

        print(len(dtw_dists))
        print(i_values[:20], j_values[:20], dtw_dists[:20], endpoint_dists[:20])
        input()

        r_pearson, p_pearson = pearsonr(dtw_dists, endpoint_dists)
        r_spearman, p_spearman = spearmanr(dtw_dists, endpoint_dists)

        print(f"Pearson r = {r_pearson:.3f}, p = {p_pearson:.3f}")
        print(f"Spearman r = {r_spearman:.3f}, p = {p_spearman:.3f}")









if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--corrected', type = str)
    parser.add_argument('-tr','--timing_ratio', type = float)

    args = parser.parse_args()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    corrected = str2bool(args.corrected)

    cor = correlation(corrected, args.timing_ratio)
    # cor.correlation_when_where()
    # cor.correlation_when_where_grasp()
    cor.correlation_when_v_grasp()
    # cor.correlation_similarity()
    # cor.read_csvfile()

