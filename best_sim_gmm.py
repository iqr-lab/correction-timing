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
from sklearn.mixture import GaussianMixture
import joblib

# the block dropping results from the best sims and get gmm from it (generated files moved to ground truth folder)

class GroundTruthGMM:
    def __init__(self, shape):

        self.shape = shape

        # sim data
        directory = '/Users/anjiabei/Documents/research/simulator/best/'

        # total_labels = []
        # total_poses = []
        poses = []
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            # Construct the full file path
            filepath = os.path.join(directory, filename)

            # Check if the file has a .pkl extension
            if filename.endswith('.pkl'):
                # for shape in ["circle", "square", "triangle", "rectangle"]:
                for shape in [self.shape]:
                    try:
                        # Open and load the pickle file
                        with open(filepath, 'rb') as file:
                            data = pickle.load(file)
                            print(f"Successfully loaded {filename}")
                            # print("Data:", len(data["triangle"]))  # Print or process the data as needed
                            # labels = []
                            # poses = []
                            for i in range(len(data[shape])):
                                # print(data[shape][i])
                                if data[shape][i]["reachable"] == True and data[shape][i]['success'] == True and data[shape][i]["pose"][2] > 0.05: # only taking positives, filter so that z > 0.4 to get spatial distr
                                # if data[shape][i]["reachable"] == True and data[shape][i]['success'] == True and data[shape][i]["pose"][1] > 0.05:
                                    poses.append(data[shape][i]['pose'])
                                    # print('append')
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
            # total_labels.append(labels)
            print(len(poses))
            # total_poses.append(poses)
        
        self.poses = np.array(poses)[:, :3] - np.array([0, 0, 0.08])  # only the positions, offset z by -0.08 so that we have distribition at z=0
        print(self.poses.shape)


    def uwhere_dist(self):

        gmm_3d, best_k = self.select_gmm_components(self.poses, max_components=10)
        print(f"Selected number of components: {best_k}")
        # input()

        # --- Fit 3D GMM ---
        gmm_3d = GaussianMixture(n_components=best_k, covariance_type='diag')
        gmm_3d.fit(self.poses)

        # --- Compute joint densities for each data point ---
        densities = np.exp(gmm_3d.score_samples(self.poses))  # shape: (N,)

        # --- 3D scatter plot with density coloring ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(self.poses[:, 0], self.poses[:, 1], self.poses[:, 2],
                        c=densities, cmap='viridis', s=20, alpha=0.3)
        fig.colorbar(sc, label='3D GMM PDF')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Data Points Colored by 3D GMM Density")
        plt.show()
        joblib.dump(gmm_3d, f'./goal_infer_files/gmm_uwhere_model_ground_truth_{self.shape}.pkl') # this was later moved to the ground truth folder.


    def plot_xy_gmm_at_z0(self, n_grid=11):
        # --- Fit GMM ---
        gmm_3d, best_k = self.select_gmm_components(self.poses, max_components=10)
        gmm_3d = GaussianMixture(n_components=best_k, covariance_type='diag')
        gmm_3d.fit(self.poses)

        # --- Create a grid for x and y ---
        x_min, x_max = self.poses[:, 0].min() - 0.1, self.poses[:, 0].max() + 0.1
        y_min, y_max = self.poses[:, 1].min() - 0.1, self.poses[:, 1].max() + 0.1
        x = np.linspace(x_min, x_max, n_grid)
        y = np.linspace(y_min, y_max, n_grid)
        xx, yy = np.meshgrid(x, y)

        # --- Evaluate GMM on the grid at z = 0 ---
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
        pdf_values = np.exp(gmm_3d.score_samples(grid_points))
        pdf_values = pdf_values.reshape(n_grid, n_grid)

        # --- Plot as heatmap ---
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, pdf_values, levels=50, cmap='viridis')
        plt.colorbar(label='PDF at z=0')
        # plt.scatter(self.poses[:, 0], self.poses[:, 1], c='red', s=10, alpha=0.5, label='Data points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D XY Distribution at Z=0 from 3D GMM')
        plt.legend()
        plt.show()



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

        # Plot AIC/BIC
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, max_components + 1), aics, label='AIC', marker='o')
        plt.plot(range(1, max_components + 1), bics, label='BIC', marker='o')
        plt.axvline(best_k + 1, color='gray', linestyle='--', label=f'Best k = {best_k + 1}')
        plt.xlabel("Number of GMM Components")
        plt.ylabel("Information Criterion")
        plt.title("GMM Model Selection")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return best_gmm, best_k + 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument('-comp','--competency', type = str) # all anyways doesn't matter here
    # parser.add_argument('-tr','--timing_ratio', type = float)
    # parser.add_argument('-s','--split', type = float) # percentage for training
    parser.add_argument('-sh','--shape', type = str)

    args = parser.parse_args()

    gt = GroundTruthGMM(shape = args.shape)
    gt.plot_xy_gmm_at_z0()
    gt.uwhere_dist()