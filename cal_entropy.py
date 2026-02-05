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

# picking the most diverse set of samples for testing which simulator for dropping blocks is the closest to the real life

class entropy:
    def __init__(self, shape):


        # Directory containing the pickle files
        directory = '/Users/anjiabei/Documents/research/simulator/square_0.03'

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
            total_labels.append(labels)
            total_poses.append(poses)
        
        self.labels = torch.tensor(total_labels, dtype=torch.int64) # n_datasets, n_datapoints
        self.poses = torch.tensor(total_poses, dtype=torch.float64)
        self.num_datasets = self.labels.shape[0]
        self.N = self.labels.shape[1]
        print(self.num_datasets, self.N)
        input()

        self.cal_prob()

    
  

    def cal_prob(self):

        # Create a tensor of shape (n_dataset, n_dataset, n_labels) for comparisons
        labels_expanded_1 = self.labels.unsqueeze(1)  # Shape: (n_dataset, 1, n_labels)
        labels_expanded_2 = self.labels.unsqueeze(0)  # Shape: (1, n_dataset, n_labels)

        # Compare element-wise (labels match: both 1 or both 0)
        self.matches = labels_expanded_1 == labels_expanded_2  # Shape: (n_dataset, n_dataset, n_labels)
        print(self.matches.shape)
        print(self.labels[:, 100])
        print(self.matches[:,:,100])




    def cal_entropy(self):

        # print(self.labels.T.float().unsqueeze(0).shape)

        # corr_matrix = self.batch_correlation_matrix(self.labels.T.float().unsqueeze(0))[0]

        # X = self.normalize(self.labels.T.float()) # (n_dadapoints, n_datasets)
        X = self.labels.T.double()
        print(X.shape)
        global_mean = X.mean(dim = 0, keepdim= True)
        global_std = X.std(dim = 0, keepdim= True, unbiased=True)
        

        num_sets = 100000 #keep this format
        num_samples = 100

        random_indices = torch.randint(0, self.N, (num_sets, num_samples), dtype=torch.int64) #(num_sets, num_samples), potential repeat

        sampled_data = X[random_indices, ...] #(num_sets, num_samples, n_datasets)
        print(sampled_data.shape, global_mean.unsqueeze(0).shape)
        x = (sampled_data - global_mean.unsqueeze(0))/(global_std.unsqueeze(0)+1e-10)
        x = ((x - x.mean(dim = 1, keepdim = True))/(x.std(dim = 1, keepdim= True, unbiased=False) + 1e-10)).abs()
        c = torch.einsum('ijk,ijl->ikl', x, x) / (num_samples)
        c = (c + c.transpose(-2, -1))/2
        # c = c.abs()
       
        # c = self.batch_correlation_matrix(sampled_data)
        print(c.shape, torch.diagonal(c[0]))



        eigenvalues = torch.linalg.eigvals(c).real
        print(eigenvalues.sum(dim = 1))
        print((eigenvalues/ eigenvalues.shape[1]).shape)
        print(torch.min(eigenvalues), torch.max(eigenvalues))

        # Calculate the sum of eigenvalues for each row (sum over 36 values)
        sum_of_eigenvalues = eigenvalues.sum(dim=1)

        entropy = -((eigenvalues/ eigenvalues.shape[1])* torch.log2(eigenvalues/ eigenvalues.shape[1] + 1e-4)).sum(dim = -1)
        print(entropy[:100])

        max_value, max_index = torch.max(entropy, dim=0)
        max_indicies = random_indices[max_index]
        print("max value", max_value)

        return max_value, max_indicies

      

    def pick_diverse_high_value_indices(self, sorted_dict, feature_matrix, num_samples=100):
        """
        Picks `num_samples` indices prioritizing high-value indices and ensuring feature diversity.

        Args:
            sorted_dict (dict): Sorted dictionary {value: [indices]} in descending order.
            feature_matrix (torch.Tensor): Tensor of shape (N, m) with N samples and m features.
            num_samples (int): Number of diverse samples to pick.

        Returns:
            selected_indices (torch.Tensor): Selected 100 indices ensuring high values and feature diversity.
        """

        all_indices = []
        for _, indices in sorted_dict.items():
            if len(all_indices) <= 100:
                all_indices.extend(indices)

        # only look at the highest entropy
        # first_value = next(iter(sorted_dict))
        # all_indices.extend(sorted_dict[first_value])

        all_indices = torch.tensor(all_indices, dtype=torch.long)  # Convert to tensor
        if len(all_indices) <= num_samples:
            return all_indices  # If fewer than 100, return all
        print(len(all_indices))

        # Start with random high-value index
        selected_indices = [all_indices[random.randrange(0, len(all_indices))]]
        # print(selected_indices)
        # print(self.labels.T[selected_indices])

        for _ in range(num_samples - 1):
            remaining_indices = list(set(all_indices.tolist()) - set(selected_indices))
            remaining_features = feature_matrix[remaining_indices,:].float()  # Get feature vectors
            # print(remaining_features.shape)

            # Compute minimum distance to already selected points
            if len(selected_indices) == 1:
                selected_features = feature_matrix[selected_indices].unsqueeze(0).float()
                # print(selected_features.shape)
            else:
                selected_features = feature_matrix[selected_indices, :].float()
                # print(selected_features.shape)
            min_distances = torch.min(torch.cdist(selected_features, remaining_features, p = 0), dim=0)[0]
            # print(torch.min(torch.cdist(selected_features, remaining_features, p = 0), dim=0)[0].shape)

            # Pick the index that maximizes the minimum distance
            best_idx = remaining_indices[torch.argmax(min_distances).item()]
            selected_indices.append(best_idx)

        return torch.tensor(selected_indices, dtype=torch.long)

    def pick_high_ent(self):

        alignment = self.matches.sum(dim = 1)/self.num_datasets # (n_dataset, n_labels) - alignment for each simulator on label x
        # alignment = alignment/alignment.sum(0) # noromalize
        
        entropy = -(alignment * torch.log2(alignment)).sum(dim=0) # (n_labels) - alignment for label x
        # print(entropy.shape)

        # Sort values and get the indices to access the corresponding data points
        sorted_values, sorted_indices = torch.sort(entropy, descending=True)
        # print(sorted_values[0])

        # Create a dictionary to group indices by their values
        value_groups = {}
        for idx, value in zip(sorted_indices, sorted_values):
            if value.item() not in value_groups:
                value_groups[value.item()] = []
            value_groups[value.item()].append(idx.item())

        selected_samples = self.pick_diverse_high_value_indices(value_groups, self.labels.T)
        # print(selected_samples)

        return selected_samples



    def test_ent(self, indices):

        # entropy, indices = self.cal_entropy()
        test_ent_indecies = self.pick_high_ent()
        test_indecies_half1 = torch.ones(50, dtype=torch.long) 
        test_indecies_half2 = torch.ones(50, dtype=torch.long) * 100
        test_indecies = torch.cat((test_indecies_half1, test_indecies_half2), dim=0)
        test_indecies_2  = torch.randint(0, self.N, (100,), dtype=torch.int64)

        sampled_label = self.labels.T[indices, ...].float()
        test_label_0 = self.labels.T[test_ent_indecies, ...].float()
        test_label_1 = self.labels.T[test_indecies, ...].float()
        test_label_2 = self.labels.T[test_indecies_2[:], ...].float()
        print(test_label_0[0])

        total_label = torch.stack((sampled_label, test_label_0, test_label_1, test_label_2), dim=0)
        print(total_label.shape)

        X = self.labels.T.double()
        global_mean = X.mean(dim = 0, keepdim= True)
        print(global_mean.shape)
        global_std = X.std(dim = 0, keepdim= True, unbiased=True)

        # x = (total_label - global_mean.unsqueeze(0))/(global_std.unsqueeze(0) + 1e-10)
        x = total_label
        print(global_mean.unsqueeze(0).shape,  x.std(dim = 1, keepdim=True, unbiased = False)[0],  x.std(dim = 1, keepdim=True, unbiased = False)[1],  x.std(dim = 1, keepdim=True, unbiased = False)[2], x.std(dim = 1, keepdim=True, unbiased = False)[3])
        x = ((x - x.mean(dim = 1, keepdim = True))/(x.std(dim = 1, keepdim=True, unbiased = False) + 1e-10)).abs()
        print(x.shape)
        # c_star  = torch.einsum('ijk,ijl->ikl', x, x) / (x.shape[1])
        c_star = torch.matmul(x.transpose(-2, -1), x)/ (x.shape[1])
        print(c_star.shape)
        print(torch.diagonal(c_star[0]), torch.diagonal(c_star[1]), torch.diagonal(c_star[2]),  torch.diagonal(c_star[3]))
        eigenvalues = torch.linalg.eigvals(c_star).real
        entropy = -((eigenvalues/ eigenvalues.shape[1])* torch.log2(eigenvalues/ eigenvalues.shape[1] + 1e-4)).sum(dim = -1)
        print(entropy)



    def pick_data(self):

        _, indices = self.cal_entropy()
        self.test_ent(indices)
        # indices = self.pick_high_ent()
        # indices = self.greedy_search()
        # indices = self.kmeans()
        sample_poses = self.poses[0][indices] # poses across all datasets are the same
        print(sample_poses.shape)

        return sample_poses, indices


    def plot_pose(self, ax, position, euler_angles):

        # Position
        x, y, z = position
        roll, pitch, yaw = euler_angles
        
        # Compute rotation matrix
        rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        
        # # Arrow direction combining x and y axes
        # combined_dir = rotation[:, 0] + rotation[:, 1]  # Add x and y axes
        # combined_dir /= np.linalg.norm(combined_dir)  # Normalize

        # Rotate a reference vector to represent orientation
        ref_vector = np.array([0, 0, 1])  # Reference vector in the z direction
        rotated_vector = rotation @ ref_vector  # Apply rotation
        
        # Plot position
        ax.scatter(x, y, z, color='blue', s=50)

        # Plot combined direction as an arrow
        ax.quiver(
            x, y, z, 
            rotated_vector[0], rotated_vector[1], rotated_vector[2], 
            color='purple', length=0.01, normalize=True, label='Combined XY rotation'
        )


    
    def f1_score(self): 

        # f1 score for each simulation labels compared to real life label and pick the index for the sim that returns the max f1 score

        true_labels = torch.randint(0, 2, (100,)).numpy()
        sim_labels = self.get_labels().numpy()
        print(sim_labels[0])

        f1_scores = []
        for labels in sim_labels:
            f1 = f1_score(true_labels, labels, average='micro')
            f1_scores.append(f1)

        max_value = max(f1_scores)
        max_index = f1_scores.index(max_value)

        return f1_scores, max_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--shape', type = str)

    args = parser.parse_args()


    ent = entropy(shape = args.shape)



    sample_data, indices = ent.pick_data()
    print(indices)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all poses
    for pose in sample_data:
        pose[2] = 0.08
        position = np.array(pose[:3])
        euler_angles = np.array(pose[3:])
        ent.plot_pose(ax, position, euler_angles)

    # position = np.zeros(3)
    # euler_angles = np.array([np.pi/2, 0, np.pi/2])
    # ent.plot_pose(ax, position, euler_angles)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display plot
    plt.show()

    data = []
    for i in range(sample_data.shape[0]):
        x = {}
        x["pose"] = np.array(sample_data[i])
        print(sample_data[i][2])
        x["index"] = int(indices[i])
        print(indices[i])
        data.append(x)
    print(data[0]["pose"], data[0]["index"])
    print(data[1]["pose"], data[1]["index"])
    print(data[2]["pose"], data[2]["index"])
    print(data[3]["pose"], data[3]["index"])
    with open("./samples_"+args.shape+".pkl", 'wb') as f:
        pickle.dump(data, f)
    # # f1, index = ent.f1_score()
    # # print(f1, index)
    