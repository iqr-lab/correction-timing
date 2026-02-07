import pickle   
import numpy as np

# adding a key to the data to match where the correction happened for the pre-planned data

class LoadData:
    def __init__(self, shape, legi = "legible", comp = "high"):


        self.shape = shape



        with open('../config/example_data.pkl', 'rb') as file:
            self.training_data = pickle.load(file)
        print(len(self.training_data))

        self.selected_data = self.get_training_data()

    
    def get_training_data(self):

        selected_data = []
        for i in range(len(self.training_data)):
            selected_data.append(self.training_data[i])
        print(len(selected_data))

        return selected_data
        


    def remapping(self):
        # indices = []
        for i in range(len(self.selected_data)):
            if self.selected_data[i]["corrected"] == True:
                waypoints = np.array(self.selected_data[i]["entire_pose_list"]).copy() # keep rotations here
                pre_traj = np.array(self.selected_data[i]["pre_pose_list"]).copy()

                dis_shortest = 10
                index = 0
                for j in range(waypoints.shape[0]):
                    dis = np.linalg.norm(waypoints[j][:3] - pre_traj[-1][:3])
                    if dis < dis_shortest:
                        dis_shortest = dis
                        index = j # index is for the last point before the correction
                    print(dis)
                # print(index, len(waypoints))
                self.selected_data[i]['rescaled_correction_timing'] = index
            # indices.append(index)
        with open('../config/example_data_rescaled.pkl', 'wb') as f:
            pickle.dump(self.selected_data, f)
        # return indices
    




if __name__ == "__main__":

    ld = LoadData(shape = "triangle")

    ld.remapping()