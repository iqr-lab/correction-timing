from pid_controller_poses import PIDController
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import argparse
import random
from tqdm import tqdm
import itertools
import os
from multiprocessing import Pool
import bz2

# extra features for trajectory based on different potential goal poses (dx = dy = 0.01m for the targets) for corrected trajectories

class extract_features():
    def __init__(self, timing_ratio, q, r):
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
        # with open('../../experiment/config/target_position.yaml','r') as file:
        with open('./target_position.yaml','r') as file:

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
        # self.split = split
        self.q = q
        self.r = r
        self.s = 0.0


        with open(f'./corrected_features_{str(int(100*self.timing_ratio))}.pkl', 'rb') as file: # all the data
            self.training_data = pickle.load(file)
        print(len(self.training_data))

        self.timing_ratio = timing_ratio

        self.filtered_data = self.filter_data(self.training_data)





    def filter_data(self, data):

        selected_data = []
        for i in range(len(data)):
            # if data[i]["comp"].strip() == str(self.comp).strip() and data[i]["features"] != False:
            if data[i]["features"] != False: # important for training
                selected_data.append(data[i])
        print(len(selected_data))
        # input()

        return selected_data

    def filter_data_uncorrected(self):

        # filter and pick the same amount of uncorrected data points
        
        # random.shuffle(self.selected_data)
        self.filtered_data = self.selected_data
        print(len(self.filtered_data))
        input()


    def normalized_dot_product(self, v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)

        dot = np.dot(v1, v2)
        # norm_factor = max(np.linalg.norm(v1)**2, np.linalg.norm(v2)**2)
        norm_factor = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_factor == 0:
            return 0.0  # Avoid division by zero for zero vectors

        return dot / norm_factor

    
    def get_features_corrected(self):

        # Define initial parameters and settings
        P, I, D = 10, 0, 5
        epsilon = 0.01  # Threshold for stopping (in meters)
        max_cmd = np.array([10.0, 10.0, 10.0])  # Max acceleration for each axis (m/s²)

        # Create the PID controller object
        pid_controller = PIDController(P, I, D, epsilon, max_cmd)

        self.offset = np.array([self.q, self.r, self.s]) # offset is the opposite of the actual relative possition

        for m in range(len(self.filtered_data)):
        # for m in [233, 344, 345, 433]: # these will have none values so skip
            ind = m 
            print("traj ", ind)
            self.filtered_data[ind]["features"] = []

            # self.filtered_data[ind]["features"] = {}



            actual_goal_index = int(self.filtered_data[ind]["target"])
            shape = self.filtered_data[ind]["shape"]
            center = np.array([self.gs[shape][0][0], 0, 0])
            target_pose = center - self.offset # dont know where the goal is but the relative position wrt offset off center; for testing

            # Set the target pose for the PID controller
            pid_controller.set_target_pose(target_pose)

            # Define the initial position and velocity of the end effector
            pre_traj = np.array(self.filtered_data[ind]["pre_pose_list"])[:,:3]
            planned_correction = np.array(self.filtered_data[ind]["rescaled_correction_timing"])
            post_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[planned_correction + 1:, :3] # +1 for after the last point of the actual traj
            planned_traj = np.concatenate((pre_traj, post_traj), axis=0)
            # if np.array(self.filtered_data[ind]["pre_pose_vel"]).ndim == 1:
            #     self.filtered_data[ind]["features"] = False # correction happns immediately
            #     # input()
            #     continue
            # pre_traj_vel = np.array(self.filtered_data[ind]["pre_pose_vel"])[:,:3]

            timestamp = np.array(self.filtered_data[ind]["pre_timestamp"])
            dts0 = timestamp[1:] - timestamp[:-1] # len(pre_traj) - 1

            # Define the time step for integration
            dt = 0.05  # Time step (seconds)

            alignment = [] # velocity alignment (optimal and current)
            alignment_2 = [] # velocity alignment (previous and current)
            distance = [] # distance to goal
            v = [] # velocity magnitude
            time = [] # time

            legi = [] # legibility
            opt = [] # optimality
            acc = [] # acceleration magnitude
            jerk = [] # jerk magnitude
            curv = [] # curvature
            alignment_3 = [] # velocity alignment (towards goal and current)
            pose_alignment = [] # position alignment (optimal and current)

            new_time = 0
            # for time list
            for i in range(planned_traj.shape[0] - 1):

                if i <= pre_traj.shape[0] - 2: # should -1 here for indexing and -1 for one less length
                    # current_velocity = pre_traj_vel[i]
                    delta_t = dts0[i]
                else:
                    # current_velocity = (planned_traj[i+1] - planned_traj[i])/0.3 # Starting velocity
                    delta_t = 0.3

                time.append(new_time) # first tine is 0, always the previous time
                new_time += delta_t

                


            prev_velocity = np.zeros([3])
            prev_acceleration = np.zeros([3])
            # new_time = 0

            for i in range(planned_traj.shape[0] - 1): # length for all lists
                pid_controller.set_target_pose(target_pose) # reset controller every time
                current_pos = planned_traj[i]
                # print(current_pos)
                # input()

                dis = np.linalg.norm(current_pos - target_pose)

                if i <= pre_traj.shape[0] - 1: # for calculating the v at the transition point as the pos_diff/delta_t
                    # current_velocity = pre_traj_vel[i]
                    current_velocity = (planned_traj[i+1] - planned_traj[i])/dts0[0] # use traj itself to cal v

                else:
                    current_velocity = (planned_traj[i+1] - planned_traj[i])/0.3 # Starting velocity

                if i == 0:
                    current_acceleration = (current_velocity - prev_velocity)/dts0[0] # off by one compared to v, its the one for the timestep before
                    current_jerk = (current_acceleration - prev_acceleration)/dts0[0]

                elif i == pre_traj.shape[0]: # transition of velocity happening
                    current_acceleration = (current_velocity - (planned_traj[i] - planned_traj[i - 1])/0.3 )/0.3 # two different v values for different time steps
                    current_jerk = (current_acceleration - prev_acceleration)/0.3
                else:
                    current_acceleration = (current_velocity - prev_velocity)/(time[i] - time[i-1]) # off by 1 compared to v
                    current_jerk = (current_acceleration - prev_acceleration)/(time[i] - time[i-1])

                # new_time += delta_t

                acc.append(np.linalg.norm(current_acceleration))
                jerk.append(np.linalg.norm(current_jerk))
                curvature = self.compute_curvature(prev_velocity, current_acceleration) # off by 1
                curv.append(curvature)
                
                v.append(np.linalg.norm(current_velocity))


                future_v = current_velocity
                future_pos = current_pos

                for k in range(5):
                    # Get the control acceleration command based on current position
                    cmd = pid_controller.get_command(future_pos)
                    # print(cmd.shape)
                    
                    if cmd is None:
                        # print("Target reached!")
                        break  # Stop the control loop once the target is reached
                    
                    # Update the robot's velocity using the acceleration command (cmd)
                    future_v = future_v + cmd * dt  # Update velocity based on acceleration

                    # Update the robot's position based on the updated velocity
                    future_pos = future_pos + future_v * dt  # Update position based on velocity
                
                
                pose_distance = np.linalg.norm(future_pos - current_pos)
                pose_alignment.append(pose_distance)

                v_to_goal = target_pose - current_pos # no need to normalize
                v_align_3 = self.normalized_dot_product(current_velocity, v_to_goal)
                alignment_3.append(v_align_3)


                v_align = self.normalized_dot_product(current_velocity, future_v)
                v_align_2 = self.normalized_dot_product(current_velocity, prev_velocity)
                prev_velocity = current_velocity
                prev_acceleration = current_acceleration

                distance.append(dis)

                alignment.append(v_align) # append the time to carry out the pid planned traj
                alignment_2.append(v_align_2)

                if i == 0:
                    l = 1/4
                    optimality = 1
                else:
                    l = self.legibliity(planned_traj[:i+1], target_pose, time)
                    optimality = self.optimality(planned_traj[:i+1], target_pose)
                legi.append(l)
                opt.append(optimality)


            correction_timing =  timestamp[-2] - timestamp[0] # correction happens one point earlier

            # get rid of one data point
            if pre_traj.shape[0] == planned_traj.shape[0]:
                del alignment[-1]
                del alignment_2[-1]
                del distance[-1]
                del v[-1]
                del legi[-1]
                del opt[-1]
                del acc[-1]
                del jerk[-1]
                del curv[-1]
                del alignment_3[-1]
                del pose_alignment[-1]
                del time[-1]
            
            else:
                    
                del alignment[pre_traj.shape[0] - 1] # dele last point of the pre traj
                del alignment_2[pre_traj.shape[0] - 1]
                del distance[pre_traj.shape[0] - 1]
                del v[pre_traj.shape[0] - 1]
                del legi[pre_traj.shape[0] - 1]
                del opt[pre_traj.shape[0] - 1]
                del acc[pre_traj.shape[0] - 1]
                del jerk[pre_traj.shape[0] - 1]
                del curv[pre_traj.shape[0] - 1]
                del alignment_3[pre_traj.shape[0] - 1]
                del pose_alignment[pre_traj.shape[0] - 1]
                del time[pre_traj.shape[0] - 1]


            data = []
            for t in range(len(time)):
                x = {}
                x["a1"] = np.array(alignment[t], dtype=np.float32)
                x["a2"] = np.array(alignment_2[t], dtype=np.float32)
                x["dis"] = np.array(distance[t], dtype=np.float32)
                x["v"] = np.array(v[t], dtype=np.float32)
                x["legi"] = np.array(legi[t], dtype=np.float32)
                x["opt"] = np.array(opt[t], dtype=np.float32)
                x["acc"] = np.array(acc[t], dtype=np.float32)
                x["jerk"] = np.array(jerk[t], dtype=np.float32)
                x["curv"] = np.array(curv[t], dtype=np.float32)
                x["a3"] = np.array(alignment_3[t], dtype=np.float32)
                x["ps"] = np.array(pose_alignment[t], dtype=np.float32)
                x["time"] = float(time[t])

                # Correction flag
                if t == len(time) - 1:
                    x["cor"] = 1.0
                elif time[t] < correction_timing:
                    x["cor"] = 0.0
                else:
                    x["cor"] = 1.0

                data.append(x)

            self.filtered_data[ind]["features"] = data


        # Round small numbers to zero
        self.offset[np.abs(self.offset) < 1e-12] = 0
        # Round to reasonable decimal places
        self.offset = np.round(self.offset, 8)

        # Make clean string for key
        key_str = '_'.join(['%g' % off for off in self.offset])

        with open(f"/home/aw797/palmer_scratch/sampled_{str(int(100*self.timing_ratio))}_finer/{key_str}.pkl", 'wb') as f: #abosulte path here for the scratch folder
            pickle.dump(self.filtered_data, f)
        print(f"saved {key_str}")


    





    def legibliity(self, traj, goal, time):


        # Compute differences between consecutive points
        diffs = np.diff(traj, axis=0)  # shape (N-1, 3)

        # Compute Euclidean norm of each segment
        segment_lengths = np.linalg.norm(diffs, axis=1)

        efficient_length =  np.linalg.norm(goal - traj[0])

        num = 0
        den = 0
        # print(len(traj))

        for i in range(len(traj) - 1):

            P_G = np.exp( -np.sum(segment_lengths[:i]) - np.linalg.norm(goal - traj[-1])) * (1/4) /np.exp(-efficient_length)
            f = time[-1] - time[i]
            # print(f)
            # input()
            num += P_G * f
            den += f
        
        L = num/den
        # print(L)
        return L
    
    def optimality(self, traj, goal):

        # Compute differences between consecutive points
        diffs = np.diff(traj, axis=0)  # shape (N-1, 3)

        # Compute Euclidean norm of each segment
        segment_lengths = np.linalg.norm(diffs, axis=1)

        # Total trajectory length
        pre_l = np.sum(segment_lengths)

        post_l =  np.linalg.norm(goal - traj[-1])

        former_pre_l = np.sum(segment_lengths[:-1])
        former_post_l = np.linalg.norm(goal - traj[-2])

        # opt = np.exp( -pre_l - post_l)/np.exp(np.linalg.norm(goal - traj[0]))
        opt = np.exp( -pre_l - post_l)/np.exp( - former_pre_l - former_post_l)

        return opt


    def compute_curvature(self, v, a):

        # Cross product and norms
        cross = np.cross(v, a)
        cross_norm = np.linalg.norm(cross)
        v_norm = np.linalg.norm(v)

        # if v_norm == 0:
        #     return 0.0  # Avoid divide-by-zero

        if v_norm < 1e-2:  # clip the v so that curvature doesn't blow up
            curvature = 0.0
        else:
            curvature = cross_norm / (v_norm ** 3)

        return curvature  # shape (N,)


   



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('-tr','--timing_ratio', type = float)
    parser.add_argument('-x','--x_offset', type = float)
    parser.add_argument('-y','--y_offset', type = float)

    args = parser.parse_args()


    ef = extract_features( args.timing_ratio, q = args.x_offset, r = args.y_offset)


    ef.get_features_corrected()

