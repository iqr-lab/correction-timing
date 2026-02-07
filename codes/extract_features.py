from pid_controller_poses import PIDController
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import argparse
import random

# extract features for both corrected and uncorrected trajectories.

class extract_features():
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
        with open('../../experiment/config/target_position.yaml','r') as file:

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
        # with open('/Users/anjiabei/Documents/research/corl_data.pkl', 'rb') as file:
        with open('/Users/anjiabei/Documents/research/rescaled_traj.pkl', 'rb') as file:
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
        input()

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
            if post_traj.shape[0] == 0:
                waypoints = np.concatenate((pre_traj, correction_traj), axis=0)
            else:
                waypoints = np.concatenate((pre_traj, correction_traj, post_traj), axis=0)
            planned_traj = np.array(self.selected_data[ind]["entire_pose_list"])
            planned_correction = np.array(self.selected_data[ind]["rescaled_correction_timing"])

            if self.selected_data[ind]["fake_correction"] == False and planned_correction <= self.timing_ratio * len(planned_traj): # even dt
                self.filtered_data.append(self.selected_data[ind])
        print("total corrected data ", len(self.selected_data))
        print("filtered data ", len(self.filtered_data))
        input()

    def filter_data_uncorrected(self):

        # all uncorrected data

        self.filtered_data = self.selected_data
        print(len(self.filtered_data))
        input()


    def normalized_dot_product(self, v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)

        dot = np.dot(v1, v2)
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


        for j in range(len(self.filtered_data)):
            ind = j
            actual_goal_index = int(self.filtered_data[ind]["target"])
            shape = self.filtered_data[ind]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

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

            for i in range(planned_traj.shape[0] - 1): # length for all lists
                pid_controller.set_target_pose(target_pose) # reset controller every time
                current_pos = planned_traj[i]

                dis = np.linalg.norm(current_pos - target_pose)

                if i <= pre_traj.shape[0] - 1: # for calculating the v at the transition point as the pos_diff/delta_t
                    # current_velocity = pre_traj_vel[i]
                    current_velocity = (pre_traj[i+1] - pre_traj[i])/dts0[0] # use traj itself to cal v
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
                        print("Target reached!")
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


            plt.figure(figsize=(8, 6))
            plt.plot(time, acc)
            plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_a2.png')
            print(self.filtered_data[ind]["legi"], self.filtered_data[ind]["success"])
            plt.show()

            # plt.figure(figsize=(8, 6))
            # plt.plot(time, alignment_3)
            # plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_a2.png')
            # print(self.filtered_data[ind]["legi"], self.filtered_data[ind]["success"])
            # plt.show()

            data = []
            for t in range(len(time)):
                x = {}
                x["a1"] = np.array(alignment[t]) # velocity alignment (optimal and current)
                x["a2"] = np.array(alignment_2[t]) # velocity alignment (previous and current)
                x["dis"] = np.array(distance[t]) # distance to goal
                x["v"] = np.array(v[t]) # velocity magnitude
                x["legi"] = np.array(legi[t]) # legibility
                x["opt"] = np.array(opt[t]) # optimality
                x["acc"] = np.array(acc[t]) # acceleration magnitude
                x["jerk"] = np.array(jerk[t]) # jerk magnitude
                x["curv"] = np.array(curv[t]) # curvature
                x["a3"] = np.array(alignment_3[t]) # velocity alignment (towards goal and current)
                x["ps"] = np.array(pose_alignment[t]) # position alignment (optimal and current)
                x["time"] = float(time[t]) # time

                if t == len(time) - 1: # make sure after getting rid of the 2 points the last point is still 1 if the correction didnt happen earlier
                    x["cor"] = 1.0
                elif time[t] < correction_timing: # correction happens at [pre_traj.shape[0] - 2]
                    x["cor"] = 0.0
                else:
                    x["cor"] = 1.0
                data.append(x)
            print(data[1]["a1"], data[1]["time"]) # n_timesteps
            self.filtered_data[ind]["features"] = data
            # with open("./features/features_"+str(j)+".pkl", 'wb') as f:
            #     pickle.dump(data, f)
        with open("/Users/anjiabei/Documents/research/features/corrected_features_"+str(int(100*self.timing_ratio))+".pkl", 'wb') as f:
            pickle.dump(self.filtered_data, f)

    def legibliity(self, traj, goal, time):

        # partial traj

        # Compute differences between consecutive points
        diffs = np.diff(traj, axis=0)  # shape (N-1, 3)

        # Compute Euclidean norm of each segment
        segment_lengths = np.linalg.norm(diffs, axis=1)

        # Total trajectory length
        # total_length = np.sum(segment_lengths)

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

        if v_norm < 1e-2:  # clip the v so that curvature doesn't blow up
            curvature = 0.0
        else:
            curvature = cross_norm / (v_norm ** 3)

        return curvature  # shape (N,)


    def get_features_uncorrected(self):

        # Define initial parameters and settings
        P, I, D = 10, 0, 5
        epsilon = 0.01  # Threshold for stopping (in meters)
        max_cmd = np.array([10.0, 10.0, 10.0])  # Max acceleration for each axis (m/s²)

        # Create the PID controller object
        pid_controller = PIDController(P, I, D, epsilon, max_cmd)

        for j in range(len(self.filtered_data)):
            ind = j
            actual_goal_index = int(self.filtered_data[ind]["target"])
            shape = self.filtered_data[ind]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

            # Set the target pose for the PID controller
            pid_controller.set_target_pose(target_pose)

            # Define the initial position and velocity of the end effector
            if np.array(self.filtered_data[ind]["pre_pose_list"]).ndim == 1:
                print(self.filtered_data[ind]["participant_id"], self.filtered_data[ind]["trial_id"], self.filtered_data[ind]["shape"])
                print(self.filtered_data[ind]["pre_pose_vel"], self.filtered_data[ind]["pre_pose_list"])
                self.filtered_data[ind]["features"] = False
                # input()
                continue
            planned_traj = np.array(self.filtered_data[ind]["pre_pose_list"])[:,:3]

            if np.array(self.filtered_data[ind]["pre_pose_vel"]).ndim == 1:
                print(self.filtered_data[ind]["participant_id"], self.filtered_data[ind]["trial_id"], self.filtered_data[ind]["shape"])
                print(self.filtered_data[ind]["pre_pose_vel"], self.filtered_data[ind]["pre_pose_list"])
                self.filtered_data[ind]["features"] = False
                # input()
                continue
            pre_traj_vel = np.array(self.filtered_data[ind]["pre_pose_vel"])[:,:3]
            # current_pos = np.array([0.0, 0.0, 0.0])  # Starting position
            print("participant id ",self.filtered_data[ind]["participant_id"])

            timestamp = np.array(self.filtered_data[ind]["pre_timestamp"])
            dts0 = timestamp[1:] - timestamp[:-1]

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

                delta_t = dts0[i]

                time.append(new_time) # first tine is 0, always the previous time
                new_time += delta_t
            time.append(new_time)


            prev_velocity = np.zeros([3])
            prev_acceleration = np.zeros([3])

            for i in range(planned_traj.shape[0]):
                pid_controller.set_target_pose(target_pose) # reset controller every time
                current_pos = planned_traj[i]

                dis = np.linalg.norm(current_pos - target_pose)

                current_velocity = pre_traj_vel[i]

                if i == 0:
                    current_acceleration = (current_velocity - prev_velocity)/dts0[0] # off by one compared to v, its the one for the timestep before
                    current_jerk = (current_acceleration - prev_acceleration)/dts0[0]
                else:
                    current_acceleration = (current_velocity - prev_velocity)/(time[i] - time[i-1]) # off by 1 compared to v
                    current_jerk = (current_acceleration - prev_acceleration)/(time[i] - time[i-1])


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
                        print("Target reached!") # for each timestep there is one target reach
                        break  # Stop the control loop once the target is reached
                    
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


            # plt.figure(figsize=(8, 6))
            # plt.plot(time, jerk)
            # # plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_a2.png')
            # print(self.filtered_data[ind]["legi"], self.filtered_data[ind]["success"])
            # plt.show()


            data = []
            for t in range(len(time)):
                x = {}
                x["a1"] = np.array(alignment[t]) # velocity alignment (optimal and current)
                x["a2"] = np.array(alignment_2[t]) # velocity alignment (previous and current)
                x["dis"] = np.array(distance[t]) # distance to goal
                x["v"] = np.array(v[t]) # velocity magnitude
                x["legi"] = np.array(legi[t]) # legibility
                x["opt"] = np.array(opt[t]) # optimality
                x["acc"] = np.array(acc[t]) # acceleration magnitude
                x["jerk"] = np.array(jerk[t]) # jerk magnitude
                x["curv"] = np.array(curv[t]) # curvature
                x["a3"] = np.array(alignment_3[t]) # velocity alignment (towards goal and current)
                x["ps"] = np.array(pose_alignment[t]) # position alignment (optimal and current)
                x["time"] = float(time[t]) # time
                x["cor"] = 0.0
                data.append(x)
            print(data[1]["a1"], data[1]["time"])
            self.filtered_data[ind]["features"] = data

        with open("/Users/anjiabei/Documents/research/features/uncorrected_features.pkl", 'wb') as f:
            pickle.dump(self.filtered_data, f)



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

    ef = extract_features(corrected, args.timing_ratio)
    if args.corrected == "True":
        ef.get_features_corrected()
    else:
        ef.get_features_uncorrected()