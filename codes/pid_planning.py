from pid_controller_poses import PIDController
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import argparse

class pid_optimal():
    def __init__(self, shape):
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

        self.shape = shape

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
        # print(self.training_data[500]["pre_pose_list"][-1], self.training_data[500]["correction_pose_list"][0])
        # input()
        
        self.selected_data = self.get_training_data()

        self.filter_data()




    def get_training_data(self):

        selected_data = []
        for i in range(len(self.training_data)):
            # if self.training_data[i]["comp"].strip() == str(self.comp).strip() and self.training_data[i]["legi"].strip() == str(self.legi).strip() and self.training_data[i]["shape"] == self.shape:
            # if self.training_data[i]["shape"] == self.shape:
            #     selected_data.append(self.training_data[i])
            if self.training_data[i]["corrected"] == True:
                selected_data.append(self.training_data[i])
            #selected_data.append(self.training_data[i])
        print(len(selected_data))

        return selected_data
    
    def filter_data(self):
        self.filtered_data = []
        for ind in range(len(self.selected_data)):
            pre_traj = np.array(self.selected_data[ind]["pre_pose_list"]).copy()
            correction_traj = np.array(self.selected_data[ind]["correction_pose_list"]).copy()
            post_traj = np.array(self.selected_data[ind]["post_pose_list"]).copy()
            # vel_traj =  np.array(self.selected_data[ind]["vel_traj"]).copy()
            # print(pre_traj.shape, correction_traj.shape, post_traj.shape)
            if post_traj.shape[0] == 0:
                waypoints = np.concatenate((pre_traj, correction_traj), axis=0)
                # waypoints = np.vstack((pre_traj, correction_traj))
            else:
                waypoints = np.concatenate((pre_traj, correction_traj, post_traj), axis=0)
                # waypoints = np.vstack((pre_traj, correction_traj, post_traj))
            # print(waypoints.shape)
            planned_traj = np.array(self.selected_data[ind]["entire_pose_list"])
            planned_correction = np.array(self.selected_data[ind]["rescaled_correction_timing"])
            # planned_correction = 10

            # if len(pre_traj) <= 0.5 * len(waypoints):
            #     self.filtered_data.append(self.selected_data[ind])
            if planned_correction <= 0.75 * len(planned_traj) and len(pre_traj) >= 0.2 * len(waypoints):
                self.filtered_data.append(self.selected_data[ind])
            # print(self.filtered_data[-1])
            # input()
        print(len(self.filtered_data))
        input()


    # def pid_planning(self):

    #     # Define initial parameters and settings
    #     P, I, D = 10, 0, 5
    #     epsilon = 0.01  # Threshold for stopping (in meters)
    #     max_cmd = np.array([1.0, 1.0, 1.0])  # Max acceleration for each axis (m/s²)

    #     # Create the PID controller object
    #     pid_controller = PIDController(P, I, D, epsilon, max_cmd)

    #     # Define the target pose (single waypoint) in 3D space (x, y, z)
    #     # target_pose = [1.0, 2.0, 3.0]  # Example target position

    #     pre_traj = np.array(self.filtered_data[ind]["pre_pose_list"]).copy()

    #     for j in range(len(self.filtered_data)):
    #         ind = 4
    #         actual_goal_index = int(self.filtered_data[ind]["target"])
    #         shape = self.filtered_data[ind]["shape"]
    #         target_pose = np.array(self.gs[shape][actual_goal_index])[:3]

    #         # Set the target pose for the PID controller
    #         pid_controller.set_target_pose(target_pose)

    #         # Define the initial position and velocity of the end effector
    #         planned_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[:, :3]
    #         # current_pos = np.array([0.0, 0.0, 0.0])  # Starting position

    #         # Define the time step for integration
    #         dt = 0.05  # Time step (seconds)

    #         planned_length = []

    #         prev_length = 0

    #         for i in range(planned_traj.shape[0] - 1):
    #             current_pos = planned_traj[i]
    #             # print(current_pos)
    #             # input()
    #             current_velocity = (planned_traj[i+1] - planned_traj[i])/0.3 # Starting velocity
    #             vel_norm = np.linalg.norm(current_velocity)
    #             print(vel_norm)
    #             # if vel_norm < 0.01:
    #             #     current_velocity = current_velocity/vel_norm * 0.01
    #             # input()

            
    #             # To store trajectory evolution
    #             positions = []

    #             # Control loop (moving the robot)
    #             while True:
    #                 # Get the control acceleration command based on current position
    #                 cmd = pid_controller.get_command(current_pos)
    #                 # print(cmd.shape)
                    
    #                 if cmd is None:
    #                     print("Target reached!")
    #                     break  # Stop the control loop once the target is reached
                    
    #                 # Update the robot's velocity using the acceleration command (cmd)
    #                 # print(current_velocity.shape, cmd.shape, current_velocity + cmd * dt)
    #                 current_velocity = current_velocity + cmd * dt  # Update velocity based on acceleration

    #                 # Update the robot's position based on the updated velocity
    #                 current_pos = current_pos + current_velocity * dt  # Update position based on velocity

    #                 # Print current position and velocity
    #                 print(f"Current Position: {current_pos}")
    #                 print(f"Current Velocity: {current_velocity}")

    #                 # Store the current position for plotting
    #                 positions.append(current_pos.copy())


    #             # Convert the list of positions to a numpy array for easy plotting
    #             positions = np.array(positions)
    #             print(positions.shape)

    #             cur_length = positions.shape[0]

    #             planned_length.append(cur_length) # append the time to carry out the pid planned traj

    #             prev_length = cur_length

    #             # Plot the trajectory evolution
    #             fig = plt.figure()
    #             ax = fig.add_subplot(111, projection='3d')

    #             # Plot the positions (trajectory)
    #             if positions.ndim == 2:
    #                 ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectory")

    #             # Plot the starting and ending points
    #             ax.scatter(planned_traj[0][0], planned_traj[0][1], planned_traj[0][2], color="green", label="Start", s=100)
    #             ax.scatter([target_pose[0]], [target_pose[1]], [target_pose[2]], color="red", label="Target", s=100)

    #             # Labels and title
    #             ax.set_xlabel('X Position')
    #             ax.set_ylabel('Y Position')
    #             ax.set_zlabel('Z Position')
    #             ax.set_title("3D Trajectory Evolution")
    #             ax.legend()

    #             # # Set the x, y, z ticks at intervals of 0.1 without limiting the axis range
    #             # ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    #             # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    #             # ax.zaxis.set_major_locator(plt.MultipleLocator(0.1))

    #             # Set the aspect ratio to be equal for all axes (so that the intervals appear uniform)
    #             ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    #             # Show plot
    #             plt.show()

    #         plt.figure(figsize=(8, 6))
    #         plt.plot(planned_length)
    #         planned_correction = np.array(self.filtered_data[ind]["rescaled_correction_timing"])
    #         print(planned_correction, planned_traj.shape)
    #         plt.axvline(x=planned_correction, color='r', linestyle='--', label="Correction")
    #         # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'.png')
    #         # plt.show()

    def pid_planning(self):

        # Define initial parameters and settings
        P, I, D = 10, 0, 5
        epsilon = 0.01  # Threshold for stopping (in meters)
        max_cmd = np.array([10.0, 10.0, 10.0])  # Max acceleration for each axis (m/s²)

        # Create the PID controller object
        pid_controller = PIDController(P, I, D, epsilon, max_cmd)

        # Define the target pose (single waypoint) in 3D space (x, y, z)
        # target_pose = [1.0, 2.0, 3.0]  # Example target position

        # pre_traj = np.array(self.filtered_data[ind]["pre_pose_list"]).copy()

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
            post_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[planned_correction:, :3]
            planned_traj = np.concatenate((pre_traj, post_traj), axis=0)
            # planned_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[:, :3]
            # print(pre_traj[-1], post_traj[0])
            # input()
            pre_traj_vel = np.array(self.filtered_data[ind]["pre_pose_vel"])[:,:3]
            # current_pos = np.array([0.0, 0.0, 0.0])  # Starting position
            print(self.filtered_data[ind]["participant_id"])

            # Define the time step for integration
            dt = 0.05  # Time step (seconds)

            planned_length = []

            prev_length = 0

            alignment = []

            for i in range(planned_traj.shape[0] - 1):
            # for i in range(pre_traj.shape[0]):
                print(i)
                pid_controller.set_target_pose(target_pose) # reset controller every time
                current_pos = planned_traj[i]
                # print(current_pos)
                # input()
                if i <= pre_traj.shape[0] - 1:
                    current_velocity = pre_traj_vel[i]
                else:
                    current_velocity = (planned_traj[i+1] - planned_traj[i])/0.3 # Starting velocity
                vel_norm = np.linalg.norm(current_velocity)
                print(current_velocity/vel_norm)

            
                # To store trajectory evolution
                positions = []

                # Get the control acceleration command based on current position
                cmd = pid_controller.get_command(current_pos)
                # print(cmd.shape)
                if cmd is None:
                    print("Target reached!")
                    break  # Stop the control loop once the target is reached
                # this will also happen at the end of every trajectory because it was planned to reach the goal
                
                
                # Update the robot's velocity using the acceleration command (cmd)
                next_velocity = current_velocity + cmd * dt  # Update velocity based on acceleration
                vel_norm = np.linalg.norm(next_velocity)
                print(next_velocity/vel_norm)

                print(current_pos, current_velocity, next_velocity, cmd)
                # input()

                v_align = self.normalized_dot_product(current_velocity, next_velocity)
                print(v_align)
                # input()

                alignment.append(v_align) # append the time to carry out the pid planned traj

                pid_controller.set_target_pose(target_pose)  #reset again
                # Control loop (moving the robot)
                while True:
                    # Get the control acceleration command based on current position
                    cmd = pid_controller.get_command(current_pos)
                    # print(cmd.shape)
                    
                    if cmd is None:
                        print("Target reached!")
                        break  # Stop the control loop once the target is reached
                    
                    # Update the robot's velocity using the acceleration command (cmd)
                    # print(current_velocity.shape, cmd.shape, current_velocity + cmd * dt)
                    current_velocity = current_velocity + cmd * dt  # Update velocity based on acceleration

                    # Update the robot's position based on the updated velocity
                    current_pos = current_pos + current_velocity * dt  # Update position based on velocity

                    # Print current position and velocity
                    # print(f"Current Position: {current_pos}")
                    # print(f"Current Velocity: {current_velocity}")

                    # Store the current position for plotting
                    positions.append(current_pos.copy())


                # Convert the list of positions to a numpy array for easy plotting
                positions = np.array(positions)
                # print(positions.shape)

                cur_length = positions.shape[0]

                planned_length.append(cur_length) # append the time to carry out the pid planned traj

                prev_length = cur_length

                # Plot the trajectory evolution
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Plot the positions (trajectory)
                if positions.ndim == 2:
                    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectory")
                
                ax.plot(pre_traj[:, 0], pre_traj[:, 1], pre_traj[:, 2])
                ax.plot(post_traj[:, 0], post_traj[:, 1], post_traj[:, 2])

                # Plot the starting and ending points
                ax.scatter(planned_traj[0][0], planned_traj[0][1], planned_traj[0][2], color="green", label="Start", s=100)
                ax.scatter([target_pose[0]], [target_pose[1]], [target_pose[2]], color="red", label="Target", s=100)

                # Labels and title
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_zlabel('Z Position')
                ax.set_title("3D Trajectory Evolution")
                ax.legend()

                # # Set the x, y, z ticks at intervals of 0.1 without limiting the axis range
                # ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
                # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
                # ax.zaxis.set_major_locator(plt.MultipleLocator(0.1))

                # Set the aspect ratio to be equal for all axes (so that the intervals appear uniform)
                ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

                # Show plot
                plt.show()

            plt.figure(figsize=(8, 6))
            plt.plot(planned_length)
            correction_timing = pre_traj.shape[0]
            # planned_correction = np.array(self.filtered_data[ind]["rescaled_correction_timing"])
            # print(planned_correction, planned_traj.shape)
            plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'.png')
            # plt.show()

    def normalized_dot_product(self, v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)

        dot = np.dot(v1, v2)
        # norm_factor = max(np.linalg.norm(v1)**2, np.linalg.norm(v2)**2)
        norm_factor = np.linalg.norm(v1) * np.linalg.norm(v2)

        if norm_factor == 0:
            return 0.0  # Avoid division by zero for zero vectors

        return dot / norm_factor

    
    def pid_v(self):

        # Define initial parameters and settings
        P, I, D = 10, 0, 5
        epsilon = 0.01  # Threshold for stopping (in meters)
        max_cmd = np.array([10.0, 10.0, 10.0])  # Max acceleration for each axis (m/s²)

        # Create the PID controller object
        pid_controller = PIDController(P, I, D, epsilon, max_cmd)

        # Define the target pose (single waypoint) in 3D space (x, y, z)
        # target_pose = [1.0, 2.0, 3.0]  # Example target position

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
            post_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[planned_correction:, :3]
            planned_traj = np.concatenate((pre_traj, post_traj), axis=0)
            # planned_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[:, :3]
            print(pre_traj[-1], post_traj[0])
            pre_traj_vel = np.array(self.filtered_data[ind]["pre_pose_vel"])[:,:3]
            # current_pos = np.array([0.0, 0.0, 0.0])  # Starting position
            print("participant id ",self.filtered_data[ind]["participant_id"])

            timestamp = np.array(self.filtered_data[ind]["pre_timestamp"])
            # num_time = timestamp.shape[0]
            dts0 = timestamp[1:] - timestamp[:-1]
            dts0 = np.insert(dts0, 0, 0) 
            # print(pre_traj.shape, dts.shape, dts[-5:])
            # input()

            # Define the time step for integration
            dt = 0.05  # Time step (seconds)

            alignment = []
            alignment_2 = []
            distance = []
            v = []
            time = []

            # for i in range(pre_traj.shape[0]):
            #     current_pos = pre_traj[i]
            #     # print(current_pos)
            #     # input()
            #     current_velocity = pre_traj_vel[i] # Starting velocity
            #     # vel_norm = np.linalg.norm(current_velocity)


            #     # Get the control acceleration command based on current position
            #     cmd = pid_controller.get_command(current_pos)
            #     # print(cmd.shape)
            #     if cmd is None:
            #         print("Target reached!")
            #         break  # Stop the control loop once the target is reached
            #     # this will also happen at the end of every trajectory because it was planned to reach the goal
                
                
            #     # Update the robot's velocity using the acceleration command (cmd)
            #     next_velocity = current_velocity + cmd * dt  # Update velocity based on acceleration

            #     v_aligh = self.normalized_dot_product(current_velocity, next_velocity)

            #     alignment.append(v_aligh) # append the time to carry out the pid planned traj

            # for i in range(post_traj.shape[0] - 1):
            #     current_pos = post_traj[i]
            #     # print(current_pos)
            #     # input()
            #     current_velocity = (post_traj[i+1] - post_traj[i])/0.3 # Starting velocity
            #     # vel_norm = np.linalg.norm(current_velocity)


            #     # Get the control acceleration command based on current position
            #     cmd = pid_controller.get_command(current_pos)
            #     # print(cmd.shape)
            #     if cmd is None:
            #         print("Target reached!")
            #         break  # Stop the control loop once the target is reached
                
                
            #     # Update the robot's velocity using the acceleration command (cmd)
            #     next_velocity = current_velocity + cmd * dt  # Update velocity based on acceleration

            #     v_aligh = self.normalized_dot_product(current_velocity, next_velocity)

            #     alignment.append(v_aligh) # append the time to carry out the pid planned traj

            prev_velocity = np.zeros([3])
            new_time = 0

            for i in range(planned_traj.shape[0] - 1):
                pid_controller.set_target_pose(target_pose) # reset controller every time
                current_pos = planned_traj[i]
                # print(current_pos)
                # input()

                dis = np.linalg.norm(current_pos - target_pose)

                if i <= pre_traj.shape[0] - 1:
                    current_velocity = pre_traj_vel[i]
                    delta_t = dts0[i]
                else:
                    current_velocity = (planned_traj[i+1] - planned_traj[i])/0.3 # Starting velocity
                    delta_t = 0.3
                # current_velocity = (planned_traj[i+1] - planned_traj[i])/0.3 # Starting velocity
                # vel_norm = np.linalg.norm(current_velocity)

                new_time += delta_t
                
                v.append(np.linalg.norm(current_velocity))
                time.append(new_time)

                # # Get the control acceleration command based on current position
                # cmd = pid_controller.get_command(current_pos)
                # # print(cmd.shape)
                # if cmd is None:
                #     print("Target reached!")
                #     break  # Stop the control loop once the target is reached

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
                    # print(current_velocity.shape, cmd.shape, current_velocity + cmd * dt)
                    future_v = future_v + cmd * dt  # Update velocity based on acceleration

                    # Update the robot's position based on the updated velocity
                    future_pos = future_pos + future_v * dt  # Update position based on velocity
                
                
                # Update the robot's velocity using the acceleration command (cmd)
                # next_velocity = current_velocity + cmd * dt  # Update velocity based on acceleration

                # print(current_pos, current_velocity, next_velocity, cmd)
                # input()

                v_align = self.normalized_dot_product(current_velocity, future_v)
                v_align_2 = self.normalized_dot_product(current_velocity, prev_velocity)
                prev_velocity = current_velocity
                # print(v_align)
                # input()

                distance.append(dis)

                alignment.append(v_align) # append the time to carry out the pid planned traj
                alignment_2.append(v_align_2)
            # total_alignment = [x + y for x, y in zip(alignment, alignment_2)]
            # print(alignment, alignment_2)

            correction_timing =  timestamp[-1] - timestamp[0]

            # plt.figure(figsize=(8, 6))
            # plt.plot(time, alignment)
            
            # # print(planned_correction, planned_traj.shape)
            
            # # correction_timing = planned_correction
            # plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_a1.png')
            # # plt.show()

            # plt.figure(figsize=(8, 6))
            # plt.plot(time, alignment_2)
            # plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_a2.png')

            # plt.figure(figsize=(8, 6))
            # plt.plot(time, distance)
            # plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_d.png')

            # plt.figure(figsize=(8, 6))
            # plt.plot(time, v)
            # plt.axvline(x=correction_timing, color='r', linestyle='--', label="Correction")
            # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_v.png')

            # # Plot the trajectory evolution
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            
            # ax.plot(pre_traj[:, 0], pre_traj[:, 1], pre_traj[:, 2])
            # ax.plot(post_traj[:, 0], post_traj[:, 1], post_traj[:, 2])

            # # Plot the starting and ending points
            # ax.scatter(planned_traj[0][0], planned_traj[0][1], planned_traj[0][2], color="green", label="Start", s=100)
            # ax.scatter([target_pose[0]], [target_pose[1]], [target_pose[2]], color="red", label="Target", s=100)

            # # Labels and title
            # ax.set_xlabel('X Position')
            # ax.set_ylabel('Y Position')
            # ax.set_zlabel('Z Position')
            # ax.set_title("3D Trajectory Evolution")
            # ax.legend()
            # plt.savefig('/Users/anjiabei/Documents/research/figures/pid_planning/'+str(ind)+'_traj.png')


            data = []
            for t in range(len(time)):
                x = {}
                x["a1"] = np.array(alignment[t])
                x["a2"] = np.array(alignment_2[t])
                x["dis"] = np.array(distance[t])
                x["v"] = np.array(v[t])
                x["time"] = float(time[t])
                if time[t] < correction_timing:
                    x["cor"] = 0.0
                else:
                    x["cor"] = 1.0
                data.append(x)
            print(data[1]["a1"], data[1]["time"])
            with open("./features/features_"+str(j)+".pkl", 'wb') as f:
                pickle.dump(data, f)

    def plot_traj(self):


        for j in range(len(self.filtered_data)):
            ind = 97
            actual_goal_index = int(self.filtered_data[ind]["target"])
            shape = self.filtered_data[ind]["shape"]
            target_pose = np.array(self.gs[shape][actual_goal_index])[:3]


            # Define the initial position and velocity of the end effector
            pre_traj = np.array(self.filtered_data[ind]["pre_pose_list"])[:,:3]
            planned_correction = np.array(self.filtered_data[ind]["rescaled_correction_timing"])
            post_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[planned_correction:, :3]
            planned_traj = np.concatenate((pre_traj, post_traj), axis=0)
            # planned_traj = np.array(self.filtered_data[ind]["entire_pose_list"])[:, :3]
            # print(pre_traj[-1], post_traj[0])
            # input()
            pre_traj_vel = np.array(self.filtered_data[ind]["pre_pose_vel"])[:,:3]
            # current_pos = np.array([0.0, 0.0, 0.0])  # Starting position
            print(self.filtered_data[ind]["participant_id"])



            # Plot the trajectory evolution
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(pre_traj[:, 0], pre_traj[:, 1], pre_traj[:, 2])
            ax.plot(post_traj[:, 0], post_traj[:, 1], post_traj[:, 2])

            # Plot the starting and ending points
            ax.scatter(planned_traj[0][0], planned_traj[0][1], planned_traj[0][2], color="green", label="Start", s=100)
            ax.scatter([target_pose[0]], [target_pose[1]], [target_pose[2]], color="red", label="Target", s=100)

            # Labels and title
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
            ax.set_title("3D Trajectory Evolution")
            ax.legend()



            # Show plot
            plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--shape', type = str)

    args = parser.parse_args()

    vt = pid_optimal(shape = args.shape)
    # vt.pid_planning()
    vt.pid_v()
    # vt.plot_traj()