import numpy as np
import time
import pid

class PIDController(object):
    """
    This class represents a PID controller for controlling the 3D end effector to a set target.
    The controller generates acceleration commands to move the end effector directly towards the target.

    Parameters:
    P, I, D        - gain terms for the PID controller
    epsilon        - proximity threshold for stopping
    max_cmd        - maximum allowed acceleration command (for each axis in 3D)
    """
    
    def __init__(self, P, I, D, epsilon, max_cmd):
        # ----- PID Parameter Setup ----- #
        self.pid = pid.PID(P, I, D, 0, 0)  # PID controller initialization
        self.epsilon = epsilon  # Threshold to stop when close enough to target
        self.max_cmd = max_cmd  # Maximum acceleration commands for each axis (3D vector)
        
        self.path_start_T = time.time()  # Path start time
        self.path_end_T = None  # Path end time (used to stop after reaching goal)

    def set_target_pose(self, target_pose):
        """Set the target position (x, y, z) for the end effector."""
        self.target_pos = np.array(target_pose)  # Assuming target_pose is a 3D tuple/list (x, y, z)
        self.pid.reset()  # Reset PID controller state for fresh trajectory planning

    def get_command(self, current_pos):
        """
        Reads the latest position and velocity of the end effector and returns an acceleration command
        to move the end effector to the target.
        
        Parameters:
            current_pos - Current 3D position of the end effector (x, y, z)
            current_velocity - Current 3D velocity of the end effector (vx, vy, vz)
        
        Returns:
            cmd - The next acceleration command to get to the updated target.
        """
        # Calculate position error (difference between target and current position)
        position_error = self.target_pos - current_pos
        # print("posistion error = ", position_error)
        
        
        # If the end effector is within the epsilon threshold, return None to stop
        if np.linalg.norm(position_error) < self.epsilon:
            print("End effector has reached the target. Stopping.")
            return None
        
        # Update PID control based on position error to calculate the acceleration
        cmd = self.pid.update_PID(position_error)  # Generate acceleration command (cmd)
        # print(cmd.shape)
        
        # Clip the command to ensure it's within the maximum allowed acceleration for each axis
        cmd = np.clip(cmd, a_min=-self.max_cmd[0], a_max=self.max_cmd[0])
        # print(cmd)
        # input()

        return cmd