# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # PID tuning values for lateral controller ()
        self.kp_lat = 4.5
        self.ki_lat = 0.6
        self.kd_lat = 0.5

        # Set the number of steps to look ahead
        self.look_ahead = 150

        # integral history term
        self.integral_error_heading = 0.0
        # derivative prev term
        self.prev_error_heading = 0.0

        # PID tuning values for longitude controller (speed of vehicle)
        self.target_vel = 12 #m/s
        self.angle_diff = np.pi/6
        self.slow_down = 10.0
        self.kp_vel = 800
        self.ki_vel = 100
        self.kd_vel = 50
        # integral history term
        self.integral_error_vel = 0.0
        # derivative prev error
        self.prev_error_vel = 0.0

        # Add additional member variables according to your need here.

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        # get the desired trajectory X steps ahead
        distance, near_index = closestNode(X, Y, trajectory)
        # wrap the index if it goes over the index of the trajectory list
        look_index = int((near_index + self.look_ahead) % len(trajectory))
        # get target X and Y positions
        target_XY = self.trajectory[look_index]

        # Determine heading from the target X and Y
        desired_heading = np.arctan2(target_XY[1] - Y, target_XY[0] - X)

        # Get error from heading
        heading_error = wrapToPi(desired_heading - psi)
        self.integral_error_heading += heading_error * delT
        der_error_heading = (heading_error - self.prev_error_heading)/(delT + 1e-10)

        #store the current heading error in prev_error_heading
        self.prev_error_heading = heading_error

        # get the new delta value
        delta = self.kp_lat * heading_error + self.ki_lat * self.integral_error_heading + self.kd_lat * der_error_heading

        # ---------------|Longitudinal Controller|-------------------------
        if np.abs(heading_error) > self.angle_diff:
            velocity = self.target_vel - self.slow_down
        else:
            velocity = self.target_vel
        vel_error = velocity - xdot
        self.integral_error_vel += vel_error * delT
        der_error_vel = (vel_error - self.prev_error_vel)/(delT + 1e-10)

        # store the current vel in prev_error_vel
        self.prev_error_vel = vel_error

        #get the new F value:
        F = self.kp_vel * vel_error + self.ki_vel * self.integral_error_vel + self.kd_vel * der_error_vel

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
