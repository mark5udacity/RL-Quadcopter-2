import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, physicsSim_Params, target_pos = None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """

        # Simulation
        self.sim = PhysicsSim(physicsSim_Params)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        target_closeness_reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        # Severely penalize the helicopter having a 'body' velocity that's too large, regardless of direction
        velocity_penalties = abs(self.sim.find_body_velocity()).sum()

        # per slack room, idea to minimize euler angles
        euler_bias = 5
        eulers_angle_penalty = abs(self.sim.pose[3:]).sum() - euler_bias

        # episode ends when hitting the ground or running out of time.  Want to fly as long as possible.
        flight_time = self.sim.time

        rotor_diff_penalty = np.std(rotor_speeds) # penalize big differences in rotor speed!

        height_reward = self.sim.pose[2] / 100. # reward going higher, since that's what we want! But not by more than distance

        total = target_closeness_reward  \
                 + flight_time           \
                 - eulers_angle_penalty  \
                 - velocity_penalties    \
                 - rotor_diff_penalty    \
                 + height_reward


        #print('rewards: ', target_closeness_reward, velocity_penalties, eulers_angle_penalty,
        #      flight_time, rotor_diff_penalty, total)
        #hyperbolic_activation = np.tanh(total)
        #if (hyperbolic_activation > -1.):
        #    print('Found rewards that crept above -1!: ', total, hyperbolic_activation)
        #return hyperbolic_activation
        return total


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state


class My_Custom_Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
