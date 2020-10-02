import copy
import math
import random
import numpy as np
import scipy as scp
from scipy import spatial
import time
from collections import deque

import gym

from gym_art.quadrotor_multi.quadrotor_single import GRAV, QuadrotorSingle
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti


class QuadrotorEnvMulti(gym.Env):
    def __init__(self,
                 num_agents,
                 dynamics_params='DefaultQuad', dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr='xyz_vxyz_R_omega', ep_time=7, obstacles_num=0, room_size=10,
                 init_random_state=False, rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV,
                 resample_goals=False, t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False,
                 quads_dist_between_goals=0.3, quads_mode='circular_config', swarm_obs=False, quads_use_numba=False):

        super().__init__()

        self.num_agents = num_agents
        self.swarm_obs = swarm_obs
        self.envs = []
        self.quads_dist_between_goals = quads_dist_between_goals
        self.single_digit_goal_list = [5, 2, 7]
        # iteratively yield the digit list
        def gentr_fn(alist): 
            while 1:
                for j in alist:
                    yield j
        self.iter_digits = gentr_fn(self.single_digit_goal_list)
        self.set_obstacles = False    # obstacle mode
        self.set_transition = True  # transition mode

        for i in range(self.num_agents):
            e = QuadrotorSingle(
                dynamics_params, dynamics_change, dynamics_randomize_every, dyn_sampler_1, dyn_sampler_2,
                raw_control, raw_control_zero_middle, dim_mode, tf_control, sim_freq, sim_steps,
                obs_repr, ep_time, obstacles_num, room_size, init_random_state,
                rew_coeff, sense_noise, verbose, gravity, t2w_std, t2t_std, excite, dynamics_simplification,
                quads_use_numba, self.swarm_obs, self.num_agents
            )
            self.envs.append(e)

        self.resample_goals = resample_goals

        self.scene = None

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

        # reward shaping
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=0., quadsettle=0.
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        ## Aux variables
        self.pos = np.zeros([self.num_agents, 3]) #Matrix containing all positions
        self.quads_mode = quads_mode


	    ## Set Goals
        self.neighbor_obs_size = 6
        self.clip_length = (num_agents-1) * self.neighbor_obs_size
        self.clip_min_box = self.observation_space.low[-self.clip_length:]
        self.clip_max_box = self.observation_space.high[-self.clip_length:]
	
        delta = quads_dist_between_goals
        pi = np.pi
        self.goal = []
        for i in range(self.num_agents):
            # Initial goal
            if self.set_transition:
                goal = self.single_digit_goal(5, i)
            else:
                goal = self.digit_goals(i)

            self.goal.append(goal)

        self.goal = np.array(self.goal)
        self.rews_settle = np.zeros(self.num_agents)
        self.rews_settle_raw = np.zeros(self.num_agents)
        self.settle_count = np.zeros(self.num_agents)

        ## Set obstacles
        self.obstacles = []
        if self.set_obstacles:
            for i in range(3):
                obstacle = [i/3-0.3, 0, 0.5]
                self.obstacles.append(obstacle)

            self.goal = np.array(self.goal)

    def single_digit_goal(self, digit, agent):
        ''' Setup goals of shape digits 0 - 9.
        We assume the number of agents is fixed and the shape is 3x5 (col x row) .
        '''
        delta = self.quads_dist_between_goals
        pi = np.pi
        set_vertical = True
        vertical_low = 1.0
        vertical_high = 3.0

        if digit == 0:
            degree = 2 * pi * agent / self.num_agents
            goal_x = delta * np.cos(degree)
            goal_y = delta * np.sin(degree)
            goal_z = 2.0
            return [goal_x, goal_z, goal_y] if set_vertical else [goal_x, goal_y, goal_z]

        elif digit == 1:
            goal_x = delta * agent
            goal_y = 0.0
            goal_z = 1.0
            return [goal_x, goal_z, goal_y] if set_vertical else [goal_x, goal_y, goal_z]

        elif digit == 2:
            # set x-axis
            if agent==2 or agent==3 or agent==4 or agent==10:
                goal_x = 0.0 * delta
            if agent==1 or agent==5 or agent==9:
                goal_x = 1.0 * delta
            if agent==0 or agent==6 or agent==7 or agent==8:
                goal_x = 2.0 * delta
            # set y-axis
            if agent==0 or agent==1 or agent==2:
                goal_y = vertical_low if set_vertical else 0.0 * delta
            if agent==3:
                goal_y = (vertical_high - vertical_low) / 4 * 1 + vertical_low if set_vertical else 1.0 * delta * vertical_factor
            if agent==4 or agent==5 or agent==6:
                goal_y = (vertical_high - vertical_low) / 4 * 2 + vertical_low if set_vertical else 2.0 * delta * vertical_factor
            if agent==7:
                goal_y = (vertical_high - vertical_low) / 4 * 3 + vertical_low if set_vertical else 3.0 * delta * vertical_factor
            if agent==8 or agent==9 or agent==10:
                goal_y = vertical_high if set_vertical else 4.0 * delta * vertical_factor
            goal_z = 1.0
            print(digit, ': ', goal_x, goal_z, goal_y)
            return [goal_x , goal_z, goal_y] if set_vertical else [goal_x, goal_y, goal_z]

        elif digit == 5:
            # set x-axis
            if agent==0 or agent==6 or agent==7 or agent==8:
                goal_x = 0.0 * delta
            if agent==1 or agent==5 or agent==9:
                goal_x = 1.0 * delta
            if agent==2 or agent==3 or agent==4 or agent==10:
                goal_x = 2.0 * delta
            # set y-axis
            if agent==0 or agent==1 or agent==2:
                goal_y = vertical_low if set_vertical else 0.0 * delta
            if agent==3:
                goal_y = (vertical_high - vertical_low) / 4 * 1 + vertical_low if set_vertical else 1.0 * delta * vertical_factor
            if agent==4 or agent==5 or agent==6:
                goal_y = (vertical_high - vertical_low) / 4 * 2 + vertical_low if set_vertical else 2.0 * delta * vertical_factor
            if agent==7:
                goal_y = (vertical_high - vertical_low) / 4 * 3 + vertical_low if set_vertical else 3.0 * delta * vertical_factor
            if agent==8 or agent==9 or agent==10:
                goal_y = vertical_high if set_vertical else 4.0 * delta * vertical_factor
            goal_z = 1.0
            print(digit, ': ', goal_x, goal_z, goal_y)
            return [goal_x, goal_z, goal_y] if set_vertical else [goal_x, goal_y, goal_z]
        elif digit == 7:
            if agent == 0:
                goal_x = 0.0
                goal_y = vertical_high if set_vertical else 0.0
            elif agent > 0 and agent < 4:
                goal_x = 1.0 * delta * agent
                goal_y = vertical_high if set_vertical else 0.0
            else:
                goal_x = 0.0
                goal_y = (vertical_high - vertical_low) / 7 * (agent - 4) + vertical_low if set_vertical else 2.0 - (agent - 4) * delta
            goal_z = 1.0
            print(digit, ': ', goal_x, goal_z, goal_y)
            return [goal_x, goal_z, goal_y] if set_vertical else [goal_x, goal_y, goal_z] 
        else:
            raise Exception('Sorry, only accept integers from 0 - 9.')

    def digit_goals(self, agent):
        delta = self.quads_dist_between_goals
        pi = np.pi
        set_vertical = True
        vertical_low = 1.0
        vertical_high = 3.0
        vertical_factor = 1.7
        horizontal_dis = 1.5

        ## digit 5
        if agent==0 or agent==6 or agent==7 or agent==8:
            goal_x = 0.0 * delta
        if agent==1 or agent==5 or agent==9:
            goal_x = 1.0 * delta
        if agent==2 or agent==3 or agent==4 or agent==10:
            goal_x = 2.0 * delta
        # set y-axis
        if agent==0 or agent==1 or agent==2:
            goal_y = vertical_low if set_vertical else 0.0 * delta
        if agent==3:
            goal_y = (vertical_high - vertical_low) / 4 * 1 + vertical_low if set_vertical else 1.0 * delta * vertical_factor
        if agent==4 or agent==5 or agent==6:
            goal_y = (vertical_high - vertical_low) / 4 * 2 + vertical_low if set_vertical else 2.0 * delta * vertical_factor
        if agent==7:
            goal_y = (vertical_high - vertical_low) / 4 * 3 + vertical_low if set_vertical else 3.0 * delta * vertical_factor
        if agent==8 or agent==9 or agent==10:
            goal_y = vertical_high if set_vertical else 4.0 * delta * vertical_factor

        ## digit 2
        # set x-axis
        if agent==13 or agent==14 or agent==15 or agent==21:
            goal_x = 0.0 * delta + horizontal_dis 
        if agent==12 or agent==16 or agent==20:
            goal_x = 1.0 * delta + horizontal_dis 
        if agent==11 or agent==17 or agent==18 or agent==19:
            goal_x = 2.0 * delta + horizontal_dis
        # set y-axis
        if agent==11 or agent==12 or agent==13:
            goal_y = vertical_low if set_vertical else 0.0 * delta
        if agent==14:
            goal_y = (vertical_high - vertical_low) / 4 * 1 + vertical_low if set_vertical else 1.0 * delta * vertical_factor
        if agent==15 or agent==16 or agent==17:
            goal_y = (vertical_high - vertical_low) / 4 * 2 + vertical_low if set_vertical else 2.0 * delta * vertical_factor
        if agent==18:
            goal_y = (vertical_high - vertical_low) / 4 * 3 + vertical_low if set_vertical else 3.0 * delta * vertical_factor
        if agent==19 or agent==20 or agent==21:
            goal_y = vertical_high if set_vertical else 4.0 * delta * vertical_factor
        goal_z = 1.0

        ##  digit 7
        if agent == 22:
            goal_x = 0.0 + horizontal_dis * 2.3
            goal_y = vertical_high if set_vertical else 0.0
        elif agent > 22 and agent < 26:
            goal_x = - (1.0 * delta * (agent - 22)) + horizontal_dis * 2.3
            goal_y = vertical_high if set_vertical else 0.0
        elif agent >= 26:
            goal_x = 0.0 + horizontal_dis * 2.3
            goal_y = (vertical_high - vertical_low) / 7 * (agent - 26) + vertical_low if set_vertical else 2.0 - (agent - 26) * delta
        goal_z = 1.0

        print(agent, goal_x, goal_z, goal_y)
        return [goal_x, goal_z, goal_y]

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def extend_obs_space(self, obs):
        obs_neighbors = []
        for i in range(len(obs)):
            observs = obs[i]
            obs_neighbor = np.array([obs[j][:self.neighbor_obs_size] for j in range(len(obs)) if j != i])
            obs_neighbor_rel = obs_neighbor - observs[:self.neighbor_obs_size]
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods
        obs_neighbors = np.clip(obs_neighbors, a_min=self.clip_min_box, a_max=self.clip_max_box)
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext


    def reset(self):
        obs, rewards, dones, infos = [], [], [], []

        models = tuple(e.dynamics.model for e in self.envs)

        # TODO: don't create scene object if we're just training and no need to visualize?
        if self.scene is None:
            self.scene = Quadrotor3DSceneMulti(
                models=models,
                w=640, h=480, resizable=True, obstacles=self.envs[0].obstacles, viewpoint=self.envs[0].viewpoint,
            )
        else:
            self.scene.update_models(models)

        for i, e in enumerate(self.envs):
            e.goal = self.goal[i]
            e.rew_coeff = self.rew_coeff

            observation = e.reset()
            obs.append(observation)
        # extend obs to see neighbors

        if self.swarm_obs and self.num_agents > 1:
            obs_ext = self.extend_obs_space(obs)
            obs = obs_ext

        self.scene.reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.obstacles)
        # self.scene.reset(tuple(e.goal for e in self.envs), self.all_dynamics())
            
        return obs

    # noinspection PyTypeChecker
    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos

        if self.swarm_obs and self.num_agents > 1:
            obs_ext = self.extend_obs_space(obs)
            obs = obs_ext

        ## SWARM REWARDS
        # -- BINARY COLLISION REWARD
        self.dist = spatial.distance_matrix(x=self.pos, y=self.pos)
        self.collisions = (self.dist < 2 * self.envs[0].dynamics.arm).astype(np.float32)
        np.fill_diagonal(self.collisions, 0.0) # removing self-collision
        self.rew_collisions_raw = - np.sum(self.collisions, axis=1)
        self.rew_collisions = self.rew_coeff["quadcol_bin"] * self.rew_collisions_raw

        for i in range(self.num_agents):
            rewards[i] += self.rew_collisions[i]
            infos[i]["rewards"]["rew_quadcol"] = self.rew_collisions[i]
            infos[i]["rewards"]["rewraw_quadcol"] = self.rew_collisions_raw[i]

        if self.quads_mode == "circular_config":
            for i, e in enumerate(self.envs):
                dis = np.linalg.norm(self.pos[i] - e.goal)
                if abs(dis) < 0.02:
                    self.settle_count[i] += 1
                else:
                    self.settle_count[i] = 0
                    break

            # drones settled at the goal for 1 sec
            control_step_for_one_sec = int(self.envs[0].control_freq * 2)
            tmp_count = self.settle_count >= control_step_for_one_sec
            if all(tmp_count):
                tmp_goals = []
                for i, env in enumerate(self.envs):
                    if self.set_transition:
                        if i ==0: # switch goal after all agents are initialized
                            digit = next(self.iter_digits)
                        tmp_single_goal = self.single_digit_goal(digit, i)
                        env.goal = tmp_single_goal
                    else:
                        tmp_goals = self.digit_goals(i)
                        env.goal = tmp_goals

                    # Add settle rewards
                    self.rews_settle_raw[i] = control_step_for_one_sec
                    self.rews_settle[i] = self.rew_coeff["quadsettle"] * self.rews_settle_raw[i]
                    rewards[i] += self.rews_settle[i]
                    infos[i]["rewards"]["rew_quadsettle"] = self.rews_settle[i]
                    infos[i]["rewards"]["rewraw_quadsettle"] = self.rews_settle_raw[i]

                self.rews_settle = np.zeros(self.num_agents)
                self.rews_settle_raw = np.zeros(self.num_agents)
                self.settle_count = np.zeros(self.num_agents)

        elif self.quads_mode == "same_goal":
            tick = self.envs[0].tick
            # teleport every 5 secs
            control_step_for_five_sec = int(5.0 * self.envs[0].control_freq)
            if tick % control_step_for_five_sec == 0 and tick > 0:
                box_size = self.envs[0].box
                x = (random.random() * 2 - 1) * box_size
                y = (random.random() * 2 - 1) * box_size
                z = random.random() * 2 * box_size
                if z < 0.25:
                    z = 0.25

                self.goal = [[x, y, z] for i in range(self.num_agents)]
                self.goal = np.array(self.goal)

                for i, env in enumerate(self.envs):
                    env.goal = self.goal[i]

        ## DONES
        if any(dones):
            obs = self.reset()
            dones = [True] * len(dones)  # terminate the episode for all "sub-envs"

        return obs, rewards, dones, infos

    def render(self, mode='human', verbose=False):
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.envs[0].control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        self.scene.render_chase(all_dynamics=self.all_dynamics(), goals=goals, obstacles=self.obstacles, mode=mode )
        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if mode == 'human' and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f'Last render + simulation time {render_time + simulation_time:.3f}')
                print(f'Rendering does not keep up, rendering every {self.render_every_nth_frame} frames')
        elif simulation_time + render_time < realtime_control_period * (self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f'We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames')

        if self.render_every_nth_frame > 4:
            self.render_every_nth_frame = 4
            print(f'Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames')

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()