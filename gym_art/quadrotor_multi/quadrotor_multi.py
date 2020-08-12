import copy
import math

import numpy as np
import scipy as scp
from scipy import spatial

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
                 resample_goals=False, t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False):

        super().__init__()

        self.num_agents = num_agents
        self.envs = []

        for i in range(self.num_agents):
            e = QuadrotorSingle(
                dynamics_params, dynamics_change, dynamics_randomize_every, dyn_sampler_1, dyn_sampler_2,
                raw_control, raw_control_zero_middle, dim_mode, tf_control, sim_freq, sim_steps,
                obs_repr, ep_time, obstacles_num, room_size, init_random_state,
                rew_coeff, sense_noise, verbose, gravity, t2w_std, t2t_std, excite, dynamics_simplification, e_id=i
            )
            self.envs.append(e)

        self.resample_goals = resample_goals

        self.scene = None

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

        # reward shaping
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=0.
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

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def reset(self):
        if self.num_agents > 1:
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

        delta = 0.0
        for i, e in enumerate(self.envs):
            # x = 0, -delta, +delta, -2*delta, +2*delta, etc.
            goal_x = ((-1) ** i) * (delta * math.ceil(i / 2))
            goal = np.array([goal_x, 0., 2.0])
            # TODO: randomize goals? more patterns?
            e.goal = goal
            e.rew_coeff = self.rew_coeff

            observation = e.reset()

            if self.num_agents == 1:
                obs = observation
            else:
                obs.append(observation)

        self.scene.reset(tuple(e.goal for e in self.envs), self.all_dynamics())

        return obs

    # noinspection PyTypeChecker
    def step(self, actions):
        if self.num_agents > 1:
            obs, rewards, dones, infos = [], [], [], []

        if self.num_agents <= 1:
            actions = [actions]

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            observation, reward, done, info = self.envs[i].step(a)

            if self.num_agents == 1:
                obs = observation
                rewards = reward
                dones = done
                infos = info
            else:
                obs.append(observation)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos

        ## SWARM REWARDS
        # -- BINARY COLLISION REWARD
        # self.dist = spatial.distance_matrix(x=self.pos, y=self.pos)

        # A little bit faster than above, 0.6s / M framesteps
        self.dist = spatial.distance.cdist(XA=self.pos, XB=self.pos)
        # print('pos: ', self.pos)

        self.collisions = (self.dist < 2 * self.envs[0].dynamics.arm).astype(np.float32)
        np.fill_diagonal(self.collisions, 0.0) # removing self-collision
        self.rew_collisions_raw = - np.sum(self.collisions, axis=1)
        self.rew_collisions = self.rew_coeff["quadcol_bin"] * self.rew_collisions_raw

        if self.num_agents == 1:
            rewards += self.rew_collisions[i]
            infos["rewards"]["rew_quadcol"] = self.rew_collisions[i]
            infos["rewards"]["rewraw_quadcol"] = self.rew_collisions_raw[i]
        else:
            for i in range(self.num_agents):
                rewards[i] += self.rew_collisions[i]
                infos[i]["rewards"]["rew_quadcol"] = self.rew_collisions[i]
                infos[i]["rewards"]["rewraw_quadcol"] = self.rew_collisions_raw[i]

        ## DONES
        if self.num_agents == 1:
            if dones == True:
                obs = self.reset()
        else:
            if any(dones):
                obs = self.reset()
                dones = [True] * len(dones)  # terminate the episode for all "sub-envs"

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        goals = tuple(e.goal for e in self.envs)
        return self.scene.render_chase(all_dynamics=self.all_dynamics(), goals=goals, mode=mode)
