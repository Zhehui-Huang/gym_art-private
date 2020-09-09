import copy
import numpy as np

import gym_art.quadrotor_multi.rendering3d as r3d

from gym_art.quadrotor_multi.quadrotor_visualization import ChaseCamera, SideCamera, GlobalCamera, quadrotor_simple_3dmodel, \
    quadrotor_3dmodel
from gym_art.quadrotor_multi.params import quad_color
from gym_art.quadrotor_multi.quad_utils import calculate_collision_matrix
from scipy import spatial


class Quadrotor3DSceneMulti:
    def __init__(
            self, w, h,
            quad_arm=None, models=None, obstacles=None, visible=True, resizable=True, goal_diameter=None,
            viewpoint='chase', obs_hw=None, obstacle_mode='no_obstacles'
    ):
        if obs_hw is None:
            obs_hw = [64, 64]

        self.window_target = None
        self.window_w, self.window_h = w, h
        self.resizable = resizable
        self.viepoint = viewpoint
        self.obs_hw = copy.deepcopy(obs_hw)

        self.quad_arm = quad_arm
        self.obstacles = obstacles
        self.obstacle_mode = obstacle_mode
        self.models = models

        self.quad_transforms, self.shadow_transforms, self.goal_transforms = [], [], []
        self.obstacles_transforms = []

        if goal_diameter:
            self.goal_forced_diameter = goal_diameter
        else:
            self.goal_forced_diameter = None

        self.diameter = self.goal_diameter = -1
        self.update_goal_diameter()

        if self.viepoint == 'chase':
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        elif self.viepoint == 'side':
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)
        elif self.viepoint == 'global':
            self.chase_cam = GlobalCamera(view_dist=self.diameter * 15)

        self.fpv_lookat = None

        self.scene = None
        self.window_target = None
        self.obs_target = None
        self.video_target = None

    def update_goal_diameter(self):
        if self.quad_arm is not None:
            self.diameter = 2 * self.quad_arm
        else:
            self.diameter = 2 * np.linalg.norm(self.models[0].params['motor_pos']['xyz'][:2])

        if self.goal_forced_diameter:
            self.goal_diameter = self.goal_forced_diameter
        else:
            self.goal_diameter = self.diameter

    def _make_scene(self):
        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

<<<<<<< HEAD
        self.quad_transforms, self.shadow_transforms, self.goal_transforms, self.collision_transforms, self.obstacle_transforms = [], [], [], [], []
=======
        self.quad_transforms, self.shadow_transforms, self.goal_transforms = [], [], []
        self.obstacles_transforms = []
>>>>>>> 906f412... Vertical pattern with global camera

        for i, model in enumerate(self.models):
            if model is not None:
                quad_transform = quadrotor_3dmodel(model, quad_id=i)
            else:
                quad_transform = quadrotor_simple_3dmodel(self.diameter)
            self.quad_transforms.append(quad_transform)

            self.shadow_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75 * self.diameter, 32))
            )
            self.collision_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.0), r3d.sphere(0.75 * self.diameter, 32))
            )

        # TODO make floor size or walls to indicate world_box
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.15, 0.25),
                                      r3d.rect((1000, 1000), (0, 100), (0, 100)))

        self.update_goal_diameter()
        self.chase_cam.view_dist = self.diameter * 15

        self.create_goals()

        bodies = [r3d.BackToFront([floor, st]) for st in self.shadow_transforms]
        bodies.extend(self.goal_transforms)
        bodies.extend(self.quad_transforms)

<<<<<<< HEAD
        if self.obstacle_mode != 'no_obstacles':
            self.create_obstacles()
            bodies.extend(self.obstacle_transforms)
=======
        # TODO: obstacles?
        # self.create_obstacles()
        # bodies.extend(self.obstacles_transforms)
>>>>>>> 906f412... Vertical pattern with global camera

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)
        self.scene = r3d.Scene(batches=[batch], bgcolor=(0, 0, 0))
        self.scene.initialize()

<<<<<<< HEAD
        # Collision spheres have to be added in the ending after everything has been rendered, as it transparent
        bodies = []
        bodies.extend(self.collision_transforms)
        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)
        self.scene.batches.extend([batch])

    def create_obstacles(self):
        for item in self.obstacles.obstacles:
            color = quad_color[0]
            if item.type == 'cube':
                obstacle_transform = r3d.transform_and_color(np.eye(4), color, r3d.box(item.size, item.size, item.size))
            elif item.type == 'sphere':
                obstacle_transform = r3d.transform_and_color(np.eye(4), color, r3d.sphere(item.size / 2, 18))
            else:
                raise NotImplementedError()

            self.obstacle_transforms.append(obstacle_transform)

    def update_obstacles(self, obstacles):
        for i, g in enumerate(obstacles.obstacles):
            self.obstacle_transforms[i].set_transform(r3d.translate(g.pos))
=======
    def create_obstacles(self):
        for i in range(3):
            color = quad_color[i]
            obstacle_transform = r3d.transform_and_color(np.eye(4), color, r3d.sphere(0.15, 2))
            self.obstacles_transforms.append(obstacle_transform)

    def update_obstacles(self, obstacles):
        for i, g in enumerate(obstacles):
            self.obstacles_transforms[i].set_transform(r3d.translate(g[0:3]))
>>>>>>> 906f412... Vertical pattern with global camera

    def create_goals(self):
        for i in range(len(self.models)):
            color = quad_color[i % len(quad_color)]
            goal_transform = r3d.transform_and_color(np.eye(4), color, r3d.sphere(self.goal_diameter / 2, 18))
            self.goal_transforms.append(goal_transform)

    def update_goals(self, goals):
        for i, g in enumerate(goals):
            self.goal_transforms[i].set_transform(r3d.translate(g[0:3]))

    def update_models(self, models):
        self.models = models

        if self.video_target is not None:
            self.video_target.finish()
            self.video_target = None
        if self.obs_target is not None:
            self.obs_target.finish()
            self.obs_target = None
        if self.window_target:
            self._make_scene()

    def reset(self, goals, dynamics, obstacles):
        first_goal = goals[0]  # TODO: make a camera that can look at all drones
        self.chase_cam.reset(first_goal[0:3], dynamics[0].pos, dynamics[0].vel)
        self.update_state(dynamics, goals, obstacles)

    def update_state(self, all_dynamics, goals, obstacles):
        if self.scene:
            self.chase_cam.step(all_dynamics[0].pos, all_dynamics[0].vel)
            self.fpv_lookat = all_dynamics[0].look_at()
            # use this to get trails on the goals and visualize the paths they follow
            # bodies = []
            # bodies.extend(self.goal_transforms)
            # world = r3d.World(bodies)
            # batch = r3d.Batch()
            # world.build(batch)
            # self.scene.batches.extend([batch])

            self.update_goals(goals=goals)
<<<<<<< HEAD
            if self.obstacle_mode != 'no_obstacles':
                self.update_obstacles(obstacles=obstacles)

            # computing collisions
            positions = np.array([dyn.pos for dyn in all_dynamics])
            collision_matrix, all_collisions = calculate_collision_matrix(positions, all_dynamics[0].arm)
            collision_sums = np.sum(collision_matrix, axis=1)
=======
            self.update_obstacles(obstacles=obstacles)
>>>>>>> 906f412... Vertical pattern with global camera

            for i, dyn in enumerate(all_dynamics):
                matrix = r3d.trans_and_rot(dyn.pos, dyn.rot)
                self.quad_transforms[i].set_transform_nocollide(matrix)

                shadow_pos = 0 + dyn.pos
                shadow_pos[2] = 0.001  # avoid z-fighting
                matrix = r3d.translate(shadow_pos)
                self.shadow_transforms[i].set_transform_nocollide(matrix)

<<<<<<< HEAD
                matrix = r3d.translate(dyn.pos)
                if collision_sums[i] > 0.0:
                    self.collision_transforms[i].set_transform_and_color(matrix, (1, 0, 0, 0.4))
                else:
                    self.collision_transforms[i].set_transform_and_color(matrix, (0, 0, 0, 0.0))


    def render_chase(self, all_dynamics, goals, mode='human', obstalces=None):
=======
    def render_chase(self, all_dynamics, goals, obstacles, mode='human'):
>>>>>>> 906f412... Vertical pattern with global camera
        if mode == 'human':
            if self.window_target is None:
                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                self._make_scene()
<<<<<<< HEAD
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstalces)
=======
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstacles)
>>>>>>> 906f412... Vertical pattern with global camera
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.window_target)
            return None
        elif mode == 'rgb_array':
            if self.video_target is None:
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                self._make_scene()
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstalces)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.video_target)
            return np.flipud(self.video_target.read())