import numpy as np
from scipy import spatial
EPS = 1e-6


class SingleObstacle():
    def __init__(self, max_init_vel=1., init_box=2.0, mean_goals=2.0, goal_central=np.array([0., 0., 2.0]), mode='no_obstacles',
                 type='sphere', size=0.0, quad_size=0.04, dt=0.05, force_mode="electron"):
        self.max_init_vel = max_init_vel
        self.init_box = init_box
        self.mean_goals = mean_goals
        self.goal_central = goal_central
        self.mode = mode
        self.type = type
        self.size = size
        self.quad_size = quad_size
        self.dt = dt
        self.force_mode = force_mode
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.omega = np.zeros(3) # angular_velocity
        self.reset()

    def reset(self, set_obstacle=False):
        if set_obstacle:
            if self.mode == 'static':
                self.static_obstacle()
            elif self.mode == 'dynamic':
                self.dynamic_obstacle()
            else:
                pass
        else:
            self.pos = np.array([20.0, 20.0, 20.0])
            self.vel = np.array([0., 0., 0.])

        size = self.get_size()
        quaternion = self.get_quaternion()
        return quaternion, size, self.pos, self.vel

    def step_electron(self, set_obstacles=False):
        force_pos = 2 * self.goal_central - self.pos
        rel_force_goal = force_pos - self.goal_central
        force_noise = np.random.uniform(low=-0.5 * rel_force_goal, high=0.5 * rel_force_goal)
        force_pos = force_pos + force_noise
        rel_force_obstacle = force_pos - self.pos
        radius = 2.0 * np.linalg.norm(rel_force_obstacle)
        radius = max(EPS, radius)

        force_direction = rel_force_obstacle / radius
        force = radius * radius * force_direction
        # Calculate acceleration, F = ma, here, m = 1.0
        acc = force
        # Calculate velocity and pos
        if set_obstacles:
            self.vel += self.dt * acc
            self.pos += self.dt * self.vel

    def step(self, quads_pos=None, quads_vel=None, set_obstacles=False):
        if self.force_mode == "electron":
            self.step_electron(set_obstacles=set_obstacles)
        elif self.force_mode == "gravity":
            self.vel = self.vel
            self.pos += self.dt * self.vel

        if np.linalg.norm(self.pos - self.goal_central) <= 0.1:
            print(np.linalg.norm(self.pos - self.goal_central))

        # The pos and vel of the obstacle give by the agents
        rel_pos_obstacle_agents = self.pos - quads_pos
        rel_vel_obstacle_agents = self.vel - quads_vel
        size = self.get_size()
        quaternion = self.get_quaternion()
        return quaternion, size, rel_pos_obstacle_agents, rel_vel_obstacle_agents

    def static_obstacle(self):
        pass

    def dynamic_obstacle(self):
        # Init position for an obstacle
        x, y, z = np.random.uniform(-self.init_box, self.init_box, size=(3,))
        z += self.mean_goals
        if z < 0.5 * self.mean_goals:
            z = 0.5 * self.mean_goals

        self.pos = np.array([x, y, z])

        # Init velocity for an obstacle
        # obstacle_vel = np.random.uniform(low=-self.max_init_vel, high=self.max_init_vel, size=(3,))
        obstacle_vel_direct = self.goal_central - self.pos
        obstacle_vel_direct_noise = np.random.uniform(low=-0.1, high=0.1, size=(3,))
        obstacle_vel_direct += obstacle_vel_direct_noise
        obstacle_vel_magn = np.random.uniform(low=0.5 * self.max_init_vel, high=self.max_init_vel)
        obstacle_vel = obstacle_vel_magn / (np.linalg.norm(obstacle_vel_direct) + EPS) * obstacle_vel_direct
        self.vel = obstacle_vel

    def cube_detection(self, pos_quads=None):
        # https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
        # Sphere vs. AABB
        collision_arr = np.zeros(len(pos_quads))
        for i, pos_quad in enumerate(pos_quads):
            rel_pos = np.maximum(self.pos - 0.5 * self.size, np.minimum(pos_quad, self.pos + 0.5 * self.size))
            distance = np.dot(rel_pos - pos_quad, rel_pos - pos_quad)
            if distance < self.quad_size ** 2:
                collision_arr[i] = 1.0

        return collision_arr

    def sphere_detection(self, pos_quads=None):
        dist = np.linalg.norm(pos_quads - self.pos, axis=1)
        collision_arr = (dist < (self.quad_size + 0.5 * self.size)).astype(np.float32)
        return collision_arr

    def collision_detection(self, pos_quads=None):
        if self.type == 'cube':
            collision_arr = self.cube_detection(pos_quads)
        elif self.type == 'sphere':
            collision_arr = self.sphere_detection(pos_quads)
        else:
            raise NotImplementedError()

        return collision_arr

    def get_size(self):
        if self.type == "sphere":
            return np.array([self.size, self.size, self.size])
        elif self.type == "cube":
            return np.array([self.size, self.size, self.size])

    def get_quaternion(self):
        omega_value = np.linalg.norm(self.omega)
        theta = omega_value / 2
        unit_omega = self.omega / max(omega_value, EPS)
        quaternion = [np.cos(theta)]
        quaternion.extend(unit_omega * np.sin(theta))
        return np.array(quaternion)