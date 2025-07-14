from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("safe_drive_env.xml")
data = mujoco.MjData(model)

class SafeDriveEnv(Env):
    def __init__(self, xml_path="safe_drive_env.xml", render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Action: left & right wheel velocity
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: 10 lidar + 2 pos + 2 vel + 2 goal vector
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.goal = np.array([5.0, 0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

    
        # Randomize goal
        self.goal = np.array([np.random.uniform(3, 6), np.random.uniform(-3, 3)])
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.model.body_pos[target_id] = np.array([*self.goal, 0.05])


        # for name in ["rand_box1", "rand_box2", "rand_box3"]:
        #     x = np.random.uniform(-6, 6)
        #     y = np.random.uniform(-3, 3)
        #     z = 0.2  # height

        #     body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        #     jnt_adr = self.model.body_jntadr[body_id]
        #     qpos_adr = self.model.jnt_qposadr[jnt_adr]

        #     self.data.qpos[qpos_adr:qpos_adr+3] = np.array([x, y, z])

        return self._get_obs(), {}

    def step(self, action):
        # Scale from [-1, 1] to [-5, 5] as per ctrlrange
        scaled_action = np.clip(action, -1, 1) * 1

        self.data.ctrl[0] = scaled_action[0]
        self.data.ctrl[1] = scaled_action[1]

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        robot_pos = self._get_robot_pos()
        distance = np.linalg.norm(robot_pos - self.goal)
        reward = -distance

        collided = self._check_collision()
        done = distance < 0.3 or collided

        return obs, reward, done, False, {}


    def _get_obs(self):
        lidar = self._simulate_lidar()
        qpos = self.data.qpos[:2]
        qvel = self.data.qvel[:2]
        rel_goal = self.goal - qpos
        return np.concatenate([lidar, qpos, qvel, rel_goal]).astype(np.float32)

    def _simulate_lidar(self, n_beams=10, fov_deg=270, max_range=10.0):
        angles = np.linspace(-fov_deg / 2, fov_deg / 2, n_beams)
        robot_pos = self.data.qpos[:2]
        robot_z = 0.1  # Sensor height
        robot_yaw = self.data.qpos[2] if self.data.qpos.shape[0] > 2 else 0.0

        lidar_readings = []

        for angle_deg in angles:
            # Convert angle to world frame
            angle_rad = np.deg2rad(angle_deg) + robot_yaw
            dir_x = np.cos(angle_rad)
            dir_y = np.sin(angle_rad)

            # Starting point and direction
            pnt_start = np.array([robot_pos[0], robot_pos[1], robot_z])
            vec = np.array([dir_x, dir_y, 0]) * max_range

            # Output containers
            geom_id = np.array([-1], dtype=np.int32)  # writable output
            dist_fraction = mujoco.mj_ray(
                self.model,
                self.data,
                pnt_start,
                vec,
                None,     # geomgroup (use all)
                1,        # flg_static: 1 to include only static geoms
                -1,       # bodyexclude: -1 to exclude none
                geom_id   # output: ID of hit geom
            )

            # Compute distance traveled
            distance = dist_fraction * max_range if geom_id[0] != -1 else max_range
            lidar_readings.append(distance)

        return np.array(lidar_readings, dtype=np.float32)



    def _get_robot_pos(self):
        return self.data.qpos[:2]

    def _check_collision(self):
        robot_keywords = ["chassis", "wheel"]
        danger_keywords = ["brick", "rand_box"]

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if geom1 and geom2:
                if (
                    any(key in geom1 for key in robot_keywords) and any(key in geom2 for key in danger_keywords)
                ) or (
                    any(key in geom2 for key in robot_keywords) and any(key in geom1 for key in danger_keywords)
                ):
                    print(f"Contact: {geom1} <-> {geom2}")
                    return True
        return False



    def render(self):
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
