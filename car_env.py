import mujoco
import mujoco.viewer
import numpy as np

class CarEnv:
    def __init__(self, xml_path='safe_drive_env.xml'):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.brick_names = ['brick1', 'brick2', 'brick3']
        self.box_names = ['rand_box1', 'rand_box2', 'rand_box3']
        self.wall_names = ['wall_left', 'wall_right', 'wall_bottom', 'wall_top']
    

    def randomize_boxes(self):
        for i in range(1, 4):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"rand_box{i}_body")
            rand_x = np.random.uniform(-4, 6)
            rand_y = np.random.uniform(-3, 3)
            z = self.model.body_pos[body_id][2]
            self.model.body_pos[body_id] = [rand_x, rand_y, z]

            # randomize orientation in the xy plane
            theta = np.random.uniform(-np.pi, np.pi)
            quat = [0, np.sin(theta / 2), np.cos(theta / 2), 0]  # Z-axis rotation
            self.model.body_quat[body_id] = quat

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.randomize_boxes()
        mujoco.mj_step(self.model, self.data)

    def step(self, action):
        self.data.ctrl = action
        mujoco.mj_step(self.model, self.data)
    
    def get_observation(self):
        rf_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"rf_{i}") for i in range(1, 11)]
        rf = [self.data.sensordata[rf_id] for rf_id in rf_ids]
        linear_vel = np.linalg.norm(self.data.qvel[:2])
        angular_vel = self.data.qvel[5]
        return np.concatenate([rf, [linear_vel, angular_vel]])

    def collided(self):
        mujoco.mj_step(self.model, self.data)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if "floor" in (geom1, geom2):
                continue

            if any(geom in (geom1, geom2) for geom in self.brick_names + self.box_names + self.wall_names):
                return True
        
        return False

    def launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)