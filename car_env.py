import mujoco
import mujoco.viewer
import numpy as np
import time
import random

class CarEnv:
    def __init__(self, xml_path='safe_drive_env.xml'):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.brick_names = ['brick1', 'brick2', 'brick3', 'brick4']
        self.box_names = ['rand_box1', 'rand_box2', 'rand_box3']
        self.wall_names = ['wall_left', 'wall_right', 'wall_bottom', 'wall_top']
        self.prev_vels = [0.0, 0.0]
        self.curr_vels = [0.0, 0.0]

    def randomize_boxes(self, easy_mode=False):
        for i in range(1, 4):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"rand_box{i}_body")
            rand_x = np.random.uniform(-4, 5)
            if easy_mode:
                rand_x = np.random.uniform(7, 8)
            rand_y = np.random.uniform(-3, 3)
            z = self.model.body_pos[body_id][2]
            self.model.body_pos[body_id] = [rand_x, rand_y, z]

            # randomize orientation in the xy plane
            theta = np.random.uniform(-np.pi, np.pi)
            quat = [0, np.sin(theta / 2), np.cos(theta / 2), 0]  # Z-axis rotation
            self.model.body_quat[body_id] = quat

    def randomize_target(self, easy_mode=False):
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target_site')
        self.model.site_pos[target_id][:2] = [6, random.randint(-3, 3)]
        if easy_mode:
            self.model.site_pos[target_id][0:2] = [0, 2]

    def reset(self, easy_mode=False):
        mujoco.mj_resetData(self.model, self.data)
        self.randomize_boxes(easy_mode=easy_mode)
        self.randomize_target(easy_mode=easy_mode)
        mujoco.mj_step(self.model, self.data)
        self.prev_vels = [0.0, 0.0]

    def step(self, action):
        self.data.ctrl = action
        mujoco.mj_step(self.model, self.data)
        self.prev_vels = self.curr_vels.copy() 
        self.curr_vels = [np.linalg.norm(self.data.qvel[:2]), self.data.qvel[5]]
        # print(f'past: {self.prev_vels}')
        # print(f'curr: {self.curr_vels}')
        # print(f'relative pos: [{self.relative_position()[0]:2f}, {self.relative_position()[1]:2f}')
        # time.sleep(0.02)
        if self.viewer is not None:
            self.viewer.sync()
    
    def get_observation(self):
        # returns 14D vector including 10D range readings, 2D current velocities and 2D previous velocities
        rf_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"rf_{i}") for i in range(1, 11)]
        rf = [self.data.sensordata[rf_id] for rf_id in rf_ids]
        return np.concatenate([rf, self.curr_vels, self.prev_vels])

    def collided(self):
        # mujoco.mj_step(self.model, self.data)
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
    
    def close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def relative_position(self):
        # Returns [dx, dy]
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        car_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "car")

        target_pos = self.data.site_xpos[target_id][:2]
        car_pos = self.data.xpos[car_id][:2]
        return car_pos - target_pos

    def done(self):
        return self.collided() or np.linalg.norm(self.relative_position()) <= 0.5
    
    def get_state(self):
        # Returns 14D vector including 10D rangefinding, 2D previous velocities, and 2D distance from target
        rf_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"rf_{i}") for i in range(1, 11)]
        rfs = [self.data.sensordata[rf_id] for rf_id in rf_ids]
        relative_pos = self.relative_position()
        return np.concatenate([rfs, self.prev_vels, relative_pos])
        