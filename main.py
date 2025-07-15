import mujoco
import mujoco.viewer
import time
import numpy as np
from data_collector import DataCollector
from car_env import CarEnv
from action_sampler import ActionSampler

model = mujoco.MjModel.from_xml_path('safe_drive_env.xml')
data = mujoco.MjData(model)
brick_names = ['brick1', 'brick2', 'brick3']
box_names = ['rand_box1', 'rand_box2', 'rand_box3']
wall_names = ['wall_left', 'wall_right', 'wall_bottom', 'wall_top']

def collided(box_num=-1):
    mujoco.mj_forward(model, data)  # ensure state is updated
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if geom1 == "floor" or geom2 == "floor":
            continue
        
        # print(f"geom1: {geom1}, geom2: {geom2}")
        # if box_num != -1 and (geom1 == f"rand_box{box_num}" or geom2 == f"rand_box{box_num}"):
        #     return True

        if any(name in (geom1, geom2) for name in brick_names + box_names + wall_names):
            return True
    return False


def randomize_boxes():
    for i in range(1, 4):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"rand_box{i}_body")
        rand_x = np.random.uniform(-4, 6)
        rand_y = np.random.uniform(-3, 3)
        z = model.body_pos[body_id][2]

        model.body_pos[body_id][:] = [rand_x, rand_y, z]

        theta = np.random.uniform(-np.pi, np.pi)
        quat = [0, np.sin(theta / 2), np.cos(theta / 2), 0]
        model.body_quat[body_id][:] = quat

def get_observation():
    rf_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"rf_{i}") for i in range(1,11)]
    # print(f'rf_ids: {rf_ids}')
    rf_values = [data.sensordata[rf_id] for rf_id in rf_ids]

    car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'car')
    linear_vel = np.linalg.norm(data.qvel[:2])
    angular_vel = data.qvel[5]
    return np.concatenate([rf_values, [linear_vel, angular_vel]])

def simulate():
    car_env = CarEnv()
    sampler = ActionSampler()
    car_env.launch_viewer()
    car_env.randomize_boxes()

    while True:
        action = sampler.sample()
        car_env.step(action=action)
        time.sleep(0.02)
        if car_env.collided():
            break

def generate_dataset(dataset_size=100000):
    car_env = CarEnv()
    data_collector = DataCollector(car_env)
    all_observations = []
    all_labels = []
    episode = 0

    while len(all_labels) < dataset_size:
        data = data_collector.run_episode()
        all_observations.extend(data['observations'])
        all_labels.extend(data['labels'])

        print(f"Episode {episode+1} finished in {len(data['observations'])} steps. Currently at {len(all_labels)} examples")
        episode += 1

    print(len(all_labels))
    X = np.array(all_observations[-dataset_size:])
    Y = np.array(all_labels[-dataset_size:])
    np.save('X.npy', X)
    np.save('Y.npy', Y)

def main():
    # generate_dataset(dataset_size=100000)
    simulate()

if __name__ == '__main__':
    main()