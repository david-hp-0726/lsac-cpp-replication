import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('safe_drive_env.xml')
data = mujoco.MjData(model)

def main():
    data.ctrl[:] = [0.5, 1]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(10000):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)


if __name__ == '__main__':
    main()