# Overview
Replication of the LSAC-APP algorithm from the paper "Safe Reinforcement Learning With Stability Guarantee for Motion Planning of Autonomous Vehicles" (Zhang et al., 2021). This project replaces the original Tensorflow + Gazebo setup with a PyTorch and a custom MuJoCo environment to train the policy. 
# Tool Stack
- **Framework**: PyTorch
- **Simulator**: MuJoCo (via gymnasium)
- **RL Algorithm**: Soft Actor-Critic (SAC) with risk-sensitive reward and Lyapunov critic
- **Environment**: Differential-drive robot with sparse LiDAR observations in a 2D obstacle world
