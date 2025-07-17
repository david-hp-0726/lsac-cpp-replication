# Real-Time Collision Prediction for Autonomous Vehicles (MuJoCo + PyTorch)

This project mainly replicates the **collision probability prediction (CPP)** component from the paper:

> *Safe Reinforcement Learning with Stability Guarantee for Motion Planning of Autonomous Vehicles*  
> IEEE TNNLS, 2021 ([link](https://doi.org/10.1109/TNNLS.2021.3084685))

The CPP model is trained on a MuJoCo-based driving simulation to predict collision risk based on range sensor inputs and velocity data. PyTorch is used as the deep learning framework. 
