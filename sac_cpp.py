import torch
import torch.nn.functional as F
import numpy as np
from car_env import CarEnv
from sac_cpp_models import Actor, Critic
from replay_buffer import ReplayBuffer 
from torch.distributions import Normal
from collision_predictor import CollisionPredictor
import time
from action_sampler import ActionSampler

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 2e-4
ALPHA = 0.3
BATCH_SIZE = 256
REPLAY_SIZE = int(1e6)
MAX_EPISODES = 10000
MAX_STEPS = 1000
# RSP = -10
RSP=-5

# Load environment and collision model
env = CarEnv()
f_phi = CollisionPredictor()
f_phi.load_state_dict(torch.load('model/best_collision_predictor.pth'))
f_phi.eval()

replay_buffer = ReplayBuffer(REPLAY_SIZE)

# SAC networks
actor = Actor()
critic1 = Critic()
critic2 = Critic()
target_critic1 = Critic()
target_critic2 = Critic()
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic1_opt = torch.optim.Adam(critic1.parameters(), lr=CRITIC_LR)
critic2_opt = torch.optim.Adam(critic2.parameters(), lr=CRITIC_LR)

def select_action(state):
    state = torch.FloatTensor(state.reshape(1, -1))
    action, _ = actor.sample(state)
    return action.detach().cpu().numpy()[0]

def compute_critic_loss(state, action, reward, next_state, done):
    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_state)
        target_q1 = target_critic1(next_state, next_action)
        target_q2 = target_critic2(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_prob
        target_value = reward + (1 - done) * GAMMA * target_q

    current_q1 = critic1(state, action)
    current_q2 = critic2(state, action)
    loss1 = F.mse_loss(current_q1, target_value)
    loss2 = F.mse_loss(current_q2, target_value)
    return loss1, loss2

def compute_actor_loss(state):
    action, log_prob = actor.sample(state)
    q1 = critic1(state, action)
    q2 = critic2(state, action)
    q = torch.min(q1, q2)
    return (ALPHA * log_prob - q).mean()

def compute_reward(state, next_state, risk, env, speed, log_reward=False):
    rel_pos = state[-2:]
    next_rel_pos = next_state[-2:]
    prev_dist = np.linalg.norm(rel_pos)
    curr_dist = np.linalg.norm(next_rel_pos)

    progress = (prev_dist - curr_dist) * 20
    arrive_bonus = 50 if curr_dist < 0.5 else 0.0
    crash_penalty = -10.0 if env.collided() else 0.0
    # speed_penalty = min(0, 0.8-speed) * 2
    speed_penalty = 0

    # reward = progress * 10 + arrive_bonus + crash_penalty + RSP * risk
    reward = progress + arrive_bonus + crash_penalty + RSP * risk + speed_penalty

    if log_reward:
        print(f"reward={reward:.2f} (progress={progress:.2f}, arrive={arrive_bonus:.2f}, crash={crash_penalty:.2f}, risk={risk:.2f}, RSP*risk={RSP*risk:.4f}, speed_penalty={speed_penalty})")
        time.sleep(0.1)
    return reward


# Training loop
total_episode_return = 0
pretrain_steps = 0
for ep in range(MAX_EPISODES):
    ### 
    if ep % 200 == 0:
        env.launch_viewer()
    if ep % 200 == 1:
        env.close_viewer()
    ###
    env.reset(easy_mode=False) ###
    state = env.get_state()
    episode_reward = 0

    action_sampler = ActionSampler(turn_momentum=0.5, drift_strength=0.2) ###

    if ep > 0 and ep <= 500 and ep % 100 == 0:
        ALPHA -= 0.05
        print(f"[Episode {ep}]: alpha={ALPHA}")

    for step in range(MAX_STEPS):
        if pretrain_steps > 0:
            action = action_sampler.sample()
        else: 
            action = select_action(state)
        pretrain_steps -= 1

        env.step(action)
        next_state = env.get_state()
        done = env.done()

        obs = env.get_observation()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        # Collision risk
        with torch.no_grad():
            logits = f_phi(torch.FloatTensor(obs).unsqueeze(0))
            risk = torch.sigmoid(logits).item()

        speed = np.abs(next_state[-4])
        reward = 0 if step == 0 else compute_reward(state, next_state, risk, env, speed, log_reward=ep%200==0 and step%3==0) ###
        # reward = 0 if step == 0 else compute_reward(state, next_state, risk, env, speed)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_episode_return += reward

        if env.done():
            break

    # Update networks
    if len(replay_buffer) > BATCH_SIZE:
        for _ in range(1):
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = [torch.FloatTensor(x) for x in batch]

            # Critic update
            loss1, loss2 = compute_critic_loss(states, actions, rewards, next_states, dones)
            critic1_opt.zero_grad()
            loss1.backward()
            critic1_opt.step()

            critic2_opt.zero_grad()
            loss2.backward()
            critic2_opt.step()

            # Actor update
            actor_loss = compute_actor_loss(states)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Soft update
            for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    if ep > 0 and ep % 200 == 0:
        print(f"[Episode {ep}] Avg Return Over 200 Ep: {total_episode_return / 200:.2f}")
        total_episode_return = 0

# Save model
torch.save(actor.state_dict(), "model/sac_cpp_actor.pth")
torch.save(critic1.state_dict(), "model/sac_cpp_critic1.pth")
