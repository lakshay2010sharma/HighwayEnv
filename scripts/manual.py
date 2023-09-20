import sys
import os
path = os.path.dirname(os.getcwd())
sys.path.insert(0, path)

import gymnasium as gym
import highway_env
from utils import record_videos, show_videos

highway_env.register_highway_envs()

env = gym.make('highway-v0', render_mode="rgb_array")
env = record_videos(env)
# env.configure({"vehicles_density": 2.5, "right_lane_reward": 0.1})
env.configure({
    "vehicles_density": 2.5,
    "manual_control": True, 
    "real_time_rendering": True
})
(obs, info), done = env.reset(), False

# Make agent
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "budget": 50,
#     "gamma": 0.7,
# }
# agent = agent_factory(env, agent_config)

# Run episode
# for step in trange(env.unwrapped.config["duration"], desc="Running..."):
#     action = agent.act(obs)
#     obs, reward, done, truncated, info = env.step(action)
while not done:
    _, _, done, _, _ = env.step(env.action_space.sample()) # with manual control, these actions are ignored
    env.render()
    
env.close()
show_videos()