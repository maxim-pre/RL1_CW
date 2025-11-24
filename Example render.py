import gymnasium as gym

env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
obs, info = env.reset()

# Render mode:
# None: no rendering
# "human": use for video

while 1:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()