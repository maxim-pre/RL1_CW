from gymnasium.wrappers import RecordVideo
import gymnasium as gym

env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="cartpole-agent",    # Folder to save videos
    name_prefix="eval",               # Prefix for video filenames
    episode_trigger=lambda x: True    # Record every episode
)


obs, info = env.reset()

while 1:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()