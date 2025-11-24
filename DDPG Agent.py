import numpy as np
import random
from keras.optimizers import SGD
import tensorflow as tf


class DDPGAgent:

    def __init__(self,
                 env,
                 critic_network,
                 actor_network,
                 critic_learning_rate=0.01,
                 actor_learning_rate=0.01,
                 discount_factor=0.8,
                 minibatch_size=64,
                 t=0.01):
        """
        :param env:

        :param critic_network: A NN that maps state action pairs to values. Input shape should be the same shape as the
        concatenated state and action (28, ). Output shape should be 1.

        :param actor_network: A NN that maps states to actions. Input shape should be the same shape as the states
        (24, ). Output shape should be the same shape as the actions (4, ).
        """
        self.env = env

        self.critic_network = critic_network
        self.actor_network = actor_network

        self.target_critic_network = self.critic_network
        self.target_actor_network = self.actor_network

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.t = t

        self.replay_buffer = []

    def select_action(self, state, add_noise=False, exploratory_noise_variance=0.25):
        # Forward pass.
        action = self.actor_network.predict(np.array([state]), verbose=0)[0]
        # Add exploratory noise.
        if add_noise:
            noise = np.random.normal(0, exploratory_noise_variance, 4)
            action += noise
            action = np.clip(action, -1, 1)
        return action

    def select_target_action(self, state):
        # Forward pass.
        action = self.target_actor_network.predict(np.array([state]), verbose=0)[0]
        return action

    def get_value(self, state, action):
        # Concatenate state and action.
        s_a = np.concatenate([state, action])
        # Get state action value.
        value = self.critic_network.predict(np.array([s_a]), verbose=0)[0][0]
        return value

    def get_target_value(self, state, action):
        # Concatenate state and action.
        s_a = np.concatenate([state, action])
        # Get state action value.
        value = self.target_critic_network.predict(np.array([s_a]), verbose=0)[0][0]
        return value

    def sample_minibatch(self):
        minibatch = random.sample(self.replay_buffer, self.minibatch_size)

        critic_xs = []
        critic_ys = []
        actor_xs = []

        for (s, a, r, s_) in minibatch:

            critic_x = np.concatenate([s, a])
            critic_xs.append(critic_x)

            target_a_ = self.select_target_action(s_)
            critic_y = r + self.discount_factor * self.get_target_value(s_, target_a_)
            critic_ys.append(critic_y)

            actor_xs.append(s)

        return np.array(critic_xs), np.array(critic_ys), np.array(actor_xs)

    def update_critic_network(self, critic_xs, critic_ys):
        # Could also use Huber loss here.
        self.critic_network.compile(optimizer=SGD(learning_rate=self.critic_learning_rate), loss="mse")
        self.critic_network.fit(critic_xs, critic_ys, epochs=1,batch_size=self.minibatch_size)

    def update_actor_network(self, actor_xs):
        # TODO: FIX THIS
        with tf.GradientTape() as tape:
            actions = self.actor_network(actor_xs, training=True)
            actor_xs_tensor = tf.convert_to_tensor(actor_xs)
            combined = tf.concat([actor_xs_tensor, actions], axis=1)
            q_values = self.critic_network(combined, training=True)
            actor_loss = -tf.reduce_mean(q_values)
            actor_grads = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            SGD(learning_rate=self.actor_learning_rate).apply_gradients(
                zip(actor_grads, self.actor_network.trainable_variables)
            )

    def update_target_weights(self):
        new_target_critic_weights = [
            self.t * w1 + (1 - self.t) * w2
            for w1, w2 in zip(self.critic_network.get_weights(), self.target_critic_network.get_weights())
        ]
        self.target_critic_network.set_weights(new_target_critic_weights)

        new_target_actor_weights = [
            self.t * w1 + (1 - self.t) * w2
            for w1, w2 in zip(self.actor_network.get_weights(), self.target_actor_network.get_weights())
        ]
        self.target_actor_network.set_weights(new_target_actor_weights)

    def learn(self, n_episodes=100):

        for n in range(n_episodes):
            print("Episode:", n)

            # Reset environment.
            state, _ = self.env.reset()

            while True:
                print(len(self.replay_buffer))

                # Select action.
                action = self.select_action(state, add_noise=True)

                # Take step.
                new_state, reward, terminal, _, _ = self.env.step(action)

                # Store transition.
                transition = (state, action, reward, new_state)
                self.replay_buffer.append(transition)

                if len(self.replay_buffer) >= self.minibatch_size:
                    # Sample minibatch.
                    critic_xs, critic_ys, actor_xs = self.sample_minibatch()

                    # Update critic network.
                    self.update_critic_network(critic_xs, critic_ys)

                    # Update actor network.
                    self.update_actor_network(actor_xs)

                # Update target weights.
                self.update_target_weights()

                if terminal:
                    break

                state = new_state








# ------------------------------------------------------ #


from keras.models import Sequential
from keras.layers import Input, Dense


critic_network = Sequential([
    Input(shape=(28,)),
    Dense(64, activation="relu", kernel_initializer="he_uniform"),
    Dense(64, activation="relu", kernel_initializer="he_uniform"),
    Dense(1, activation="linear", kernel_initializer="he_uniform")  # No activation.
])

actor_network = Sequential([
    Input(shape=(24,)),
    Dense(64, activation="relu", kernel_initializer="he_uniform"),
    Dense(64, activation="relu", kernel_initializer="he_uniform"),
    Dense(4, activation="tanh", kernel_initializer="glorot_uniform")  # Tanh to map outputs to [-1, 1].
])


# ------------------------------------------------------ #


import gymnasium as gym

env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")


# ------------------------------------------------------ #


test = DDPGAgent(env, critic_network, actor_network)
test.learn()
