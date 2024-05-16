import gym
import time

import numpy as np
import tensorflow as tf

from collections import deque



class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode='none')
        self.name = ''
        self.rewards = deque([])
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        self.model = self.generate_model()

    def generate_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=64, input_dim=4))
        model.add(tf.keras.layers.Dense(units=16, input_dim=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=2, input_dim=16, activation='softmax'))
        model.compile(optimizer = self.optimizer)
        return model
    
    def generate_action(self, state):
        action_prob_tensor = np.array(self.model(tf.convert_to_tensor([state]))[0])
        action = np.random.choice(2, 1, p=action_prob_tensor)[0]
        return action, action_prob_tensor[action]
    
    def make_episode(self, epsilon=0.1):
        T = []
        observation = self.env.reset()
        for t in range(500):
            state = observation[0] if t == 0 else observation
            action, prob = self.generate_action(state)
            observation, reward, terminated, _, _ = self.env.step(action)
            T.append((state, prob, action, reward))
            if terminated:
                break
        return T
    
    def train_model(self, T):
        states = []
        actions = []
        for t in T:
            states.append(t[0])
            act = [0, 0]
            act[t[2]] = 1
            actions.append(act)
        
        rewards = []
        G = 0    
        for t in T[::-1]:
            G = G * 0.99 + t[3]
            rewards.append(G)
        rewards = rewards[::-1]
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
    
        with tf.GradientTape() as tape:
            policies = self.model(states)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * rewards)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def test_episode(self, epsilon=0):
        test_env = gym.make('CartPole-v1', render_mode='human')
        T = []
        observation = test_env.reset()
        for t in range(500):

            state = observation[0] if t == 0 else observation
            action_prob_tensor = np.array(self.model(tf.convert_to_tensor([state]))[0])
            action = list(action_prob_tensor).index(max(action_prob_tensor))
            observation, reward, terminated, _, _ = test_env.step(action)
            T.append(state)
            if terminated:
                break
        return T

    def learn(self):
        episode = 1000
        for e in range(episode):
            T = self.make_episode()
            print(e, len(T))
            self.train_model(T)
            if e %100 == 0:
                t = self.test_episode()
                print(e, len(t))
                time.sleep(1)
        

if __name__ == "__main__":
    mc_cartpole = CartPole()
    mc_cartpole.learn()
    # mc_cartpole.learn()
    