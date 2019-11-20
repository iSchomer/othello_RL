from final_project.othello_env import Board, OthelloGame
import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import random


class OthelloAgent:
    def __init__(self):
        self.state_size = 64
        self.action_size = 64
        self.memory = deque(maxlen=2000)
        self.gamma = 1.0  # episodic --> undiscounted
        self.epsilon = .10
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self.build_model()

    def build_model(self):
        # Feed-forward NN
        model = tf.keras.Sequential()
        model.add(layers.Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='sigmoid'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, 7), random.randint(0, 7)]

        # Take an action based on the Q function
        act_values = self.model.predict(state)
        # TODO - chance action from int(0, 63) to [int(0, 7), int(0, 7)]
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            # TODO: change target_f to a more intuitive name
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # optional epsilon decay feature
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    # initialize agent and environment
    agent = OthelloAgent()
    game = OthelloGame(interactive=False, show_steps=True)

    # agent.load("final_project/othello_backup_v1")
    done = False
    batch_size = 32
    episodes = 1

    for e in range(episodes):
        game.reset()
        game.start()
        state = game.get_state()  # 8x8 numpy array
        state = np.reshape(state, [1, 64])   # 1x64 vector

        for move in range(60):   # max amount of moves
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            next_state = np.reshape(next_state, [1, 64])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                # terminal reward is 1 for win, -1 for lose
                # use this as an indexing code to get the result
                results = ['Tie', 'Win', 'Loss']
                result = results[reward]
                print("episode {}: {} moves, Result: {}, e: {:.2}"
                      .format(e, move, result, agent.epsilon))
                break
            # TODO - maybe only update every batch_size moves
            #       (instead of every move after batch_size)?
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("final_project/othello_backup_v1")
