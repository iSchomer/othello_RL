from final_project.othello_env import OthelloGame
import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import RandomUniform
import random
from time import process_time
import matplotlib.pyplot as plt


class OthelloAgent:
    def __init__(self, ep, model_type='dense'):
        self.state_size = 64
        self.action_size = 64
        self.tile = 'X'
        self.memory = deque(maxlen=2000)
        self.gamma = 1.0  # episodic --> undiscounted
        self.epsilon = 0.1
        self.epsilon_min = 0.05
        self.epsilon_step = (self.epsilon - self.epsilon_min)/ep
        self.learning_rate = 0.02
        self.model_type = model_type
        self.model = self.build_model()

    def build_model(self):
        init = RandomUniform(minval=-0.5, maxval=0.5)
        if self.model_type == 'dense':
            # Feed-forward NN
            model = tf.keras.Sequential()
            model.add(layers.Dense(50, input_dim=self.state_size, activation='sigmoid', kernel_initializer=init))
            model.add(layers.Dense(64, activation='sigmoid'))
        else:
            # convolutional neural network
            model = tf.keras.Sequential()
            # convolve to a 6x6 grid
            model.add(layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=(8, 8, 1)))
            model.add(layers.Flatten())
            model.add(layers.Dense(50, input_dim=self.state_size, activation='sigmoid', kernel_initializer=init))
            # add a dense layer
            model.add(layers.Dense(64, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, s):
        valid_actions = game.board.get_valid_moves(self.tile)
        if np.random.rand() <= self.epsilon and not testing:
            random.shuffle(valid_actions)
            return valid_actions[0]
        else:
            # Take an action based on the Q function
            all_values = self.model.predict(s)
            # return the VALID action with the highest network value
            # use an action_grid that can be indexed by [x, y]
            action_grid = np.reshape(all_values[0], newshape=(8, 8))
            q_values = [action_grid[v[0], v[1]] for v in valid_actions]
            return valid_actions[np.argmax(q_values)]

    def replay(self, batch_size):
        """
        Perform backpropogation using stochastic gradient descent.
        Only want to update the state-action pair that is selected (the target for all
                 other actions are set to the NN estimate so that the estimate is zero)
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_NN = self.model.predict(state)
            target_NN[0][action] = target   # only this Q val will be updated
            self.model.fit(state, target_NN, epochs=1, verbose=0)

    def epsilon_decay(self):
        # optional epsilon decay feature
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def store_results():
    # present the timed results
    t_stop = process_time()
    print('Runtime: {}hr.'.format((t_stop - t_start)/3600.))
    # create and save a figure
    if storing:
        t = [i for i in range(len(results_over_time))]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, results_over_time)
        ax.set_xlabel("Episode")
        ax.set_title("Percent Wins During Training")
        plt.savefig(save_filename + '.png')
        print('A PNG file containing the run was stored')


if __name__ == "__main__":
    try:
        episodes = 500
        storing = False
        loading = False
        testing = False
        # initialize agent and environment
        agent = OthelloAgent(episodes, model_type='cnn')
        game = OthelloGame(interactive=False, show_steps=False)

        # FILENAME CONVENTION
        #      'saves/NN-type_opponent_num-episodes'
        if storing:
            save_filename = 'final_project/saves/cnn_rand_20000_2'
        if loading:
            load_filename = 'final_project/saves/cnn_rand_12000_2'
            agent.load(load_filename + ".h5")

        terminal = False
        batch_size = 32
        if loading and not testing:
            prev_data = np.load(load_filename + '.npy')
            avg_result = prev_data[-1]
            episode_start = len(prev_data)
            results_over_time = np.append(prev_data, np.zeros(episodes))
        else:
            avg_result = 0
            results_over_time = np.zeros(episodes)
            episode_start = 0

        # time it
        t_start = process_time()
        for e in range(episode_start, episode_start + episodes):
            game.reset()
            game.start()
            state = game.get_state()  # 8x8 numpy array
            if agent.model_type == 'dense':
                state = np.reshape(state, [1, 64])
            else:
                state = state.reshape(1, state.shape[0], state.shape[1], 1)

            for move in range(100):   # max amount of moves in an episode
                action = agent.get_action(state)
                reward, next_state, terminal = game.step(action)
                if agent.model_type == 'dense':
                    next_state = np.reshape(next_state, [1, 64])
                else:
                    next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1], 1)
                if not testing:
                    agent.remember(state, action, reward, next_state, terminal)
                state = next_state
                if terminal:
                    # terminal reward is 0 for loss, 0.5 for tie, 1 for win
                    # use this as an indexing code to get the result
                    outcomes = ['Loss', 'Tie', 'Win']
                    result = outcomes[int(reward*2)]
                    if result == 'Win':
                        n = 1
                    else:
                        n = 0
                    avg_result += (1/(e+1))*(n - avg_result)
                    results_over_time[e] = avg_result
                    print("episode {}: {} moves, Result: {}, e: {:.2}"
                          .format(e, move, result, agent.epsilon))
                    print("Average win/loss ratio: ", avg_result)
                    break
                # Question - maybe only update every batch_size moves
                #       (instead of every move after batch_size)?
                if len(agent.memory) > batch_size and not testing:
                    agent.replay(batch_size)

            agent.epsilon_decay()
            if e % 100 == 0 and e > 0 and storing:
                # save name as 'saves/model-type_training-opponent_num-episodes.h5'
                agent.save(save_filename + ".h5")
                np.save(save_filename + '.npy', results_over_time)
        store_results()
    except KeyboardInterrupt:
        # change the length of our numpy array to be whatever we stopped at
        save_data = results_over_time[[i < 100 or r > 0 for i, r in enumerate(results_over_time)]]
        np.save(save_filename + '.npy', save_data)
        store_results()
