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
from datetime import datetime


class OthelloAgent:
    def __init__(self, ep, model_type='dense'):
        self.state_size = 64
        self.action_size = 64
        self.tile = 'X'
        self.memory = deque(maxlen=2000)
        self.gamma = 1.0  # episodic --> no discount
        self.episodes = ep
        self.epsilon = 0.1
        self.epsilon_min = 0.08
        self.epsilon_step = (self.epsilon - self.epsilon_min) / self.episodes
        self.learning_rate = 0.02
        self.model_type = model_type
        self.model = self.build_model()

    def build_model(self):
        init = RandomUniform(minval=-0.5, maxval=0.5)
        if self.model_type == 'dense':
            # Feed-forward NN
            model = tf.keras.Sequential()
            model.add(layers.Dense(50, input_dim=self.state_size, activation='sigmoid', kernel_initializer=init))
            model.add(layers.Dense(64, activation='sigmoid', kernel_initializer=init))
        else:
            # convolutional neural network
            model = tf.keras.Sequential()
            # convolve to a 6x6 grid
            model.add(layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(8, 8, 1)))
            model.add(layers.Flatten())
            # add a dense layer
            model.add(layers.Dense(30, input_dim=self.state_size, activation='sigmoid', kernel_initializer=init))
            model.add(layers.Dense(64, activation='sigmoid', kernel_initializer=init))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def remember(self, st, act, rw, next_st, vld_moves, done):
        self.memory.append((st, act, rw, next_st, vld_moves, done))

    def get_action(self, st, test):
        valid_actions = game.board.get_valid_moves(self.tile)
        if np.random.rand() <= self.epsilon and not test:
            random.shuffle(valid_actions)
            return valid_actions[0]
        else:
            # Take an action based on the Q function
            all_values = self.model.predict(st)
            # return the VALID action with the highest network value
            # use an action_grid that can be indexed by [x, y]
            action_grid = np.reshape(all_values[0], newshape=(8, 8))
            q_values = [action_grid[v[1], v[0]] for v in valid_actions]
            return valid_actions[np.argmax(q_values)]

    def replay(self, bat_size):
        """
        Perform back-propagation using stochastic gradient descent.
        Only want to update the state-action pair that is selected (the target for all
                 other actions are set to the NN estimate so that the estimate is zero)
        """
        mini_batch = random.sample(self.memory, bat_size)
        for st, act, rw, next_st, vld_moves, done in mini_batch:
            target = rw
            if not done:
                all_values = self.model.predict(next_st)
                action_grid = np.reshape(all_values[0], newshape=(8, 8))
                q_values = [action_grid[v[1], v[0]] for v in vld_moves]
                target = rw + self.gamma * np.amax(q_values)
            target_nn = self.model.predict(st)
            target_nn[0][act[1] * 8 + act[0]] = target  # only this Q val will be updated
            self.model.fit(st, target_nn, epochs=1, verbose=0)

    def epsilon_decay(self):
        # linear epsilon decay feature
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def store_results(data):
    # present the timed results
    t_stop = process_time()
    print('Runtime: {}hr.'.format((t_stop - t_start) / 3600.))
    # create and save a figure
    if storing:
        t1 = [i for i in range(len(data))]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t1, data)
        test_points = [0] + [test_interval * (i + 1) for i in range(int(len(data) / test_interval))]
        ax.scatter(test_points, test_result, marker='o')
        ax.set_xlabel("Episode")
        ax.set_title("Percent Wins During Training")
        plt.savefig(save_filename + datetime.now().strftime("(%m-%d--%H:%M)") + '_training' + '.png')
        # plt.show()


if __name__ == "__main__":
    episodes = 500
    storing = True
    loading = False
    testing = False  # keep this at False unless loading is True
    alpha = .01

    terminal = False
    batch_size = 32

    test_interval = 100
    test_length = 10

    outcomes = ['Loss', 'Tie', 'Win']
    move_counter = 0

    # initialize agent and environment
    agent = OthelloAgent(episodes, model_type='cnn')
    game = OthelloGame(opponent='rand', interactive=False, show_steps=False)

    # FILENAME CONVENTION
    #      'saves/NN-type_opponent_num-episodes'
    if storing:
        save_filename = 'final_project/saves/c16-3d40_rand_2000'
    if loading:
        load_filename = 'final_project/saves/c12-2d50_rand_600'
        agent.load(load_filename + ".h5")

    if loading and not testing:
        prev_data = np.load(load_filename + '.npy')
        avg_result = prev_data[-1]
        episode_start = len(prev_data) + 1
        results_over_time = np.append(prev_data, np.zeros(episodes))
    else:
        avg_result = 0
        results_over_time = np.zeros(episodes)
        episode_start = 1

        test_result = []

    # time it
    t_start = process_time()
    for e in range(episode_start, episode_start + episodes):

        # perform a 100-episode test with greedy policy
        if e % test_interval == 0 or e == 1:
            testing = True

        if testing is True:
            test_result.append(0)
            for test_ep in range(test_length):
                game.reset()
                game.start()
                state = game.get_state()  # 6x6 numpy array
                if agent.model_type == 'dense':
                    state = np.reshape(state, [1, 64])
                else:
                    state = state.reshape(1, state.shape[0], state.shape[1], 1)

                for move in range(100):
                    action = agent.get_action(state, testing)
                    reward, next_state, valid_moves, terminal = game.step(action)
                    if agent.model_type == 'dense':
                        next_state = np.reshape(next_state, [1, 64])
                    else:
                        next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1], 1)
                    state = next_state
                    if terminal:
                        # terminal reward is 0 for loss, 0.5 for tie, 1 for win
                        # use this as an indexing code to get the result
                        result = outcomes[int(reward * 2)]
                        if result == 'Win':
                            n = 1
                        else:
                            n = 0
                        test_result[-1] += (1 / (test_ep % test_interval + 1)) * (n - test_result[-1])
                        if test_ep % 10 == 0 and test_ep > 0:
                            print('testing' + "episode {}: {} moves, Result: {}".format(test_ep, move, result))
                            print("Average win/loss ratio: ", test_result[-1])
                        break
            testing = False

        game.reset()
        game.start()
        state = game.get_state()  # 8x8 numpy array
        if agent.model_type == 'dense':
            state = np.reshape(state, [1, 64])
        else:
            state = state.reshape(1, state.shape[0], state.shape[1], 1)

        for move in range(100):  # max amount of moves in an episode
            move_counter += 1
            action = agent.get_action(state, testing)
            reward, next_state, valid_moves, terminal = game.step(action)
            if agent.model_type == 'dense':
                next_state = np.reshape(next_state, [1, 64])
            else:
                next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1], 1)
            agent.remember(state, action, reward, next_state, valid_moves, terminal)
            state = next_state
            if terminal:
                # terminal reward is 0 for loss, 0.5 for tie, 1 for win
                # use this as an indexing code to get the result
                result = outcomes[int(reward * 2)]
                if result == 'Win':
                    n = 1
                else:
                    n = 0

                avg_result += alpha * (n - avg_result)
                results_over_time[e - 1] = avg_result

                if e % 10 == 0:
                    print("episode {}: {} moves, Result: {}, e: {:.2}"
                          .format(e, move, result, agent.epsilon))
                    print("Average win/loss ratio: ", avg_result)
                break

            # Question - maybe only update every batch_size moves
            #       (instead of every move after batch_size)?
            if move_counter % batch_size == 0:  # if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        agent.epsilon_decay()
        if e % test_interval == 0 and storing:
            # save name as 'saves/model-type_training-opponent_num-episodes.h5'
            agent.save(save_filename + datetime.now().strftime("(%m-%d--%H)") + ".h5")
            # save only the length of the array that we stopped at
            save_data = results_over_time[:e]
            np.save(save_filename + datetime.now().strftime("(%m-%d--%H)") + '.npy', save_data)
            store_results(save_data)
    store_results(results_over_time)
