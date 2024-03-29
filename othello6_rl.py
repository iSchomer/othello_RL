from othello6_env import OthelloGame
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomUniform
import random
from time import process_time
import matplotlib.pyplot as plt
from datetime import datetime


class OthelloAgent:
    def __init__(self, ep):
        self.state_size = 36
        self.action_size = 36
        self.tile = 'X'
        self.memory = deque(maxlen=2000)
        self.gamma = 1.0  # episodic --> no discount
        self.episodes = ep
        self.epsilon = 0.1
        self.epsilon_min = 0.0
        self.epsilon_step = (self.epsilon - self.epsilon_min)/self.episodes
        self.learning_rate = 0.01
        self.model = self.build_model()

    def build_model(self):
        # Feed-forward NN
        model = tf.keras.Sequential()
        init = RandomUniform(minval=-0.5, maxval=0.5)
        model.add(layers.Dense(30, input_dim=self.state_size, activation='sigmoid', kernel_initializer=init))
        model.add(layers.Dense(36, activation='sigmoid', kernel_initializer=init))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
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
            action_grid = np.reshape(all_values[0], newshape=(6, 6))
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
                action_grid = np.reshape(all_values[0], newshape=(6, 6))
                q_values = [action_grid[v[1], v[0]] for v in vld_moves]
                target = rw + self.gamma * np.amax(q_values)
            target_nn = self.model.predict(st)
            target_nn[0][act[1]*6+act[0]] = target   # only this Q val will be updated
            self.model.fit(st, target_nn, epochs=1, verbose=0)

        self.memory.clear()

    def epsilon_decay(self):
        # linear epsilon decay feature
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
        t1 = [i for i in range(len(results_over_time))]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t1, results_over_time)
        ax.set_xlabel("Episode")
        ax.set_title("Percent Wins During Training")
        plt.savefig(save_filename + datetime.now().strftime("%m%d%H:%M") + '_training' + '.png')

        t2 = [i for i in range(len(test_result))]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t2, test_result)
        ax.set_xlabel("Episode")
        ax.set_title("Percent Wins During Testing")
        plt.savefig(save_filename + datetime.now().strftime("%m%d%H:%M") + '_testing' + '.png')


if __name__ == "__main__":
    try:
        storing = True
        loading = False
        testing = False

        terminal = False
        batch_size = 360
        episodes = 20000

        test_interval = 2000
        test_length = 1000

        outcomes = ['Loss', 'Tie', 'Win']
        move_counter = 0

        # initialize agent and environment
        agent = OthelloAgent(episodes)
        game = OthelloGame(opponent='heur', interactive=False, show_steps=False)

        # FILENAME CONVENTION
        #      'saves/NN-type_opponent_num-episodes'
        if storing:
            save_filename = './saves/othello6_basic-sequential_rand_2000'
        if loading:
            load_filename = './saves/othello6_basic-sequential_rand_2000'
            agent.load(load_filename + ".h5")

        if loading and not testing:
            prev_data = np.load(load_filename + '.npy')
            avg_result = prev_data[-1]
            episode_start = len(prev_data)
            results_over_time = np.append(prev_data, np.zeros(episodes))
        else:
            avg_result = 0
            results_over_time = np.zeros(episodes)
            episode_start = 0

            test_result = []

        # time it
        t_start = process_time()
        for e in range(episode_start, episode_start + episodes):

            # perform a 100-episode test with greedy policy
            if e % test_interval == 0:
                testing = True

            if testing is True:
                test_result.append(0)
                for test_ep in range(test_length):
                    game.reset()
                    game.start()
                    state = game.get_state()  # 6x6 numpy array
                    state = np.reshape(state, [1, 36])  # 1x36 vector

                    for move in range(100):
                        action = agent.get_action(state, testing)
                        reward, next_state, valid_moves, terminal = game.step(action)
                        next_state = np.reshape(next_state, [1, 36])
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
            state = game.get_state()  # 6x6 numpy array
            state = np.reshape(state, [1, 36])   # 1x36 vector

            for move in range(100):   # max amount of moves in an episode
                move_counter += 1
                action = agent.get_action(state, testing)
                reward, next_state, valid_moves, terminal = game.step(action)
                next_state = np.reshape(next_state, [1, 36])
                agent.remember(state, action, reward, next_state, valid_moves, terminal)
                state = next_state
                if terminal:
                    # terminal reward is 0 for loss, 0.5 for tie, 1 for win
                    # use this as an indexing code to get the result
                    result = outcomes[int(reward*2)]
                    if result == 'Win':
                        n = 1
                    else:
                        n = 0

                    avg_result += (1/(e+1))*(n - avg_result)
                    results_over_time[e] = avg_result

                    if e % 10 == 0 and e > 0:
                        print("episode {}: {} moves, Result: {}, e: {:.2}"
                              .format(e, move, result, agent.epsilon))
                        print("Average win/loss ratio: ", avg_result)
                    break

                # Question - maybe only update every batch_size moves
                #       (instead of every move after batch_size)?
                if move_counter % batch_size == 0:
                # if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            agent.epsilon_decay()
            if e % 100 == 0 and e > 0 and storing:
                # save name as 'saves/model-type_training-opponent_num-episodes.h5'
                # agent.save(save_filename + datetime.now().strftime("%m%d%H:%M") + ".h5")
                # np.save(save_filename + datetime.now().strftime("%m%d%H:%M") + '.npy', results_over_time)
                pass
        store_results()
    except KeyboardInterrupt:
        # change the length of our numpy array to be whatever we stopped at
        save_data = results_over_time[[i < 100 or r > 0 for i, r in enumerate(results_over_time)]]
        np.save(save_filename + datetime.now().strftime("%m%d%H:%M") + '.npy', save_data)
        store_results()
