# Othello

import random
import sys
import numpy as np


# static methods
def is_on_board(x, y):
    # Returns True if the coordinates are located on the board.
    return 0 <= x <= 7 and 0 <= y <= 7


def is_on_corner(x, y):
    # Returns True if the position is in one of the four corners.
    return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 7)


class Board:

    def __init__(self):
        self.board = []
        for _ in range(8):
            self.board.append([' ']*8)
        self.reset()

    def draw(self):

        HLINE = '  +----+----+----+----+----+----+----+----+'

        print('     1    2    3    4    5    6    7    8')
        print(HLINE)
        for y in range(8):
            print(y + 1, end=' ')
            for x in range(8):
                print('| %s' % (self.board[x][y]), end='  ')
            print('|')
            print(HLINE)

    def reset(self):
        # Blanks out the board it is passed, except for the original starting position.
        for x in range(8):
            for y in range(8):
                self.board[x][y] = ' '

        # Starting pieces: X = black, O = white.
        self.board[3][3] = 'X'
        self.board[3][4] = 'O'
        self.board[4][3] = 'O'
        self.board[4][4] = 'X'

    def is_valid_move(self, tile, xstart, ystart):
        # Returns False if the player's move on space xstart, ystart is invalid.
        # If it is a valid move, returns a list of spaces that would become the player's if they made a move here.
        if self.board[xstart][ystart] != ' ' or not is_on_board(xstart, ystart):
            return False

        self.board[xstart][ystart] = tile  # temporarily set the tile on the board.

        if tile == 'X':
            other_tile = 'O'
        else:
            other_tile = 'X'

        tilesToFlip = []
        for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = xstart, ystart
            x += xdirection  # first step in the direction
            y += ydirection  # first step in the direction
            if is_on_board(x, y) and self.board[x][y] == other_tile:
                # There is a piece belonging to the other player next to our piece.
                x += xdirection
                y += ydirection
                if not is_on_board(x, y):
                    continue
                while self.board[x][y] == other_tile:
                    x += xdirection
                    y += ydirection
                    if not is_on_board(x, y):  # break out of while loop, then continue in for loop
                        break
                if not is_on_board(x, y):
                    continue
                if self.board[x][y] == tile:
                    # There are pieces to flip over. Go in the reverse direction until we reach the original space,
                    # noting all the tiles along the way.
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        tilesToFlip.append([x, y])

        self.board[xstart][ystart] = ' '  # restore the empty space
        if len(tilesToFlip) == 0:  # If no tiles were flipped, this is not a valid move.
            return False
        return tilesToFlip
    
    def get_valid_moves(self, tile):
        # Returns a list of [x,y] lists of valid moves for the given player on the given board.
        valid_moves = []

        for x in range(8):
            for y in range(8):
                if self.is_valid_move(tile, x, y):
                    valid_moves.append([x, y])
        return valid_moves

    def get_score(self):
        # Determine the score by counting the tiles. Returns a dictionary with keys 'X' and 'O'.
        xscore = 0
        oscore = 0
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == 'X':
                    xscore += 1
                if self.board[x][y] == 'O':
                    oscore += 1
        return {'X': xscore, 'O': oscore}

    def make_move(self, tile, xstart, ystart):
        # Place the tile on the board at xstart, ystart, and flip any of the opponent's pieces.
        # Returns False if this is an invalid move, True if it is valid.
        tiles_to_flip = self.is_valid_move(tile, xstart, ystart)

        if not tiles_to_flip:
            return False

        self.board[xstart][ystart] = tile
        for x, y in tiles_to_flip:
            self.board[x][y] = tile
        return True

    def copy(self):
        # Make a duplicate of the board list and return the duplicate.
        dupe_board = Board()

        for x in range(8):
            for y in range(8):
                dupe_board.board[x][y] = self.board[x][y]

        return dupe_board

    def copy_with_valid_moves(self, tile):
        # ONLY TO BE USED WITH PLAYER INTERACTION. NOT FOR TRAINING
        # Returns a new board with . marking the valid moves the given player can make.
        dupe_board = self.copy()

        for x, y in dupe_board.get_valid_moves(tile):
            dupe_board.board[x][y] = '.'
        return dupe_board

    def list_to_array(self):
        state = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if self.board[j][i] == 'X':
                    state[i, j] = 1
                elif self.board[j][i] == 'O':
                    state[i, j] = -1
                else:
                    state[i, j] = 0
        return state


class OthelloGame:

    def __init__(self, interactive=True, show_steps=False):
        self.board = Board()
        self.player_tile = 'X'
        self.computer_tile = 'O'
        self.player_score = 0
        self.computer_score = 0
        self.interactive = interactive
        self.stepper = show_steps

    def reset(self):
        self.board.reset()
        self.player_score = 0
        self.computer_score = 0

    def get_state(self):
        return self.board.list_to_array()

    def choose_player_tile(self):
        # Lets the player type which tile they want to be.
        # Returns a list with the player's tile as the first item, and the computer's tile as the second.
        tile = ''
        while not (tile == 'X' or tile == 'O'):
            print('Do you want to be X or O? X always moves first.')
            tile = input().upper()

        # the first element in the list is the player's tile, the second is the computer's tile.
        if tile == 'X':
            assigned_tiles = ['X', 'O']
        else:
            assigned_tiles = ['O', 'X']

        self.player_tile, self.computer_tile = assigned_tiles

    def get_player_move(self):
        # Let the player type in their move given a board state.
        # Returns the move as [x, y] (or returns the strings 'hints' or 'quit')
        valid_digits = '1 2 3 4 5 6 7 8'.split()
        while True:
            print('Enter your move, or type quit to end the game, or hints to turn off/on hints.')
            move = input().lower()
            if move == 'quit':
                return 'quit'
            if move == 'hints':
                return 'hints'

            if len(move) == 2 and move[0] in valid_digits and move[1] in valid_digits:
                x = int(move[0]) - 1
                y = int(move[1]) - 1
                if not self.board.is_valid_move(self.player_tile, x, y):
                    continue
                else:
                    break
            else:
                print('That is not a valid move. Type the x digit (1-8), then the y digit (1-8).')
                print('For example, 81 will be the top-right corner.')

        return [x, y]

    def get_computer_move(self):
        # Given a board and the computer's tile, determine where to
        # move and return that move as a [x, y] list.
        possible_moves = self.board.get_valid_moves(self.computer_tile)

        # randomize the order of the possible moves
        random.shuffle(possible_moves)

        # always go for a corner if available.
        for x, y in possible_moves:
            if is_on_corner(x, y):
                return [x, y]

        # Go through all the possible moves and remember the best scoring move
        best_score = -1
        best_move = []
        for x, y in possible_moves:
            dupe_board = self.board.copy()
            dupe_board.make_move(self.computer_tile, x, y)
            score = dupe_board.get_score()[self.computer_tile]
            if score > best_score:
                best_move = [x, y]
                best_score = score
        return best_move
    
    def show_points(self):
        # Prints out the current score.
        scores = self.board.get_score()
        print('You have %s points. The computer has %s points.' % (scores[self.player_tile], scores[self.computer_tile]))

    def calculate_reward(self, result):
        return result

    def start(self):
        if self.interactive:
            self.run_interactive()
        else:
            # Reset the board and game.
            self.board.reset()
            self.choose_player_tile()
            showHints = True
            if self.player_tile == 'X':
                turn = 'player'
            else:
                turn = 'computer'

    def step(self, action):
        # TODO - make a function that takes a player's action and returns the next state and reward
        #        and also indicates whether a terminal state is reached
        reward = 0
        done = False     # indicates terminal state
        next_board = self.board.copy()    # TODO - update board based on action
        # option to display visuals while learning how to train
        if self.stepper:
            next_board.draw()
            print(next_board.list_to_array())
            print('Reward on step: {0}'.format(reward))
        self.board = next_board
        return reward, next_board, done

    def run_interactive(self):
        print('Welcome to Othello!')
        while True:
            # Reset the board and game.
            main_board = Board()
            self.choose_player_tile()
            showHints = True
            if self.player_tile == 'X':
                turn = 'player'
            else:
                turn = 'computer'

            while True:
                if turn == 'player':
                    # Player's turn.
                    if showHints:
                        valid_moves_board = main_board.copy_with_valid_moves(self.player_tile)
                        valid_moves_board.draw()
                        print(main_board.list_to_array())
                    else:
                        main_board.draw()

                    self.show_points(main_board)
                    move = self.get_player_move(main_board)

                    if move == 'quit':
                        print('Thanks for playing!')
                        sys.exit()  # terminate the program
                    elif move == 'hints':
                        showHints = not showHints
                        continue
                    else:
                        main_board.make_move(self.player_tile, move[0], move[1])

                    if not main_board.get_valid_moves(self.computer_tile):
                        print('Your opponent has no legal move. It is your turn.')
                        if not main_board.get_valid_moves(self.player_tile):
                            print('You also have no legal move. The game is over.')
                            break
                        pass
                    else:
                        turn = 'computer'
                else:
                    # Computer's turn.
                    main_board.draw()
                    self.show_points(main_board)
                    input('Press Enter to see the computer\'s move.\n')
                    x, y = self.get_computer_move(main_board)
                    main_board.make_move(self.computer_tile, x, y)

                    if not main_board.get_valid_moves(self.player_tile):
                        print('You have no legal move. It is the computer\'s turn.')
                        if not main_board.get_valid_moves(self.computer_tile):
                            print('Your opponent also has no legal move. The game is over')
                            break
                        pass
                    else:
                        turn = 'player'

            # Display the final score.
            main_board.draw()
            scores = main_board.get_score()
            self.player_score = scores[self.player_tile]
            self.computer_score = scores[self.computer_tile]
            margin = self.player_score - self.computer_score
            print('The player scored %s points. The computer scored %s points.' % (self.player_score, self.computer_score))
            if self.player_score > self.computer_score:
                print('You beat the computer by %s points! Congratulations!' % margin)
                return self.calculate_reward(1)
            elif self.player_score < self.computer_score:
                print('You lost. The computer beat you by %s points.' % margin)
                return self.calculate_reward(-1)
            else:
                print('The game was a tie!')
                return self.calculate_reward(0)


if __name__ == '__main__':
    othello = OthelloGame()
    othello.start()
