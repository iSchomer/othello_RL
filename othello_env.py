# Othello

import random
import sys
import numpy as np


class Othello:

    def draw_board(self, board):

        HLINE = '  +----+----+----+----+----+----+----+----+'

        print('     1    2    3    4    5    6    7    8')
        print(HLINE)
        for y in range(8):
            print(y + 1, end=' ')
            for x in range(8):
                print('| %s' % (board[x][y]), end='  ')
            print('|')
            print(HLINE)

    def reset_board(self, board):
        # Blanks out the board it is passed, except for the original starting position.
        for x in range(8):
            for y in range(8):
                board[x][y] = ' '

        # Starting pieces: X = black, O = white.
        board[3][3] = 'X'
        board[3][4] = 'O'
        board[4][3] = 'O'
        board[4][4] = 'X'

    def get_new_board(self):
        # Creates a brand new, blank board data structure.
        board = []
        for i in range(8):
            board.append([' '] * 8)

        return board

    def is_valid_move(self, board, tile, xstart, ystart):
        # Returns False if the player's move on space xstart, ystart is invalid.
        # If it is a valid move, returns a list of spaces that would become the player's if they made a move here.
        if board[xstart][ystart] != ' ' or not self.is_on_board(xstart, ystart):
            return False

        board[xstart][ystart] = tile  # temporarily set the tile on the board.

        if tile == 'X':
            otherTile = 'O'
        else:
            otherTile = 'X'

        tilesToFlip = []
        for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x, y = xstart, ystart
            x += xdirection  # first step in the direction
            y += ydirection  # first step in the direction
            if self.is_on_board(x, y) and board[x][y] == otherTile:
                # There is a piece belonging to the other player next to our piece.
                x += xdirection
                y += ydirection
                if not self.is_on_board(x, y):
                    continue
                while board[x][y] == otherTile:
                    x += xdirection
                    y += ydirection
                    if not self.is_on_board(x, y):  # break out of while loop, then continue in for loop
                        break
                if not self.is_on_board(x, y):
                    continue
                if board[x][y] == tile:
                    # There are pieces to flip over. Go in the reverse direction until we reach the original space,
                    # noting all the tiles along the way.
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        tilesToFlip.append([x, y])

        board[xstart][ystart] = ' '  # restore the empty space
        if len(tilesToFlip) == 0:  # If no tiles were flipped, this is not a valid move.
            return False
        return tilesToFlip

    def is_on_board(self, x, y):
        # Returns True if the coordinates are located on the board.
        return 0 <= x <= 7 and 0 <= y <= 7

    def get_board_with_valid_moves(self, board, tile):
        # Returns a new board with . marking the valid moves the given player can make.
        dupeBoard = self.get_board_copy(board)

        for x, y in self.get_valid_moves(dupeBoard, tile):
            dupeBoard[x][y] = '.'
        return dupeBoard

    def get_valid_moves(self, board, tile):
        # Returns a list of [x,y] lists of valid moves for the given player on the given board.
        validMoves = []

        for x in range(8):
            for y in range(8):
                if self.is_valid_move(board, tile, x, y):
                    validMoves.append([x, y])
        return validMoves

    def get_board_score(self, board):
        # Determine the score by counting the tiles. Returns a dictionary with keys 'X' and 'O'.
        xscore = 0
        oscore = 0
        for x in range(8):
            for y in range(8):
                if board[x][y] == 'X':
                    xscore += 1
                if board[x][y] == 'O':
                    oscore += 1
        return {'X': xscore, 'O': oscore}

    def choose_player_tile(self):
        # Lets the player type which tile they want to be.
        # Returns a list with the player's tile as the first item, and the computer's tile as the second.
        tile = ''
        while not (tile == 'X' or tile == 'O'):
            print('Do you want to be X or O? X always moves first.')
            tile = input().upper()

        # the first element in the list is the player's tile, the second is the computer's tile.
        if tile == 'X':
            return ['X', 'O']
        else:
            return ['O', 'X']

    # def who_goes_first(self):
    #     # Randomly choose the player who goes first.
    #     if random.randint(0, 1) == 0:
    #         return 'computer'
    #     else:
    #         return 'player'

    # def play_again(self):
    #     # This function returns True if the player wants to play again, otherwise it returns False.
    #     print('Do you want to play again? (yes or no)')
    #     return input().lower().startswith('y')

    def make_move(self, board, tile, xstart, ystart):
        # Place the tile on the board at xstart, ystart, and flip any of the opponent's pieces.
        # Returns False if this is an invalid move, True if it is valid.
        tilesToFlip = self.is_valid_move(board, tile, xstart, ystart)

        if not tilesToFlip:
            return False

        board[xstart][ystart] = tile
        for x, y in tilesToFlip:
            board[x][y] = tile
        return True

    def get_board_copy(self, board):
        # Make a duplicate of the board list and return the duplicate.
        dupeBoard = self.get_new_board()

        for x in range(8):
            for y in range(8):
                dupeBoard[x][y] = board[x][y]

        return dupeBoard

    def is_on_corner(self, x, y):
        # Returns True if the position is in one of the four corners.
        return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 7)

    def get_player_move(self, board, playerTile):
        # Let the player type in their move.
        # Returns the move as [x, y] (or returns the strings 'hints' or 'quit')
        DIGITS1TO8 = '1 2 3 4 5 6 7 8'.split()
        while True:
            print('Enter your move, or type quit to end the game, or hints to turn off/on hints.')
            move = input().lower()
            if move == 'quit':
                return 'quit'
            if move == 'hints':
                return 'hints'

            if len(move) == 2 and move[0] in DIGITS1TO8 and move[1] in DIGITS1TO8:
                x = int(move[0]) - 1
                y = int(move[1]) - 1
                if not self.is_valid_move(board, playerTile, x, y):
                    continue
                else:
                    break
            else:
                print('That is not a valid move. Type the x digit (1-8), then the y digit (1-8).')
                print('For example, 81 will be the top-right corner.')

        return [x, y]

    def get_computer_move(self, board, computerTile):
        # Given a board and the computer's tile, determine where to
        # move and return that move as a [x, y] list.
        possibleMoves = self.get_valid_moves(board, computerTile)

        # randomize the order of the possible moves
        random.shuffle(possibleMoves)

        # always go for a corner if available.
        for x, y in possibleMoves:
            if self.is_on_corner(x, y):
                return [x, y]

        # Go through all the possible moves and remember the best scoring move
        bestScore = -1
        bestMove = []
        for x, y in possibleMoves:
            dupeBoard = self.get_board_copy(board)
            self.make_move(dupeBoard, computerTile, x, y)
            score = self.get_board_score(dupeBoard)[computerTile]
            if score > bestScore:
                bestMove = [x, y]
                bestScore = score
        return bestMove

    def show_points(self, mainBoard, playerTile, computerTile):
        # Prints out the current score.
        scores = self.get_board_score(mainBoard)
        print('You have %s points. The computer has %s points.' % (scores[playerTile], scores[computerTile]))

    def calculate_reward(self, result):
        return result

    def list_to_array(self, mainBoard):
        state = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if mainBoard[j][i] == 'X':
                    state[i, j] = 1
                elif mainBoard[j][i] == 'O':
                    state[i, j] = 2
                else:
                    state[i, j] = 3
        return state

    def run_othello(self):
        print('Welcome to Othello!')

        while True:
            # Reset the board and game.
            mainBoard = self.get_new_board()
            self.reset_board(mainBoard)
            playerTile, computerTile = self.choose_player_tile()
            showHints = True
            if playerTile == 'X':
                turn = 'player'
            else:
                turn = 'computer'
            # print('The ' + turn + ' will go first.')

            while True:
                if turn == 'player':
                    # Player's turn.
                    if showHints:
                        validMovesBoard = self.get_board_with_valid_moves(mainBoard, playerTile)
                        self.draw_board(validMovesBoard)
                        print(self.list_to_array(mainBoard))
                    else:
                        self.draw_board(mainBoard)

                    self.show_points(mainBoard, playerTile, computerTile)
                    move = self.get_player_move(mainBoard, playerTile)

                    if move == 'quit':
                        print('Thanks for playing!')
                        sys.exit()  # terminate the program
                    elif move == 'hints':
                        showHints = not showHints
                        continue
                    else:
                        self.make_move(mainBoard, playerTile, move[0], move[1])

                    if not self.get_valid_moves(mainBoard, computerTile):
                        print('Your opponent has no legal move. It is your turn.')
                        if not self.get_valid_moves(mainBoard, playerTile):
                            print('You also have no legal move. The game is over.')
                            break
                        pass
                    else:
                        turn = 'computer'

                else:
                    # Computer's turn.
                    self.draw_board(mainBoard)
                    self.show_points(mainBoard, playerTile, computerTile)
                    input('Press Enter to see the computer\'s move.\n')
                    x, y = self.get_computer_move(mainBoard, computerTile)
                    self.make_move(mainBoard, computerTile, x, y)

                    if not self.get_valid_moves(mainBoard, playerTile):
                        print('You have no legal move. It is the computer\'s turn.')
                        if not self.get_valid_moves(mainBoard, computerTile):
                            print('Your opponent also has no legal move. The game is over')
                            break
                        pass
                    else:
                        turn = 'player'

            # Display the final score.
            self.draw_board(mainBoard)
            scores = self.get_board_score(mainBoard)
            print('X scored %s points. O scored %s points.' % (scores['X'], scores['O']))
            if scores[playerTile] > scores[computerTile]:
                print('You beat the computer by %s points! Congratulations!' % (scores[playerTile] - scores[computerTile]))
                return self.calculate_reward(1)
            elif scores[playerTile] < scores[computerTile]:
                print('You lost. The computer beat you by %s points.' % (scores[computerTile] - scores[playerTile]))
                return self.calculate_reward(-1)
            else:
                print('The game was a tie!')
                return self.calculate_reward(0)

            # if not self.play_again():
            #     break


if __name__ == '__main__':
    othello = Othello()
    othello.run_othello()
