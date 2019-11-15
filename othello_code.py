# Reversi

import random
import sys


def draw_board(board):
    # This function prints out the board that it was passed. Returns None.
    HLINE = '  +-----+-----+-----+-----+-----+-----+-----+-----+'
    VLINE = '  |     |     |     |     |     |     |     |     |'

    print('    1   2   3   4   5   6   7   8')
    print(HLINE)
    for y in range(8):
        print(VLINE)
        print(y + 1, end=' ')
        for x in range(8):
            print('|  %s' % (board[x][y]), end='  ')
        print('|')
        print(VLINE)
        print(HLINE)


def reset_board(board):
    # Blanks out the board it is passed, except for the original starting position.
    for x in range(8):
        for y in range(8):
            board[x][y] = ' '

    # Starting pieces:
    board[3][3] = 'X'
    board[3][4] = 'O'
    board[4][3] = 'O'
    board[4][4] = 'X'


def get_new_board():
    # Creates a brand new, blank board data structure.
    board = []
    for i in range(8):
        board.append([' '] * 8)

    return board


def is_valid_move(board, tile, xstart, ystart):
    # Returns False if the player's move on space xstart, ystart is invalid.
    # If it is a valid move, returns a list of spaces that would become the player's if they made a move here.
    if board[xstart][ystart] != ' ' or not is_on_board(xstart, ystart):
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
        if is_on_board(x, y) and board[x][y] == otherTile:
            # There is a piece belonging to the other player next to our piece.
            x += xdirection
            y += ydirection
            if not is_on_board(x, y):
                continue
            while board[x][y] == otherTile:
                x += xdirection
                y += ydirection
                if not is_on_board(x, y):  # break out of while loop, then continue in for loop
                    break
            if not is_on_board(x, y):
                continue
            if board[x][y] == tile:
                # There are pieces to flip over. Go in the reverse direction until we reach the original space, noting all the tiles along the way.
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


def is_on_board(x, y):
    # Returns True if the coordinates are located on the board.
    return x >= 0 and x <= 7 and y >= 0 and y <= 7


def get_board_with_valid_moves(board, tile):
    # Returns a new board with . marking the valid moves the given player can make.
    dupeBoard = get_board_copy(board)

    for x, y in get_valid_moves(dupeBoard, tile):
        dupeBoard[x][y] = '.'
    return dupeBoard


def get_valid_moves(board, tile):
    # Returns a list of [x,y] lists of valid moves for the given player on the given board.
    validMoves = []

    for x in range(8):
        for y in range(8):
            if is_valid_move(board, tile, x, y):
                validMoves.append([x, y])
    return validMoves


def get_board_score(board):
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


def choose_player_tile():
    # Lets the player type which tile they want to be.
    # Returns a list with the player's tile as the first item, and the computer's tile as the second.
    tile = ''
    while not (tile == 'X' or tile == 'O'):
        print('Do you want to be X or O?')
        tile = input().upper()

    # the first element in the list is the player's tile, the second is the computer's tile.
    if tile == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']


def who_goes_first():
    # Randomly choose the player who goes first.
    if random.randint(0, 1) == 0:
        return 'computer'
    else:
        return 'player'


def play_again():
    # This function returns True if the player wants to play again, otherwise it returns False.
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')


def make_move(board, tile, xstart, ystart):
    # Place the tile on the board at xstart, ystart, and flip any of the opponent's pieces.
    # Returns False if this is an invalid move, True if it is valid.
    tilesToFlip = is_valid_move(board, tile, xstart, ystart)

    if not tilesToFlip:
        return False

    board[xstart][ystart] = tile
    for x, y in tilesToFlip:
        board[x][y] = tile
    return True


def get_board_copy(board):
    # Make a duplicate of the board list and return the duplicate.
    dupeBoard = get_new_board()

    for x in range(8):
        for y in range(8):
            dupeBoard[x][y] = board[x][y]

    return dupeBoard


def is_on_corner(x, y):
    # Returns True if the position is in one of the four corners.
    return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 7)


def get_player_move(board, playerTile):
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
            if not is_valid_move(board, playerTile, x, y):
                continue
            else:
                break
        else:
            print('That is not a valid move. Type the x digit (1-8), then the y digit (1-8).')
            print('For example, 81 will be the top-right corner.')

    return [x, y]


def get_computer_move(board, computerTile):
    # Given a board and the computer's tile, determine where to
    # move and return that move as a [x, y] list.
    possibleMoves = get_valid_moves(board, computerTile)

    # randomize the order of the possible moves
    random.shuffle(possibleMoves)

    # always go for a corner if available.
    for x, y in possibleMoves:
        if is_on_corner(x, y):
            return [x, y]

    # Go through all the possible moves and remember the best scoring move
    bestScore = -1
    for x, y in possibleMoves:
        dupeBoard = get_board_copy(board)
        make_move(dupeBoard, computerTile, x, y)
        score = get_board_score(dupeBoard)[computerTile]
        if score > bestScore:
            bestMove = [x, y]
            bestScore = score
    return bestMove


def show_points(playerTile, computerTile):
    # Prints out the current score.
    scores = get_board_score(mainBoard)
    print('You have %s points. The computer has %s points.' % (scores[playerTile], scores[computerTile]))


if __name__ == '__main__':
    print('Welcome to Reversi!')

    while True:
        # Reset the board and game.
        mainBoard = get_new_board()
        reset_board(mainBoard)
        playerTile, computerTile = choose_player_tile()
        showHints = False
        turn = who_goes_first()
        print('The ' + turn + ' will go first.')

        while True:
            if turn == 'player':
                # Player's turn.
                if showHints:
                    validMovesBoard = get_board_with_valid_moves(mainBoard, playerTile)
                    draw_board(validMovesBoard)
                else:
                    draw_board(mainBoard)

                show_points(playerTile, computerTile)
                move = get_player_move(mainBoard, playerTile)

                if move == 'quit':
                    print('Thanks for playing!')
                    sys.exit()  # terminate the program
                elif move == 'hints':
                    showHints = not showHints
                    continue
                else:
                    make_move(mainBoard, playerTile, move[0], move[1])

                if get_valid_moves(mainBoard, computerTile) == []:
                    break
                else:
                    turn = 'computer'

            else:
                # Computer's turn.
                draw_board(mainBoard)
                show_points(playerTile, computerTile)
                input('Press Enter to see the computer\'s move.\n')
                x, y = get_computer_move(mainBoard, computerTile)
                make_move(mainBoard, computerTile, x, y)

                if not get_valid_moves(mainBoard, playerTile):
                    break
                else:
                    turn = 'player'

        # Display the final score.
        draw_board(mainBoard)
        scores = get_board_score(mainBoard)
        print('X scored %s points. O scored %s points.' % (scores['X'], scores['O']))
        if scores[playerTile] > scores[computerTile]:
            print('You beat the computer by %s points! Congratulations!' % (scores[playerTile] - scores[computerTile]))
        elif scores[playerTile] < scores[computerTile]:
            print('You lost. The computer beat you by %s points.' % (scores[computerTile] - scores[playerTile]))
        else:
            print('The game was a tie!')

        if not play_again():
            break
