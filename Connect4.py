"""
Rules: https://pl.wikipedia.org/wiki/Czw%C3%B3rki
Authors: Wiktor Krieger & Sebastian Augustyniak

Instructions:
Start a game in terminal,
Make a move by choosing number from 1 to 7, click enter.
Repeat until you or your opponent has connected four characters
"""
import numpy as np
from easyAI import TwoPlayerGame


class ConnectFour(TwoPlayerGame):
    def __init__(self, players, board=None):
        """
        Constructor of the ConnectFour class, initializes the Connect Four game

        :param players: A list of two players.
        :param board: An array representing the state of the board, default is an empty board.
        """
        self.players = players
        self.board = (
            board
            if (board is not None)
            else (np.array([[0 for i in range(7)] for j in range(6)]))
        )
        self.current_player = 1  # Zaczyna gracz numer 1
        self.last_move = None  # Dodane pole do przechowywania informacji o ostatnim ruchu

    def show(self):
        """
        Displays the current state of the Connect Four game board
        """
        colors = ["\u001b[0m", "\u001b[32m",
                  "\u001b[34m"]  # ANSI Escape Codes dla kolorów (0m - domyślny, 32m - zielony, 34m - niebieski)
        print("\n1 2 3 4 5 6 7")
        for row in range(6):
            row_str = ""
            for col in range(7):
                cell = self.board[5 - row][col]
                cell_color = colors[cell]
                row_str += f"{cell_color}{'O' if cell == 1 else 'X' if cell == 2 else '.'}\u001b[0m "
            print(row_str)

    def possible_moves(self):
        """
        Returns a list of available moves, which are the columns where a player can place a character

        :return: List of available columns (from 1 to 7).
        """
        return [
            i + 1 for i in range(7)
            if (self.board[:, i].min() == 0)
        ]

    def make_move(self, column):
        """
        Makes a move by placing a player's character in the selected column

        :param column: The column number where the player places a piece (from 1 to 7)
        """
        column -= 1
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.current_player
        self.last_move = (line, column, self.current_player)

    def lose(self):
        """
        Checks if the current player has lost the Connect Four game by finding a sequence of four pieces

        :return: True if the current player has lost otherwise False
        """
        return find_four(self.board, self.opponent_index)

    # -----------------

    def is_over(self):
        """
        Checks if the game is over, meaning the board is full or one of the players has lost

        :return: True if the current player has lost otherwise False
        """
        return (self.board.min() > 0) or self.lose()

    def scoring(self):
        """
        Determines the game's score. Assigns a score of -100 if the current player has lost otherwise, 0
        :return: The game's score value
        """
        return -100 if self.lose() else 0


def find_four(board, current_player):
    """
    A helper function that checks if there is a sequence of four characters belonging to a specific player on the board
    :param board: An array representing the state of the board
    :param current_player: The number of the current player (1 or 2)
    :return: True if a sequence of four characters is found; otherwise False
    """
    for pos, direction in POS_DIR:
        if has_four_in_a_row(board, pos, direction, current_player):
            return True
    return False


def has_four_in_a_row(board, start_pos, direction, current_player):
    """
    A helper function that checks for a sequence of four characters in a specific direction (vertical, horizontal, diagonal)
    :param board: An array representing the state of the board
    :param start_pos: The starting position from which to check the sequence
    :param direction: The direction in which the sequence is checked
    :param current_player: The number of the current player (1 or 2).
    :return: True if a sequence of four characters is found otherwise False
    """
    streak = 0
    while is_valid_position(start_pos):
        if board[start_pos[0], start_pos[1]] == current_player:
            streak += 1
            if streak == 4:
                return True
        else:
            streak = 0
        start_pos = move_position(start_pos, direction)
    return False


def is_valid_position(pos):
    """
    A helper function that checks if a position is within the board's boundaries.
    :param pos: The position to be checked
    :return: True if the position is within the board's boundaries otherwise False
    """
    return 0 <= pos[0] <= 5 and 0 <= pos[1] <= 6


def move_position(pos, direction):
    """
    A helper function that moves a position in a specific direction
    :param pos: The current position
    :param direction: The direction of movement
    :return: The new position after the movement
    """
    return pos + direction


"""
A list containing all possible directions in which sequences of four characters are checked on the Connect Four board
"""
POS_DIR = np.array(
    # Vertical
    [[[i, 0], [0, 1]] for i in range(6)]
    # Horizontal
    + [[[0, i], [1, 0]] for i in range(7)]
    # Diagonal
    + [[[i, 0], [1, 1]] for i in range(1, 3)]
    + [[[0, i], [1, 1]] for i in range(4)]
    + [[[i, 6], [1, -1]] for i in range(1, 3)]
    + [[[0, i], [1, -1]] for i in range(3, 7)]
)

if __name__ == "__main__":

    from easyAI import AI_Player, Negamax, Human_Player

    ai = Negamax(5)
    human = Human_Player
    game = ConnectFour([Human_Player(human), AI_Player(ai)])
    game.play()

    if game.lose():
        print("Player %d wins." % game.opponent_index)
    else:
        print("We have a draw")

    if game.last_move:
        line, column, player = game.last_move
        print(f"Last move: Player {player} in column {column + 1}, row {line + 1}.")
