# Controller part of the MVC architecture for an implementation of the Go game

import pyglet
from model import Model
import client

class Controller(object):
    """ Controller class which enables communication between the client and the model"""

    def __init__(self, grid_size, player_1_name, player_2_name):
        """Initialize the controller/game and grid size and pass initial data from the model to the client
        Arguments
            grid_size (int): number of squares (n x n)
        Variables changed by this function
            self.n
            self.model
            self.window
        """
        # Initialize the controller and start communicating between the view and model
        self.n = int(grid_size)
        self.player_1 = player_1_name
        self.player_2 = player_2_name
        self.model = Model(self.n)
        self.window = client.Window(self, self.n)
        self.window.receive_data(self.model.get_data())

        # Display the start message and start the game.
        self.window.info.text = "Let's start, " + self.player_1 + "!"
        pyglet.app.run()
    

    def _update_window(self):
        """Send updated Data to the View
        Variables changed by this function
            self.window
        """
        self.window.receive_data(self.model.get_data())
        
    
    def play(self, pos):
        """Place a stone on a certain position on the grid
        Arguments
            pos (tuple): x and y coordinate to place the stone
        Variables changed by this function
            self.model
            self.window.info.text
        """

        # Get the name of the current player
        player_name = self.player_1 if self.model.get_data()['color'] else self.player_2

        # If valid move
        if(self.model.place_stone(pos[0], pos[1])):
            self.window.info.text = 'Nice Move ' + player_name + '!'
        # If invalid move
        else:
            self.window.info.text = 'Invalid move ' + player_name + '!'
        
        self._update_window()
    

    def passing(self):
        """Pass on the turn of the current player
        Variables changed by this function
            self.window.info.text
            self.model
        """
        self.window.info.text = "You passed on your turn"
        
        # If able to pass on the turn
        if self.model.passing():
            # If the game is over
            if self.model.get_data()['game_over']:
                self.window.info.text = 'Game is Over! Both players passed their turn!'
            # If the game continues
            else:
                self.window.info.text = 'Continue playing!'

        self._update_window()
    

    def mark_territory(self, pos):
        """Mark the grid with color-circles to indicate points. Only possible at the end of a game.
        Arguments
            pos (tuple): x and y coordinate to mark territory
        Variables changed by this function
            self.model
        """
        self.model.mark_territory(pos[0], pos[1])
        self._update_window()


    def new_game(self):
        """Create a new game
        Variables changed by this function
            self.model
            self.window
            self.window.into.text
        """
        self.model.__init__(self.n)
        self.window.new_game(self.model.get_data())
        self.window.info.text = "New game, let's start, " + self.player_1 + "!"