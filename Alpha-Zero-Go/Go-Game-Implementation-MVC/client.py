import pyglet
from pyglet.sprite import Sprite
from pyglet.text import Label
from graphics import Grid, Button, Circle

# constant
BLACK = True
WHITE = False


BLACK_TERRITORY = (0, 0, 0, 255)
WHITE_TERRITORY = (255, 255, 255, 255)

MAX_STONE_SCALING = 0.6     # 1 is original size
MAX_GRID_SIZE = 0.7         # 1 is full size of window
LITTLE_STONE_SIZE = 0.2
SCREEN_SIZE = 700
DEFAULT_BOARD_SIZE = 9

class Window(pyglet.window.Window):

	def __init__(self, controller, size=DEFAULT_BOARD_SIZE):
		super(Window, self).__init__(SCREEN_SIZE,SCREEN_SIZE, fullscreen=False, caption='Go Board')

		# link the view to the controller
		self.controller = controller
		# pass gameplay info from controller
		self.data = { 'size': size,
				'stones': [[None for x in range(size)] for y in range(size)],
				'territory': [[None for x in range(size)] for y in range(size)],
				'color' : None,
				'game_over': False,
				'score': [0, 0] }
		# Set default background color
		pyglet.gl.glClearColor(0.5,0.5,0.5,1)

		#loading images
		self.loading_images()
		#center black and white stone images
		self.init_resources()
		#draw the initial display
		self.init_display()


	def loading_images(self):
		self.image_background = pyglet.resource.image('images/Background.png')
		self.image_black_stone = pyglet.resource.image('images/BlackStone.png')
		self.image_white_stone = pyglet.resource.image('images/WhiteStone.png')


	def init_resources(self):
		"""Center black and white stones for proper visualization
		Attributes updated by this function:
		self.image_black_stone
		self.image_white_stone
		"""

		def center_image(image):
			"""Set an image's anchor point to its center
			Arguments:
				image (pyglet.resource.image) - image to center
			Attributes updated by this function:
				image
			"""
			image.anchor_x = image.width/2
			image.anchor_y = image.height/2
			
		center_image(self.image_black_stone)
		center_image(self.image_white_stone)


	def init_display(self):
		"""Gather all graphical elements together and draw them simutaneously.
		Attributes updated by this function:
		self.batch
		self.background
		self.grid
		self.info
		"""

		#create a batch to display all graphics
		self.batch = pyglet.graphics.Batch()
		# Graphic groups (groups of lower index get drawn first)
		self.grp_back = pyglet.graphics.OrderedGroup(0)
		self.grp_grid = pyglet.graphics.OrderedGroup(1)
		self.grp_label = pyglet.graphics.OrderedGroup(2)
		self.grp_stones = pyglet.graphics.OrderedGroup(3)
		self.grp_territory = pyglet.graphics.OrderedGroup(4)
		#background
		self.background = Sprite(self.image_background,batch=self.batch,group=self.grp_back)
		#grid
		self.grid = Grid(x=self.width/2,
			 y=self.height/2,
			 width=self.width*MAX_GRID_SIZE,
			 height=self.height*MAX_GRID_SIZE,
			 batch=self.batch,
			 group=self.grp_grid,
			 n=self.data['size'])
		#label (quite complex so factored out into another method)
		self.init_label()
		#stones
		
		#territory
		


	def init_label(self):
		"""Load all labels and buttons.
		Attributes updated by this function:
		self.info
		self.score_black
		self.black_label_stone
		self.score_white
		self.white_label_stone
		self.player_color
		self.current_player_stone
		self.button_pass
		self.button_newgame
		"""
		# Game Information Display
		label_y = 670
		label_font_size = 12
		label_text_color = (0, 0, 0, 255)
		
		# Controller-Info Panel
		# The Text of this label is directly changed inside the controller
		self.info = Label(x=10, y=10, text="Let's start!", color=label_text_color,
				  font_size=label_font_size, batch=self.batch, group=self.grp_label)

		# Score-Label
		Label(x=10, y=label_y, text='Score:', color=label_text_color,
				  font_size=label_font_size, bold=True, batch=self.batch, group=self.grp_label)

		# SCORES BLACK PLAYER
		self.score_black = Label(x=100, y=label_y, text=str(self.data['score'][1]), color=label_text_color,
				  font_size=label_font_size, batch=self.batch, group=self.grp_label)
		self.black_label_stone = Sprite(self.image_black_stone,
				   batch=self.batch, group=self.grp_label,
				   x=0, y=0)
		self.black_label_stone.scale = LITTLE_STONE_SIZE
		self.black_label_stone.set_position(80, label_y + self.black_label_stone.height/4)

		# SCORES WHITE PLAYER
		self.score_white = Label(x=170, y=label_y, text=str(self.data['score'][0]), color=label_text_color,
				  font_size=label_font_size, batch=self.batch, group=self.grp_label)
		self.white_label_stone = Sprite(self.image_white_stone,
				   batch=self.batch, group=self.grp_label,
				   x=0, y=0)
		self.white_label_stone.scale = LITTLE_STONE_SIZE
		self.white_label_stone.set_position(150, label_y + self.white_label_stone.height/4)

		# CURRENT PLAYER STONE
		self.player_color = Label(x=550, y=label_y, text="Your color: ", color=label_text_color,
			font_size=label_font_size, bold=True, batch=self.batch, group=self.grp_label)

		# INITIAL PLAYER STONE
		self.current_player_stone = Sprite(self.image_black_stone,
				   batch=self.batch, group=self.grp_label,
				   x=0, y=0)
		self.current_player_stone.scale = LITTLE_STONE_SIZE
		self.current_player_stone.set_position(660, label_y + self.current_player_stone.height/4)

		# Game Buttons  
		# Button that can be pressed to pass on current round
		self.button_pass = Button(pos=(600,40), text='Pass', batch=self.batch)
		
		# New-Game Button
		self.button_newgame = Button(pos=(480,40), text='New Game')

	

	def update(self, *args):
		"""This function does all the calculations when the data gets updated.
		For other games that require permanent simulations you would add
		the following line of code at the end of __init__():
		pyglet.clock.schedule_interval(self.update, 1/30)
		Attributes updated by this function:
		self.batch_stones
		self.stone_sprites
		self.image_black_stone
		self.image_white_stone
		self.batch_territory
		self.score_black
		self.score_white
		self.current_player_stone
		"""
		# Game Information Updates
		# Scores of each player
		self.update_stones()
		self.update_territories()
		self.update_scores()
		self.update_current_player()
			
		# If the new size in the data is different than the current size
		if self.data['size'] != self.grid.size:
			self.init_display()


	def update_stones(self):
		"""Update the black and white stones on the game board.
		Attributes updated by this function:
		self.batch_stones
		self.stone_sprites
		self.image_black_stone
		self.image_white_stone
		"""
		# Display the stones on the regular batch
		self.batch_stones = self.batch
		self.stone_sprites = []
		# Place all stones on the grid
		# Scale stone images
		scaling = self.grid.field_width / self.image_black_stone.width

		# Limit max size of stones
		if scaling > MAX_STONE_SCALING:
			scaling = MAX_STONE_SCALING

		# Iterate trough all data stones and place the corresponding black or
		# white stone on the grid
		for i in range(0, self.data['size']):
			for j in range(0, self.data['size']):
				if self.data['stones'][j][i] != None:
					# Get x and y grid coordinates
					x_coord, y_coord = self.grid.get_coords(i, j)

					# Get the stone color to place
					stone_color = self.image_black_stone if self.data['stones'][j][i] == BLACK else None
					stone_color = self.image_white_stone if self.data['stones'][j][i] == WHITE else stone_color

					# Place the stone on the grid
					if stone_color:
						_s = Sprite(stone_color,
									batch=self.batch_stones,
									group=self.grp_stones,
									x=x_coord,
									y=y_coord)
						_s.scale = scaling
						self.stone_sprites.append(_s)

	def update_territories(self):
		"""Update the black and white territories on the board.
		Attributes updated by this function:
		self.batch_territory
		"""
		# Display the territory an the regular batch
		# Display the stones on the regular batch
		self.batch_territory = self.batch
		
		rad = 5
		
		# Iterate trough all territory indicators and place the corresponding
		# black or white circle on the grid or above stones
		for i in range(0, self.data['size']):
			for j in range(0, self.data['size']):
				if self.data['territory'][j][i] != None:
					x_coord, y_coord = self.grid.get_coords(i, j)
					if self.data['territory'][j][i] == BLACK:
						Circle(x_coord,
							   y_coord,
							   color=BLACK_TERRITORY,
							   r=rad,
							   batch=self.batch_territory,
							   group=self.grp_territory)
					elif self.data['territory'][j][i] == WHITE:
						Circle(x_coord,
							   y_coord,
							   color=WHITE_TERRITORY,
							   r=rad,
							   batch=self.batch_territory,
							   group=self.grp_territory)

	def update_scores(self):
		"""Update scores for BLACK and WHITE.
		Attributes updated by this function:
		self.score_black
		self.score_white
		"""
		self.score_black.text = str(self.data['score'][1])
		self.score_white.text = str(self.data['score'][0])

	def update_current_player(self):
		"""Update stone of current player.
		Attributes updated by this function:
		self.current_player_stone
		"""
		# Remve the last current player stone
		self.current_player_stone.delete()

		# If its the BLACK players turn
		if self.data['color']:
			self.current_player_stone = Sprite(self.image_black_stone,
						   batch=self.batch, group=self.grp_label,
						   x=0, y=0)
			self.current_player_stone.scale = LITTLE_STONE_SIZE
			self.current_player_stone.set_position(660, 670 + self.current_player_stone.height/4)
		# If its the WHITE players turn
		else:
			self.current_player_stone = Sprite(self.image_white_stone,
						   batch=self.batch, group=self.grp_label,
						   x=0, y=0)
			self.current_player_stone.scale = LITTLE_STONE_SIZE
			self.current_player_stone.set_position(660, 670 + self.current_player_stone.height/4)


	def on_draw(self):
		"""Draw the interface"""
		self.clear()
		self.batch.draw()

		if self.data['game_over']:
			self.button_newgame.draw()

	def on_mouse_press(self, mousex, mousey, button, modifiers):
		"""Function called on any mouse button press"""
		# Check for clicks on New Game Button only when game is Over
		if(self.data['game_over']):
			if (mousex, mousey) in self.button_newgame:
				self.controller.new_game()
				
			if button == pyglet.window.mouse.LEFT:
				# Mark territory if the game is over
				pos = self.grid.get_indices(mousex, mousey)
				if pos != None:
					self.controller.mark_territory(pos)

		# Handle clicks during game
		
		# Check if pass-button was pressed
		if (mousex, mousey) in self.button_pass:
			self.controller.passing()

		# Grid position clicked (only if above buttons)
		elif button == pyglet.window.mouse.LEFT and mousey > 60:
			# Place a stone at clicked position
			pos = self.grid.get_indices(mousex, mousey)
			if pos != None:
				self.controller.play(pos)

	def receive_data(self, data):
		"""Receive data from the controller and update view.
		Attributes updated by this function:
		self.data
		"""
		self.data.update(data)
		self.update()

	def new_game(self, data):
		"""Receive data from the controller and start a new game.
		Attributes updated by this function:
		self.data
		"""
		# Initialize the display
		self.data.update(data)
		self.init_display()
		self.update()


	def on_key_press(self, symbol, modifiers):
		"""Function that gets called on any key press (keyboard)"""
		pass




# main program
# open the window
if __name__ == '__main__':
	window = Window()
	pyglet.app.run()