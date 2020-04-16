# NeuralNetworkGame

Python project that solves logical game using neural network and Monte Carlo tree search (MCTS) algorithm

Requirements:
python 3.6.x - 3.7.x
  
pip install numpy
pip install sklearn
pip install pygame

Main parameters:
GAMES - number of games to play for training or demonstration
  TABLE_HEIGHT - height of game board 
  TABLE_WIDTH - width of game board

  INPUTS - number of neural net inputs
  TREE_SEARCHES - number of MCTS iterations
  TREE_DEPTH - depth of each MCTS iteration
  DISCOUNT = reward discount rate at each depth level
  EPSILON - randomness rate during training process

  TRAIN - True: train net, False: demonstrate net with given weights
  SAVE_PROGRESS - True: save network weights after successful training
  SHOW_TURNS - True: show text output of each turn
  SHOW_INFO - True: show technical details of each turn
