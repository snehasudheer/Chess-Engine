# CIS667_Chess_AI

**Attribution and link to the code used in the project:**

https://github.com/mikolaj-skrzypczak/chess-engine https://healeycodes.com/building-my-own-chess-engine

**Installation guidelines:**
Version of Python used: Python 3 

**Libraries imported: **

random, pygame 
‘pygame ’ can be installed through the terminal using the command ‘pip install
pygame’.

**Steps to run the interactive domain program:** 

Clone the repository. 
An option to select the board size will be displayed where you can select a number as the board size.
According to the preference, select the control strategy for players. There are 3 options to choose for each player. 1)Human : A person playing. 2)Baseline-AI : AI will be playing by choosing random actions. 3)Tree-based AI : AI will be playing by using Alpha-Beta pruning algorithm.
Run the ChessMain.py file.

**Steps to run the computer experiments for Tree-AI part:**

In 'ChessMain.py' run the game in 'demo' mode if you want to run the code. If want to generate the experiments of the Tree-AI then run the code in 'exp' mode. The mode assignment is in main. In 'demo' mode you can choose the game size and control startegies for both the players. In 'exp' mode the player one is assigned either Random-AI or Tree-AI at random and the player two is assigned AI opposite to player one. 

**Steps to run the computer experiments for Tree-AI-NN part:**

In the ‘ChessMain.py’ run the game in 'train' mode to generate the training dataset for games and than run the code in 'nn' mode to run the experiments for Tree AI Neural Network. The 'nn' mode is similar to 'exp' mode, it just use functions in nn.py file.
