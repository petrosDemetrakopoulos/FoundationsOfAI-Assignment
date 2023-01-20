## Foundations of AI Course assignments - Group 09
This repo hosts the assignments our group delivered in the course "Foundations of AI" during the graduate prgramme DS&AI of TU/e.  
The assignments are implementing 4 different adversary agents for the "Competitive Sudoku" (described below) game.

### Group members
Petros Demetrakopoulos  
Stefanos Karamperas  
Dimitar Glavikov

### Game description and rules
(The followingg description and rules have been taken as-is from the relative course material file)  
  
Competitive sudoku is an adversarial two-player game, in which players take
turns filling out cells of a given sudoku puzzle. Players score points by com-
pleting regions; when the puzzle is completed, the player with the most points
wins. Because the puzzle-aspect of the sudoku is non-trivial—it is in general
non-obvious which value should be entered in which cell—the game is played
under the supervision of an oracle that ensures the puzzle remains solvable af-
ter every move. Note that the game may be played with improper puzzles—in
particular, the game may be started with a completely empty grid. We now
present the full rules in detail.  

**Start**: Both players start with 0 points. The starting player is chosen using
some unspecified criterion; the choice may be random or based on external
circumstance such as tournament rules. Beginning with the starting player, the
players consecutively take turns until the game is finished
