# Neural Network Fight Simulator
An evolutionary neural network designed to gradually improve it's ability to play a Mario Bros like mini-game (simply jumping on the head of the opponent). The program has a population of 100 candidate neural networks. It tests all of their performances against an opponent (whose behavior the user can partially control) and rates them based on how they do (using a function also partially controllable by the user). The program then terminates some number of networks, normally 55 but also any networks whose score was below the "Cutoff" and allows the remaining ones to reproduce. If all of the networks are killed off, the simulator starts over with totally random networks. The surviving networks after each generation are randomly modified and the cycle continues.

## How the networks are rated
The networks are scored based on their ability to jump on the head of their opponent. The opponent behaves like a Goomba from Mario, simply moving back and forth at a constant rate. The goal of the networks is to control their player who can move left to right and jump such that they jump on the head of the opponent. This is defined to occur when the relative velocity between the player and opponent is downward and the two collide. If the relative velocity is negative, that is counted as a win for the network. If it is non-negative, that is counted as a loss. If the player jumps off the edge of the stage, that is a fall. Finally, if the player survives for some maximum amount of time, it is recorded as a survival. The user can enter values to assign to each of these outcomes using the program's main window.

run 
```sh
$ python3 fightingEvolutionaryNeuralNetwork.py
```
to start the program. A window will appear.
![alt Startup window](https://github.com/patricknaughton01/nn-fight-simulator/blob/master/pretty%20good%20settings.png?raw=true "Startup window")

# Labels
The label at the top notifies you of what generation you are on (which one just completed). The "Wins", "Losses," and "Falls" labels and graphs display how many neural networks won, lost, or fell in the last iteration. The bottom label displays the current state of the simulator.

# Settings
The user can control how the opponent moves and how each network is scored. To modify a setting, simply click its box and begin typing. Pressing backspace will do what you expect. The focused field will be outlined by a blue rectangle.
## Evaluation
The "Win," "Lose," "Fall," and "Surv" constants are numbers that are added to the network's score whenever the network wins, loses, falls, or survives respectively. The "Cutoff" is the minimum score a network has to have to move on to the next round.
"Time," "Close," and "Top" "Muls" are the multipliers assigned to certain actions the player can take. 
- The time multiplier is divided by the reciprocal of the time the player is alive to encourage the player to hit the enemy quickly. 
- The close multiplier multiplies the amount of time the player spends "close" (within 20 pixels) to the enemy. 
- The top multiplier is multiplied by the number of pixels above the enemy the player is at the end of the match.
## Game Parameters
- "Jump" is a number from 0-1 that defines how often the enemy should jump, higher numbers meaning they are more likely to.
- "Height" is how high the enemy should jump in pixels.
- "Max Time" is the number of seconds (when simulating at "real time") each network should get before their turn is terminated.

# Buttons
The buttons conrol the progression of the algorithm.
- "Do Step By Step": Show the user all 100 networks controlling the player to fight the enemy. Once clicked, there is no way to skip the animations, but you can speed up playback speed by pressing one of the number keys (this will set the playback speed to (n+1)x, so pressing 0 is "real time," 1 is double speed, 9 is 10x speed).
- "ASAP": Do one generation as soon as possible. This will simulate all 100 networks and kill off the ones that do not perform as well without showing the user the animation.
- "ALAP": Do generations for as long as possible. This is essentially the same as "ASAP" but a new generation will start as soon as the previous one finishes. To exit this mode, hold ESC.
- "Extinction": This will kill the 90 worst networks. This is typically done if they get stuck in a rut. Another way to start over is to raise the cutoff to an absurd number so that all the networks get killed.

# NumPy :(
Unfortunately, I wrote this program before I knew about NumPy. If I ever revisit it, that will be the first thing I change to make it run faster.
