This is a open-source version of the game Mind Unbind. Currently, only the core scoring algorithm - MovePredictionEngine - is published.

Mind Unbind is an abstract, free-form drawing game. The player draws a continuous and ever-changing curve, aiming to minimize repetition.
The obstacle is the player's own doodling habits: tendencies to repeat patterns, even with a best effort to vary them. 
The score is feedback on whether intention to change, or ingrained habit, was stronger (and to what degree). 

Ultimately, it explores 'free-will' in a gamified way. 

Note that the game segments the continuous curve (as in the drawings below) into a series of discrete 'moves', which MovePredictionEngine processes. 

------------
![ScoreExamples1](https://github.com/user-attachments/assets/9e9a3438-3b29-4fa6-bbc3-2c9c2bfe2bfc)

The goal is to draw a continuous curve, varying its directional flow, changing how it changes. Done well, this creates intuitively complex drawings.

Scores range 0 to 200: from no variation (a straight line, or circle), to optimally varied shapes. A random, truly mindless scribbler - lacking both intent and habit - averages 100.

Human variation is far from optimal: scores above 120 are rare, as subtle doodling defaults inevitably creep into our scribbling.

While you endeavor to vary curves as best you can, your signature habits lurk in the background, challenging your efforts. Even matching randomness can be quite difficult!

![ScoreExampeles2](https://github.com/user-attachments/assets/4eb274db-d709-40cd-b6be-008c2966efe6)
![ScoreExamples3](https://github.com/user-attachments/assets/c40513c1-b1be-43a8-9ba3-40d932432232)


------------
![Examples4](https://github.com/user-attachments/assets/dedb216b-194d-459c-ba88-cde80eda1227)


