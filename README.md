[Mario.webm](https://github.com/sacchinbhg/Mario-Reinforcement-Learning/assets/61612220/41dc549b-9dda-4bbb-8615-1b6f2fa9ff7d)# Mario-Reinforcement-Learning
## Configuring GYM
In my project, I taught a computer to play the Super Mario Bros game. To help it understand the game, I used a special way to look at what's happening on the screen.

Imagine the game like a big grid made up of small squares. Each square can be different things, like a block, a pipe, or an enemy. But the computer doesn't see it like we do. It looks at the game's memory, which is like a list of all the squares in the grid. There's a tricky part because the game's screen only shows some of the squares at a time. When it reaches the end, it starts over and shows different squares. But the memory keeps track of all of them.

I also needed to tell the computer where Mario and the enemies are on the screen. Instead of saying "Mario is here," I had to say how many squares Mario is from the left and from the top. It's like telling someone where to find something on a big map.

To make it easier for the computer, I gave each square a number. For example, Mario got the number 2, and blocks got the number 1. This made it simpler for the computer to understand what's happening. However, it sometimes got confused with enemies like Piranha Plants, treating them as if they were Goombas.

I made sure the computer could keep track of what's happening over time. It remembered the last few moments in the game, almost like when you watch a slow-motion replay of a sports game.

I had to choose between two ways for the computer to learn. One way was to make it look at the actual pictures on the screen, but that would have been too slow and hard for my computer to learn. So, I made it read the memory instead, which was faster and simpler for my computer.

I also used a scoring system to teach the computer. If Mario went a long way in the game and didn't take too long, it got a high score. But if Mario fell or got caught by an enemy, it got a lower score.


## Training
I employed the PPO agent from SB3, using the MlpPolicy, which consisted of two layers of Dense(64). I used the default settings for most parameters, along with a linear annealing learning rate scheduler that gradually decreased the learning rate to zero by the end of each training session. This addition notably enhanced the training process's reliability. Each model underwent training for 10 million steps, and the entire training session typically lasted approximately 7.5 hours. I am using a GeForce GTX 1080 Ti

## Result

Regrettably, despite the agent's success in completing level 1-1, it faced challenges in other stages, indicating that it may not have grasped every aspect of the game's mechanics as intended. While it learned to jump when encountering enemies or detecting bottomless pits, the stark contrast between its flawless performance in 1-1 and repeated failures in other levels suggests that it relied heavily on memorizing the layout of each level to optimize its path from start to finish. This reliance on level-specific learning may not generalize well to new levels.

To address this issue, I consider a few potential solutions:

Introducing randomness in the starting location within a level, although the gym environment may not support this directly. We could experiment with letting the game run for a random number of initial steps before the agent takes control. However, this approach might still favor the early part of the stage.

Exploring transfer learning, where we use a pre-trained agent and train it on new levels for a shorter duration. With exposure to more diverse levels, the agent may develop a broader understanding of the game rather than memorizing specific level layouts. However, there's a concern about potential forgetting as it adapts to new levels while retaining prior knowledge.

Training the agent on a subset of stages. In each training episode, the agent would tackle a different stage randomly selected from a subset of similar levels. This approach, feasible after release 7.4.0, seems promising and straightforward to implement. However, there's a question of whether the policy can effectively converge, but it's worth experimenting with nonetheless.


[Mario.webm](https://github.com/sacchinbhg/Mario-Reinforcement-Learning/assets/61612220/477348d4-aadd-49ef-9714-a56d160c4e1f)


