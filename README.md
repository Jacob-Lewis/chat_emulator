# Engagement Optimization
This is a project established July 2020 to continue research, experimentation, and development of engagement optimization algorithms with applications to consumer media.
Contained in this repo is:
Environment for engagement optimization that includes chat emulator
Algorithms for engagement optimization. Pytorch results shared in Results directory

This emulator simulates the challenge of surfacing the content type returned from various data sources that is most likely to generate a click from the user. Below is a diagram of the challenge:
![Alt text](docs/emulator_problem.png?raw=true "Title")

### Guide to the problem setup
##### Step 1:
The user sends a signal to the bot. This signal will be the state that we will want to choose the best action for.
##### Step 2: 
The bot sends retrieves content options based on the user signal. These content options constitute the action space for the RL agent because the RL agent must choose one of these pieces of response content to return to the user. The bot passes the signal (state) and response content options (action space) to the RL agent. 
##### Step 3: 
The RL agent choose the best response content option (action) based on the available options (action space) to show to the user based on the user's input signal (state). The RL will choose which response content option to show based on it's policy, which is designed to maximize the clicks of the user over the long term. The policy is where the agent manages its tradeoff between exploration and exploitation. For example, sometimes the agent may choose a different content type (action) to show the user based on her signal (state), just to see what will happen.
##### Step 4: 
The user will choose to either click or not click (reward) on the content shown by the bot. The user's behavior is dictated by a behavior matrix. This matrix defines the true behavior of the user. The number of rows is equal to the dimensions of the user utterance (state space) and the number of columns is equal to the dimensions of the different types of content (action space). Each element shows the probability of the user clicking on that content type based on the magnitude of that dimension in the state vector. For example behavior_matrix[2,1] shows the probability of the user clicking on content type 1 if their input signal was 100% dimension 2 (Learn). 
##### Step 5: 
This feedback is used by the RL agent to update its model of the behavior_matrix.

Resources:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://github.com/ShangtongZhang/DeepRL
https://github.com/navneet-nmk/pytorch-rl
