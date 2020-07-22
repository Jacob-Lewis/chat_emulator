

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from numpy.random                       import default_rng
from torch.utils.tensorboard            import SummaryWriter
from model                              import DQN
from environment                        import Environment
from replay_memory                      import ReplayMemory, Transition

rng = default_rng()
BATCH_SIZE = 128
GAMMA = 0.999
EPS = 0.25
TARGET_UPDATE = 10

writer = SummaryWriter('runs/chat_emulator_experiments_v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Environment()
n_actions = env.get_action_space_size()
state_size = env.get_state_space_size()

policy_net = DQN(state_size, n_actions).to(device)
target_net = DQN(state_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(20000)


steps_done = 0

def get_valid_action(UR_matrix, action_space):
    i = 0
    valid_action = None
    while i < UR_matrix.shape[1] and not valid_action: #maybe infinite loop
        rank = torch.argsort(UR_matrix, descending=True)
        optimal_action = rank[:,i]
        if optimal_action.item() in action_space:
            valid_action = optimal_action
        i += 1
    action = torch.tensor([[valid_action]], device=device, dtype=torch.long)
    return action

def select_true_action(state, action_space):
    behavior_matrix = env._get_behavior_matrix()
    UR_matrix = state.matmul(behavior_matrix)
    return get_valid_action(UR_matrix, action_space)

def select_action(state, action_space):
    global steps_done
    sample = random.random()
    steps_done += 1
    if sample > EPS:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            UR_matrix = policy_net(state)
            return get_valid_action(UR_matrix, action_space)
    else:
        random_action = random.sample(action_space, 1)
        action = torch.tensor([random_action], device=device, dtype=torch.long)
        return action

def optimize_model(i_episode):
    #Task always results in done so may not need to set q(s',a') to 0
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    #non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    #next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (0 * GAMMA) + reward_batch #Set q(s',a') to 0

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    reward = reward_batch.sum()/BATCH_SIZE
    writer.add_scalar('Agent reward', reward, i_episode)
    writer.add_scalar('Loss', loss, i_episode)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

state, actions = env.reset()
writer.add_graph(policy_net, state)
num_episodes = 100000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state, actions = env.reset()

    #Establish true benchmark
    true_action = select_true_action(state, actions)
    true_reward,_ = env._sample_reward(true_action)
    writer.add_scalar('True reward', true_reward, i_episode)

    # Select and perform an action
    action = select_action(state, actions)
    _, reward, done, _ = env.step(action)
    reward = torch.tensor([reward], device=device)
    next_state = None

    if i_episode % 1000 == 0:
        print(f"At iteration {i_episode}")
    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Perform one step of the optimization (on the target network)
    optimize_model(i_episode)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
