import numpy as np
import torch as T

from dddqn_network import DeepQNetwork

"""## Storing Replay Memories
In this section we will create a mechanism for the agent ot keep track of statesm actions, rewards, new states, and the final state.

All of these factors will be used in the calculation of the Target for the loss function of the DQN.

"""


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DDDQNAgent(object):
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        n_actions,
        input_dims,
        mem_size,
        batch_size,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        algo=None,
        env_name=None,
        chkpt_dir="tmp/dqn",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name=self.env_name + "_" + self.algo + "_q_eval",
            chkpt_dir=self.chkpt_dir,
        )
        self.q_eval.to(self.q_eval.device)
        
        self.q_next = DeepQNetwork(
            self.lr,
            self.n_actions,
            input_dims=self.input_dims,
            name=self.env_name + "_" + self.algo + "_q_next",
            chkpt_dir=self.chkpt_dir,
        )
        self.q_next.to(self.q_next.device)

    # Epsilon greedy action selection
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Add dimension to observation to match input_dims x batch_size by placing in list, then converting to tensor
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            # As our forward function now has both state and advantage, fetch latter for actio selection
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    # Replay action selection
    def choose_best_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        # Replace target network if appropriate
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        # Fetch states and advantage actions for current state using eval network
        # Also fetch the same for next state using target network
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        # Eval network calculation of next state V and A
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        # Indices for matrix multiplication
        indices = np.arange(self.batch_size)

        # Calculate current state Q-values and next state max Q-value by aggregation, subtracting constant advantage mean
        # Along first dimension, which is action dimension, keeping original matrix dimensions

        # recall [indices,actions] is used to maintain array shape of (batch_size) instead of (batch_size,actions)
        # Essentilly by adding a a batch index to our vector array we ensure that calculated Q_pred is not tabular, but applicable for a batch update
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        # For q_next, fetch max along the action dimension. 0th element, because max returns a tuple,
        # of which 0th position are values and 1st position the indices.
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        # QEval q-values for DDQN
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        # Build your target using the current state reward and q_next, DDQN setup
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss.item()
