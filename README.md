# Udacity Reinforcement Learning Project1

//////////////////////////////////////////
/// Work in progress
/////////////////////////////////////////

## Introduction 

This is the first project of the Udacity Deep Reinforcement Learning course. In this project Udacity provides a Unity3D application that is used as a training environment for Deep Q network. The goal of this environment is to collect all the yellow bananas and not touch the blue ones. The environment provides ray casts which return the distance to the floor, bananas and walls. The resulting vector is passed to the Jupyter Notebook as a state.

```python
Number of agents: 1
Number of actions: 4
States look like: [1.         0.         0.         0.         0.84408134 0.
 0.         1.         0.         0.0748472  0.         1.
 0.         0.         0.25755    1.         0.         0.
 0.         0.74177343 0.         1.         0.         0.
 0.25854847 0.         0.         1.         0.         0.09355672
 0.         1.         0.         0.         0.31969345 0.
 0.        ]
States have length: 37
```

The agent tries to find the action with the most future cumulative reward, and thus trains the deep Neural network to predict the best action, given a random state.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/fbF0UsxEx5Y/0.jpg)](https://www.youtube.com/watch?v=fbF0UsxEx5Y). 

*Training in progress*

## Setup the environment in Windows 10

As with most machine learning projects, its best to start with setting up a virtual environment. This way the packages that need to be imported, don't conflict with Python packages that are already installed. For this project i used the Anaconda environment based on Python 3.6. 

While the example project provides a requirements.txt, i ren into this error while adding the required packages to your project

```python
!pip -q install ./
ERROR: Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0) (from versions: 0.1.2, 0.1.2.post1, 0.1.2.post2)
ERROR: No matching distribution found for torch==0.4.0 (from unityagents==0.4.0)
```

The solution, is to install a downloaded wheel file form the PyTourch website yourself. I downloaded "torch-0.4.1-cp36-cp36m-win_amd64.whl" from the PyTourch site https://pytorch.org/get-started/previous-versions/

```
(UdacityRLProject1) C:\Clients\Udacity\deep-reinforcement-learning\[your project folder]>pip install torch-0.4.1-cp36-cp36m-win_amd64.whl
Processing c:\clients\udacity\deep-reinforcement-learning\[your project folder]\torch-0.4.1-cp36-cp36m-win_amd64.whl
Installing collected packages: torch
Successfully installed torch-0.4.1
```

After resolving the dependencies, i still had a code issue, because the action returned a numpy.int64 instead of an in32.

```
packages\unityagents\environment.py", line 322, in step
for brain_name in list(vector_action.keys()) + list(memory.keys()) + list(text_action.keys()):
AttributeError: 'numpy.int64' object has no attribute 'keys'
```

Here i had to change 'action' (numpy.int64) to an int32 by using 'action.astype(int)'

```Python
# 2. do the step in the actual environment, and recieve a next state and reward
env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment
```

When all dependencies and issues are resolved, the training can begin.

## Training the agent with code provided by the course

To start, and make sure the environment works, I have used the DQN_agent that came with the workspace solution for DQN networks. My first training result took a while before eventually capping with an average score of around 15.x.

![alt text](https://github.com/fuzzballb/UdacityRFlearningProject1/blob/master/images/Eps_decay_0_995.PNG "Training with default epsilon decay")

I figured that it didn't explore the random paths enough, because the Eps_decay was quite high, at 0.995. Meaning that the amound of randomness over time diminished quite fast, making the agent stick to what it already knows 

After changing the Eps_decay to 0.905 the initial "Average score" went up a lot faster, and almost reached 16.5 within 1300 episodes

```
Start (eps_decay=0.905)
Episode 100	Average Score: 3.42
Episode 200	Average Score: 8.55
Episode 300	Average Score: 12.29
Episode 400	Average Score: 14.83
Episode 500	Average Score: 16.37
Episode 600	Average Score: 15.27
Episode 700	Average Score: 15.24
Episode 800	Average Score: 15.51
Episode 900	Average Score: 15.46
Episode 1000	Average Score: 16.30
Episode 1100	Average Score: 16.49
Episode 1200	Average Score: 16.20
Episode 1300	Average Score: 16.34
Episode 1330	Average Score: 16.02
When finished, you can close the environment.
```

![alt text](https://github.com/fuzzballb/UdacityRFlearningProject1/blob/master/images/Eps_decay_0_905.PNG "Training with diminished epsilon decay")

It seems that exploring states and the resulting rewards, beyond the current policy pays off.


## project environment

The README describes the project environment details (i.e., the state and action spaces, and when the environment is considered solved).

## Learning Algorithm

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

Now we know the basic theory about Policy learning (Sampling actions based on past recorded experiences) we can analyze the code.

 
1.	First we initialize the agent in the Navigation notebook

Navigation.ipynb

```Python
        from dqn_agent import Agent
            # initialise an agent
            agent = Agent(state_size=37, action_size=4, seed=0)
```

2.	This sets the state and action size for the agent and creates two Neural networks that both map a state to an action values. 

Dqn_agent.py

```Python
       self.state_size = state_size
       self.action_size = action_size
       self.seed = random.seed(seed)

       # Q-Network
       self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
       self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
```


Model.py

```Python
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```


Lastly a replay buffer is created to store previous experiences

Dqn_agent.py

```Python
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
```

3.	The envirionment is set up and a score system is initialized for each 100 episodes. After this, we start looping trough the timesteps, and the agent perfoms an act

Navigation.ipynb

```Python
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # state = env.reset()
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            # 1. change to a version of agent.act
            action = agent.act(state, eps)
```

4.	The agent returns an action given the state as per current policy. The state is passed to the local qnetwork and this retuns all action values. The sum of all these values is one. Then depending of epsilon (exploitation vs exploration) ether the highest action value is taken, or a random value. In the beginning all values return 0, because there is no reward to give.

Dqn_agent.py

```Python
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # .astype(int)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size)).astype(int)
```


5.	Now we know what action we are going to take, we can set a step in the envirionment using this action. The result will be the next_state, reward and done (terminal state)

Navigation.ipynb

```Python
            # 2. do the step in the actual environment, and recieve a next state and reward
            env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            
            # 3. change to a version of agent.step
            agent.step(state, action, reward, next_state, done)
```

6.	Now we know which state we where in, the action we took, the reward we got, the next state we are in and if we are done. This information is stored in the replay buffer. If there are engough experiences in the replay buffer, we can start to learn from these experiences in batches. 

Dqn_agent.py

```Python
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
```

Here we have the replay buffer and two networks, qnetwork_target and qnetwork_local. The actions, rewards and the next states that where recorded from the environment while taking steps, are recorded in the replay buffer. When there are at least 100 steps in the replay buffer, the learning process can begin.


##The target network

The target network gets the maximum predicted Q values (Q_targets_next), given the next_states. It can make this prediction, because later on in the code, the wights for the local network are copied to this target network. 

But we don't need the Q_targets_next values, instead we need the Q targets of the current state. To calculate the Q targets of the current state we use the rewards of the current states.

```Python
         rewards
```

plus the discounted rewards of the expected cumulative rewards of the previsouly predicted rewards of the next state.  

```Python
         (gamma * Q_targets_next 
```

if there are no next states, then the dones are 0 and thus this part of the equation is 0. and only the rewards remain

```Python
         * (1 - dones))
```

the full equation is 

```Python
         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
```

##The local network 

Now that we have a target network that can predict the Q_targets for the current states, based on the next_states. We can train the local network to generate the same results while having states and actions as input.

First we predict what the local network expects the Q values to be given the current states and actions

```Python
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
```

Then we train the local network to generate Q_expected resutls that are closer to the Q_target results. 

```Python
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

Essentially what is happening here is that the replay buffer has the current states and actions and the corresponding future states and rewards. By calculating the Q_targets of the current state, based on future values (rewards and next_states) we van optimise the prediction of the local network, that has the current states and actions as input.

Now that the weights of the local network have been optimised, we update the network targets with the same wights, so it can also make a better prediction

```Python
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   
```




## Hyper parameters




## Ideas for Future Work




## GPU acceleration

GPU acceleration didn't do a lot for speeding up the training. It still was about a second per episode






