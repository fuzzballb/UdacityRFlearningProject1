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


## Hyper parameters




## Ideas for Future Work




## GPU acceleration

GPU acceleration didn't do a lot for speeding up the training. It still was about a second per episode






