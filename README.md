# Udacity Reinforcement Learning Project1

//////////////////////////////////////////
/// Work in progress
/////////////////////////////////////////

In this first project .....

[picture of game]

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/fbF0UsxEx5Y/0.jpg)](https://www.youtube.com/watch?v=fbF0UsxEx5Y)


I have used the DQN_agent that came with the workspace solution for DQN networks. My first traning result took a while before eventually capping around 15.x.

I figured that it didn't explore the random paths engough, because the Eps_decay was quite high, at 0.995. Meaning that the amound of randomness over time deminished quite fast, thus nog following the best possible path 

After changing the Eps_decay to 0.985 the initial "Average score" whent up a lot faster, and almost reached 16.5 within 1300 episodes



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




[Check]
GPU accelleration didn't do a lot for speeding up the training. It still was about a second per episode



[run localy]

The examples in the GitHub repository have this as their first cell. When Running on Windows10, it can't find torch 0.4.0

!pip -q install ./
ERROR: Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0) (from versions: 0.1.2, 0.1.2.post1, 0.1.2.post2)
ERROR: No matching distribution found for torch==0.4.0 (from unityagents==0.4.0)

I downloaded "torch-0.4.1-cp36-cp36m-win_amd64.whl" from the pytorch site

(UdacityRLProject1) C:\Users\ABloemenkamp>cd C:\Clients\Udacity\deep-reinforcement-learning\python
(UdacityRLProject1) C:\Clients\Udacity\deep-reinforcement-learning\python>pip install torch-0.4.1-cp36-cp36m-win_amd64.whl
Processing c:\clients\udacity\deep-reinforcement-learning\python\torch-0.4.1-cp36-cp36m-win_amd64.whl
Installing collected packages: torch
Successfully installed torch-0.4.1

Code issue 

packages\unityagents\environment.py", line 322, in step
for brain_name in list(vector_action.keys()) + list(memory.keys()) + list(text_action.keys()):
AttributeError: 'numpy.int64' object has no attribute 'keys'

Here i had to change 'action' (numpy.int64) to an int32 by using 'action.astype(int)'

# 2. do the step in the actual environment, and recieve a next state and reward
env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment

