import math
import numpy as np
import matplotlib.pyplot as plt

initial_state = [0, 0]
KalmanGain = 0.2

#config
XSysVariance = 5
VSysVariance = 5
SensXVariance = 50
SensVVariance = 50
timecount = 100
dt = 1

true_position = [[0]*2]*int(timecount/dt)     # システムの真の状態 [x^true, v^true]
estimate_position = [[0]*2]*int(timecount/dt) # カルマンフィルタによって推定されるシステムの状態 [x^est, v^est]
odometory = [[0]*2]*int(timecount/dt)
observation = [[0]*2]*int(timecount/dt)
input_value = [0.1]*int(timecount/dt)           # システムへの入力u(t)

#calc true_position
for t in range(int(timecount/dt)):
    if t == 0:
        true_position[t] = initial_state
        continue
    true_position[t] = [true_position[t-1][0]+true_position[t-1][1]*dt+0.5*input_value[t]*dt*dt+np.random.normal(0,XSysVariance),
                        true_position[t-1][1]+input_value[t]*dt+np.random.normal(0,VSysVariance)]

#Kalman Filter
for t in range(int(timecount/dt)):
    if t == 0:
        estimate_position[0] = initial_state
        continue
    odometory[t] = [estimate_position[t-1][0]+estimate_position[t-1][1]*dt+0.5*input_value[t]*dt*dt,
                            estimate_position[t-1][1]+input_value[t]*dt]
    observation[t] = [true_position[t][0]+np.random.normal(0,SensXVariance), true_position[t][1]+np.random.normal(0,SensVVariance)]
    estimate_position[t] = [(1-KalmanGain) * odometory[t][0] + KalmanGain * observation[t][0],
                            (1-KalmanGain) * odometory[t][1] + KalmanGain * observation[t][1]]

fig = plt.figure()
ax_x = fig.add_subplot(2,1,1,xlabel='t',ylabel='x')
ax_v = fig.add_subplot(2,1,2,xlabel='t',ylabel='v')

#ax_x.plot(range(0,timecount,dt), [i[0] for i in odometory], label='odometory')
#ax_v.plot(range(0,timecount,dt), [i[1] for i in odometory], label='odometory')
ax_x.plot(range(0,timecount,dt), [i[0] for i in observation], label='observation')
ax_v.plot(range(0,timecount,dt), [i[1] for i in observation], label='observation')
ax_x.plot(range(0,timecount,dt), [i[0] for i in true_position], label='true_position')
ax_v.plot(range(0,timecount,dt), [i[1] for i in true_position], label='true_position')
ax_x.plot(range(0,timecount,dt), [i[0] for i in estimate_position], label='estimate_position')
ax_v.plot(range(0,timecount,dt), [i[1] for i in estimate_position], label='estimate_position')

ax_x.legend()
ax_v.legend()
plt.show()
