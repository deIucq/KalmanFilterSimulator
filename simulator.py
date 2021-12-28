import numpy as np
import matplotlib.pyplot as plt

initial_state = np.array([0, 0])
KalmanGain = np.array([[0.1, 0  ],
                       [0  , 0.2]])

#config
XSysVariance = 1
VSysVariance = 1
SensXVariance = 200
SensVVariance = 5
timecount = 100
dt = 0.1

input_value = [0.5]*int(timecount/dt)                   # システムへの入力u(t) 今回は等加速度運動とした
true_position = [np.array([0,0])]*int(timecount/dt)     # システムの真の状態 [x^true, v^true] 以下，状態は[x,v]の形
estimate_position = [np.array([0,0])]*int(timecount/dt) # カルマンフィルタによって推定されるシステムの状態
odometory = [np.array([0,0])]*int(timecount/dt)         # システムの入力から推定されるシステムの状態
observation = [np.array([0,0])]*int(timecount/dt)       # センサーの値から(観測，推定される)システムの状態

#calc true_position
for t in range(int(timecount/dt)):
    if t == 0:
        true_position[t] = initial_state
        continue
    true_position[t] = np.array([[1,dt],[0,1]])@true_position[t-1]\
                       + np.array([0.5*dt*dt,dt])*input_value[t] \
                       + np.array([np.random.normal(0,XSysVariance),np.random.normal(0,VSysVariance)])

#Kalman Filter
for t in range(int(timecount/dt)):
    if t == 0:
        estimate_position[0] = initial_state
        continue
    odometory[t] = np.array([[1,dt],[0,1]])@estimate_position[t-1]\
                   + np.array([0.5*dt*dt,dt])*input_value[t]
    observation[t] = np.array([true_position[t][0]+np.random.normal(0,SensXVariance), true_position[t][1]+np.random.normal(0,SensVVariance)])
    estimate_position[t] = (np.eye(2) - KalmanGain)@odometory[t] + KalmanGain@observation[t]

fig = plt.figure()
ax_x = fig.add_subplot(2,1,1,xlabel='t',ylabel='x')
ax_v = fig.add_subplot(2,1,2,xlabel='t',ylabel='v')

#ax_x.plot(range(0,timecount,dt), [i[0] for i in odometory], label='odometory')
#ax_v.plot(range(0,timecount,dt), [i[1] for i in odometory], label='odometory')
ax_x.plot(np.arange(0, timecount, dt), [i[0] for i in observation], label='observation')
ax_v.plot(np.arange(0, timecount, dt), [i[1] for i in observation], label='observation')
ax_x.plot(np.arange(0, timecount, dt), [i[0] for i in true_position], label='true_position')
ax_v.plot(np.arange(0, timecount, dt), [i[1] for i in true_position], label='true_position')
ax_x.plot(np.arange(0, timecount, dt), [i[0] for i in estimate_position], label='estimate_position')
ax_v.plot(np.arange(0, timecount, dt), [i[1] for i in estimate_position], label='estimate_position')

ax_x.legend()
ax_v.legend()
plt.show()
