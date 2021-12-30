import numpy as np
import matplotlib.pyplot as plt

# config
XSysVariance = 5
VSysVariance = 2
XSensVariance = 500
VSensVariance = 50
timecount = 100
dt = 0.1

# figure
fig = plt.figure()
ax_x = fig.add_subplot(2,1,1,xlabel='t',ylabel='x')
ax_v = fig.add_subplot(2,1,2,xlabel='t',ylabel='v')

# system
# 初期状態
initial_state = np.array([0, 0])
# システムへの入力(加速度)u(t) 今回は等加速度運動
input_value = [2]*int(timecount/dt)
# システムの真の状態 [x^true, v^true] 以下，各状態は[x,v]の形
true_position = [np.array([0,0])]*int(timecount/dt)
# カルマンフィルタによって推定されるシステムの状態
estimate_position = [np.array([0,0])]*int(timecount/dt)
# システムの入力から推定されるシステムの状態
odometory = [np.array([0,0])]*int(timecount/dt)
# センサーの値から推定されるシステムの状態
observation = [np.array([0,0])]*int(timecount/dt)
# カルマンゲイン
kalmangain = [np.array([[0,0],[0,0]])]*int(timecount/dt)
# 最適カルマンゲイン(最初に与えられた分散を用いて計算されたもの。現実のシステムではSysVarianceは事前に与えられないため，逐次計算する必要がある。)
optimized_kalmangain = [np.array([[XSysVariance/(XSysVariance+XSensVariance), 0],[0, VSysVariance/(VSysVariance+VSensVariance)]])]*int(timecount/dt)

# calculate true_position & observation
for t in range(int(timecount/dt)):
    if t == 0:
        true_position[t] = initial_state
        continue
    true_position[t] = np.array([[1,dt],[0,1]])@true_position[t-1] \
                       + np.array([0.5*dt*dt,dt])*input_value[t] \
                       + np.array([np.random.normal(0,XSysVariance),np.random.normal(0,VSysVariance)])
    observation[t] = true_position[t] + np.array([np.random.normal(0,XSensVariance), np.random.normal(0,VSensVariance)])
ax_x.plot(np.arange(0, timecount, dt), [i[0] for i in true_position], label='true position', zorder=2)
ax_v.plot(np.arange(0, timecount, dt), [i[1] for i in true_position], label='true position', zorder=2)

# Kalman Filter
for t in range(int(timecount/dt)):
    if t == 0:
        estimate_position[0] = initial_state
        continue
    odometory[t] = np.array([[1,dt],[0,1]])@estimate_position[t-1] \
                    + np.array([0.5*dt*dt,dt])*input_value[t]
    estimate_position[t] = (np.eye(2) - optimized_kalmangain[t])@odometory[t] + optimized_kalmangain[t]@observation[t]
ax_x.plot(np.arange(0, timecount, dt), [i[0] for i in observation], label='observation', zorder=1)
ax_v.plot(np.arange(0, timecount, dt), [i[1] for i in observation], label='observation', zorder=1)
ax_x.plot(np.arange(0, timecount, dt), [i[0] for i in estimate_position], label='estimated position')
ax_v.plot(np.arange(0, timecount, dt), [i[1] for i in estimate_position], label='estimated position')

# Kalman Filter with KalmanGain optimized in advance.
for t in range(int(timecount/dt)):
    if t == 0:
        estimate_position[0] = initial_state
        continue
    odometory[t] = np.array([[1,dt],[0,1]])@estimate_position[t-1] \
                    + np.array([0.5*dt*dt,dt])*input_value[t]
    observation[t] = true_position[t] + np.array([np.random.normal(0,XSensVariance), np.random.normal(0,VSensVariance)])
    estimate_position[t] = (np.eye(2) - optimized_kalmangain[t])@odometory[t] + optimized_kalmangain[t]@observation[t]

# calculate only odometory
for t in range(int(timecount/dt)):
    if t == 0:
        estimate_position[0] = initial_state
        continue
    estimate_position[t] = np.array([[1,dt],[0,1]])@estimate_position[t-1] \
                   + np.array([0.5*dt*dt,dt])*input_value[t]
ax_x.plot(np.arange(0, timecount, dt), [i[0] for i in estimate_position], label='only odometory')
ax_v.plot(np.arange(0, timecount, dt), [i[1] for i in estimate_position], label='only odometory')

ax_x.legend()
ax_v.legend()
plt.show()
