from pykalman import KalmanFilter
import kalmanlib as klib
import numpy as np
import matplotlib.pyplot as plt

#kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
#measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
#kf = kf.em(measurements, n_iter=5)
#(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
#(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

A = np.eye(2)
B = np.array([[1.9999,0.31],[-0.2,1.911]])
Ry = np.array([[2,0.00002],[0.00002,3]])*10
Rx = np.array([[1.3,0.0002],[0.0002,3]])*100
T=500

sigGen = klib.LinearGaussianSignalGenerator(A, B, Ry, Rx)

startMean=np.array([10,5])
startCov=np.eye(2)*0.1

start = np.random.multivariate_normal(mean=startMean,cov=startCov)#np.array([10,5])
ys, xs = sigGen.generate(T,start)
#xs = xs[...,1:]
#ys = ys[...,1:]

kf = KalmanFilter(transition_matrices = [[1, 0], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
measurements = np.transpose(xs)  # observations
kf = kf.em(measurements,n_iter=5000,em_vars = ["transition_matrices","observation_matrices","transition_covariance","observation_covariance", "initial_state_mean","initial_state_covariance"])
(sig, errs) = kf.filter(measurements)
(smoothedSig, smoothedErrs) = kf.smooth(measurements)
smoothedSig = np.transpose(smoothedSig)
smoothedErrs = np.transpose(smoothedErrs,(1,2,0))
sig = np.transpose(sig)
errs = np.transpose(errs,(1,2,0))



f,(ax1,ax2) = plt.subplots(1,2,figsize=(11,5))

ts = np.arange(T)
ax1.grid()
ax1.set_title("Показатель номер 1", fontsize=16)
ax1.set_xlabel("t", fontsize=14)
ax1.plot(ts, ys[0,:])
ax1.plot(ts, xs[0,:])
ax1.plot(ts, sig[0,:])
ax1.plot(ts, smoothedSig[0,:])
ax1.fill_between(ts,smoothedSig[0,:]-1.96*np.sqrt(smoothedErrs[0,0,:]), smoothedSig[0,:]+1.96*np.sqrt(smoothedErrs[0,0,:]), alpha=0.4 )
ax1.fill_between(ts,sig[0,:]-1.96*np.sqrt(errs[0,0,:]), sig[0,:]+1.96*np.sqrt(errs[0,0,:]), alpha=0.4 )


ax2.grid()
ax2.set_title("Показатель номер 2", fontsize=16)
ax2.set_xlabel("t", fontsize=14)
ax2.plot(ts, ys[1,:])
ax2.plot(ts, xs[1,:])
ax2.plot(ts, sig[1,:])
ax2.plot(ts, smoothedSig[1,:])
ax2.fill_between(ts,smoothedSig[1,:]-1.96*np.sqrt(smoothedErrs[1,1,:]), smoothedSig[1,:]+1.96*np.sqrt(smoothedErrs[1,1,:]), alpha=0.4 )
ax2.fill_between(ts,sig[1,:]-1.96*np.sqrt(errs[1,1,:]), sig[1,:]+1.96*np.sqrt(errs[1,1,:]), alpha=0.4 )

ax2.legend(["Сигнал(Y)","Измерения(X)","Отфильтрованный сигнал","Сглаженнный сигнал"])
# plt.savefig('filtered.pdf')
plt.show()
