import numpy as np
import kalmanlib as klib
import matplotlib.pyplot as plt


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

filter = klib.KalmanFilter(A, B, Ry, Rx,startMean,startCov)
sig,errs,aprSig,aprErrs = filter.filterSignal(xs)
smoothedSig, smoothedErrs = filter.smoothSignal(sig,errs,aprSig,aprErrs)


f,(ax1,ax2) = plt.subplots(1,2,figsize=(11,5))

ax1.grid()
ax1.set_title("Показатель номер 1", fontsize=16)
ax1.set_xlabel("t", fontsize=14)
ax1.plot(np.arange(T), ys[0,:])
ax1.plot(np.arange(T), xs[0,:])
ax1.plot(np.arange(T), sig[0,:])
ax1.plot(np.arange(T), smoothedSig[0,:])
ax1.fill_between(np.arange(T),smoothedSig[0,:]-1.96*np.sqrt(smoothedErrs[0,0,:]), smoothedSig[0,:]+1.96*np.sqrt(smoothedErrs[0,0,:]), alpha=0.4 )
ax1.fill_between(np.arange(T),sig[0,:]-1.96*np.sqrt(errs[0,0,:]), sig[0,:]+1.96*np.sqrt(errs[0,0,:]), alpha=0.4 )


ax2.grid()
ax2.set_title("Показатель номер 2", fontsize=16)
ax2.set_xlabel("t", fontsize=14)
ax2.plot(np.arange(T), ys[1,:])
ax2.plot(np.arange(T), xs[1,:])
ax2.plot(np.arange(T), sig[1,:])
ax2.plot(np.arange(T), smoothedSig[1,:])
ax2.fill_between(np.arange(T),smoothedSig[1,:]-1.96*np.sqrt(smoothedErrs[1,1,:]), smoothedSig[1,:]+1.96*np.sqrt(smoothedErrs[1,1,:]), alpha=0.4 )
ax2.fill_between(np.arange(T),sig[1,:]-1.96*np.sqrt(errs[1,1,:]), sig[1,:]+1.96*np.sqrt(errs[1,1,:]), alpha=0.4 )

ax2.legend(["Сигнал(Y)","Измерения(X)","Отфильтрованный сигнал","Сглаженнный сигнал"])
# plt.savefig('filtered.pdf')
plt.show()
