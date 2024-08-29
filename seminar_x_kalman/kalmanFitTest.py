import numpy as np
import kalmanlib as klib
import matplotlib.pyplot as plt




#1d example (take any numbers basically)
#A = np.array([[0.8]])#dynamics matrix
#B = np.array([[1]])#observation matrix
#Ry = np.eye(1)#
#Rx = np.eye(1)*35#
#startMean=np.array([5])
#startCov=np.eye(1)*0.1


#2d example
A = np.array([[1,0.3],[-0.001,1]])#np.eye(2)*1.0000000001 #uniform-velocity movement
B = np.array([[1,0],[0,0.5]])
Ry = np.eye(2)#
Rx = np.eye(2)*35#
startMean=np.array([5,10])
startCov=np.eye(2)*0.1

T=100#number of steps to simulate

sigGen = klib.LinearGaussianSignalGenerator(A, B, Ry, Rx)

start = np.random.multivariate_normal(mean=startMean,cov=startCov)
ys, xs = sigGen.generate(T,start)

filter = klib.KalmanFilter(A=np.eye(2)*0.2, B=B, Ry=np.eye(2), Rx=35*np.eye(2), startMean=np.ones([2]), startCov=np.eye(2)*0.1 )
filter.fit(xs, Niter=50000, fixB=True, fixRx =True)
print(filter)
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
ax1.legend(["Сигнал(Y)","Измерения(X)","Отфильтрованный сигнал","Сглаженнный сигнал"])

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
#ax2.legend(["Сигнал(Y)","Отфильтрованный сигнал","Сглаженнный сигнал"])
#plt.savefig('filtered.pdf')
plt.show()
