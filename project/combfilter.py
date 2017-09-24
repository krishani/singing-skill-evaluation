import numpy as np
import math
import matplotlib.pyplot as plt
F=100
x1=4800
x2=6000
'''
pxf=[[0 for x in range(x1,x2)]for y in range(F)]
print pxf
'''
def gaussianComb(x,F):
	sigma=0
	r=16
	l=1.0/(math.sqrt(2*np.pi*r))
	for i in range(100):	
		z=-1.0*math.pow(x-F-100*i,2)/(2.0*256)
		sigma += math.exp(z)
	
	return l*sigma
'''
for f in range(F):
	for x in range(x1,x2):
		pxf[f][x-x1]=gaussianComb(x,f)
y=range(4800,6000)
f,(ax1,ax2)=plt.subplots(1,2, sharey=True)
ax1.plot(pxf[0],y)
ax1.set_xlabel('P(x,f)')
ax1.set_ylabel('cents')

ax2.plot(pxf[1],y)

ax1.set_title('F=78')
ax2.set_title('F=19')

plt.show()
'''
#print pxf
