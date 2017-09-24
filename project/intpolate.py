
import numpy as np
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt


'''
x = np.linspace(0, 100, 9)
y = x**2
xi = np.linspace(0, 150, 101)


rbf = Rbf(x, y)
fi = rbf(xi)


plt.plot(xi, np.sin(xi), 'bo',xi,fi,'g')

plt.title('Interpolation using RBF - multiquadrics')
plt.show()
'''
def nextPoint(x,y,xnew,predicted):
	rbf=Rbf(x,y)
	predicted=rbf(xnew)
	
def fit(x,y):
	z=np.polyfit(x,y,3)
	p=np.poly1d(z)
	return p
'''
x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
p=fit(x,y)
print p
xnew=np.arange(-10,10)
ynew=p(xnew)

plt.plot(x,y,'b-')
plt.show()
plt.plot(xnew,ynew,'b-')
plt.show()
'''

