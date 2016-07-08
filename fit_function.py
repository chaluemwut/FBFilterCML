import numpy as np
from numpy.polynomial import polynomial as P
from matplotlib import pyplot

# x = np.linspace(-1,1,51) # x "data": [-1, -0.96, ..., 0.96, 1]
# y = x**3 - x + 0.01*np.random.randn(len(x)) # x^3 - x + N(0,1) "noise"
x = [0, 25, 50, 75 ,100, -25, -50, -75 , -100]
y = [0, 4.914710779, 6.179001863, 8.20972476, 32.95841755, 4.914710779, 6.179001863, 8.20972476, 32.95841755]

c, stats = P.polyfit(x,y,2,full=True)
print c # c[0], c[2] should be approx. 0, c[1] approx. -1, c[3] approx. 1

xm = np.linspace(0, 100, 101)
ym = c[2]*np.power(xm, 2*np.ones((1,101)))+\
c[1]*np.power(xm, 1*np.ones((1,101)))+\
c[0]

pyplot.plot(x[0:5], y[0:5], 'ob-')
pyplot.plot(xm, ym[0], '*r--')
pyplot.show()

print stats # note the large SSR, explaining the rather poor results