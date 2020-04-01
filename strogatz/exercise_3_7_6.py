import matplotlib.pyplot as plt
import math

aa      = [1,2,3]
bs      = [0.1,0.2,0.3]
h       = 0.1
l       = 10.0
colours = ['r','g','b','c','m','y']
i       = 0

us      = [h*u for u in range(0,int(l/h))]
u2s     = [math.exp(-u) for u in us]
plt.plot(us,u2s,c='k')

for a in aa:
    for b in bs:
        u1s = [a - b *u for u in us]
        plt.plot(us,u1s,c=colours[i],label='a={0},b={1}'.format(a,b))
        i+=1
        i%=len(colours)
        
plt.legend()
plt.show()