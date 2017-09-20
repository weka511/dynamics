import rki,matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def roessler(y,a=0.2,b=0.2,c=5.7):
    dx=-y[1] - y[2]
    dy = y[0] + a*y[1]
    dz = b + y[2]*(y[0]-c)
    return [dx,dy,dz]

if __name__=='__main__':
    a=0.2
    b=0.2
    c=5.7    
    rk=rki.ImplicitRungeKutta2(lambda y: roessler(y,a,b,c),10,0.000000001)
    driver = rki.Driver(rk,0.000000001,0.5,1.0,0.000000001)

    try:
        nn=2000
        plt.suptitle(r'R\"ossler Equation: {0} iterations'.format(nn))
        plt.title(r'$\dot x=-y-z,\dot y=x+ay,\dot z = b + z(x-c),a={0},b={1},c={2}$'.format(a,b,c))
        plt.xlabel('x')
        plt.ylabel('y')
        for y in [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]:
            label='({0},{1})'.format(y[0],y[1])
            xs=[]
            ys=[]
            for i in range(nn):
                y= driver.step(y)
                xs.append(y[0])
                ys.append(y[1])
            plt.plot(xs,ys,label=label)

        plt.legend()
        plt.savefig('roessler.png')
        plt.show()
    except rki.ImplicitRungeKutta.Failed as e:
        print ("caught!",e)