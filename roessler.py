import rki,matplotlib.pyplot as plt

def roessler(y,a=0.2,b=0.2,c=5.7):
    dx=-y[1] - y[2]
    dy = y[0] + a*y[1]
    dz = b + y[2]*(y[0]-c)
    return [dx,dy,dz]

if __name__=='__main__':
    rk=rki.ImplicitRungeKutta2(roessler,10,0.000000001)
    driver = rki.Driver(rk,0.000000001,0.5,1.0,0.000000001)
    try:
        nn=1000
        plt.title("Roeesler Equation: {0} iterations".format(nn))
        plt.xlabel('x')
        plt.ylabel('y')
        for y in [[0,1,0]]:
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