import rki,matplotlib.pyplot as plt

def duffing(y):
    dx=y[1]
    dy = -0.15*y[1] + y[0] - y[0]**3
    return [dx,dy]

if __name__=='__main__':
    rk=rki.ImplicitRungeKutta2(duffing,10,0.000000001)
    driver = rki.Driver(rk,0.000000001,0.5,1.0,0.000000001)
    try:
        nn=1000
        y=[0,1]
        plt.title("Duffing's Equation ({0},{1})".format(y[0],y[1]))
        xs=[]
        ys=[]
        for i in range(nn):
            y= driver.step(y)
            xs.append(y[0])
            ys.append(y[1])
        plt.plot(xs,ys)
        plt.xlabel('x')
        plt.ylabel('y')

        plt.show()
    except rki.ImplicitRungeKutta.Failed as e:
        print ("caught!",e)