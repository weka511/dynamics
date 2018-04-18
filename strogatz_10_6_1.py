import matplotlib.pyplot as plt

if __name__=='__main__':
    R0      = 2.9
    R1      = 4
    N       = 200
    M       = 200
    epsilon = 0.000001
    x0      = 0.5
    step=(R1-R0)/N
    for i in range(N):
        r = R0+i*step
        x = 0.5
        rs=[]
        xs=[]        
        for j in range(M):
            x=r*x*(1-x)
            rs.append(r)
            xs.append(x)
            if abs(x-x0)<epsilon:
                break
        plt.scatter(rs,xs,s=1,edgecolors='b',c='b')
     
    plt.show()       
    