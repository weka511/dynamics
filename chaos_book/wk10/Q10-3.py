from math              import sqrt
from matplotlib.pyplot import figure, legend, plot, show
from numpy             import  count_nonzero,  vectorize
from numpy.random      import rand

alpha = (3 - sqrt(5))/2
Lambda = (sqrt(5)+1)/2
M      = 1000
N      = 10000
K      = 5
def pw_map(x):
    return alpha + Lambda*x if x<alpha else 1 - Lambda*(x-alpha)

def get_census(x):
    n0   = count_nonzero(x<alpha)
    n1   = count_nonzero(x>=alpha)
    rho0 = n0/alpha
    rho1 = n1/(1-alpha)
    return rho0,rho1

if __name__ == '__main__':
    figure(figsize=(12,12))
    for k in range(K):
        x  = rand(M)
        mu = []
        for i in range(N):
            x          = vectorize(pw_map)(x)
            rho0, rho1 = get_census(x)
            mu.append(rho0/M)
        plot(mu,label=f'{k+1}')
    legend()
    show()
