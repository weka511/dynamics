from math              import sqrt
from matplotlib.pyplot import figure, legend, plot, savefig, scatter, show, title
from numpy             import  count_nonzero,  vectorize
from numpy.random      import rand

alpha  = (3 - sqrt(5))/2
Lambda = (sqrt(5)+1)/2
M      = 1000
N      = 500000
n      = 250000
K      = 5

def pw_map(x):
    return alpha + Lambda*x if x<alpha else 1 - Lambda*(x-alpha)

def get_densities(x,
                  L0 = alpha*M,
                  L1 = (1-alpha)*M):
    n0   = count_nonzero(x<alpha)
    n1   = count_nonzero(x>=alpha)
    rho0 = n0/L0
    rho1 = n1/L1
    return rho0,rho1

if __name__ == '__main__':
    pw_map_vectorized = vectorize(pw_map)

    fig = figure(figsize=(20,12))
    fig.add_subplot(121)
    x  = rand(M)
    y  = pw_map_vectorized(x)
    scatter(x,y,
            c = 'b',
            s = 1)
    title('The Map')
    fig.add_subplot(122)

    for k in range(K):
        x  = rand(M)
        mu = []
        for i in range(N):
            x          = pw_map_vectorized(x)
            rho0, rho1 = get_densities(x)
            if i>n:
                mu.append(rho0)
        plot(mu,
             label      = f'{k+1}',
             linewidth  = 1,
             markersize = 1)

    legend()
    title(f'M={M}, N= {N}, n= {n}')
    savefig('Q10-3')
    show()
