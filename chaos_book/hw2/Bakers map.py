def f(pt):
    x=pt[0]
    y=pt[1]
    if y<=2:
        return (x/3,2*y)
    else:
        return (x/3+1/2,2*y-1)

def area(u):
    return 0.5*(  u[0][0]*u[1][1] + u[1][0]*u[2][1] + u[2][0]*u[3][1] + u[3][0]*u[0][1]
                - u[1][0]*u[0][1] - u[2][0]*u[1][1] - u[3][0]*u[2][1] - u[0][0]*u[3][1])
    
u = [(0,0), (1,0), (1,1), (0,1)]

v = [f(pt) for pt in u]

print (area(v)/area(u))