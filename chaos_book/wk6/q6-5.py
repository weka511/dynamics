def cycle(s,m=3,N=12):
    return [s[i%len(s)] for i in range(m,m+N)]

def create_w(s):
    w = s[0:1]
    for i in range(1,len(s)):
        w.append(w[-1] if s[i]==0 else 1-w[-1])
    return w

def get_gamma(w,N=10):
    return sum([w[i%len(w)]/2**(i+1)  for i in range(N)])

def gamma_max(orbit):
    max_value = 0
    for m in range(len(orbit)):
        s     = cycle(orbit,m)
        w     = create_w(s)
        max_value = max(max_value,get_gamma(w))
    return max_value


if __name__=='__main__':
    for orbit in [[1],
                  [0,1],
                  [0,0,1],
                  [0,1,0,1,1],
                  [1,0,1,1,1,0]]:
        print (orbit,gamma_max(orbit))

