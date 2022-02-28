def cycle(s,
          m = 3,
          N = 24):
    return [s[i%len(s)] for i in range(m,m+N)]

def create_w(s):
    w = s[0:1]
    for i in range(1,len(s)):
        w.append(w[-1] if s[i]==0 else 1-w[-1])
    assert len(w)==len(s)
    return w

def get_gamma(w,
              N = 24):
    return sum([w[i%len(w)]/2**(i+1)  for i in range(N)])

def gamma_max(orbit):
    return max([get_gamma(create_w(cycle(orbit, m=m))) for m in range(len(orbit))])


if __name__=='__main__':
    # s = cycle([1,0],m=0)
    # print (s)
    # w = create_w(s)
    # print (w,get_gamma(w))
    for orbit in [
            [1],
            [0,1],
            [0,0,1],
            [0,1,0,1,1],
            [1,0,1,1,1,0]
        ]:
        print (orbit,gamma_max(orbit))

