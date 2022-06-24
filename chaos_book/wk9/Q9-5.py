N = [0,1,1,1]
for n in range(4,8):
    N.append(2*n+1)
N.append(25)

for n in range(9,26):
    N.append(N[n-1] + 2*N[n-4] - N[n-8])
print (N[1:])
print (len(N),N[-1])
