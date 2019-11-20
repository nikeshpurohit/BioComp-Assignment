i = 1
list = []
N = 70
cs = 10

for i in range(0,N):
    list.append(i)
    i += 1

print(list)

def chunkIt(L, num):
    avg = len(L) / float(num)
    out = []
    last = 0.0

    while last < len(L):
        out.append(L[int(last):int(last + avg)])
        last += avg

    return out

list = chunkIt(list, cs)

print(list)