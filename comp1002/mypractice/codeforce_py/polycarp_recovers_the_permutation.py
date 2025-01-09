from collections import deque

ans = []

def solve(x, a):
    q = deque([x])
    i, j = 0, len(a)-1
    while i<j:
        while i<j and a[i] < a[j]: 
            q.appendleft(a[i])
            i+=1
        while i<j and a[j] < a[i]: 
            q.append(a[j])
            j-=1



for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))
    
    res = solve(a[0], a[1:])
    if res[0] == -1:
        res = solve(a[-1], a[:-1])
    
    res = " ".join([str(x) for x in res])
    ans.append(res)

print("\n".join(ans))