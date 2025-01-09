def isKill(a, h, k):
    for i in range(len(a)-1):
        h -= min(k, a[i+1]-a[i])
    h-=k
    return h<=0
    
def bSearch(a, n, h):
    r = 10**18
    l = 1

    while l<r:
        mid = (r+l)>>1
        if isKill(a, h, mid):
            r = mid
        else:
            l = mid+1
    
    return l

ans = []

for _ in range(int(input())):

    n, h = map(int, input().split())
    a = list(map(int, input().split()))

    ans.append(str(bSearch(a, n, h)))

print("\n".join(ans))
