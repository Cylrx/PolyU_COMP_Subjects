ans = []

for _ in range(int(input())):
    n = int(input())
    
    arr = list(map(int, input().split()))
    arr = [[arr[i], i+1] for i in range(n)]
    arr.sort(key = lambda x: x[0], reverse=True)

    res1 = 0
    res2 = [0]*(n+1)
    cur = 1
    for x in arr:
        res2[x[1]] = cur
        res1+=abs(cur)*x[0]*2
        cur = -cur
        if cur > 0:
            cur += 1
        
    
    ans.append(str(res1))
    ans.append(" ".join([str(x) for x in res2]))

print("\n".join(ans))
        
    