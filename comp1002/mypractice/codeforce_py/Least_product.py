ans = []
for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))
    n_cnt = 0
    z_cnt = 0 
    for x in a:
        if x<0:
            n_cnt += 1
        if x==0:
            z_cnt += 1
    
    if n_cnt%2==0 and z_cnt==0:
        ans.append("0\n1 0")
    else:
        ans.append("0")

print("\n".join(ans))