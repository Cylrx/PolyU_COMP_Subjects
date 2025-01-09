ans = []

for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))
    a.sort()
    for i in range(1, (n>>1)+1):
        ans.append(f"{a[i]} {a[0]}")
    
print("\n".join(ans))