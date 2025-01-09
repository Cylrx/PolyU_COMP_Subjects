import functools

ans = []
for _ in range(int(input())):
    n, k = map(int, input().split())
    b = list(map(int, input().split()))

    prod = functools.reduce(lambda x, y: x*y, b)

    if 2023%prod == 0:
        ans.append("YES")
        ans.append("1 "*(k-1)+str(2023//prod))
    else:
        ans.append("NO")
    
print("\n".join(ans))

