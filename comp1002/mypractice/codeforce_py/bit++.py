n = int(input())
ans = 0
for _ in range(n):
    s = set(input())
    if "-" in s: ans -= 1
    else: ans += 1
print(ans, end="")