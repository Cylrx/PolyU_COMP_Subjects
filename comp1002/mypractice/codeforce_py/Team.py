n = int(input())
ans = 0
for _ in range(n):
    a, b, c = map(int, input().split())
    if a&b or b&c or a&c: ans+=1
print(ans, end="")