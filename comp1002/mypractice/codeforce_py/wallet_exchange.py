ans = []
for _ in range(int(input())):
    a, b = map(int, input().split())
    if (a+b) % 2 == 0:
        ans.append("Bob")
    else:
        ans.append("Alice")

print("\n".join(ans))

