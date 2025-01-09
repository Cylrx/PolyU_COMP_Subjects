n, k = map(int, input().split())
arr = list(map(int, input().split()))
ans = 0
for x in arr:
    if x >= arr[k-1] and x>0:
        ans += 1
print(ans, end="")