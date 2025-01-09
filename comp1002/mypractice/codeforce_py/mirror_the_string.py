t = int(input())
ans = []

for _ in range(t):
    n = int(input())
    s = input()
    i = 1
    if n>=2 and s[0]==s[1]:
        ans.append(s[:2])
        continue

    while i<n:
        if ord(s[i-1]) < ord(s[i]):
            ans.append(s[:i]+s[i-1::-1])
            break
        i+=1

    if i>=n: 
        ans.append(s[:]+s[::-1])
    
print("\n".join(ans))