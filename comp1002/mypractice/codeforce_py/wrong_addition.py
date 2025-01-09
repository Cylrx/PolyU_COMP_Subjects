def solve(a, s):
    b = ""
    
    for digit in a[::-1]:
        if not len(s):
            return "-1"
        if int(digit) <= int(s[-1]): 
            b += str(int(s[-1]) - int(digit))
            s = s[:-1]
        else:
            if len(s)<2: return "-1"
            tmp = int(s[-2:]) - int(digit)
            if tmp > 9 or tmp < 0: return "-1"
            b += str(tmp)
            s = s[:-2]

    if s:
        b+=s[::-1]

    return b[::-1]



t = int(input())
ans = []

for _ in range(t):
    a, s = input().split()
    b = solve(a, s)
    ans.append(str(int(b)))

print("\n".join(ans))
