ans = []
for _ in range(int(input())):
    n = int(input()) + 2
    s = input() + "ba"
    ss = "".join(list(s))
    res = ""
    s = s.replace("e", "a")
    s = s.replace("c", "b")
    s = s.replace("d", "b")
    i = 2
    while i<n-1:
        if s[i]==s[i+1]:
            res+=ss[i-2:i+1]
            i+=3
        else:
            res+=ss[i-2:i]
            i+=2
        res+='.'
    ans.append(res[:-1]) 
print("\n".join(ans))

    
