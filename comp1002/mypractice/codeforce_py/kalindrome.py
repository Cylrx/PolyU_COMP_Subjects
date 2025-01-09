ans = []

for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))
    
    i = 0
    j = n-1

    while i<j and a[i]==a[j]:
        i+=1
