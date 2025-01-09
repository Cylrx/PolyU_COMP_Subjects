def gcd(a, b):
    if not b: return a
    return gcd(b, a%b)

def search(arr):
    n = len(arr)
    if n==1: return arr[0]
    if n==2: return gcd(arr[0], arr[1])
    return gcd(search(arr[:n>>1]), search(arr[n>>1:]))

def check(arr, d):
    for x in arr:
        if x % d == 0:
            return False
    return True

t = int(input())
ans = []

for _ in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    odd = [arr[i] for i in range(n) if i%2==1]
    even = [arr[i] for i in range(n) if i%2==0]

    odd_gcd = search(odd)
    even_gcd = search(even)

    if check(even, odd_gcd):
        ans.append(str(odd_gcd))
    elif check(odd, even_gcd): 
        ans.append(str(even_gcd))
    else:
        ans.append("0")

print("\n".join(ans)) 