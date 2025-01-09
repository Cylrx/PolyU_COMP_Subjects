def isPrime(x):
    if x==1: return False
    for i in range(2, x):
        if x % i == 0:
            return False
    return True
    
def gcd(a, b):
    if not b: return a
    return gcd(b, a%b)

def lcm(a, b):
    a, b = max(a, b), min(a, b)
    g = gcd(a, b)
    if b//g==1:
        return a*a//g
    else:
        return a*b//g
ans = []
for _ in range(int(input())):
    a, b = map(int, input().split())
    ans.append(str(lcm(a, b)))

print("\n".join(ans))