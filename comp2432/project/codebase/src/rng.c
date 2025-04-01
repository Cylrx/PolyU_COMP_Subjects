#include "rng.h"

static unsigned long long s[2];
static const unsigned long long s1_base = 0xf2e6bcd65ef0803c;
static const unsigned long long s2_base = 0xc51b10664c184979;

void seed(unsigned long long s1, unsigned long long s2) {
    s[0] = s1 ? s1 ^ s1_base : s1_base;
    s[1] = s2 ? s2 ^ s2_base : s2_base;
}

static unsigned long long next() {
    unsigned long long a = s[0], b = s[1];
    s[0] = b;
    a ^= a << 23;
    a ^= a >> 17;
    a ^= b ^ (b >> 26);
    s[1] = a;
    return s[0] + a;
}

double randd() {
    return ((double)(next() >> 11)) * 0x1.0p-53; 
    // extract top 53 bits, then multiply by 2^-53 -> [0, 1)
    // 0x1.0p-53 = hex representation of 2^-53
}

int randi(int l, int r) {
    unsigned long long next_val = next();
    int next_int = (int)(next_val >> 32 ^ next_val);
    return next_int % (r - l) + l;
}
