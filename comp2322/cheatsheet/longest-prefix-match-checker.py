#!/usr/bin/env python3
import sys
sys.setrecursionlimit(10**7)

def read_bin_str():
    # read a line, strip spaces, then remove all internal spaces
    s = sys.stdin.readline()
    if not s:
        return None
    s = s.strip().replace(" ", "")
    return s

#
# 1)  Parse input
#

line = read_bin_str()
if line is None:
    sys.exit(0)
N = int(line)

# read N-1 ranges
ranges = []
for _ in range(N-1):
    s1 = read_bin_str()   # 32‐bit binary with spaces every 8 bits
    s2 = read_bin_str()
    p  = int(read_bin_str())
    a = int(s1, 2)
    b = int(s2, 2)
    if b < a:
        a, b = b, a
    ranges.append((a, b, p))

# default (Nth) port is labeled N-1
default_port = N - 1

# read M prefixes
M = int(read_bin_str())
prefixes = []
for _ in range(M):
    sb = read_bin_str()     # e.g. "10101010 00000000 000" → length ≤ 32
    port = int(read_bin_str())
    if sb == "":
        plen = 0
        ival = 0
    else:
        plen = len(sb)
        ival = int(sb, 2)
    prefixes.append((ival, plen, port))


#
# 2) Build the “ground‐truth” intervals over [0..2^32−1]
#
#    We will sort the N−1 explicit ranges by start, fill in the gaps
#    with the default_port, and produce a list orig_intervals[] of
#    disjoint, sorted, cover‐all intervals (start,end,port).
#
ranges.sort(key=lambda x: x[0])
orig_intervals = []
prev_end = -1
FULL_END = (1<<32) - 1

for (a, b, p) in ranges:
    if a > prev_end + 1:
        # gap → implicit default
        orig_intervals.append((prev_end+1, a-1, default_port))
    orig_intervals.append((a, b, p))
    prev_end = b

if prev_end < FULL_END:
    orig_intervals.append((prev_end+1, FULL_END, default_port))


#
# 3) Build a binary‐trie of the M prefixes
#
class Node:
    __slots__ = ("port","kids")
    def __init__(self):
        self.port = None
        self.kids = {}   # 0 → child, 1 → child

root = Node()
for (ival, plen, port) in prefixes:
    node = root
    for i in range(plen):
        bit = (ival >> (plen - 1 - i)) & 1
        if bit not in node.kids:
            node.kids[bit] = Node()
        node = node.kids[bit]
    node.port = port    # set/override

#
# 4) From that trie, do a full DFS to carve the entire [0..2^32−1] space
#    into disjoint intervals of “longest‐prefix” → port.
#
prefix_intervals = []

def dfs(node, depth, prefix_val, inherited_port):
    # inherited_port = the port of the most‐recent prefix‐node on the path
    cur_port = node.port if node.port is not None else inherited_port

    if depth == 32:
        # single 32-bit address
        prefix_intervals.append((prefix_val, prefix_val, cur_port))
        return

    if node.kids:
        # for each branch 0 and 1
        for bit in (0,1):
            if bit in node.kids:
                dfs(node.kids[bit],
                    depth+1,
                    (prefix_val<<1) | bit,
                    cur_port)
            else:
                # whole missing‐branch subtree gets cur_port
                new_pref = (prefix_val<<1) | bit
                m = 32 - (depth+1)
                st = new_pref << m
                ed = ((new_pref+1)<<m) - 1
                prefix_intervals.append((st, ed, cur_port))
    else:
        # no children: entire subtree under this node
        m = 32 - depth
        st = prefix_val << m
        ed = ((prefix_val+1)<<m) - 1
        prefix_intervals.append((st, ed, cur_port))

# start DFS from the root, with default_port as the inherited
dfs(root, 0, 0, default_port)

# sort by start
prefix_intervals.sort(key=lambda x: x[0])


#
# 5) Scan the prefix_intervals and for each interval check
#    that its port matches the ground‐truth port at that address.
#    We only need to check the mapping at the start of each interval,
#    because by construction both sets of intervals are a partition
#    of [0..2^32−1].
#
j = 0   # pointer into orig_intervals
L = len(orig_intervals)

for (st, ed, pfx_port) in prefix_intervals:
    # advance j so that orig_intervals[j] covers 'st'
    while j < L and orig_intervals[j][1] < st:
        j += 1
    if j>=L:
        # shouldn't happen
        break
    oa, ob, gport = orig_intervals[j]
    if not (oa <= st <= ob):
        # gap in orig_intervals? also shouldn't happen
        break
    if pfx_port != gport:
        # found mismatch: address 'st' is our counter-example
        ip32 = f"{st:032b}"
        ipstr = " ".join(ip32[i:i+8] for i in range(0,32,8))
        print("False")
        print(ipstr)
        print(f"expected port {gport}, but prefix table gives port {pfx_port}")
        sys.exit(0)

# if we reached here → everything matched
print("True")
