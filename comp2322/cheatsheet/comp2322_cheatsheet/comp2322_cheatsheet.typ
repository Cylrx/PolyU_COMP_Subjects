#import "@preview/lovelace:0.3.0": *
#set page(
  paper: "us-letter",
  margin: 0.085in
)
#set par(justify: true)
#set text(
  size: 6.5pt,
)

#let sps = {v(2pt)}
#let dd(y, x) = {$(diff #y) / (diff #x)$}
#let ddx(y) = {$(diff #y)/(diff x)$}
#let big(x) = {$upright(bold(#x))$}
#let code(c) = {highlight(raw(c), fill: silver)}
#let head(h) = {text(fill: maroon, weight: "black", style: "oblique")[#h]}
#let subhead(h) = {text(fill: blue, weight: "black", style: "oblique")[#h]}
#let def(x, y, g: 1em) = {grid(columns: 2, gutter: g, [#set text(style: "italic"); #x], [#y])}

#columns(3, gutter: 6pt)[
== #head("General Knowledge")
- *Structure*: End points $<->$ Comm Links $<->$ Switch $<->$ End points
  #def(
    [
        End-points #subhead([(Network Edge)])\
        Comm Links #subhead([(Access Net / Phys Media)])\
        Packet Switchs #subhead()[(Network Core)]\
    ],
    [

      PC, Servers, Phones, ...\
      Copper, Fiber, WiFi, 5G, ...\
      Router, Switches, ...\
    ], 
    g: 0.5em
  )

- *High-level Structure*: 
  #def(
    [
      Network of Networks\ 
      Network Protocols\ 
      Internet Standards
    ],
    [
      Interconnected ISP\
      Rules that govern how data exchange\
      Ensure diff. HW & SW work together
    ]
  )

=== #subhead([Access Network])

#def(
  [
    Digital Subscriber\ Line (DSL)\
    Cable Networks\
    Ethernet\ 
    Wireless Access\ Network
  ],
  [
    Use *existing* telephone lines. Direct link to telecom company's central office (*dedicated access*)\
    Fiber coax cable: homes $<->$ ISP (*shared access*)\ 
    Fiber optic cables. Used in institutions / company\
    #big([Wireless LAN]): short range. E.g., WiFi\
    #big([Widearea Wireless Access]) E.g., Cellular\
  ]
)

=== #subhead([Physical Media])
#def(
  [
    Coaxial Cable #v(1.4em)
    Fiber Optical Cable #v(1.1em)
    Radio Signal\
  ],
  [
    (*Guided Media*): 2 concentric copper conductors
    - Bidirectional, broadband (multiple channel)
    (*Guided Media*): Glass fiber, light pulses. High Speed
    - Low err. rate. Immune electromagnetic noise
    (*Unguided Media*): electromagnetic signals\
    - Stability: reflection / obstruction / interference
    - Wireless, Bidirectional
  ]
)

=== #subhead([Network Core])
#big([Packet Switching]): 
+ Host break applicaiton-layer message into _packets_. 
+ Host send packets to edge router
+ Router forward one to the next *(wait for entire packet arrive)*

#big([Key Functions])
- Routing: determine source-destination route
- Forwarding: move packet to next router (input $->$ output)

#big([Switching Modes])\
Frequency Division Multiplexing (FDM) & Time Division ... (TDM)
#grid(columns: 2, gutter: -0.5em, 
  [#image("fdm_tdm.png", width: 100%)], 
  [#image("circ_switching.png") #image("packet_switching.png")]
)

== #head([Transmission Loss & Delay])

#big("Delay at 1 Node: "): $d_"nodal" = d_"proc" + d_"queue" + d_"trans" + d_"prop"$

*$d_"proc"$ processing delay*: check bit error, determine output link\
*$d_"queue"$ queueing delay*: time waiting at output link (congesetion)\
*$d_"trans"$ transmission delay*: $d_"trans" = sum_i "packet size (bits)" / ("trasmission rate at hop "i "(bits/sec)")$\
*$d_"prop"$ propagation delay*: $d_"prop" =sum_i ("length of link "i) / ("propagation speed in medium")$
#v(.5em)

*More on Queueing Delay*:\
$L$: packet length (bits); $R$: link bandwidth (bps); $a$: average arrival rate\


#grid(columns: 2, gutter: 1em,
[
#def(
  [
    *Avg service time*\
    *Network Intensity*\
    *Avg queuing delay*
  ],
  [
    $T = L/R$\
    $ rho = (L a) / R = T a$\
    $d_"queue" = (T rho) / (1 - rho) = (L rho) / (R (1- rho))$
  ]
)
], 
[
Notice, \
$rho -> 0, d_"queue" -> 0$, \
$rho -> 1, d_"queue" -> oo$\
]
)

#v(.5em)
*Packet Loss*\
When $L a$ > $R$: Packet queue at router. Packet lost if memory full.

*Throughput*\
Rate (*bits/time unit*) which bits transferred between sender/receiver\ 
- instantaneous: rate at given point in time
- average: rate over longer period of time

Theoretical Upperbound: #highlight([*max-flow min-cut*]) of the network

== #head([Protocol Layers & Service Model])

=== #subhead([TCP/IP])
#def(
  [
    #big([Application])\
    #big([Transport])\
    #big([Network])\
    #big([Link])\
    #big([Physical])
  ],
  [
    Application-specific functionality (*Data*) FTP, HTTP\
    Process $<->$ process w/ports (*Segments*) TCP, UDP\ 
    Host $<->$ host w/IP (*Packets*) IP, routing protocols.\
    Node $<->$ node w/MAC (*Frame*) Ethernet, WiFi, PPP\
    *Binary Bits* "in" physical medium
  ]
)
=== #subhead([ISO/OSI Reference Model])
/ *Seven Layers*:  
  Application $->$ Presentation $->$ Session\ $->$ Transport $->$ Network $->$ Data Link $->$ Physical
  
  *Two more than TCP/IP*: 
  - _Presentation_: data format, encryption, compression
  - _Session_: syncrhonization, checkpointing, recovery of data

== #head([Application Layer])\
- Non-persistent HTTP: 
  - Each object request $->$ new TCP connection
  - *2 RTT* to fetch every object
- Persistent HTTP
  - Server leaves TCP connection open after init
  - Object sent over same TCP connection
  - Minimum of *1 RTT* to fetch all objects

=== #subhead([HTTP Request])
#def(
  [
    #image("http-req.png", width: 100%)
  ],
  [
    #image("http-req-example.png", width: 100%)
  ],
  g: -3pt
)
- cr: carriage return; lf: line feed
- Method types: 
  - *`HTTP/1.0`*: GET, POST, HEAD
  - *`HTTP/1.1`*: GET, POST, HEAD, PUT (uploads file in entity body to path psecified in URL field) DELETE (deletes file specified in URL field)
    - `POST` method: data in request message body. High security.
    - `URL` (or `GET`) method: data in URL field. Low security.

=== #subhead([HTTP Response])
#def(
  [
    #image("http-res.png", width: 100%)
  ],
  [
    #image("http-res-example.png", width: 100%)
  ],
  g: -3pt
)
- Status code: 
  - 200: OK, 301: Moved Permanently, 400: Bad Request, 404: Not Found, 505: HTTP Version Not Supported

=== #subhead([Caching])
- Total delay = Internet delay + access delay + LAN delay
- HTTP request: `If-modified-since: <date>`
- HTTP response: `304 Not Modified` or `200 OK`

=== #subhead([Email])
- Send: SMTP (Simple Mail Transfer Protocol)
- Recv: POP3 (Post Office Prot) / IMAP (Internet Mail Access Prot)
- User Agent $->$ Mail Server A $->$ Mail Server B $->$ User Agent
- `CRLF.CRLF` to end message

*POP3* (Post Officee Protocol):  Deletes message, stateless
- Authorization phase: 
  - Client command to server: `user <username>`, `pass <password>`
  - Server response: `+OK`, `-ERR`
- Transaction phase: 
  - `list`: list message nuumbers
  - `retr <msg_number>`: retrieve message by number
  - `dele <msg_number>`: delete message by number
  - `quit`: end session
*IMAP* (Internet Mail Access Protocol): Keep state
- Keep all messages on server
- Allows user to organize messages in folders

=== #subhead([DNS (Domain Name System)])
- *Root Name Servers*: 13 root servers. return IP of TLD server
- *TLD Name Servers*: return IP of authoritative name server
  - Responsible for com, org, net, edu, ... uk, fr, ca, ...
- *Authoritative Name Servers*: return IP of host
  - Usually organization's own DNS server (e.g., Cloudflare)
- *Local DNS*: each ISP has one. not part of hierarchy.
  - If have cache, return (might outdate). Else, forward into hierarchy.

_Resource Records (RR)_: format `(name, value, type, ttl)`
- *A*: hostname $->$ IP. `name` = hostname, `value` = IP
- *NS*: name $->$ auth DNS. `name` = domain, `value` = auth DNS hostname
- *MX*: mail server for domain. `name` = domain, `value` = server name
- *CNAME*: canonical name. `name` = alias, `value` = real name

== #head([Transport Layer])
- *Multiplexing*: assign app data to different ports, add these port headers to outgoing packets. Let multiple apps to use same network.
- *Demultiplexing*: based on incoming packet port numbers, direct data to correct app.

=== #subhead([Reliable Data Transfer (RDT)])
- *RDT 1.0*: do nothing, assume no error
- *RDT 2.0*: solve packet corruption.
  - Adds checksum to detect error. 
  - Receiver send ACK/NAK to acknowledge sender. 
- *RDT 2.1*: solve ACK/NAK packet corruption
  - Important: assumes stop-and-wait protocol (1 packet at a time)
  - Sender Rule: 
    - Send packet 0, wait for ACK
    - If `NAK` or `ACK/NAK` corrupt, resend packet 0
    - If `ACK`, send packet 1, wait for ACK
  - Receiver Rule:
    - If packet corrupt, send NAK
    - If packet 0 received, when prev is 1, send ACK, flip 0/1
    - If packet 1 received, when prev is 0, send ACK, flip 0/1
    - If packet 0 received two times, delete duplicate, send ACK again
- *RDT 2.2*: NAK-free protocol - Send `ACK 0/1` instead of `NAK`
  - Receiver ACK just the last correct packet if new packet corrupt
  - Sender retransmit packet if duplicate ACK received
- *RDT 3.0*: Add timeout to sender (Solves ACK packet loss)

*Performance of RDT 3.0*: 
- Transmission Time = $D_"trans" = "Length" / "Rate" + "RTT"$
- Utilization (sender) = $U_"sender" = (L slash R) / ("RTT" + L slash R)$

*Pipelining*: 
Utilization (sender): $U_"sender" = (N times L slash R) / ("RTT" + L slash R)$ 
  - where $N$ is the number of packet transmitted at once (parallization)

*Go-Back-N (GBN)*:\
- Key Idea: ensure #underline[in-order delivery].\
  - Receiver discard out-of-order pkt, only ACK packet $i-1$
  - *Cumulative ACK* sender get any ACK $i$, mark all pkts $<= i$ ACK
- Sender: 
  - Send $N$ (window size) packets, wait for ACK
  - Receive any valid ACK $i$, move window to $[i + 1, i + N]$
  - If 1st pkt in window timeout, resend entire window
- Receiver: 
  - Only accept next in-order pkt (window size = 1, conceptually)
  - Only accept pkt $i$ *if and only if* $i$ is the next \#seq
  - If got out-of-order pkt $j > i$, discard and send ACK $i-1$
  - If got in-order pkt $i$, send ACK $i$
Prevent Corruption: $N <= K - 1$, where $K$ is the \#seq range

*Selective Repeat (SR)*:\
Prevent Corruption: $N <= floor(K slash 2)$, where $K$ is the \#seq range
- Sender: 
  - Send $N$ (window size) packets, wait for ACK
  - Track timeout of *every* single packet. Resend whichever timeouts
- Receiver: 
  - ACK (not cumulative) every single valid pkt (even out-of-order)
  - Buffer out-of-order packets within its window (window size = $N$)
  - Send buffered pkt to application layer once prior pkts arrive.

=== #subhead([TCP (Transmission Control Protocol)]) 
#image("tcp-segment.png", width: 70%)
- *Sequence Num*: 
  - Byte number of 1st byte in the current segment
  - #underline[NOT the same] as \#seq in GBN and SR
  - Example: `SEQ=1000` data length=500 bytes (Sent bytes 1000\~1499)
- *Acknowledgement Num*: 
  - Sequence number of next byte expected from the other side
  - Example (Con't): `RECV ACK=1500` (i.e., next byte expected is 1500)
- 3-way-handshake w/ `SYN` and `ACK` flags
  - $A -> B$: `SYN, seq = x` A initiate connection
  - $B -> A$: `SYN-ACK, seq = y, ack = x+1` B acknowledge A's SYN
  - $A -> B$: `ACK, ack = y+1`. A acknowledge B's SYN
  - $A$ and $B$ maintain randomly initialized unique seq \#

*TCP Timeout Control*\
$ "EstimateRTT"_n = (1-alpha) dot "EstimateRTT"_(n-1) + alpha dot "SampleRTT"_n\
"DevRTT"_n = (1-beta) dot "DevRTT"_(n-1) + beta dot |"SampleRTT"_n - "EstimateRTT"_n| \
"TimeoutInterval"_n = "EstimateRTT"_n + 4 dot "DevRTT"_n
$

- where $alpha = 0.125, beta = 0.25$ typically, EMA
- Timeout = avg RTT + 4 \* deviation (for safety margin)
  - too short: premature timenout, unnecessary retransmission
  - too long: slow reaction to segment loss

*TCP Receiver Actions*:\ 
- Recv in-order seg, previous received all already ACK'ed
  - Action: Wait 500ms for new pkt, if no, send ACK of current pkt.
- Recv in-order seg, previous received pkt not yet ACK'ed
  - Action: immediately send *single* accumulative ACK (all previous)
- Recv out-of-order seg $j > i$ (Gap Detected)
  - Action: immediately send *duplicate* ACK $i-1$, with seq\# = $i$
- Recv seg that fills *lower-end* gap caused in the above case.
  - Action: immediately send ACK.
- If sender recv 3 duplicate ACK of same data, resend un-ACK'ed segment with smallest seq \# immediately.
- Receiver "advertise" free buffer space via *`rwnd`* value, 
  - sender limit in-flight data in response to this value

*TCP Congestion Control*\
- Additive Increase, Multiplicative Decrease (AIMD): 
  - Increase `cwnd` by 1 MSS (Maximum Segment Size) *every RTT*. 
  - Cut `cwnd` by half when loss detected.
  - $"rate" approx "cwnd" div "RTT" "(bytes/sec)"$
- _TCP Reno_: 
  - timeout loss: `cwnd = 1 MSS`, exp grow to `ssthresh`, then lin grow
  - 3 duplicate ACK loss: `ssthresh` and `cwnd` $div 2$, then linearly grow
- _TCP Tahoe_: always set `cwmd` to 1 MSS when loss detected
- *avg TCP throughput*: $3/4 dot "Window Size"/"RTT"$ bytes/sec

== #head([Network Layer])
- _*forwarding (data plane)*_: move packets from a router's input to output
- _*routing (control plane)*_: determine route from source to destination
  - _Traditional routing algorithms_: implemented in routers (local decision)
  - _Software-defined networking (SDN)_: implemented in (remote) servers

#place(right, image("input-port.png", width: 50%), dx: 0pt, dy: -5pt)
=== #subhead([Router])
#place(right, image("router-architecture.png", width: 35%), dx: 15pt, dy: 5pt)
*Input Port* (3 components): 
  + Line termination (physical layer), bit-level reception
  + link layer protocol (data link layer, e.g., Ethernet)
  + Forwarding: 
    - Use header field values $->$ determine output port
    - If datagram arrive fast, queue at _input port buffer_
*Input Port Queuing*\
- _Head-of-the-line (HOL) blocking_: in same input port, packet infront of line blocking packets behind (e.g., when head packet's output port is busy)
*Switching Fabric* (3 types of methods): 
+ _Memory_: CPU direct control. packet copied to RAM, speed bottlenecked.
+ _Bus_: datagram move thru. shared bus. Limited by bus bandwidth.
+ _Crossbar_: high parallelism, $n^2$ crosspoints. 
#image("switching-fabric-types.png")
#place(right, image("output-port.png", width: 65%), dx: 0pt, dy: -17pt)
*Output Port* (3 components): 
+ Datagram buffer. Required when datagram arrives $>$ transmission rate
+ Link layer protocol (send) 3. Line termination
=== #subhead([Internet Protocol (IP)])
#image("ip-datagram.png")
*IPv4 Address*: 32-bits, 2 levels, 5 classes (A, B, C, D, E)
- A: begin w/ $(0)_2$ 0-127, 8/24bit, $1/2$ addr space, mask: 255.0.0.0
- B: begin w/ $(10)_2$ 128-191, 16/16bit, $1/4$ addr space, mask: 255.255.0.0
- C: begin w/ $(110)_2$ 192-223, 24/8bit, $1/8$ addr space, mask: 255.255.255.0
- D: begin w/ $(1110)_2$ 224-239, multicast, $1/16$ addr space
- E: begin w/ $(1111)_2$ 240-255, reserved (future proof), $1/16$ addr space

*Subnets*: devices / interfaces with same netid (network part of IP address), physically reach each other without router.

*Classless Inter-Domain Routing (CIDR)*: 
- subnet part (netid) is arbitrary length
- format: *`a.b.c.d/x`*, where *`x`* is the \# bits in subnet portion of address

*IP Fragmentation*:\
_Purpose_: 
- Network links have _*MTU*_ (_Max Transmission Unit_)
- Need to fragment datagrams, then reassemble at destination.
_Header_:
- `length` field: total length of datagram (header + data)
- `offset` field: data length $div$ 8
- `fragflag`: 0 for last fragment, 1 for not last

*Destination-based Forwarding*:\
- Longest prefix matching: forward table contains list of prefixes, find longest prefix match

*DHCP* (Dynamic Host Configuration Protocol):\
#underline[_Purpose_: let host dynamically get IP addr. from network server when join]
- Host: boradcast *DHCP discover* msg to find DHCP server
- DHCP server: respond *DHCP offer* msg
- Host: requests IP address via *DHCP request* msg
- DHCP server: send *DHCP ack* msg, which returns: 
  - allocatd IP, first-hop router addr, network mask, name and IP of DNS

*ICANN* (Internet Corporation for Assigned Names and Numbers):\
  - Assign blocks of IP addresses to the RIRs (Regional Internet Registries); RIRs assign to ISPs or large organizations

=== #subhead([Routing Protocols])
*Bellman-Ford Algorithm*:
- Same as Dijkstra's algorithm: $d(x) = min(d(x), d(y) + l(y, x))$
- Except, don't choose min node each iteration. 
- *for* $n-1$ iterations ($n$ = number of nodes), 
  - *for* each $m$ edge $(y, x)$: 
    - $d(x) = min(d(x), d(y) + l(y, x))$ *if* $y -> x$
    - $d(y) = min(d(y), d(x) + l(x, y))$ *if* $x -> y$
  
*Distance Vector Algorithm*:\
- Same as Bellman-Ford, except asynchronous, distributed, iterative
- Maintains a #text(size: 0.8em)[多源最短路矩阵]
- Each node: 
  + *wait* for change in local link cost, or msg from neighbor
  + *re-compute* distance vector estimate using Bellman-Ford
  + if DV to *any* dest changed, notify *all* neighbors
- _Characteristics_: 
  - *Good news travel fast* (adj edge cost $arrow.b$): node cost unchange / improve. Advertise to all neighbors. Update ripple out one hop at a time (fast).


  - *Bad news travel slow* (adj edge cost $arrow.t$): *$A$* drop routes that use the bad edge, seek alternative from neighbor *$B$*. But neighbor *$B$* might route through *$A$*, its shorter only because it is oudated. End up “bouncing” distance updates back and forth (slow, _count-to-infinity problem_).
    - *Solution*: if *$B$* routes through *$A$*, set *$B$*'s distance to $oo$.

*Link-State Routing* (Dijkstra) *vs.* *Distance Vector* (Bellman-Ford)
#table(
  columns: (auto, 1fr, auto),
  inset: (x: 1pt, y: 3pt),
  stroke: (x: none, y: none),
  align: (left, center, center),
  table.hline(),
  [], [LS (Dijkstra)], [DV (Bellman-Ford)],
  table.hline(),
  [Message Complexity], [$O(n m)$], [exchange msg between adj nodes only],
  [Convergence Speed], [$O(n^2)$], [vary. loops / count-to-infinity],
  table.hline(),
)
*AS*: Autonomous System, group of routers under same admin control\
* Intra-AS Routing*:\
- _Interior Gateway Protocols_ (IGPs): 
  - _RIP (Routing Information Protocol)_: 
    - distance metric: #underline[number of hops] (max 15 hops)
    - update interval: DV advertise every 30 seconds
    - routing table managed by application-level process route-d (daemon)
- _Open Shortest Path First_ (OSPF): 
  - LS-based, use Dijkstra's algorithm
  - OSPF message directly over IP (*not* TCP/UDP). Use `protocol` field 89
  - All OSPF messages authenticated; Multiple same-cost paths allowed
  - Each link, multiple cost metric for different services 
  - Supports multicast (MOSPF) and unicast
  - *Hierarchical OSPF* in large domains

*Inter-AS Routing*:\
- Forwarding table configured by both intra- and inter-AS routing algorithm
- _Hot Potato Routing_: 
  - When destination reachable via multiple AS, greedily choose gateway w/ least cost. Don't care about cost of inter-AS routes.
- _BGP_ (Border Gateway Protocol): 
  - Types: 
    - _eBGP_: obtain subnet reachability from neighboring AS
    - _iBGP_: propagate reachability to AS-internal routers
  - BGP Messages:
    - `OPEN`: open semi-persistent TCP connection to remote BGP peers
    - `UPDATE`: advertises new paths (or withdraw old)
    - `KEEPALIVE`: keep connection alive in absence of `UPDATE`, and ack `OPEN`
    - `NOTIFICATION`: report errors in previous msg; also for closing conn.
  - BGP Route Selection: 
    + Shortest AS-path
    + Closest next-hop router
    + local preference value attribute: policy-based selection
    + Aadditional criterias

== #head([Link Layer])
#place(right, image("2d-parity.png", width: 45%), dy: -25pt)
*Parity Checking*:\
- _Single Bit Parity_: *detect* single bit error
- _Two-Dimensional Parity_: *detect* and *correct* single bit error

#place(right, 
  [
    #box($
    (D dot 2^r "xor" R) mod G = 0\
    D dot 2^r "xor" R = n G\
    D dot 2^r = n G "xor" R\
    R = (D dot 2^r) mod G
    $, fill: luma(84%))
  ], dy: -3pt, dx: 5pt
)
*Cyclic Redundancy Check (CRC)*:\
+ Choose divisor $G$ (*must* be $r + 1$ bits long)
+ Left shift data bits $D$ by $r$ bits $=> D dot 2^r$
+ Obtain remainder: $R = (D dot 2^r) mod G$ ($r$ bits)
+ Form final code: concatenate $D$ with $R$ (equivalently, $D dot 2^r "xor" R$)
- #text(fill: red)[*IMPORTANT*: both addition in "$n G$" and subtraction in "$div G$" is *XOR*]
- _*Intuition*_: 
  - Normally: $big(A) * big(B) = (A_0 dot 2^0 dot big(B)) + (A_1 dot 2^1 dot big(B)) + ... + (A_n dot 2^n dot big(B))$
  - But in CRC: $big(A) * big(B) = (A_0 dot 2^0 dot big(B)) "xor" dots "xor" (A_n dot 2^n dot big(B))$
  - Normally in long division, subtract each $(A_i dot 2^i dot big(B))$ from dividend
  - But in CRC, we $"xor"$ each $(A_i dot 2^i dot big(B))$ from dividend
  - Intuitively, both $+$ and $-$ replaced  with *xor*. 
  - Crucially, this works because #underline[*xor* is its own inverse]

*Multiple Access Protocols*:\
- Idealy goals: *1)* When 1 node, send at rate $R$ (channel max rate) *2)* When $M$ nodes, each send at average rate $R slash M$ *3)* Fully decentralized: no central controller / clock for synchronization *4)* Simple
- *MAC (medium access control) protocols*: 
  + _*Channel Partitioning MAC protocols*_: 
    + TDMA (Time Division Multiple Access): 
     - Channel time divided into fixed length slots
     - Each station gets a slot; unused slots go idle
    + FDMA (Frequency Division Multiple Access): 
      - Channel spectrum divided into frequency bands
      - Each station assign a fixed frequency. Unused freq bands go idle
  + _*Random Access MAC protocols*_: 
    + Slotted ALOHA: 
      - Each node: attempt to transmit as soon as possible
      - When no collision: transmit at full channel rate
      - When collision: no node can send (slot wasted)
        - each node retry in subsequent slot with prob. $p$ until success
      - _Assumptions_: all frame same size; time divided into equal-sized slots; nodes are synchronized, send only at time slot boundary;
      - *Pros*: single node full rate; highly decentralized; simple
      - *Cons*: collision waste slots; clock synchronization needed
      - *Low efficiency*: $lim_(N -> oo) N p(1-p)^(N-1) = 1/e approx 37%$, for $N$ nodes each transmit in a slot with probability $p$
    + Pure (unslotted) ALOHA: 
      - Each node send ASAP (no slot, no boundary, thus no sync needed)
      - Collision when $>=2$ nodes overlap. *Worse efficiency*: 
        - $lim_(N -> oo) N p(1-p)^(2(N-1)) = 1/2e approx 18%$
    + CSMA (Carrier Sense Multiple Access):
      - Listen before transmit. Send if idle; defer if busy. 
      - Collision still happen (two nodes think channel is idle)
    + CSMA/CD (Carrier Sense Multiple Access with Collision Detection):
      - Added collision detection. Better performance than ALOHA
  + _*"Taking turns" MAC protocols*_: 
    + Polling: master node polls one by one from "slave" devices
      - Polling overhead, latency, single point of failure
    + Token ring:  token passed around ring. Only token-holder can send
      - Token overhead, latency, single point of failure (token)

=== #subhead([Local Area Networks (LANs)])
*MAC Address* (aka LAN Address)\
- used "locally" to send frame between interfaces (same network)
- 48 bit address, burned in NIC ROM, hex (e.g., `1A-2F-BB-76-09-AD`)
- Alloc. administered by IEEE, manufacturer obtain portion of addr space
- *ARP* (addr. resolution protocol): ip$->$mac; *RARP* (reverse ARP): mac$->$ip
  - each node maintain a ARP cache (ARP table: `<IP, MAC>`), with TTL
  - A _boradcast_ ARP requesst containing B's IP address
  - B receives ARP packet and _unicast_ reply packet to A
- *Addressing*: 
  + datagram created with IP src/dst, unchanged throughout transmission
  + At each node: lookup next-hop IP $->$ lookup MAC in ARP table$->$ add self MAC and next-hop MAC to frame header as `MAC src` and `MAC dst`
#place(right, image("ethernet-frame-structure.png", width: 60%), dy: -5pt)
*Ethernet*
- `preamble`: *7 bytes* of $(10101010)_2$ + *1 byte* of $(10101011)_2$ 
  - $(10101010)_2$ creates fixed wave pattern, let receiver sync w/ sender freq.
  - $(10101011)_2$ known as _Start Frame Delimiter_ (SFD), tells receiver to start
- `dest address` and `src address`: *6 bytes* each (contains MAC address)
  - If interface receives a frame w/ `dest addr` not matching its own, discard
- `type`: *2 bytes* upper layer protocol (e.g., mostly IP)
- `data` (payload): *46-1500 bytes*
- `CRC`: *4 bytes*. If error detected, frame dropped
*Ethernet Characteristics*: 
- Connectionless & unreliable: no handshake, no ack. It's upper layer's job
- Uses *Unslotted CSMA/CD w/ binary backoff*
  + NIC receives datagram from network layer $=>$ create frame
  + If sense channel idle, transmit frame. Else, wait until idle
  + If transmit w/o collision, done; Otherwise, abort.
  + After abort, _*exponential backoff*_: for $m$-th collision, wait $K times 512$ bit time ($K$ sample from ${0, 1, dots, 2^(m-1)}$). Then return to *step 2* above

*Switches*
+ _Hub_: bits come in on one port $=>$ broadcast to *all* other ports at same rate
  - All connecting nodes can collide w/ one another
  - No frame buffering; no CSMA/CD at hub: host NICs detect collision
+ _Switch_: examine incoming frame's `dest address` $=>$ selective forward
  - Buffer incoming frames at switch. CSMA/CD before forwarding.
  - Transparent (invisible to host); Plug-and-play (no config needed)
  - *Switch table*:
    - `<MAC, Port, TTL>`: records which port reach the host with the `MAC`
    - Learning: when recv frame, record sender location in switch table
  
  #pseudocode-list(booktabs: true, title: [Switch Update & Forwarding])[
    + *Assume*: frame $F$ received at switch $S$ from port $p$
    + *Given*: switch table at $S$.table
    + $"src", "dst"$ $<-$ unpack($F$) \/\/ src and dst MAC addr in $F$
    + $S$.table[$"src"$] $=$ $p$
    + *if* $"dst"$ *in* $S$.swith_table *then*:
      + *if* $S$.table[$"dst"$] == $S$.table[$"src"$] *then* drop $F$
      + *else* forward $F$ to interface at $S$.table[$"dst"$]
    + *else* flood \/\/ forward $F$ to all interface, except arriving interface
  ]















]

