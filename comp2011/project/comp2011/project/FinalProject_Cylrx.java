package comp2011.project;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

import static comp2011.project.Helper.*;

/*
    Final Project for COMP2011

    Time Complexity Analysis:
    - O (n) for primes < Integer.MAX_VALUE
    - O (n log log n) for primes < Long.MAX_VALUE

    Theoretical Upper Limit:
    - Sieve up to Long.MAX_VALUE
    - Assuming 8GB RAM and unlimited compute time
 */
public class FinalProject_Cylrx {

    private static final long S = 1_048_576L; // MUST be at least 1e6 and power of 2
    private static final long LIM = (long) 1e19;


    private static long maxPrime;
    private static long priCnt = 0; // number of primes found
    private static long storeCnt = 0; // numbers of primes stored
    private static long priBound; // the upperbound of stored primes
    private static long[][] pri; // stored primes
    private static int pr = 0, pc = 0;


    /*
    Block Sieve with 2-3-5 wheel factorization
    Time Complexity: O(n log log n)
     */
    static void blockSieve() {
        int bitsetSize = (int) (S >> 6); // 2^20 -> 2^14
        long[] notPri = new long[bitsetSize];
        long[] curPri = new long[((int) pi(S)) * 2];
        int curi = 0;

        final long last = (pc == 0) ? pri[pr - 1][pri[pr - 1].length - 1] : pri[pr][pc - 1];
        int[] wheel = new int[8];

        // construct the wheel
        long prev = last;
        for (int i = 0; i < 8; i++) {
            long j = prev + 1;
            while ((j & 1L) == 0 || j % 3 == 0 || j % 5 == 0) j++;
            wheel[i] = (int) (j - prev);
            prev = j;
        }

        int ptr = wheel[0]; // ptr for notPri[]
        int wi = 0;

        // starts from [last, last + S)
        for (long b = 0; last + b * S < LIM; b++) {
            Arrays.fill(notPri, 0);
            long l = last + b * S, r = l + S; // sieve [l, r)

            for (long pi = 3; pi < storeCnt; pi++) { // skip 2, 3, 5
                int i = (int) (pi >> 20); // pi / 2^20
                int j = (int) (pi & (S - 1)); // pi % 2^20
                long p = pri[i][j];
                if (p * p > r) break;
                long factor = (l + p - 1) / p;
                for (long k = factor * p - l; k < S; k += p) {
                    int ik = (int) k;
                    int ikIndex = ik >> 6;
                    int ikBit = ik & 63;
                    notPri[ikIndex] |= (1L << ikBit);
                }
            }

            int len = (int) Math.min(S, LIM - l);
            for (; ptr < len; ptr += wheel[wi]) {
                int ptrIndex = ptr >> 6;
                int ptrBit = ptr & 63;
                if ((notPri[ptrIndex] & (1L << ptrBit)) == 0L) { // is prime

                    // store the prime if its within priBound
                    long p = l + ptr;
                    maxPrime = p;
                    if (p < priBound) {
                        try {
                            pri[pr][pc++] = p;
                        } catch (NullPointerException e) {
                            System.err.println("pri[pr]?? " + pri[pr]);
                            System.err.println("pr?? " + pr);
                            System.err.println("len?? " + pri.length);
                        }
                        if (pc >= pri[pr].length) {
                            pr++;
                            pc = 0;
                            storeCnt++;
                        }
                    }
                    curPri[curi++] = p;
                    priCnt++;
                }
                wi = (wi + 1) & 7;
            }

            if ((b & 127) == 0) {
                System.out.print("[Eratosthenes Sieve]: sieved up to " + sci(r) + "\r");
            }

            Writer.write(curPri, curi);
            curi = 0;
            ptr &= 1_048_575; // ptr % 2^20
        }
    }


    /*
    Euler Sieve with 2-3-5 wheel factorization
    Time Complexity: O(n)
     */
    public static long[] eulerSieve() {
        long start = System.currentTimeMillis();
        //int n = 100_000_000;
        int n = (int) Math.min(eulerSieveHelper(mem()), Integer.MAX_VALUE);
        int bitsetSize = (n >> 6) + 1;
        long[] notPrime = new long[bitsetSize];
        long[] initPrimes = new long[(int) pi(n)];
        int pi = 3;

        System.out.println("[Euler Sieve]: sieving first " + n + " primes");

        initPrimes[0] = 2;
        initPrimes[1] = 3;
        initPrimes[2] = 5;

        int[] wheel = {4, 2, 4, 2, 4, 6, 2, 6};
        int wi = -1;

        for (int i = 7; i < n; i += wheel[wi]) {
            int index = i >> 6; // index = i / 64
            int bit = i & 63;  // bit = i % 64
            if ((notPrime[index] & (1L << bit)) == 0) {
                initPrimes[pi++] = i;
            }
            for (int j = 0; j < pi; j++) {
                long prime = initPrimes[j];
                long k = (long) i * prime;
                if (k > n) break;
                int ik = (int) k;
                int ikIndex = ik >> 6;
                int ikBit = ik & 63;
                notPrime[ikIndex] |= (1L << ikBit);
                if (i % prime == 0) break;
            }
            wi = (wi + 1) & 7; // wi % 8
        }

        storeCnt = priCnt = pi;
        maxPrime = initPrimes[pi - 1];
        long end = System.currentTimeMillis();
        System.out.println("[Euler Sieve]: sieved " + pi + " primes in " + ((double) (end - start) / 1000.0) + " seconds");
        System.out.println("[Euler Sieve]: largest prime found " + maxPrime + "\n");
        return Arrays.copyOf(initPrimes, pi);
    }


    /*
    Initialize the program by precomputing the first few primes using Euler Sieve up to Integer.MAX_VALUE
    Then, we store the primes in RAM and calculate the upper bound of the prime numbers we can store
    Finally, we allocate the necessary memory to store the primes and start the block sieve
     */
    static void initialize() {
        long freeMem = mem() - (S << 3);
        long n = (freeMem >> 3); // maximum number of primes we can store in RAM
        int maxRows = (int) (n / S); // maximum amount of rows we can store. In practice, much lower, due to memory fragmentation

        pri = new long[maxRows][];

        long[] ips = eulerSieve();
        Writer.write(ips);
        long ipn = ips.length;

        for (long i = 0; i < ipn; i += S) {
            long ipr = ipn - i; // not yet stored length
            int len = (int) (Math.min(ipr, S));
            pri[pr] = new long[(int) S];
            for (pc = 0; pc < len; pc++) pri[pr][pc] = ips[(int) (i + pc)];
            if (len == S) pr++;
        }

        ips = null;
        System.gc();

        long cells = 0;
        int rows = pr + 1;
        while (rows < maxRows) {
            try {
                pri[rows] = new long[(int) S];
                cells += S;
                rows++;
            } catch (OutOfMemoryError m1) {
                break;
            }
        }
        priBound = rows * S;

        System.out.println("\n[Memory Usage]: ");
        System.out.println("\t Successfully allocated " + sci(cells << 3) + " Bytes of RAM to store prime numbers.");
        System.out.println("\t This allows my program to stores " + sci(cells) + " prime numbers (up to " + sci(upper(cells)) + ")");

        double upper = upper(n);
        double upperLimit = upper * upper;
        System.out.println("\n[Theoretical Limit]: ");
        System.out.println("\t My program can sieve up to ~" + sci(upperLimit));
        System.out.println("\t ... since we only need prime numbers up to sqrt(n) to sieve n");
        System.out.println("\t Which equates to ~" + sci(pi(upperLimit)) + " prime numbers\n\n");

    }


    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        try {
            initialize();
            blockSieve();
        } catch (OutOfMemoryError outOfMemoryError) {
            System.err.println("Array size too large");
            System.err.println("Max JVM memory: " + Runtime.getRuntime().maxMemory());
        }
        long end = System.currentTimeMillis();

        final long last = (pc == 0) ? pri[pr - 1][pri[pr - 1].length - 1] : pri[pr][pc - 1];
        System.out.println("[Final Result]: Found " + priCnt + " primes");
        System.out.println("[Final Result]: Largest prime found is " + last);
        System.out.println("[Time elapsed]: " + (double) (end - start) / 1000.0 + " seconds");
        String.valueOf(3);
    }
}

/*
    Helper class for writing prime numbers to the file efficiently
    Utilizes custom implemented buffered writing mechanism and direct memory allocation
    to achieve high throughput and low latency

    BENCHMARKS: writes 23,000,000 primes / sec on M1 Macbook Pro
    RESULTS MAY VARY DEPENDING ON THE HARDWARE
 */
class Writer {
    private static final String FILENAME = "Cylrx.txt";
    private static final int BUFFER_SIZE = 1024 * 1024 * 16;  // 16 MB
    private static final int THRESHOLD = 32;

    private static final FileOutputStream fos;
    private static final FileChannel fc;
    private static final ByteBuffer bb;

    static {
        try {
            fos = new FileOutputStream(FILENAME, false);
            fc = fos.getChannel();
            bb = ByteBuffer.allocateDirect(BUFFER_SIZE);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void write(long[] primes) {
        write(primes, primes.length);
    }

    public static void write(long[] primes, int len) {
        try {
            for (int i = 0; i < len; i++) {
                long prime = primes[i];
                if (bb.remaining() < THRESHOLD) {
                    bb.flip();
                    while (bb.hasRemaining()) fc.write(bb);
                    bb.clear();
                } else toBuffer(prime);
            }
            bb.flip();
            while (bb.hasRemaining()) fc.write(bb);
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
    }

    private static void toBuffer(long num) {
        byte[] tmp = new byte[20];
        int pos = tmp.length;
        while (num > 0) {
            tmp[--pos] = (byte) ('0' + (num % 10));
            num /= 10;
        }
        bb.put(tmp, pos, tmp.length - pos);
        bb.put((byte) ' ');
    }
}

class Helper {

    public static long mem() {
        long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        return Runtime.getRuntime().maxMemory() - usedMemory;
    }

    /*
    Find the max number of prime numbers we can compute via Euler Sieve given the memory limit.
    Since we need two arrays:
    - isPrime[] for the sieve
    - primes[] to store found primes
    It is a constraint problem of x/8 + 8 * x / ln(x) < MEMORY_LIMIT
    We use binary search to find x.
    Since x / ln(x) is a loose upperbound, our x will always be underestimated.
    This naturally leaves us some extra memory space, preventing overflow.
     */
    public static long eulerSieveHelper(long maxMem) {
        long mid, l = 0, r = ((long) Integer.MAX_VALUE) << 3;

        while (l < r) {
            mid = (l + r) >> 1;
            long y = mid / 8 + 8 * pi(mid);
            if (y < maxMem) l = mid + 1;
            else r = mid;
        }

        assert l + 8 * pi(l) < maxMem;
        return l;
    }

    /*
    Returns a loose, but safe non-asymptotic bound on the prime-counting function
     */
    public static long pi(double x) {
        double res = 16; // primes < 55
        double logx = Math.log(x);
        if (x >= 355991) res = x / logx * (1 + 1 / logx + 2.51 / (logx * logx));
        else if (x >= 60184) res = x / (logx - 1.1);
        else if (x >= 55) res = x / (logx - 4);
        return (long) Math.ceil(res);
    }

    /*
    Returns the upper bound of the n'th prime number
     */
    public static long upper(double n) {
        double lgn = Math.log(n);
        double lglgn = Math.log(Math.log(n));

        double res = n;

        if (n >= 688_383) res *= (lgn + lglgn - 1 + (lglgn - 2) / lgn);
        else if (n >= 27_076) res *= (lgn + lglgn - 1 + (lglgn - 1.8) / lgn);
        else if (n >= 2) res *= (lgn + lglgn - 0.5);
        else res = 2;

        return (long) Math.ceil(res);
    }

    public static String sci(double x) {
        int pow = 0;
        while (x / 10 > 1) {
            pow++;
            x /= 10;
        }
        String val = String.format("%.3f", x);
        return "(" + val + " * 10 ^ " + pow + ")";
    }
}

