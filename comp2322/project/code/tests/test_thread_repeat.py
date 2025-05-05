import os, time, math, threading
import matplotlib.pyplot as plt
from client import HTTPClient

# based on test_multi_thread in test_server.py
def single_run(client, rel, N):
    times = [0]*N
    def worker(i):
        t0=time.perf_counter()
        status,_,_ = client.request('GET', rel)
        t1=time.perf_counter()
        if status!=200: raise RuntimeError(f"Got {status}")
        times[i]=t1-t0
    th=[threading.Thread(target=worker,args=(i,)) for i in range(N)]
    for t in th: t.start()
    for t in th: t.join()
    xs=list(range(N))
    mx=sum(xs)/N; my=sum(times)/N
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,times))
    vx=sum((x-mx)**2 for x in xs)
    vy=sum((y-my)**2 for y in times)
    return abs(cov/math.sqrt(vx*vy)) if vx*vy>0 else 0.0

def main():
    # params
    M = 100        # number of repeats
    N = 10        # threads per run
    host = os.getenv('TEST_SRVR_HOST','127.0.0.1')
    port = int(os.getenv('TEST_SRVR_PORT','8080'))
    rel  = '/resources/big.txt'
    client = HTTPClient(host, port)

    # check resource
    try:
        client.request('GET', rel)
    except Exception as e:
        print("Error: cannot fetch", rel, "-", e)
        return

    # collect absolute correlations
    from tqdm import tqdm
    corrs = [ single_run(client, rel, N) for _ in tqdm(range(M)) ]

    # plot
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True,
        gridspec_kw={'height_ratios':[3,1]})
    # histogram
    bins = min(10, M)
    ax1.bar(
        *zip(*[(b,corrs.count(b)) for b in map(lambda i: round(i/bins,2), range(bins+1))]),
        width=1/bins, color='skyblue', edgecolor='white')
    # better: use plt.hist
    ax1.cla()
    ax1.hist(corrs, bins=bins, range=(0,1), color='skyblue', edgecolor='white')
    ax1.axvline(0.8, color='red', linestyle='--')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of |r|')
    # box plot
    ax2.boxplot(corrs, vert=False, patch_artist=True,
        boxprops=dict(facecolor='lightgreen', color='green'),
        medianprops=dict(color='darkgreen'))
    ax2.axvline(0.8, color='red', linestyle='--')
    ax2.set_xlabel('Absolute Correlation (0.0 to 1.0)')
    ax2.set_yticks([])
    plt.tight_layout()
    plt.show()
    plt.savefig('thread-repeat.png')

if __name__=='__main__':
    main()
