import unittest
import os
import threading
import time
import math
from pathlib import Path
from client import HTTPClient
from email.utils import parsedate_to_datetime

class TestHTTPServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # basic info
        cls.host = os.getenv('TEST_SRVR_HOST', '127.0.0.1')
        cls.port = int(os.getenv('TEST_SRVR_PORT', '8080'))
        cls.timeout = int(os.getenv('TEST_SRVR_TIMEOUT', '15'))
        cls.max_conn = int(os.getenv('TEST_SRVR_MAX_CONN', '100'))
        cls.client = HTTPClient(cls.host, cls.port)

        print(f"Environment Variables:")
        print(f"\thost: {cls.host}")
        print(f"\tport: {cls.port}")
        print(f"\ttimeout: {cls.timeout}")
        print(f"\tmax_conn: {cls.max_conn}")

        # check if server is up
        try:
            status, _, _ = cls.client.request('GET', '/')
            if status != 200:
                raise RuntimeError(f"Expected 200 on GET /, got {status}")
        except Exception as e:
            raise unittest.SkipTest(f"Cannot reach server at {cls.host}:{cls.port}: {e}")

        project_root = Path(__file__).parent.parent
        cls.res_dir = project_root / 'resources'
        if not cls.res_dir.exists():
            raise FileNotFoundError(f"Resources dir not found at {cls.res_dir}")

        fixtures = {
            'test.txt' : 'Hello, this is a test.txt file.',
            'test.xyz' : 'unsupported extension',
            'test.json': '{"foo":"bar"}',
            'test.xml':  '<root></root>',
            'test.css': 'body {color: blue;}',
            'test.js':  'console.log("js test");',
        }
        for name, content in fixtures.items():
            (cls.res_dir / name).write_text(content)

    def test_get_root(self):
        status, hdr, body = self.client.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn('content-type', hdr)
        self.assertIn('text/html', hdr['content-type'])
        self.assertTrue(body.lower().startswith(b'<!doctype html') or b'<html' in body.lower())
        self.assertTrue('connection' in hdr)
        self.assertEqual(hdr['connection'], 'keep-alive')
        self.assertIn('last-modified', hdr)

    def test_get_txt(self):
        status, hdr, body = self.client.request('GET', '/resources/test.txt')
        self.assertEqual(status, 200)
        self.assertEqual(hdr.get('content-type'), 'text/plain')
        self.assertEqual(body.decode('utf-8'), 'Hello, this is a test.txt file.')
        self.assertIn('last-modified', hdr)

    def test_head_txt(self):
        status, hdr, body = self.client.request('HEAD', '/resources/test.txt', is_head=True)
        self.assertEqual(status, 200)
        self.assertEqual(hdr.get('content-type'), 'text/plain')
        self.assertEqual(int(hdr.get('content-length', -1)),
                         len('Hello, this is a test.txt file.'))
        self.assertEqual(body, b'')
        self.assertIn('last-modified', hdr)

    def test_404(self):
        for is_head in [False, True]:
            s, h, b = self.client.request('GET', '/resources/no_such.html', is_head=is_head)
            self.assertEqual(s, 404)
            if not is_head:
                self.assertTrue('Error 404' in b.decode('utf-8'))

    def test_403(self):
        for is_head in [False, True]:
            s, h, b = self.client.request('GET', '/server.py', is_head=is_head)
            self.assertEqual(s, 403)
            if not is_head:
                self.assertTrue('Error 403' in b.decode('utf-8'))

    def test_415(self):
        for is_head in [False, True]:
            s, h, b = self.client.request('GET', '/resources/test.xyz', is_head=is_head)
            self.assertEqual(s, 415)
            if not is_head:
                self.assertTrue('Error 415' in b.decode('utf-8'))

    def test_400_mth(self):
        for is_head in [False, True]:
            s, h, b = self.client.request('POST', '/', is_head=is_head)
            self.assertEqual(s, 400)
            if not is_head:
                self.assertTrue('Error 400' in b.decode('utf-8'))

    def test_400_ver(self):
        for is_head in [False, True]:
            s, h, b = self.client.request('GET', '/', version='HTTP/2.0', is_head=is_head)
            self.assertEqual(s, 400)
            if not is_head:
                self.assertTrue('Error 400' in b.decode('utf-8'))

    def test_400_path(self):
        for is_head in [False, True]:
            s, h, b = self.client.request('GET', '/../../etc/passwd', is_head=is_head)
            self.assertEqual(s, 400)
            if not is_head:
                self.assertTrue('Error 400' in b.decode('utf-8'))

    def test_304(self):
        # first fetch to get Last‑Modified
        s, h, b = self.client.request('GET', '/resources/test.txt')
        self.assertEqual(s, 200)
        self.assertIn('last-modified', h)
        lm = h.get('last-modified')
        self.assertIsNotNone(lm)
        # now re‑request with same date
        s2, h2, b2 = self.client.request(
            'GET',
            '/resources/test.txt',
            headers={'If-Modified-Since': lm})
        self.assertEqual(s2, 304)
        self.assertEqual(b2, b'')

    def test_old_date(self):
        # pretend an old date → we should get 200
        # various dates for robustness
        olds = [ 
            'Mon, 01 Jan 1990 00:00:00 GMT',
            'Tue, 15 Feb 2000 12:30:45 GMT',
            'Wed, 31 Dec 2003 23:59:59 GMT',
            'Thu, 29 Feb 2004 00:00:00 GMT',
            'Fri, 01 Jan 2010 01:01:01 GMT',
            'Tue, 30 Jun 2015 23:59:60 GMT',  
            'Wed, 29 Feb 2012 00:00:00 GMT',  
            'Fri, 01 Jan 2015 00:00:00 GMT',  
            'Sat, 01 Jan 2000 00:00:00 GMT',  
            'Sun, 31 Dec 1999 23:59:59 GMT',  
            'Mon, 15 Mar 2021 12:34:56 GMT'   
        ]
        for old in olds:
            s, h, b = self.client.request(
                'GET', '/resources/test.txt',
                headers={'If-Modified-Since': old})
            self.assertEqual(s, 200)
            self.assertTrue(len(b) > 0)
            self.assertIn('last-modified', h)
            self.assertLess(compare_http_date(old, h['last-modified']), 0)

    def test_new_date(self):
        news = [
            'Mon, 01 Jan 2038 03:14:07 GMT',
            'Tue, 15 Feb 2038 12:30:45 GMT',
            'Wed, 31 Dec 2038 23:59:59 GMT',
            'Thu, 29 Feb 2038 00:00:00 GMT',
            'Fri, 01 Jan 2038 01:01:01 GMT',
        ]
        for new in news:
            s, h, b = self.client.request(
                'GET', '/resources/test.txt',
                headers={'If-Modified-Since': new})
            self.assertEqual(s, 304)
            self.assertEqual(b, b'')
            self.assertIn('last-modified', h)
            self.assertLess(compare_http_date(h['last-modified'], new), 0)

    def test_bad_date(self):
        s, h, b = self.client.request(
            'GET', '/resources/test.txt',
            headers={'If-Modified-Since': 'garbage-date'})
        self.assertEqual(s, 400)

    def test_alive_http11(self):
        with self.client.open(version='HTTP/1.1') as conn:
            s1, h1, b1 = conn.send('GET', '/resources/test.txt')
            s2, h2, b2 = conn.send('GET', '/resources/test.js')
            self.assertEqual(s1, 200)
            self.assertEqual(s2, 200)
            self.assertEqual(h1.get('connection'), 'keep-alive')
            self.assertEqual(h2.get('connection'), 'keep-alive')
            self.assertIn('timeout=', h1['keep-alive'])
            self.assertIn('max=',     h1['keep-alive'])
            self.assertIn('last-modified', h1)
            self.assertIn('timeout=', h2['keep-alive'])
            self.assertIn('max=',     h2['keep-alive'])
            self.assertIn('last-modified', h2)
    
    def test_close_http11(self):
        with self.client.open(version='HTTP/1.1') as conn:
            s, h, b = conn.send('GET', '/resources/test.txt', {'Connection': 'close'})
            self.assertIn('last-modified', h)
            self.assertEqual(s, 200)
            try: 
                s, h, b = conn.send('GET', '/resources/test.js')
                self.assertTrue(s == None and h == {} and b == b'')
            except Exception as e:
                self.assertIn('Connection reset by peer', str(e))

    def test_close_http10(self):
        conn = self.client.open(version='HTTP/1.0')
        s1, h1, b1 = conn.send('GET', '/resources/test.txt')
        self.assertEqual(s1, 200)
        self.assertEqual(h1.get('connection'), 'close')
        self.assertIn('last-modified', h1)
        try: 
            s, h, b = conn.send('GET', '/resources/test.js')
            self.assertTrue(s == None and h == {} and b == b'')
        except Exception as e:
            self.assertIn('Connection reset by peer', str(e))
        conn.close()
    
    def test_alive_http10(self):
        conn = self.client.open(version='HTTP/1.0')
        s1, h1, b1 = conn.send('GET', '/resources/test.txt', {'Connection': 'keep-alive'})
        self.assertEqual(s1, 200)
        self.assertEqual(h1.get('connection'), 'keep-alive')
        self.assertTrue('keep-alive' in h1)
        self.assertIn('last-modified', h1)
        try: 
            s2, h2, b2 = conn.send('GET', '/resources/test.js')
            self.assertEqual(s2, 200)
        except Exception as e:
            self.fail(f"Unexpected error: {e}")
        conn.close()
   
    def test_multi_thread(self):
        big_rel = '/resources/big.txt'
        big_file = self.res_dir / 'big.txt'
        if not big_file.exists():
            self.skipTest(f"big.txt not found in {self.res_dir}")

        N = 10
        durations = [None] * N
        errors = []

        def worker(idx):
            try:
                start = time.perf_counter()
                status, hdr, body = self.client.request('GET', big_rel)
                stop = time.perf_counter()
                self.assertEqual(status, 200, f"Thread {idx} got {status}")
                durations[idx] = stop - start
            except Exception as e:
                errors.append((idx, e))

        # launch N threads at (almost) the same time
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads: t.start()
        for t in threads: t.join()

        if errors:
            self.fail(f"Errors in workers: {errors}")

        # pearson correlation
        xs = list(range(N))
        mean_x = sum(xs) / N
        mean_y = sum(durations) / N
        cov = sum((x - mean_x)*(y - mean_y) for x, y in zip(xs, durations))
        var_x = sum((x - mean_x)**2 for x in xs)
        var_y = sum((y - mean_y)**2 for y in durations)

        r = cov / math.sqrt(var_x * var_y)
        print(f"Pearson Correlation = {r:.3f}")

        # if server is non-threaded, r close to 1.0
        self.assertLess(abs(r), 0.8,
            f"High index-duration correlation ({r:.3f}) MIGHT suggest non-threaded server") 

    def test_timeout(self):
        with self.client.open(version='HTTP/1.1') as conn:
            s, h, b = conn.send('GET', '/')
            self.assertEqual(s, 200)
            time.sleep(self.timeout + 2)
            try: 
                s2, h2, b2 = conn.send('GET', '/')
                self.assertTrue(s2 == None and h2 == {} and b2 == b'')
            except Exception as e:
                self.assertIn('Connection reset by peer', str(e))
    
    def test_stress(self): 
        N = [
            self.max_conn // 2, 
            self.max_conn, 
            self.max_conn + 1
        ]
        ans = [True, True, False]

        for a, n in zip(ans, N): 
            try: 
                responses: list[tuple[int, dict, bytes]] = self.client.request_N('GET', '/resources/style.css', is_head=False, N=n)
                if a: 
                    for s, h, b in responses: 
                        self.assertEqual(s, 200)
                    self.assertEqual(len(responses), n)
                else:
                    self.fail('Should have failed due to exceeding max exchanges per connection')
            except ConnectionResetError:
                if a: 
                    self.fail('Should not have failed due to exceeding max exchanges per connection')
            except Exception as e:
                self.fail(f'Unexpected error: {e}')
        



def parse_http_date(date_str):
    parts = date_str.strip().split()
    day   = int(parts[1])
    month_str = parts[2]
    year  = int(parts[3])
    month_map = {
        'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4,
        'May':5, 'Jun':6, 'Jul':7, 'Aug':8,
        'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12
    }
    month = month_map.get(month_str, 0)
    hh, mm, ss = parts[4].split(':')
    hour   = int(hh)
    minute = int(mm)
    second = int(ss)
    
    return (year, month, day, hour, minute, second)

def compare_http_date(date1, date2):
    """
    Compare two HTTP-Date strings.
    Returns:
      >0 if date1 > date2,
      <0 if date1 < date2,
       0 if equal.
    """
    t1 = parse_http_date(date1)
    t2 = parse_http_date(date2)
    
    for a, b in zip(t1, t2):
        if a != b:
            return a - b
    return 0


if __name__ == '__main__':
    unittest.main()
