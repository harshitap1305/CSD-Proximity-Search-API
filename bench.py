"""Run: python test_api.py   (references share no code with app.py)"""
import sys, functools, http.server, importlib, io, math, os, random, tempfile, threading, time
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "locations.csv")); N = len(df); S = 100
LAT, LON, CAT, ID = df.Latitude.values, df.Longitude.values, df.Category.values, df.ID.values
cats = sorted(set(CAT)); R = lambda x: round(float(x), 9)

# ------------------------------------------------------------------ road files
def make_links(p_missing, seed):
    rng = random.Random(seed); lines = []
    for i in range(S):
        for j in range(S):
            k = i * S + j + 1
            if j + 1 < S and rng.random() >= p_missing: lines.append(f"{k} {k+1}")
            if i + 1 < S and rng.random() >= p_missing: lines.append(f"{k} {k+S}")
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"

# independent reference: parse text -> scipy Dijkstra -> filter -> sort
def ref(text, q, cat, rad, hops):
    e = [tuple(map(int, l.replace(",", " ").split()[:2])) for l in text.splitlines()
         if len(l.split()) >= 2 and l.strip()[0] not in "#%"]
    e = [(a - 1, b - 1) for a, b in e if 1 <= a <= N and 1 <= b <= N and a != b]
    u = [a for a, b in e] + [b for a, b in e]; v = [b for a, b in e] + [a for a, b in e]
    w = [1.0 if hops else math.hypot(LAT[a]-LAT[b], LON[a]-LON[b]) for a, b in e] * 2
    G = coo_matrix((w, (u, v)), shape=(N, N)).tocsr()
    src = int(np.argmin((LAT-q[0])**2 + (LON-q[1])**2))
    d = dijkstra(G, indices=src)
    out = []
    for k in range(N):
        eu = math.hypot(LAT[k]-q[0], LON[k]-q[1])
        if CAT[k] == cat and R(eu) <= R(rad) and np.isfinite(d[k]): out.append((R(d[k]), R(eu), int(ID[k])))
    return [x[2] for x in sorted(out)[:10]]

def load(mode="coords", **env):
    os.environ.update(DIST_MODE=mode, LINKS_FILE="/nonexistent", SNAP="1", **env)
    import app; importlib.reload(app); return app

def queries(n, seed):
    rng = random.Random(seed)
    for t in range(n):
        if t % 2: k = rng.randrange(N); q = (LAT[k], LON[k])        # on a location
        else:     q = (rng.random(), rng.random())                  # off-grid
        yield q, rng.choice(cats), rng.choice([.08, .15, .3, .6, 1.5])


# ======================================================================= bench.py
"""Reproduces the numbers in the report.
Usage: python bench.py validate baselines timing ties     (any subset; no args = these four)
       python bench.py scale                              (1M-node synthetic network; ~1 GB RAM)
       python bench.py http http://HOST:PORT              (end-to-end latency of a RUNNING server)
"""
import resource, statistics as st, tempfile
os.environ.update(DIST_MODE="coords", LINKS_FILE="/nonexistent", SNAP="1")
import app as A
idx = A.index

def refgraph(text):
    e = np.array([list(map(int, l.split())) for l in text.splitlines()]) - 1
    w = np.hypot(LAT[e[:, 0]] - LAT[e[:, 1]], LON[e[:, 0]] - LON[e[:, 1]])
    return coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(N, N)).tocsr()

def refq(G, q, cat, rad):
    s = int(np.argmin((LAT - q[0]) ** 2 + (LON - q[1]) ** 2)); d = dijkstra(G, indices=s)
    eu = np.hypot(LAT - q[0], LON - q[1]); m = (CAT == cat) & (np.round(eu, 9) <= round(rad, 9)) & np.isfinite(d)
    c = np.where(m)[0]; key = sorted((R(d[k]), R(eu[k]), int(ID[k])) for k in c); return [k[2] for k in key[:10]]

def point(rng, i):                                     # alternate: on a location / off-grid
    if i % 2: k = rng.randrange(N); return (LAT[k], LON[k])
    return (rng.random(), rng.random())

def validate():
    rng = random.Random(99); tot = bad = 0
    for p in (0.0, 0.05, 0.15, 0.30):
        txt = make_links(p, int(p * 100) + 3); G = refgraph(txt); g = idx.graph_from_text(txt)
        for t in range(1000):
            q = point(rng, t); cat = rng.choice(cats); rad = rng.choice([.05, .1, .2, .4, 1.5])
            tot += 1; bad += idx.search(q[0], q[1], cat, rad, g) != refq(G, q, cat, rad)
    print(f"validate: {tot} queries (0/5/15/30% roads missing), mismatches vs scipy Dijkstra: {bad}")

def baselines():
    rng = random.Random(7)
    print("baselines: mean matches of 10 vs the link-aware answer (600 queries per row, radii .15/.25/.4)")
    print("  missing | Euclid-only | Manhattan-only")
    for p in (0.02, 0.05, 0.10, 0.15, 0.30):
        g = idx.graph_from_text(make_links(p, 77)); me = ma = n = 0
        while n < 600:
            q = point(rng, n); cat = rng.choice(cats); rad = rng.choice([.15, .25, .4])
            tr = idx.search(q[0], q[1], cat, rad, g)
            if len(tr) < 10: continue
            eu = np.hypot(LAT - q[0], LON - q[1]); c = np.where((CAT == cat) & (np.round(eu, 9) <= round(rad, 9)))[0]
            e10 = set(ID[c[np.lexsort((ID[c], np.round(eu[c], 9)))[:10]]].tolist())
            me += len(e10 & set(tr)); ma += len(set(idx.search(q[0], q[1], cat, rad, None)) & set(tr)); n += 1
        print(f"  {int(p*100):5d}%  | {me/n:11.2f} | {ma/n:.2f}")

def timing():
    rng = random.Random(3); txt = make_links(.15, 1)
    t0 = time.perf_counter(); idx.graph_from_text(txt); idx._cache.clear()
    t0 = time.perf_counter(); g = idx.graph_from_text(txt); parse = (time.perf_counter() - t0) * 1000
    ts = []
    for _ in range(300):
        q = (rng.random(), rng.random()); c = rng.choice(cats); r = rng.choice([.1, .3, 1.5])
        t0 = time.perf_counter(); idx.search(q[0], q[1], c, r, g); ts.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter(); A.Index(os.path.join(HERE, "locations.csv")); load = (time.perf_counter() - t0) * 1000
    print(f"timing: 10k nodes, {len(txt.splitlines())} links: CSV load+index {load:.0f} ms, parse+build {parse:.0f} ms, "
          f"search median {np.median(ts):.2f} ms, p99 {np.percentile(ts, 99):.2f} ms, max {max(ts):.2f} ms")

def ties():
    rng = random.Random(11); txt = make_links(.15, 1); qs = []
    while len(qs) < 800:
        qs.append((point(rng, len(qs)), rng.choice(cats), rng.choice([.1, .2, .3])))
    g = idx.graph_from_text(txt); a = [idx.search(q[0], q[1], c, r, g) for q, c, r in qs]
    A.HOPS = True; idx._cache.clear(); g = idx.graph_from_text(txt)
    b = [idx.search(q[0], q[1], c, r, g) for q, c, r in qs]; A.HOPS = False; idx._cache.clear()
    v = [i for i in range(len(qs)) if len(a[i]) == 10]
    print(f"ties: coords vs hops, {len(v)} queries with 10 answers: sets differ in {100*sum(a[i] != b[i] for i in v)/len(v):.1f}%, "
          f"avg overlap {np.mean([len(set(a[i]) & set(b[i])) for i in v]):.2f}/10")

def scale():
    M = 1000; n = M * M; cs = np.random.default_rng(0); ii = np.arange(n) // M; jj = np.arange(n) % M
    d = tempfile.mkdtemp()
    pd.DataFrame({"ID": np.arange(1, n + 1), "Latitude": np.round(ii / (M - 1), 6), "Longitude": np.round(jj / (M - 1), 6),
                  "Category": cs.choice(cats, n)}).to_csv(f"{d}/big.csv", index=False)
    right = np.where(jj < M - 1)[0]; down = np.where(ii < M - 1)[0]
    E = np.vstack([np.c_[right + 1, right + 2], np.c_[down + 1, down + M + 1]]); E = E[cs.random(len(E)) >= .05]; cs.shuffle(E)
    text = "\n".join(f"{a} {b}" for a, b in E) + "\n"; del E
    t0 = time.perf_counter(); B = A.Index(f"{d}/big.csv"); load = time.perf_counter() - t0
    t0 = time.perf_counter(); gb = B.graph_from_text(text); parse = time.perf_counter() - t0
    rng = random.Random(1); ts = []
    for _ in range(30):
        q = (rng.random(), rng.random()); t0 = time.perf_counter(); B.search(q[0], q[1], rng.choice(cats), .05, gb); ts.append((time.perf_counter() - t0) * 1000)
    print(f"scale: 1M nodes, {len(text.splitlines()):,} links (5% missing): CSV load+index {load:.1f} s, parse+build {parse:.1f} s, "
          f"search median {np.median(ts):.1f} ms, max {max(ts):.1f} ms, peak process memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f} GB")

def http(url):
    import httpx
    rng = random.Random(5); body = open(os.path.join(HERE, "sample_links.txt"), "rb").read(); url = url.rstrip("/") + "/search/"
    client = httpx.Client(timeout=60)                  # one reused connection (not a new client per request)
    def run(n, f):
        ts = []
        for _ in range(n):
            d = dict(lat=rng.random(), long=rng.random(), cat=rng.choice(cats), rad=rng.choice([.15, .3, .6]))
            t0 = time.perf_counter(); r = client.post(url, data=d, files={"link": ("l.txt", body)} if f else None)
            assert r.status_code == 200, r.text; ts.append((time.perf_counter() - t0) * 1000)
        return ts
    first = run(1, True)[0]; a = run(200, True); b = run(200, False); q = lambda x: sorted(x)[int(.95 * len(x)) - 1]
    print(f"http {url}: first request {first:.1f} ms")
    print(f"  file uploaded every request: median {st.median(a):.1f} ms, p95 {q(a):.1f} ms, max {max(a):.1f} ms")
    print(f"  no file (fallback)         : median {st.median(b):.1f} ms, p95 {q(b):.1f} ms, max {max(b):.1f} ms")

if __name__ == "__main__":
    args = sys.argv[1:] or ["validate", "baselines", "timing", "ties"]
    for a in args:
        if a == "http": continue
        if a.startswith("http"): http(a)
        else: globals()[a]()
