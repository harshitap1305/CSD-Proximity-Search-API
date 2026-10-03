import importlib, math, os, random, tempfile, time
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from fastapi.testclient import TestClient

df = pd.read_csv("locations.csv"); N = len(df); S = 100
rows = list(df.itertuples(index=False)); cats = sorted(df.Category.unique())
ULAT = sorted(df.Latitude.unique()); ULON = sorted(df.Longitude.unique())
STEP = float(np.median(np.diff(ULAT)))
R = lambda x: round(x, 9)

def ref_plain(q, cat, rad, snap, hops):               
    a, o = (min(ULAT, key=lambda u: abs(u-q[0])), min(ULON, key=lambda u: abs(u-q[1]))) if snap else q
    out = []
    for r in rows:
        e = math.hypot(r.Latitude-q[0], r.Longitude-q[1])
        if r.Category == cat and R(e) <= R(rad):
            d = abs(r.Latitude-a) + abs(r.Longitude-o)
            out.append((float(round(d/STEP)) if hops else R(d), R(e), r.ID))
    return [x[2] for x in sorted(out)[:10]]

def ref_graph(q, cat, rad, hops, blocked=()):           
    bad = {frozenset(b) for b in blocked}; u, v, w = [], [], []
    for i in range(S):
        for j in range(S):
            k = i*S+j
            for a, b in ((i+1, j), (i, j+1)):
                if a < S and b < S:
                    m = a*S+b
                    if frozenset((k+1, m+1)) in bad: continue
                    h = 1.0 if hops else abs(df.Latitude[k]-df.Latitude[m]) + abs(df.Longitude[k]-df.Longitude[m])
                    u += [k, m]; v += [m, k]; w += [h, h]
    G = coo_matrix((w, (u, v)), shape=(N, N)).tocsr()
    src = min(range(S), key=lambda i: abs(ULAT[i]-q[0]))*S + min(range(S), key=lambda j: abs(ULON[j]-q[1]))
    d = dijkstra(G, indices=src)
    out = [(R(d[r.ID-1]), R(math.hypot(r.Latitude-q[0], r.Longitude-q[1])), r.ID) for r in rows
           if r.Category == cat and R(math.hypot(r.Latitude-q[0], r.Longitude-q[1])) <= R(rad)]
    return [x[2] for x in sorted(out)[:10]]

def load(blocked="/nonexistent", snap="1", mode="coords"):
    os.environ.update(BLOCKED_EDGES=blocked, SNAP=snap, DIST_MODE=mode)
    import app; importlib.reload(app); return app

def rand_q(rng, t):
    if t % 3 == 0: return (rng.random(), rng.random())                                  
    if t % 3 == 1: k = rng.randrange(N); return (df.Latitude[k], df.Longitude[k])       
    return (rng.choice([0, 1, .5, -.1, 1.1]), rng.random())                              
    
for snap in ("1", "0"):
    for mode in ("coords", "hops"):
        A = load(snap=snap, mode=mode); cl = TestClient(A.app); rng = random.Random(1); T = 150
        for t in range(T):
            q = rand_q(rng, t); cat = rng.choice(cats); rad = rng.choice([.05, .1, .2, .4, 1.5])
            got = cl.get("/search/", params=dict(lat=q[0], long=q[1], cat=cat, rad=rad)).json()["ids"]
            assert got == ref_plain(q, cat, rad, snap == "1", mode == "hops"), (snap, mode, q, cat, rad)
        print(f"[1] SNAP={snap} DIST_MODE={mode:6s}: {T}/{T} match brute force")

for mode in ("coords", "hops"):
    A = load(mode=mode); rng = random.Random(3)
    for t in range(12):
        k = rng.randrange(N); q = (df.Latitude[k], df.Longitude[k]); cat = rng.choice(cats); rad = rng.choice([.1, .3, 1.5])
        assert A.index.search(*q, cat, rad) == ref_graph(q, cat, rad, mode == "hops"), (mode, q)
    print(f"[2] {mode:6s}: 12/12 == scipy Dijkstra on full lattice")

rng = random.Random(7); blocked = []
for i in range(S):
    for j in range(S):
        k = i*S+j+1
        if j+1 < S and rng.random() < .15: blocked.append((k, k+1))
        if i+1 < S and rng.random() < .15: blocked.append((k, k+S))
f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); pd.DataFrame(blocked, columns=["ID1", "ID2"]).to_csv(f.name, index=False)
for mode in ("coords", "hops"):
    A = load(blocked=f.name, mode=mode); good = 0
    for t in range(20):
        q = rand_q(rng, t); q = (min(max(q[0], 0), 1), min(max(q[1], 0), 1)); cat = rng.choice(cats); rad = rng.choice([.15, .3, .6])
        good += A.index.search(*q, cat, rad) == ref_graph(q, cat, rad, mode == "hops", blocked)
    print(f"[3] blocked roads, {mode:6s}: {good}/20 == scipy Dijkstra"); assert good == 20

A = load(); cl = TestClient(A.app)
assert cl.get("/search/", params=dict(lat=.5, long=.5, cat="zzz", rad=.1)).status_code == 400
assert cl.get("/search/", params=dict(lat=.5, long=.5)).status_code == 422
assert cl.post("/search/", json=dict(lat=.5, long=.5, cat="BANK", rad=.1)).json() == cl.get("/search/", params=dict(lat=.5, long=.5, cat="bank", rad=.1)).json()
ids = cl.get("/search/", params=dict(lat=.2, long=.9, cat="bank", rad=.15)).json()["ids"]
assert len(ids) == len(set(ids)) and all(df.Category[i-1] == "bank" for i in ids)
assert all(math.hypot(df.Latitude[i-1]-.2, df.Longitude[i-1]-.9) <= .15+1e-9 for i in ids)
print("[4] validation, POST==GET, category + radius constraints, no duplicates: OK")

rng = random.Random(0); t0 = time.perf_counter()
for _ in range(300): A.index.search(rng.random(), rng.random(), "cafe", .3)
print(f"[5] 10k points: {(time.perf_counter()-t0)/300*1000:.2f} ms/query")
big = pd.DataFrame({"ID": np.arange(1, 10**6+1), "Latitude": np.random.default_rng(0).random(10**6),
                    "Longitude": np.random.default_rng(1).random(10**6), "Category": np.random.default_rng(2).choice(cats, 10**6)})
big.to_csv("/tmp/big.csv", index=False); B = A.Index("/tmp/big.csv")
for r_ in (.05, 1.5):
    t0 = time.perf_counter(); [B.search(rng.random(), rng.random(), "cafe", r_) for _ in range(20)]
    print(f"    1M points, rad={r_}: {(time.perf_counter()-t0)/20*1000:.1f} ms/query")
print("ALL TESTS PASSED")
