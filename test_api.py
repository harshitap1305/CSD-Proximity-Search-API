import functools, http.server, importlib, io, math, os, random, tempfile, threading, time
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "locations.csv")); N = len(df); S = 100
LAT, LON, CAT, ID = df.Latitude.values, df.Longitude.values, df.Category.values, df.ID.values
cats = sorted(set(CAT)); R = lambda x: round(float(x), 9)

def make_links(p_missing, seed):
    rng = random.Random(seed); lines = []
    for i in range(S):
        for j in range(S):
            k = i * S + j + 1
            if j + 1 < S and rng.random() >= p_missing: lines.append(f"{k} {k+1}")
            if i + 1 < S and rng.random() >= p_missing: lines.append(f"{k} {k+S}")
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"

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

TXT = make_links(.15, 1)
FULL = make_links(0, 2)

for mode in ("coords", "hops"):
    A = load(mode); cl = TestClient(A.app); ok = 0; T = 60
    for q, cat, rad in queries(T, 11):
        r = cl.post("/search/", data=dict(lat=q[0], long=q[1], cat=cat, rad=rad), files={"link": ("links.txt", TXT)})
        assert r.status_code == 200, r.text
        assert r.json()["ids"] == ref(TXT, q, cat, rad, mode == "hops"), (mode, q, cat, rad); ok += 1
    print(f"[1] upload, {mode:6s}, 15% roads missing: {ok}/{T} == scipy Dijkstra")

A = load(); cl = TestClient(A.app)
open(os.path.join(HERE, "_links_test.txt"), "w").write(TXT)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE))
threading.Thread(target=srv.serve_forever, daemon=True).start(); url = f"http://127.0.0.1:{srv.server_port}/_links_test.txt"
q, cat, rad = (0.3, 0.7), "park", 0.25; base = dict(lat=q[0], long=q[1], cat=cat, rad=rad); exp = ref(TXT, q, cat, rad, False)
pairs = [[int(x) for x in l.split()] for l in TXT.splitlines()]
tries = {
 "POST multipart file":   cl.post("/search/", data=base, files={"link": ("l.txt", TXT)}),
 "GET  multipart file":   cl.request("GET", "/search/", data=base, files={"link": ("l.txt", TXT)}),
 "POST form text":        cl.post("/search/", data={**base, "link": TXT}),
 "POST JSON text":        cl.post("/search/", json={**base, "link": TXT}),
 "POST JSON pair list":   cl.post("/search/", json={**base, "link": pairs}),
 "GET  query = URL":      cl.get("/search/", params={**base, "link": url}),
 "GET  query = filename": cl.get("/search/", params={**base, "link": "_links_test.txt"}),
}
for name, r in tries.items():
    assert r.status_code == 200 and r.json()["ids"] == exp, (name, r.status_code, r.text)
    print(f"[2] {name:22s} OK")
os.remove(os.path.join(HERE, "_links_test.txt")); srv.shutdown()

# 3. messy link file ----------------------------------------------------------
messy = "# comment\r\n\r\n" + TXT.replace("\n", "\r\n") + "5 5\n999999 3\n1,2\n  2   1  \n3 abc\nfoo\n"
for q, cat, rad in queries(15, 21):
    r = cl.post("/search/", data=dict(lat=q[0], long=q[1], cat=cat, rad=rad), files={"link": ("l.txt", messy)})
    assert r.json()["ids"] == ref(TXT, q, cat, rad, False)
print("[3] CRLF, comments, blanks, commas, duplicates, self-loops, unknown IDs, junk lines: 15/15 OK")

# 4. sparse / disconnected networks ------------------------------------------
sparse = make_links(.45, 5); good = 0; n = 0
for q, cat, rad in queries(30, 31):
    r = cl.post("/search/", data=dict(lat=q[0], long=q[1], cat=cat, rad=rad), files={"link": ("l.txt", sparse)}).json()["ids"]
    n += 1; good += (r == ref(sparse, q, cat, rad, False))
print(f"[4] 45% roads missing (many unreachable nodes): {good}/{n} == scipy Dijkstra"); assert good == n

# 5. complete network == no-link fallback ------------------------------------
for q, cat, rad in queries(40, 41):
    p = dict(lat=q[0], long=q[1], cat=cat, rad=rad)
    a = cl.post("/search/", data=p, files={"link": ("l.txt", FULL)}).json()["ids"]
    b = cl.post("/search/", data=p).json()["ids"]
    assert a == b == ref(FULL, q, cat, rad, False), (q, cat, rad, a, b)
print("[5] full road file == no `link` (Manhattan fallback) == Dijkstra: 40/40")

# 6. errors & contract -------------------------------------------------------
P = dict(lat=.5, long=.5, cat="bank", rad=.2)
assert cl.post("/search/", data={**P, "cat": "zzz"}).status_code == 400
assert cl.post("/search/", data={"lat": .5}).status_code == 422
assert cl.post("/search/", data={**P, "lat": "x"}).status_code == 422
assert cl.post("/search/", data={**P, "rad": -1}).status_code == 400
assert cl.post("/search/", data=P, files={"link": ("l.txt", "garbage\nmore garbage\n")}).status_code == 400
assert cl.post("/search/", data={**P, "cat": "BANK"}, files={"link": ("l.txt", TXT)}).json() == cl.post("/search/", data=P, files={"link": ("l.txt", TXT)}).json()
ids = cl.post("/search/", data=P, files={"link": ("l.txt", TXT)}).json()["ids"]
assert len(ids) == 10 == len(set(ids)) and all(CAT[i-1] == "bank" and math.hypot(LAT[i-1]-.5, LON[i-1]-.5) <= .2 + 1e-9 for i in ids)
print("[6] error codes, case-insensitive category, 10 unique ids, category+radius constraints: OK")
print("ALL TESTS PASSED")
