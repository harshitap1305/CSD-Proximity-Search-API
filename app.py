import heapq
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

K = 10
EPS = 9  
DATA = os.environ.get("LOCATIONS_CSV", os.path.join(os.path.dirname(__file__), "locations.csv"))
SNAP = os.environ.get("SNAP", "1") != "0"
HOPS = os.environ.get("DIST_MODE", "coords").lower() == "hops"
BLOCKED = os.environ.get("BLOCKED_EDGES", os.path.join(os.path.dirname(__file__), "blocked_edges.csv"))

class Index:
    def __init__(self, csv_path: str, blocked_path: str | None = None):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        self.ids = df["ID"].to_numpy(np.int64)
        self.lat = df["Latitude"].to_numpy(np.float64)
        self.lon = df["Longitude"].to_numpy(np.float64)
        self.cat = df["Category"].astype(str).str.strip().str.lower().to_numpy()

        self.ulat, self.ulon = np.unique(self.lat), np.unique(self.lon)
        pairs = pd.DataFrame({"a": self.lat, "b": self.lon}).duplicated().sum() == 0
        self.lattice = bool(pairs and self.ulat.size * self.ulon.size == self.ids.size and self.ulat.size > 1)
        self.step = float(np.median(np.diff(self.ulat))) if self.ulat.size > 1 else 1.0

        self.by_cat = {}
        for c in np.unique(self.cat):
            idx = np.where(self.cat == c)[0]
            idx = idx[np.argsort(self.lat[idx], kind="stable")]
            self.by_cat[c] = (idx, self.lat[idx])

        self.graph = None
        if blocked_path and os.path.exists(blocked_path):
            self._build_graph(blocked_path)

    def _build_graph(self, blocked_path):
        ulat, ulon = self.ulat, self.ulon
        li = np.searchsorted(ulat, self.lat)
        lj = np.searchsorted(ulon, self.lon)
        node = {(a, b): k for k, (a, b) in enumerate(zip(li, lj))}   
        self.node, self.li, self.lj = node, li, lj
        blocked = pd.read_csv(blocked_path)
        pos = {int(i): k for k, i in enumerate(self.ids)}
        bad = {frozenset((pos[int(a)], pos[int(b)])) for a, b in blocked.iloc[:, :2].to_numpy()}
        adj = defaultdict(list)
        for (a, b), k in node.items():
            for da, db in ((1, 0), (0, 1)):
                m = node.get((a + da, b + db))
                if m is None or frozenset((k, m)) in bad:
                    continue
                w = 1.0 if HOPS else abs(self.lat[k] - self.lat[m]) + abs(self.lon[k] - self.lon[m])
                adj[k].append((m, w)); adj[m].append((k, w))
        self.graph = adj

    def _dijkstra_rank(self, qlat, qlon, targets: dict):
        a = int(np.abs(self.ulat - qlat).argmin()); b = int(np.abs(self.ulon - qlon).argmin())
        src = self.node[(a, b)]
        d0 = abs(self.lat[src] - qlat) + abs(self.lon[src] - qlon)   
        dist = {src: d0}; pq = [(d0, src)]; done = set(); found = []; cutoff = None
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            if cutoff is not None and round(d, EPS) > cutoff:
                break
            done.add(u)
            if u in targets:
                found.append((round(d, EPS), round(targets[u], EPS), int(self.ids[u])))
                if len(found) == K:
                    cutoff = round(d, EPS)      
            for v, w in self.graph[u]:
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; heapq.heappush(pq, (nd, v))
        return sorted(found)[:K]

    def search(self, qlat, qlon, category, rad):
        entry = self.by_cat.get(category.strip().lower())
        if entry is None:
            return []
        idx, lats = entry
        lo = np.searchsorted(lats, qlat - rad, side="left")
        hi = np.searchsorted(lats, qlat + rad, side="right")
        cand = idx[lo:hi]
        dlat, dlon = self.lat[cand] - qlat, self.lon[cand] - qlon
        euc = np.sqrt(dlat ** 2 + dlon ** 2)
        inside = round_le(euc, rad)
        cand, dlat, dlon, euc = cand[inside], dlat[inside], dlon[inside], euc[inside]
        if cand.size == 0:
            return []
        if self.graph is not None:
            return [r[2] for r in self._dijkstra_rank(qlat, qlon, dict(zip(cand.tolist(), euc.tolist())))]
        if SNAP and self.lattice:          
            sa = self.ulat[np.abs(self.ulat - qlat).argmin()]
            so = self.ulon[np.abs(self.ulon - qlon).argmin()]
            man = np.abs(self.lat[cand] - sa) + np.abs(self.lon[cand] - so)
        else:
            man = np.abs(dlat) + np.abs(dlon)
        man = np.round(man / self.step) if HOPS else np.round(man, EPS)
        order = np.lexsort((self.ids[cand], np.round(euc, EPS), man))[:K]
        return self.ids[cand[order]].tolist()


def round_le(euc, rad):
    return np.round(euc, EPS) <= round(rad, EPS)


app = FastAPI(title="Nearest Locations API")
index = Index(DATA, BLOCKED)


class Q(BaseModel):
    lat: float
    long: float
    cat: str
    rad: float


def _run(lat, lon, cat, rad):
    if rad < 0:
        raise HTTPException(400, "rad must be >= 0")
    if cat.strip().lower() not in index.by_cat:
        raise HTTPException(400, f"unknown category; valid: {sorted(index.by_cat)}")
    return {"ids": index.search(lat, lon, cat, rad)}


@app.get("/search/")
@app.get("/search", include_in_schema=False)
def search_get(lat: float = Query(...), long: float = Query(...), cat: str = Query(...), rad: float = Query(...)):
    return _run(lat, long, cat, rad)


@app.post("/search/")
@app.post("/search", include_in_schema=False)
def search_post(q: Q):
    return _run(q.lat, q.long, q.cat, q.rad)


@app.get("/")
def root():
    return {"status": "ok", "usage": "/search/?lat=0.5&long=0.5&cat=bank&rad=0.1", "n": int(index.ids.size)}
