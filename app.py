import hashlib
import heapq
import os
import urllib.request
from collections import OrderedDict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

K = 10
EPS = 9                      # decimals when comparing distances (kills float noise)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LOCATIONS_CSV", os.path.join(HERE, "locations.csv"))
DEFAULT_LINKS = os.environ.get("LINKS_FILE", os.path.join(HERE, "links.txt"))
SNAP = os.environ.get("SNAP", "1") != "0"
HOPS = os.environ.get("DIST_MODE", "coords").lower() == "hops"

class Graph:
    """Undirected road graph in CSR form (python lists -> fast scalar access)."""
    def __init__(self, indptr, indices, weights, n_edges):
        self.indptr, self.indices, self.weights, self.n_edges = indptr, indices, weights, n_edges

class Index:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        self.ids = df["ID"].to_numpy(np.int64)
        self.lat = df["Latitude"].to_numpy(np.float64)
        self.lon = df["Longitude"].to_numpy(np.float64)
        self.cat = df["Category"].astype(str).str.strip().str.lower().to_numpy()
        self.n = self.ids.size

        order = np.argsort(self.ids, kind="stable")          # ID -> row lookup
        self._sorted_ids, self._sorted_rows = self.ids[order], order

        self.ulat, self.ulon = np.unique(self.lat), np.unique(self.lon)
        unique_pairs = not pd.DataFrame({"a": self.lat, "b": self.lon}).duplicated().any()
        self.lattice = bool(unique_pairs and self.ulat.size * self.ulon.size == self.n and self.ulat.size > 1)
        self.step = float(np.median(np.diff(self.ulat))) if self.ulat.size > 1 else 1.0

        # per-category rows sorted by latitude -> bounding band via binary search
        self.by_cat = {}
        for c in np.unique(self.cat):
            idx = np.where(self.cat == c)[0]
            idx = idx[np.argsort(self.lat[idx], kind="stable")]
            self.by_cat[c] = (idx, self.lat[idx])

        self._cache: "OrderedDict[str, Graph]" = OrderedDict()

    # ------------------------------------------------------------------ links
    def rows_of(self, id_arr):
        pos = np.searchsorted(self._sorted_ids, id_arr)
        pos = np.clip(pos, 0, self.n - 1)
        ok = self._sorted_ids[pos] == id_arr
        return self._sorted_rows[pos], ok

    def graph_from_text(self, text: str) -> Graph:
        key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest() + ("h" if HOPS else "c")
        g = self._cache.get(key)
        if g is not None:
            self._cache.move_to_end(key)
            return g
        a, b = [], []
        for line in text.splitlines():
            p = line.replace(",", " ").split()
            if len(p) < 2 or p[0][0] in "#%":
                continue
            try:
                x, y = int(p[0]), int(p[1])
            except ValueError:
                try:
                    x, y = int(float(p[0])), int(float(p[1]))
                except ValueError:
                    continue
            a.append(x); b.append(y)
        if not a:
            raise ValueError("link file contains no valid 'a b' lines")
        ra, oka = self.rows_of(np.asarray(a, np.int64))
        rb, okb = self.rows_of(np.asarray(b, np.int64))
        keep = oka & okb & (ra != rb)
        ra, rb = ra[keep], rb[keep]
        if ra.size == 0:
            raise ValueError("none of the links refer to known location IDs")
        lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)       # undirected, de-duplicated
        code = np.unique(lo.astype(np.int64) * self.n + hi)
        lo, hi = code // self.n, code % self.n
        w = np.ones(lo.size) if HOPS else np.hypot(self.lat[lo] - self.lat[hi], self.lon[lo] - self.lon[hi])
        src = np.concatenate([lo, hi]); dst = np.concatenate([hi, lo]); ww = np.concatenate([w, w])
        o = np.argsort(src, kind="stable")
        indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=self.n))])
        g = Graph(indptr.tolist(), dst[o].tolist(), ww[o].tolist(), int(lo.size))
        self._cache[key] = g
        while len(self._cache) > 3:
            self._cache.popitem(last=False)
        return g

    # ---------------------------------------------------------------- queries
    def _candidates(self, qlat, qlon, category, rad):
        entry = self.by_cat.get(category.strip().lower())
        if entry is None:
            return None
        idx, lats = entry
        lo = np.searchsorted(lats, qlat - rad, side="left")
        hi = np.searchsorted(lats, qlat + rad, side="right")
        cand = idx[lo:hi]
        dlat, dlon = self.lat[cand] - qlat, self.lon[cand] - qlon
        euc = np.sqrt(dlat ** 2 + dlon ** 2)
        inside = np.round(euc, EPS) <= round(rad, EPS)
        return cand[inside], euc[inside]

    def search(self, qlat, qlon, category, rad, graph: Graph | None = None):
        got = self._candidates(qlat, qlon, category, rad)
        if got is None or got[0].size == 0:
            return []
        cand, euc = got
        if graph is not None:
            src = int(np.argmin((self.lat - qlat) ** 2 + (self.lon - qlon) ** 2))   # nearest node
            return self._dijkstra(graph, src, dict(zip(cand.tolist(), euc.tolist())))
        # ---- no link file: complete lattice, grid distance = Manhattan distance
        if SNAP and self.lattice:
            sa = self.ulat[np.abs(self.ulat - qlat).argmin()]
            so = self.ulon[np.abs(self.ulon - qlon).argmin()]
            man = np.abs(self.lat[cand] - sa) + np.abs(self.lon[cand] - so)
        else:
            man = np.abs(self.lat[cand] - qlat) + np.abs(self.lon[cand] - qlon)
        man = np.round(man / self.step) if HOPS else np.round(man, EPS)
        order = np.lexsort((self.ids[cand], np.round(euc, EPS), man))[:K]
        return self.ids[cand[order]].tolist()

    def _dijkstra(self, g: Graph, src: int, targets: dict):
        """Early-stopping Dijkstra. targets: row -> Euclidean distance from query."""
        indptr, indices, weights = g.indptr, g.indices, g.weights
        ids = self.ids
        dist = {src: 0.0}
        pq = [(0.0, src)]
        done = set()
        found, cutoff, remaining = [], None, len(targets)
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            if cutoff is not None and round(d, EPS) > cutoff:
                break                                   # everything tied with the K-th is settled
            done.add(u)
            e = targets.get(u)
            if e is not None:
                found.append((round(d, EPS), round(e, EPS), int(ids[u])))
                remaining -= 1
                if len(found) == K:
                    cutoff = round(d, EPS)
                if remaining == 0:
                    break
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                nd = d + weights[k]
                if nd < dist.get(v, 1e300):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        found.sort()
        return [f[2] for f in found[:K]]

app = FastAPI(title="Nearest Locations API")
index = Index(DATA)

def _read_link(value) -> str | None:
    """Turn whatever was sent as `link` into the text of the road file."""
    if value is None:
        return None
    if isinstance(value, UploadFile):
        return None   # handled by caller (async read)
    s = str(value)
    if s.strip() == "":
        return None
    t = s.strip()
    if t.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(t, timeout=30) as r:
            return r.read().decode("utf-8-sig", "replace")
    if "\n" not in t and len(t) < 260:                     # maybe a file inside the app folder
        p = os.path.abspath(os.path.join(HERE, t))
        if p.startswith(HERE + os.sep) and os.path.isfile(p):
            with open(p, encoding="utf-8-sig", errors="replace") as f:
                return f.read()
    return s

def _solve(params: dict, link_text: str | None):
    try:
        lat, lon, rad = float(params["lat"]), float(params["long"]), float(params["rad"])
        cat = str(params["cat"])
    except KeyError as e:
        raise HTTPException(422, f"missing field: {e.args[0]} (need lat, long, cat, rad, link)")
    except (TypeError, ValueError):
        raise HTTPException(422, "lat, long and rad must be numbers")
    if rad < 0:
        raise HTTPException(400, "rad must be >= 0")
    if cat.strip().lower() not in index.by_cat:
        raise HTTPException(400, f"unknown category; valid: {sorted(index.by_cat)}")
    if link_text is None and os.path.isfile(DEFAULT_LINKS):
        with open(DEFAULT_LINKS, encoding="utf-8-sig", errors="replace") as f:
            link_text = f.read()
    graph = None
    if link_text is not None:
        try:
            graph = index.graph_from_text(link_text)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ids": index.search(lat, lon, cat, rad, graph)}


async def _search(request: Request):
    params = dict(request.query_params)
    ctype = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
        form = await request.form()
        for k, v in form.multi_items():
            params[k] = v
    elif "application/json" in ctype:
        try:
            body = await request.json()
            if isinstance(body, dict):
                params.update(body)
        except Exception:
            raise HTTPException(400, "invalid JSON body")
    raw = params.get("link")
    if isinstance(raw, UploadFile):
        text = (await raw.read()).decode("utf-8-sig", "replace")
    elif isinstance(raw, (list, tuple)):                    # JSON list of [a, b] pairs
        text = "\n".join(f"{p[0]} {p[1]}" for p in raw)
    else:
        try:
            text = await run_in_threadpool(_read_link, raw)
        except Exception as e:
            raise HTTPException(400, f"could not read link: {e}")
    return JSONResponse(await run_in_threadpool(_solve, params, text))



# ---- routes: thin wrappers so that /docs (Swagger) shows real input fields -------------
_CATS = sorted(index.by_cat)
_DESC = {
    "lat": "Current latitude (number)",
    "long": "Current longitude (number)",
    "cat": "Category to search for",
    "rad": "Search radius (circular / Euclidean distance)",
}
_POST_DOC = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["lat", "long", "cat", "rad"],
    "properties": {
        "lat": {"type": "number", "example": 0.5, "description": _DESC["lat"]},
        "long": {"type": "number", "example": 0.5, "description": _DESC["long"]},
        "cat": {"type": "string", "enum": _CATS, "description": _DESC["cat"]},
        "rad": {"type": "number", "example": 0.1, "description": _DESC["rad"]},
        "link": {"type": "string", "format": "binary",
                 "description": "Road file (.txt), one 'a b' pair of location IDs per line. Optional."},
    }}}}}}
_GET_DOC = {"parameters": [
    {"name": "lat", "in": "query", "required": True, "schema": {"type": "number", "example": 0.5}, "description": _DESC["lat"]},
    {"name": "long", "in": "query", "required": True, "schema": {"type": "number", "example": 0.5}, "description": _DESC["long"]},
    {"name": "cat", "in": "query", "required": True, "schema": {"type": "string", "enum": _CATS}, "description": _DESC["cat"]},
    {"name": "rad", "in": "query", "required": True, "schema": {"type": "number", "example": 0.1}, "description": _DESC["rad"]},
    {"name": "link", "in": "query", "required": False, "schema": {"type": "string"},
     "description": "Optional: URL of the road file, or a file name inside the app folder. (To upload a file, use POST.)"},
]}

@app.post("/search/", openapi_extra=_POST_DOC, summary="Search (upload the road file here)")
async def search_post(request: Request):
    return await _search(request)

@app.get("/search/", openapi_extra=_GET_DOC, summary="Search (no file upload)")
async def search_get(request: Request):
    return await _search(request)

@app.api_route("/search", methods=["GET", "POST"], include_in_schema=False)
async def search_noslash(request: Request):
    return await _search(request)

@app.get("/")
def root():
    return {"status": "ok", "n_locations": int(index.n),
            "usage": "POST /search/ with lat, long, cat, rad and link (road file)"}
