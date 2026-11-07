# RESTOCK AI — Retail Inventory Replenishment Intelligence | ONE Google Colab cell
# Never run out. Never overstock.
APP_PASSWORD = ""   # optional: e.g. "exam2026" protects the public link (username: restock). Leave "" for open access.
import subprocess, sys, os, re, zipfile, html, math, datetime, hashlib, warnings
from statistics import NormalDist
warnings.filterwarnings("ignore")
try:
    import gradio, plotly, sklearn, statsmodels
    assert int(gradio.__version__.split(".")[0]) in (4, 5)
except Exception:
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scikit-learn>=1.2", "statsmodels", "scipy", "pandas", "numpy"])
import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

C = dict(fg="#183B35", sg="#C3E8B7", am="#E7B96C", iv="#F6F3EA", ch="#263C37", li="#D8C8F0", co="#D98276", gg="#A5B5AD")
G = {}; TESTS = []; _D = {}
BL = ["", "nan", "none", "n/a", "na", "-", "null", "nat"]
ST = ["Critical Stock", "Stockout Risk", "Reorder Now", "Reorder Soon", "Healthy Stock", "Overstocked", "Slow Moving", "Needs data"]
STC = dict(zip(ST, [C["co"], C["co"], C["am"], C["am"], C["sg"], C["li"], C["gg"], C["gg"]])); GL = dict(zip(ST, ["▲▲", "▲", "●●", "●", "✓", "■", "◇", "?"]))
S0 = dict(sl=.95, review=14, ocm=1.0, hpm=1.0, oc=500.0, hp=22.0, excess=120, slow=1.0, wc=.35, ws=.35, wr=.30, ltscale=1.0)
SC0 = dict(dm=1.0, ltm=1.0, ltadd=0.0, sl=None, ssm=1.0, qm=1.0, moqm=1.0, cm=0.0)
ASOF = pd.Timestamp("2026-11-06"); NDAYS = 395; STORES = ["Hyderabad Central", "Pune Mall", "Chennai Express"]
esc = lambda x: html.escape(str(x)); nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
nf = lambda v: "—" if nul(v) else ("∞" if np.isinf(v) else format(v, ",.0f")); fx = lambda v, d=1: "—" if nul(v) else ("∞" if np.isinf(v) else format(v, "." + str(d) + "f"))
pc = lambda v, d=0: "—" if nul(v) else format(v, "." + str(d) + "%"); dts = lambda v: "—" if nul(v) else pd.Timestamp(v).strftime("%d %b %Y")
def money(v):
    if nul(v) or np.isinf(v): return "—"
    a = abs(v); s = "-" if v < 0 else ""
    return s + "₹" + (format(a / 1e7, ".2f") + "Cr" if a >= 1e7 else format(a / 1e5, ".1f") + "L" if a >= 1e5 else format(a, ",.0f"))
pill = lambda s: "<span class='pill' style='background:%s'>%s %s</span>" % (STC.get(s, C["gg"]), GL.get(s, ""), esc(s))
def note(t, k="info"): return "<div class='note' style='border-left:4px solid %s'>%s</div>" % ({"info": C["gg"], "ok": C["fg"], "warn": C["am"], "err": C["co"], "ai": C["li"]}[k], t)
def card(l, v, d, c=C["fg"]): return "<div class='kc'><div class='kl'>%s</div><div class='kv' style='color:%s'>%s</div><div class='kd'>%s</div></div>" % (l, c, v, d)
SEC = lambda t: "<div class='sec'>%s</div>" % t
def tbl(d, cols, fmt=None):
    fmt = fmt or {}; h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows(): h += "<tr>" + "".join("<td>%s</td>" % (fmt[k](r[k]) if k in fmt else ("—" if nul(r[k]) else esc(r[k]))) for k, _n in cols) + "</tr>"
    return h + "</table></div>"
def sty(f, h=340, t=""):
    f.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["ch"]), height=h, title=dict(text=t, x=.02), margin=dict(l=55, r=25, t=50, b=45), legend=dict(orientation="h", y=-.25), colorway=[C["fg"], C["am"], C["li"], C["co"], C["gg"]]); return f
def empty(m):
    f = go.Figure(); f.add_annotation(text=m, showarrow=False, font=dict(color=C["gg"], size=14)); f.update_xaxes(visible=False); f.update_yaxes(visible=False); return sty(f, 280)
def banner(): return note("<b>Illustrative synthetic data.</b> Fictional retailer, products and suppliers with planted situations; not a real company's performance.", "warn") if G.get("demo") else note("Using your uploaded data.", "info")
def safe(n):
    def deco(fn):
        def w(*a):
            if "X" not in G: m = note("Load data first.")
            else:
                try: return fn(*a)
                except Exception as e: m = note("Could not render this page: " + esc(e), "err")
            return m if n == 1 else (m,) + (None,) * (n - 1)
        return w
    return deco
nm_ = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
def clean_str(s): t = s.astype(str).str.strip(); return t.where(~(s.isna() | t.str.lower().isin(BL)), np.nan)
def to_num(s): return s.astype(float) if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s.astype(str).str.replace(r"[₹$,%\s]", "", regex=True), errors="coerce")
def to_dt(s):
    if pd.api.types.is_datetime64_any_dtype(s): return s
    try: return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception: return pd.to_datetime(s, errors="coerce")
def read_csv_any(f):
    p = f if isinstance(f, str) else getattr(f, "name", f)
    try: d = pd.read_csv(p)
    except Exception: d = pd.read_csv(p, sep=None, engine="python", encoding_errors="replace")
    if d.empty: raise ValueError("A file contains no rows.")
    return d

# ------------------------------ FICTIONAL synthetic retailer (fixed seed) ------------------------------
ITEMS = {"Staples": ["Basmati Rice", "Wheat Atta", "Toor Dal", "Sugar", "Cooking Oil", "Poha"], "Beverages": ["Tea", "Filter Coffee", "Cola", "Mango Drink", "Energy Drink", "Green Tea"], "Snacks": ["Namkeen", "Potato Chips", "Cookies", "Cream Biscuits", "Roasted Peanuts", "Wafers"],
         "Personal Care": ["Shampoo", "Soap", "Toothpaste", "Face Wash", "Body Lotion", "Hair Oil"], "Household": ["Detergent", "Dish Wash", "Floor Cleaner", "Garbage Bags", "Tissue Roll", "Agarbatti"], "Dairy & Packaged": ["Paneer", "Butter", "Ghee", "Cheese Slices", "Instant Noodles", "Pasta"]}
SIZES = ["Small pack", "Standard pack", "Family pack"]
SUPS = [("SUP1", "Sahyadri Foods", 5, .5, .97, 1.00), ("SUP2", "Deccan Distributors", 8, 1, .94, .98), ("SUP3", "Bharat Agro", 12, 2, .90, .95), ("SUP4", "Konkan Beverages", 15, 3, .85, .93), ("SUP5", "Gangetic FMCG", 22, 7, .72, .90), ("SUP6", "Malabar Traders", 10, 1.5, .93, 1.02), ("SUP7", "Metro Wholesale", 28, 5, np.nan, .88)]
def make_data(seed=42):
    r = np.random.default_rng(seed); dates = pd.date_range(ASOF - pd.Timedelta(days=NDAYS - 1), ASOF); T = len(dates); dow = dates.dayofweek.values; doy = dates.dayofyear.values; N = 108; ns = len(STORES)
    typ = np.array(["fast"] * 16 + ["stable"] * 30 + ["seasonal"] * 20 + ["promo"] * 12 + ["slow"] * 30); r.shuffle(typ); share = r.dirichlet([4, 3, 3], N); MU = np.zeros((T, N)); PR = np.zeros((T, N)); prods = []
    for c in ITEMS:
        for it in ITEMS[c]:
            for sz in SIZES: prods.append(("SKU%03d" % (len(prods) + 1), "%s %s" % (it, sz), c))
    skus = [p[0] for p in prods]
    for k in range(N):
        t = typ[k]; base = {"fast": r.uniform(28, 70), "stable": r.uniform(6, 22), "seasonal": r.uniform(5, 18), "promo": r.uniform(8, 24), "slow": r.uniform(.2, .9)}[t]; mu = base * (1 + .2 * (dow >= 5)) * (1 + r.uniform(-.15, .15) * np.linspace(-1, 1, T))
        if t == "seasonal": mu = mu * (1 + .6 * np.cos(2 * np.pi * (doy - int(r.choice([60, 150, 250, 330]))) / 365))
        if t == "promo":
            for a in range(int(r.integers(5, 30)), T - 4, 35): PR[a:a + 4, k] = 1
            mu = mu * (1 + .9 * PR[:, k])
        MU[:, k] = mu
    U = r.poisson(MU[:, :, None] * share[None, :, :] * r.lognormal(0, .2, (T, N, 1))); rate = U[-90:].sum(2).mean(0)
    sales = pd.DataFrame(dict(date=np.repeat(dates.values, N * ns), sku=np.tile(np.repeat(skus, ns), T), store=np.tile(np.tile(STORES, N), T), units=U.reshape(-1), promo=np.repeat(PR, ns, axis=1).reshape(-1).astype(int)))
    prod, sup, inv, inc, cost = [], [], [], [], []; nfast = npo = nst = 0
    for k in range(N):
        t = typ[k]; sku, pn, cat = prods[k]; rt = max(float(rate[k]), .05); c0 = float(np.clip(r.lognormal(4.2, .7), 15, 900)) * (3 if k % 17 == 0 else 1); price = c0 * r.uniform(1.2, 1.5)
        nsup = 1 + int(r.random() < .8) + int(r.random() < .4); first = (k * 5 + k // 6) % 7; others = [int(j) for j in r.permutation(7) if j != first][:nsup - 1]; offs = []
        for j in [first] + others:
            s = SUPS[j]; pack = int(r.choice([1, 6, 12, 24])); moq = pack * int(math.ceil(int(r.choice([12, 24, 48, 100])) / pack)); lt = max(2, int(round(s[2] * r.uniform(.9, 1.1))))
            offs.append(dict(supplier_id=s[0], supplier_name=s[1], sku=sku, lead_time_days=lt, lead_time_sd=round(s[3] * r.uniform(.8, 1.2), 2), moq=moq, case_pack=pack, unit_cost=round(c0 * s[5] * r.uniform(.97, 1.03), 2), on_time_rate=s[4]))
        sup += offs; lt0 = offs[0]["lead_time_days"]; sid = offs[0]["supplier_id"]
        if t == "fast":
            nfast += 1; oh = int(rt * lt0 * r.uniform(.25, .5)) if nfast <= 6 else int(rt * (lt0 + r.uniform(15, 40)))
            if nfast <= 3: npo += 1; inc.append(dict(po_id="PO%04d" % npo, sku=sku, supplier_id=sid, qty=int(rt * lt0 * .6), expected_date=ASOF + pd.Timedelta(days=int(lt0 + r.integers(2, 6)))))
        elif t == "slow": oh = int(max(rt * r.uniform(250, 500), 30))
        elif t == "stable" and nst < 8: nst += 1; oh = int(rt * r.uniform(180, 300))
        else:
            oh = int(rt * (lt0 + r.uniform(10, 45)))
            if t == "stable" and r.random() < .25: npo += 1; inc.append(dict(po_id="PO%04d" % npo, sku=sku, supplier_id=sid, qty=int(rt * lt0 * .8), expected_date=ASOF + pd.Timedelta(days=int(r.integers(2, max(3, lt0))))))
        parts = np.floor(oh * share[k]).astype(int); parts[0] += oh - parts.sum()
        for si, s in enumerate(STORES): inv.append(dict(sku=sku, location=s, on_hand=int(parts[si]), allocated=int(parts[si] * .04)))
        prod.append(dict(sku=sku, product_name=pn, category=cat, unit_cost=round(c0, 2), unit_price=round(price, 2), importance=3 if t == "fast" else int(r.integers(1, 4)))); cost.append(dict(sku=sku, ordering_cost=int(r.integers(300, 900)), holding_pct=float(r.choice([18, 20, 22, 25]))))
    return dict(sales=sales, prod=pd.DataFrame(prod), inv=pd.DataFrame(inv), sup=pd.DataFrame(sup), inc=pd.DataFrame(inc), cost=pd.DataFrame(cost))
def demo_raw(seed=42):
    if ("r", seed) not in _D: _D[("r", seed)] = make_data(seed)
    return _D[("r", seed)]

# ------------------------------ ingestion and validation ------------------------------
def _sp(s):
    o = {}
    for it in s.split(","):
        p = it.strip().split(":"); o[p[0]] = (p[1], p[2] == "1", p[3].split("|") if len(p) > 3 and p[3] else [])
    return o
SPEC = dict(prod=_sp("sku:s:1:sku_id|item_id,product_name:s:0:name,category:s:0,unit_cost:n:1:cost,unit_price:n:0:price,importance:n:0"), sales=_sp("date:d:1,sku:s:1:sku_id,units:n:1:units_sold|quantity,store:s:0:location,promo:n:0:promotion"), inv=_sp("sku:s:1,location:s:0:store,on_hand:n:1:on_hand_qty|stock,allocated:n:0:reserved"),
 sup=_sp("supplier_id:s:1:supplier,supplier_name:s:0,sku:s:1,lead_time_days:n:1:lead_time,lead_time_sd:n:0,moq:n:0:min_order_qty,case_pack:n:0:pack_size,unit_cost:n:1:cost,on_time_rate:n:0:on_time"), inc=_sp("po_id:s:0,sku:s:1,supplier_id:s:0,qty:n:1:quantity,expected_date:d:1:eta"), cost=_sp("sku:s:1,ordering_cost:n:0,holding_pct:n:0:holding_cost_pct"))
TN = dict(sales="sales_history", prod="product_master", inv="current_inventory", sup="supplier_info", inc="incoming_purchase_orders", cost="product_costs"); REQ = ("sales", "prod", "inv", "sup")
KEY = dict(prod=["sku"], sales=["date", "sku", "store"], inv=["sku", "location"], sup=["supplier_id", "sku"], inc=["po_id", "sku"], cost=["sku"])
RNG = dict(sup=[("lead_time_days", lambda s: s <= 0, "lead time must be above 0"), ("unit_cost", lambda s: s <= 0, "unit_cost must be above 0"), ("on_time_rate", lambda s: s > 100, "on_time_rate above 100")], prod=[("unit_cost", lambda s: s <= 0, "unit_cost must be above 0")], cost=[("holding_pct", lambda s: s > 100, "holding_pct above 100")])
def parse_map(text):
    ov = {}
    for p in re.split(r"[,;\n]", text or ""):
        if "=" in p: k, v = p.split("=", 1); ov[nm_(k)] = nm_(v)
    return ov
def map_cols(raw, t, ov, iss):
    d = raw.copy(); d.columns = [nm_(c) for c in d.columns]; ren = {}
    for f, (k, req, al) in SPEC[t].items():
        c = ov.get(t + "_" + f) or ov.get(f); c = c if c in d.columns else next((a for a in [f] + [nm_(x) for x in al] if a in d.columns and a not in ren), None)
        if c is None or c in ren:
            if req: iss.append(("error", "%s: required field '%s' not found. Columns found: %s. Use the mapping box, e.g. %s=your_column." % (TN[t], f, ", ".join(map(str, raw.columns[:12])), f)))
            continue
        if c != f and f in d.columns: d = d.drop(columns=[f])
        ren[c] = f
    d = d.rename(columns=ren); return d.loc[:, ~d.columns.duplicated()]
def ctab(raw, t, ov, iss, ex):
    d = map_cols(raw, t, ov, iss)
    if any(l == "error" for l, _ in iss): return d
    d = d.copy(); d["_row"] = d.index + 2; bad = pd.Series(False, index=d.index)
    for f, (k, req, al) in SPEC[t].items():
        if f not in d.columns: d[f] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]") if k == "d" else np.nan
        elif k == "n": d[f] = to_num(d[f])
        elif k == "d": d[f] = to_dt(d[f])
        else: d[f] = clean_str(d[f])
    if "sku" in d: d["sku"] = d["sku"].str.upper().str.replace(r"\s+", "", regex=True)
    def flag(m, why):
        nonlocal bad
        m = m.fillna(False) & ~bad
        if m.any(): ex.append((TN[t], why, int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: %s (rows %s)" % (TN[t], m.sum(), why, ", ".join(str(int(i)) for i in d["_row"][m].values[:6])))); bad = bad | m
    for f, (k, req, al) in SPEC[t].items():
        if req: flag(d[f].isna(), "missing or invalid " + f)
        if k == "n": flag(d[f] < 0, "negative " + f)
    for f, fn, why in RNG.get(t, []): flag(fn(d[f]), why)
    flag(d.duplicated(KEY[t], keep="first") & d[KEY[t]].notna().all(axis=1), "duplicate " + "+".join(KEY[t])); return d[~bad].drop(columns="_row").reset_index(drop=True)
_ctr = [0]
def load_all(raw, mapping=""):
    iss, ex, ov, D = [], [], parse_map(mapping), {}; rows_in = {t: len(raw[t]) for t in SPEC if raw.get(t) is not None}
    for t in REQ:
        if raw.get(t) is None: iss.append(("error", "Missing required table: %s." % TN[t]))
    if any(l == "error" for l, _ in iss): raise ValueError(" | ".join(m for l, m in iss if l == "error"))
    for t in rows_in: D[t] = ctab(raw[t], t, ov, iss, ex)
    errs = [m for l, m in iss if l == "error"]
    if errs: raise ValueError(" | ".join(errs))
    for t in ("inc", "cost"):
        if t not in D: D[t] = pd.DataFrame({f: pd.Series(dtype="datetime64[ns]" if k == "d" else float if k == "n" else object) for f, (k, _, _) in SPEC[t].items()}); iss.append(("info", "No %s table: %s." % (TN[t], "no incoming orders are assumed" if t == "inc" else "default ordering and holding costs from Settings are used (labelled assumption)")))
    for t in ("prod", "sales"):
        if D[t].empty: raise ValueError("No valid rows remain in %s after validation." % TN[t])
    ids = set(D["prod"]["sku"])
    for t in ("sales", "inv", "sup", "inc", "cost"):
        m = ~D[t]["sku"].isin(ids)
        if m.any(): ex.append((TN[t], "SKU not in product master", int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: SKU not found in the product master (%s)." % (TN[t], m.sum(), ", ".join(D[t]["sku"][m].unique()[:5])))); D[t] = D[t][~m].reset_index(drop=True)
    s = D["sup"]
    if s["on_time_rate"].notna().any() and s["on_time_rate"].max() > 1.5: s["on_time_rate"] = s["on_time_rate"] / 100
    if s["on_time_rate"].isna().any(): iss.append(("info", "%d supplier offers have no on_time_rate: reliability is shown as unavailable, never invented." % s["on_time_rate"].isna().sum()))
    if s["lead_time_sd"].isna().any(): iss.append(("info", "%d supplier offers have no lead_time_sd: lead-time variability is unknown (treated as 0 and flagged lower confidence)." % s["lead_time_sd"].isna().sum()))
    if s["moq"].isna().any() or s["case_pack"].isna().any(): iss.append(("info", "Missing MOQ / case pack are treated as no constraint (MOQ 0, pack 1): an explicit assumption."))
    iv = D["inv"]
    if iv["allocated"].isna().any(): iss.append(("info", "Missing allocated quantities are treated as 0 (assumption)."))
    inc = D["inc"]; bi = ~inc["supplier_id"].isna() & ~inc["supplier_id"].isin(s["supplier_id"])
    if bi.any(): iss.append(("warn", "%d incoming orders reference suppliers not in the supplier table (kept)." % bi.sum()))
    P = D["prod"].reset_index(drop=True); P["category"] = P["category"].fillna("Uncategorised"); P["product_name"] = P["product_name"].fillna(P["sku"])
    if P["unit_price"].isna().any(): iss.append(("info", "Missing unit_price: set to 1.3 × unit_cost for revenue-at-risk only (assumption)."))
    P["unit_price"] = P["unit_price"].fillna(P["unit_cost"] * 1.3); P["importance"] = P["importance"].fillna(2).clip(1, 3); D["prod"] = P; skus = list(P["sku"])
    sl = D["sales"]; sl["date"] = sl["date"].dt.normalize(); asof = sl["date"].max(); dates = pd.date_range(sl["date"].min(), asof)
    M = sl.groupby(["date", "sku"])["units"].sum().unstack("sku").reindex(index=dates, columns=skus); fill = float(M.isna().mean().mean())
    if fill > .001: iss.append(("info", "%.1f%% of SKU-days had no sales row and were treated as zero demand (assumption; verify your extract is not missing days)." % (100 * fill)))
    D["M"] = M.fillna(0.0).values.astype(float); D["dates"] = dates; D["asof"] = asof; D["skus"] = skus; D["catc"] = pd.factorize(P["category"])[0]
    if len(dates) < 120: iss.append(("warn", "Only %d days of sales history: forecasting falls back to a baseline and results are unreliable." % len(dates)))
    D.update(issues=iss, excluded=ex, rows_in=rows_in); _ctr[0] += 1; D["key"] = _ctr[0]; return D

# ------------------------------ forecasting (chronological backtest) ------------------------------
HS = [1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 45, 60, 75, 90]; FE = ["r7", "r91", "rsd", "rdm", "dow", "mon", "h", "cat", "lsc"]; MN = ["Naive (seasonal, last week)", "Moving average (28d)", "Holt-Winters (weekly)", "Gradient boosting (ML)"]
def fc_hw(y, h):
    y = np.asarray(y, float)
    if y.sum() <= 0: return np.zeros(h)
    try:
        f = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7, initialization_method="estimated").fit(optimized=True).forecast(h)
        if not np.all(np.isfinite(f)): raise ValueError("x")
        return np.maximum(f, 0)
    except Exception: return np.full(h, y[-28:].mean())
def mlrows(M, dates, catc, origins, hs, lim=None):
    T, N = M.shape; z = np.zeros((1, N)); cs = np.vstack([z, np.cumsum(M, 0)]); cs2 = np.vstack([z, np.cumsum(M ** 2, 0)]); out = []; mj = -1
    for i in origins:
        m7 = (cs[i + 1] - cs[i - 6]) / 7; m28 = (cs[i + 1] - cs[i - 27]) / 28; m91 = (cs[i + 1] - cs[i - 90]) / 91; sd = np.sqrt(np.maximum((cs2[i + 1] - cs2[i - 27]) / 28 - m28 ** 2, 0)); sc = m28 + .5
        for h in hs:
            j = i + h
            if lim is not None and j >= lim: continue
            k0 = -(-h // 7); ids = j - 7 * (k0 + np.arange(8)); ids = ids[ids >= 0]; dm = M[ids].mean(0) if len(ids) else m28; d = dates[i] + pd.Timedelta(days=h)
            out.append(pd.DataFrame(dict(r7=m7 / sc, r91=m91 / sc, rsd=sd / sc, rdm=dm / sc, dow=d.dayofweek, mon=d.month, h=h, cat=catc, lsc=np.log(sc), y=(M[j] / sc if j < T else np.nan), sc=sc)))
            if j < T: mj = max(mj, j)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), mj
def wape_cols(act, pred):
    den = act.sum(0); num = np.abs(pred - act).sum(0); return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
def snaive(M, t0, h): return np.stack([M[t0 - 7 + (d % 7)] for d in range(h)])
def run_forecast(M, dates, catc):
    T, N = M.shape; nzc = M > 0; first = np.where(nzc.any(0), nzc.argmax(0), T); hd = T - first; msg = []; ma = lambda t0, h: np.tile(M[t0 - 28:t0].mean(0), (h, 1))
    if T < 120: return dict(F=ma(T, 90), method=np.array([MN[1] + " (baseline: history too short)"] * N, dtype=object), wape=np.full(N, np.nan), perf=pd.DataFrame(), bt={}, act=None, cut=None, split=None, msg=["History is %d days: moving-average baseline for every SKU, no backtest." % T], hd=hd, names=MN)
    c = T - 28; act = M[c:]; bt = {MN[0]: snaive(M, c, 28), MN[1]: ma(c, 28), MN[2]: np.stack([fc_hw(M[:c, k], 28) for k in range(N)], axis=1)}; mj = -1; ml_ok = c >= 250; mdl_full = None
    if ml_ok:
        tr, mj = mlrows(M, dates, catc, range(90, c - 1, 7), HS, lim=c); te, _ = mlrows(M, dates, catc, [c - 1], range(1, 29)); mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=.08, max_depth=5, random_state=0).fit(tr[FE], tr["y"]); bt[MN[3]] = (np.maximum(mdl.predict(te[FE]), 0) * te["sc"].values).reshape(28, N)
    else: msg.append("ML disabled: needs at least 250 days of history before the backtest window.")
    names = list(bt); W = np.vstack([wape_cols(act, bt[n]) for n in names]); Wf = np.where(np.isnan(W), np.inf, W); best = Wf.argmin(0); allnan = np.isinf(Wf.min(0)); best = np.where(allnan, 1, best); base = hd < 90; best = np.where(base, 1, best)
    perf = pd.DataFrame([dict(method=n, wape=float(np.abs(bt[n] - act).sum() / max(act.sum(), 1e-9)), mae=float(np.abs(bt[n] - act).mean()), rmse=float(np.sqrt(((bt[n] - act) ** 2).mean())), chosen=int((best == i).sum())) for i, n in enumerate(names)])
    if ml_ok:
        tr2, _ = mlrows(M, dates, catc, range(90, T - 1, 7), HS, lim=T); te2, _ = mlrows(M, dates, catc, [T - 1], range(1, 91)); mdl_full = HistGradientBoostingRegressor(max_iter=150, learning_rate=.08, max_depth=5, random_state=0).fit(tr2[FE], tr2["y"]); MLF = (np.maximum(mdl_full.predict(te2[FE]), 0) * te2["sc"].values).reshape(90, N)
    F = np.zeros((90, N)); nv = snaive(M, T, 90); mv = ma(T, 90)
    for k in range(N):
        n = names[best[k]]; F[:, k] = nv[:, k] if n == MN[0] else mv[:, k] if n == MN[1] else fc_hw(M[:, k], 90) if n == MN[2] else MLF[:, k]
    method = np.array([names[b] + (" (baseline: short history)" if base[k] else " (baseline: no demand in backtest)" if allnan[k] and not base[k] else "") for k, b in enumerate(best)], dtype=object)
    return dict(F=F, method=method, wape=np.where(allnan | base, np.nan, Wf.min(0)), perf=perf, bt=bt, act=act, cut=c, split=(mj, c), msg=msg, hd=hd, names=names)
def get_fc(D):
    k = ("fc", D["key"])
    if k not in _D: _D[k] = run_forecast(D["M"], D["dates"], D["catc"])
    return _D[k]

# ------------------------------ replenishment engine ------------------------------
zval = lambda sl: NormalDist().inv_cdf(min(max(sl, .5), .9999))
def rnd(q, moq, pack):
    q = np.asarray(q, float); pk = np.maximum(np.nan_to_num(np.asarray(pack, float), nan=1.0), 1.0); mq = np.maximum(np.nan_to_num(np.asarray(moq, float), nan=0.0), 0.0); r = np.where(q > 1e-9, np.ceil(q / pk - 1e-9) * pk, 0.0); return np.where((r > 0) & (r < mq), np.ceil(mq / pk) * pk, r)
def eoq(D, S_, H_): return np.where((D > 0) & (H_ > 0) & (S_ >= 0), np.sqrt(2 * D * S_ / np.where(H_ > 0, H_, 1)), 0.0)
def score_offers(g, urgent, S):
    g = g.copy(); eff = g["lead_time_days"] + g["lead_time_sd"].fillna(0); cs = g["unit_cost"].min() / g["unit_cost"]; sp = eff.min() / eff; ws = S["ws"] * (2 if urgent else 1); rel = g["on_time_rate"]
    num = S["wc"] * cs + ws * sp + np.where(rel.notna(), S["wr"] * rel.fillna(0), 0.0); den = S["wc"] + ws + np.where(rel.notna(), S["wr"], 0.0); g["score"] = num / np.maximum(den, 1e-9); g["urgent"] = urgent; g["cheapest"] = g["unit_cost"] == g["unit_cost"].min(); g["cost_idx"] = g["unit_cost"] / g["unit_cost"].min(); return g
def plan(X, S, sc=None, explain=False):
    sc = dict(SC0, **(sc or {})); B = X["B"].copy(); N = len(B); F = X["F"] * sc["dm"]; CF = np.cumsum(F, 0); ar = np.arange(N); rate = F[:30].mean(0); sd = B["sd"].values.astype(float) * sc["dm"]
    lt = np.maximum(B["lt"].values.astype(float) * S["ltscale"] * sc["ltm"] + sc["ltadd"], 0); sdlt = np.nan_to_num(B["sdlt"].values.astype(float)) * S["ltscale"] * sc["ltm"]; z = zval(sc["sl"] if sc["sl"] is not None else S["sl"]); valid = ~B["miss"].values.astype(bool); lt = np.where(valid, lt, 0.0)
    idx = np.clip(np.ceil(lt).astype(int), 1, 90); ltd = np.where(lt > 0, CF[idx - 1, ar] + np.maximum(lt - 90, 0) * rate, 0.0); ss = z * np.sqrt(np.maximum(lt * sd ** 2 + rate ** 2 * sdlt ** 2, 0)) * sc["ssm"]; rop = ltd + ss; tgt = rop + rate * S["review"]
    oh = np.nan_to_num(B["on_hand"].values.astype(float)); al = B["alloc"].values.astype(float); inc = B["inc"].values.astype(float); sell = np.maximum(oh - al, 0); pos = oh - al + inc
    PS = sell[None, :] + X["ARRc"] - CF; PP = pos[None, :] - CF; neg = PS < -1e-6; so = np.where(neg.any(0) & (F.sum(0) > 0), neg.argmax(0) + 1, np.inf); mask = np.arange(1, 91)[:, None] <= idx[None, :]; short = np.where(rate > 0, np.maximum(0, -np.where(mask, PS, np.inf).min(0)), 0.0)
    hp_ = PP <= rop[None, :] + 1e-9; rday = np.where(pos <= rop + 1e-9, 0, np.where(hp_.any(0), hp_.argmax(0) + 1, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"): dos = np.where(rate > 0, sell / rate, np.inf); dosp = np.where(rate > 0, np.maximum(pos, 0) / rate, np.inf)
    qty = rnd(np.where(valid & (pos <= rop + 1e-9), np.maximum(tgt - pos, 0), 0.0) * sc["qm"], B["moq"].values * sc["moqm"], B["pack"].values)
    crit = valid & (rate > 0) & ((sell <= 0) | (so <= 3)); risk = valid & (rate > 0) & ~crit & (so < np.maximum(lt, 1)); now = valid & ~crit & ~risk & (pos <= rop + 1e-9) & (qty > 0); soon = valid & (rate > 0) & ~crit & ~risk & ~now & (pos - rop <= rate * S["review"])
    slow = valid & ~(crit | risk | now | soon) & (rate < S["slow"]); over = valid & ~(crit | risk | now | soon | slow) & (dosp > S["excess"]); st = np.select([~valid, crit, risk, now, soon, slow, over], [ST[7], ST[0], ST[1], ST[2], ST[3], ST[6], ST[5]], default=ST[4])
    P = B.copy(); P["rate"] = rate; P["sd"] = sd; P["lt"] = lt; P["ltd"] = ltd; P["ss"] = ss; P["rop"] = rop; P["tgt"] = tgt; P["pos"] = pos; P["sell"] = sell; P["dos"] = dos; P["dosp"] = dosp; P["so"] = so; P["short"] = short; P["rday"] = rday; P["qty"] = np.where(valid, qty, 0.0); P["state"] = st
    for h in (7, 14, 30, 90): P["f%d" % h] = F[:h].sum(0)
    P["oc_e"] = B["oc"].values * S["ocm"]; P["hp_e"] = B["hp"].values * S["hpm"]; P["est_cost"] = P["qty"].values * B["pcost"].values * (1 + sc["cm"] / 100); P["inv_value"] = np.maximum(oh, 0) * B["cost"].values; P["hold_cost"] = P["inv_value"].values * P["hp_e"].values / 100
    P["excess_units"] = np.where(valid, np.where(rate > 0, np.maximum(sell - rate * S["excess"], 0), sell), 0.0); P["excess_value"] = P["excess_units"].values * B["cost"].values; P["expo"] = short * B["price"].values; P["cv"] = np.where(rate > 0, sd / np.where(rate > 0, rate, 1), np.nan); P["turn"] = np.where(P["inv_value"].values > 0, rate * 365 * B["cost"].values / np.where(P["inv_value"].values > 0, P["inv_value"].values, 1), np.nan)
    rc = np.select([st == ST[0], st == ST[1], st == ST[2], st == ST[3], np.isin(st, [ST[5], ST[6]])], [1, .85, .65, .35, .1], 0.0); mx = lambda a: a / max(np.nanmax(np.where(np.isfinite(a), a, np.nan)) if np.isfinite(a).any() else 1, 1e-9)
    P["pscore"] = 100 * (rc > 0) * (.35 * rc + .20 * mx(rate * B["price"].values) + .15 * mx(lt) + .10 * B["imp"].values / 3 + .20 * mx(P["expo"].values)); P["priority"] = np.select([P["pscore"].values >= 55, P["pscore"].values >= 30, P["pscore"].values > 0], ["High", "Medium", "Low"], "—")
    asof = X["asof"]; P["reorder_date"] = [asof + pd.Timedelta(days=int(max(d, 1))) if np.isfinite(d) else pd.NaT for d in rday]; P["arrival"] = [rd + pd.Timedelta(days=int(math.ceil(l))) if (q > 0 and pd.notna(rd)) else pd.NaT for rd, l, q in zip(P["reorder_date"], lt, P["qty"])]
    unk = B["lt_src"].str.contains("unknown").values; low = (B["hdays"].values < 90) | (B["wape"].values > .6) | B["miss"].values.astype(bool) | unk; P["conf"] = np.where(low, "Low", np.where((B["hdays"].values >= 365) & (B["wape"].values <= .35), "High", "Medium"))
    if explain: P["expl"] = [expl(r, S) for r in P.to_dict("records")]
    return P
def expl(r, S):
    h = "%s (%s)" % (r["sku"], r["pname"]); s = r["state"]
    if s == ST[7]: return "%s: needs data (%s), so no recommendation is made and no value is assumed." % (h, r["why"])
    ev = "Evidence: available %s, allocated %s, incoming %s, position %s; forecast %s/day; lead time %s d (%s); safety stock %s; reorder point %s." % (nf(r["sell"]), nf(r["alloc"]), nf(r["inc"]), nf(r["pos"]), fx(r["rate"], 1), fx(r["lt"], 0), r["lt_src"], nf(r["ss"]), nf(r["rop"]))
    if s in (ST[0], ST[1]): w = "projected to run out in %s days%s; a new order placed today cannot arrive for %s days. About %s of sales is at risk" % (fx(r["so"], 0), " (available stock is already %s)" % nf(r["sell"]) if r["sell"] <= 0 else "", fx(r["lt"], 0), money(r["expo"])); n = "Order %s units now from %s and chase any open PO." % (nf(r["qty"]), r["sname"])
    elif s == ST[2]: w = "inventory position %s is at or below the reorder point %s" % (nf(r["pos"]), nf(r["rop"])); n = "Order %s units from %s (arrives about %s)." % (nf(r["qty"]), r["sname"], dts(r["arrival"]))
    elif s == ST[3]: w = "position %s will reach the reorder point %s in about %s days" % (nf(r["pos"]), nf(r["rop"]), fx(r["rday"], 0)); n = "Plan an order on %s; no order yet." % dts(r["reorder_date"])
    elif s == ST[5]: w = "stock covers %s days, above the %d-day limit; excess value %s" % (fx(r["dosp"], 0), S["excess"], money(r["excess_value"])); n = "Do not reorder; review promotions or transfers."
    elif s == ST[6]: w = "demand is only %s units/day, below the %s threshold, with %s days of stock" % (fx(r["rate"], 2), fx(S["slow"], 1), fx(r["dosp"], 0)); n = "Keep orders minimal; review range."
    else: w = "stock is within policy (cover %s days)" % fx(r["dosp"], 0); n = "No order needed."
    return "%s: %s. %s %s Confidence: %s." % (h, w, n, ev, r["conf"])
def mkarr(N, incs):
    A = np.zeros((90, N)); tot = np.zeros(N)
    for i, d, q in incs:
        tot[i] += q
        if d <= 90: A[max(int(d), 1) - 1, i] += q
    return A, tot
def analyze(D, S, fc):
    P_ = D["prod"]; skus = D["skus"]; N = len(skus); ix = {s: i for i, s in enumerate(skus)}; M = D["M"]; asof = D["asof"]; T = M.shape[0]; F = fc["F"]; inv = D["inv"]
    oh = inv.groupby("sku")["on_hand"].sum().reindex(skus); al = inv.groupby("sku")["allocated"].sum().reindex(skus).fillna(0.0); locs = {}
    for r in inv.itertuples():
        lc = r.location if isinstance(r.location, str) else "All"; locs.setdefault(r.sku, {}); locs[r.sku][lc] = locs[r.sku].get(lc, 0) + r.on_hand
    incs = []; over = np.zeros(N, bool)
    for r in D["inc"].itertuples():
        off = (r.expected_date.normalize() - asof).days
        if off <= 0: off = 1; over[ix[r.sku]] = True
        incs.append((ix[r.sku], off, r.qty))
    ARR, itot = mkarr(N, incs); ARRc = np.cumsum(ARR, 0); sell0 = np.maximum(oh.fillna(0).values - al.values, 0); PS0 = sell0[None, :] + ARRc - np.cumsum(F, 0); ng = PS0 < -1e-6; so0 = np.where(ng.any(0) & (F.sum(0) > 0), ng.argmax(0) + 1, np.inf)
    gb = dict(tuple(D["sup"].groupby("sku"))); rows = []; offers = []; cst = D["cost"].set_index("sku") if len(D["cost"]) else pd.DataFrame()
    nzc = M > 0; first = np.where(nzc.any(0), nzc.argmax(0), T); lastsale = np.where(nzc.any(0), np.argmax(nzc[::-1], 0), T); sd90 = M[-90:].std(0, ddof=1) if T > 2 else np.zeros(N)
    for i, s in enumerate(skus):
        g = gb.get(s); pr = P_.iloc[i]; oc = cst["ordering_cost"].get(s, np.nan) if len(cst) else np.nan; hp = cst["holding_pct"].get(s, np.nan) if len(cst) else np.nan; why = []; b = None
        if g is None: why.append("no supplier offer")
        else:
            g = g.copy(); g["moq"] = g["moq"].fillna(0); g["case_pack"] = g["case_pack"].fillna(1).clip(lower=1); g = score_offers(g, bool(so0[i] <= g["lead_time_days"].max()), S); b = g.loc[g["score"].idxmax()]; g["chosen"] = g.index == g["score"].idxmax(); g["pname"] = pr["product_name"]; offers.append(g)
        if nul(oh.iloc[i]): why.append("no inventory record")
        rows.append(dict(sku=s, pname=pr["product_name"], cat=pr["category"], sup=b["supplier_id"] if b is not None else "—", sname=(b["supplier_name"] if b is not None and not nul(b["supplier_name"]) else (b["supplier_id"] if b is not None else "—")), cost=pr["unit_cost"], pcost=b["unit_cost"] if b is not None else np.nan, price=pr["unit_price"], moq=b["moq"] if b is not None else 0.0, pack=b["case_pack"] if b is not None else 1.0,
                         lt=float(b["lead_time_days"]) if b is not None else np.nan, sdlt=b["lead_time_sd"] if b is not None else np.nan, lt_src=("stated; SD supplied" if b is not None and not nul(b["lead_time_sd"]) else "stated; SD unknown" if b is not None else "missing"), rel=b["on_time_rate"] if b is not None else np.nan, on_hand=oh.iloc[i], alloc=al.iloc[i], inc=itot[i], sd=sd90[i], hdays=T - first[i], nz=int(nzc[:, i].sum()), last_sale=lastsale[i],
                         hp=hp if not nul(hp) else S["hp"], oc=oc if not nul(oc) else S["oc"], cost_src="supplied" if not (nul(hp) or nul(oc)) else "default (assumption)", imp=pr["importance"], wape=fc["wape"][i], method=fc["method"][i], miss=bool((b is None) or nul(oh.iloc[i])), why="; ".join(why), overdue=bool(over[i])))
    B = pd.DataFrame(rows); X = dict(B=B, F=F, ARR=ARR, ARRc=ARRc, asof=asof, skus=skus, locs=locs, S=S, fc=fc, D=D, offers=pd.concat(offers, ignore_index=True) if offers else pd.DataFrame()); X["P"] = plan(X, S, None, True); return X
def demo_D(seed=42):
    if ("d", seed) not in _D: _D[("d", seed)] = load_all(demo_raw(seed))
    return _D[("d", seed)]
def demo_X(seed=42, S=None):
    S = dict(S or S0); k = ("X", seed, tuple(S.values()))
    if k not in _D: D = demo_D(seed); _D[k] = analyze(D, S, get_fc(D))
    return _D[k]

# ------------------------------ what-if simulation (common random numbers) ------------------------------
def simulate(X, S, pol, world, H=90, seed=7, adapt=False):
    B = X["B"]; N = len(B); rng = np.random.default_rng(seed); Uu = rng.random((H, N)); Zg = rng.standard_normal((H, N)); Zl = rng.standard_normal((H, N)); pl = dict(pol)
    if adapt: pl.update(dm=world["dm"], ltadd=world["ltadd"])
    Pp = plan(X, S, pl); rop = np.nan_to_num(Pp["rop"].values); tgt = np.nan_to_num(Pp["tgt"].values); valid = (Pp["state"].values != ST[7]); qm = pl.get("qm", 1.0); moq = B["moq"].values * pl.get("moqm", 1.0); pack = B["pack"].values
    rate = X["F"][:30].mean(0) * world["dm"]; rel = np.where(rate > 0, np.clip(B["sd"].values * world["dm"] / np.where(rate > 0, rate, 1), 0, 2.5), 0.0); ltw = np.maximum(np.nan_to_num(B["lt"].values) * S["ltscale"] * world["ltm"] + world["ltadd"], 1); sdl = np.nan_to_num(B["sdlt"].values) * S["ltscale"]
    cost = np.nan_to_num(B["cost"].values); pc_ = np.nan_to_num(B["pcost"].values) * (1 + world["cm"] / 100); price = B["price"].values; hc = cost * B["hp"].values * S["hpm"] / 100 / 365; dis = world["dis"] / 100; surge = np.ones(H)
    s0 = int(world["ps"]); surge[s0:s0 + int(world["pl"])] = 1 + world["psg"] / 100; Ao = np.zeros((H + 130, N)); Ad = np.zeros((H + 130, N))
    for t in range(90):
        sh = t + int(world["ltadd"])
        if sh < H + 130: Ao[sh] += X["ARR"][t]; Ad[sh] += X["ARR"][t] * (1 - dis)
    bal = np.maximum(np.nan_to_num(B["on_hand"].values) - B["alloc"].values, 0.0); bal0 = bal.copy(); pipe = B["inc"].values.astype(float).copy(); rec = np.zeros(N); sold_t = np.zeros(N); lost = np.zeros(N); dem_t = 0.0; norders = 0; spend = 0.0; hold = 0.0; invv = 0.0
    for t in range(H):
        bal = bal + Ad[t]; rec += Ad[t]; pipe = pipe - Ao[t]; pos = bal + pipe; q = rnd(np.where(valid & (pos <= rop + 1e-9), np.maximum(tgt - pos, 0) * qm, 0.0), moq, pack); ix = np.nonzero(q > 0)[0]
        if len(ix): L = np.clip(np.rint(ltw + sdl * Zl[t]), 1, 120).astype(int); np.add.at(Ao, (t + L[ix], ix), q[ix]); np.add.at(Ad, (t + L[ix], ix), q[ix] * (1 - dis)); pipe[ix] += q[ix]; norders += len(ix); spend += float((q[ix] * (1 - dis) * pc_[ix]).sum())
        lam = np.maximum(rate * surge[t] * np.exp(rel * Zg[t] - rel ** 2 / 2), 1e-9); dem = poisson.ppf(Uu[t], lam); sold = np.minimum(bal, dem); lost += dem - sold; bal = bal - sold; sold_t += sold; dem_t += float(dem.sum()); hold += float(bal @ hc); invv += float(bal @ cost)
    ok = bool(np.allclose(bal0 + rec - sold_t, bal, atol=1e-6)); exc = float((np.maximum(bal - rate * S["excess"], 0) * cost).sum())
    return dict(fill=1 - lost.sum() / dem_t if dem_t > 0 else np.nan, so_skus=int((lost > 0).sum()), lost_rev=float((lost * price).sum()), hold=hold, spend=spend, excess=exc, orders=norders, wc=invv / H, end_inv=float(bal @ cost), ok=ok)

# ------------------------------ pages ------------------------------
def fl(P, c, s, l, p, st):
    m = pd.Series(True, index=P.index)
    if c: m &= P["cat"].isin(c)
    if s: m &= P["sup"].isin(s)
    if p: m &= P["priority"].isin(p)
    if st: m &= P["state"].isin(st)
    if l: m &= P["sku"].map(lambda k: any(v > 0 for q, v in G["X"]["locs"].get(k, {}).items() if q in l))
    return P[m]
def getP(c, s, l, p, st):
    P = fl(G["X"]["P"], c, s, l, p, st)
    if P.empty: raise ValueError("No SKUs match the filters.")
    return P
@safe(5)
def r_ov(c, s, l, p, st):
    X = G["X"]; P = getP(c, s, l, p, st); n = len(P); cnt = P["state"].value_counts(); imm = int(P["state"].isin(ST[:3]).sum()); atr = int(P["state"].isin(ST[:2]).sum()); T = X["D"]["M"].shape[0]
    h = banner() + "<div class='kg'>" + card("Active SKUs", nf(n), "with a product record") + card("Inventory units", nf(P["on_hand"].sum()), "on hand across locations") + card("Inventory valuation", money(P["inv_value"].sum()), "on hand × unit cost") + card("Annual holding cost", money(P["hold_cost"].sum()), "valuation × holding %", C["am"]) + card("Need replenishment now", nf(imm), "▲ critical, stockout risk or reorder now", C["co"]) + \
        card("Approaching reorder point", nf(cnt.get(ST[3], 0)), "● reorder soon", C["am"]) + card("Stockout-risk SKUs", nf(atr), "will run out before a new order could arrive", C["co"]) + card("Overstocked / slow", "%d / %d" % (cnt.get(ST[5], 0), cnt.get(ST[6], 0)), "excess value %s" % money(P["excess_value"].sum()), C["li"]) + card("Procurement requirement", money(P["est_cost"].sum()), "recommended orders at supplier cost") + card("Inventory health", pc(cnt.get(ST[4], 0) / n), "share of SKUs in healthy stock", C["fg"]) + "</div>"
    h += note("All figures are computed from the active data (%d days of history to %s). Recommendations are proposals for buyers; no order is placed with any supplier." % (T, dts(X["asof"])), "info"); top = P[P["pscore"] > 0].sort_values("pscore", ascending=False).head(10)
    h += SEC("Top products requiring attention") + tbl(top, [("sku", "SKU"), ("pname", "Product"), ("state", "Status"), ("priority", "Priority"), ("qty", "Order qty"), ("sname", "Supplier"), ("expl", "Why and evidence")], dict(state=pill, qty=nf)); rec = P[P["qty"] > 0].sort_values("reorder_date").head(8)
    h += SEC("Recent replenishment recommendations") + (tbl(rec, [("sku", "SKU"), ("pname", "Product"), ("qty", "Qty"), ("sname", "Supplier"), ("reorder_date", "Order on"), ("arrival", "Expected arrival"), ("est_cost", "Est. cost")], dict(qty=nf, reorder_date=dts, arrival=dts, est_cost=money)) if len(rec) else note("No orders recommended in this view.", "ok"))
    f1 = sty(go.Figure(go.Bar(x=[GL[x] + " " + x for x in ST], y=[int(cnt.get(x, 0)) for x in ST], marker_color=[STC[x] for x in ST])), 330, "SKUs by inventory status"); f1.update_xaxes(tickangle=-25); g = P.groupby("cat")["inv_value"].sum().sort_values(); f2 = sty(go.Figure(go.Bar(y=g.index, x=g.values, orientation="h", marker_color=C["fg"])), 320, "Inventory value by category (₹)")
    ix = [X["skus"].index(k) for k in P["sku"]]; M = X["D"]["M"][:, ix].sum(1); w = pd.Series(M, index=X["D"]["dates"]).resample("W").sum().iloc[1:-1]; fw = pd.Series(X["F"][:, ix].sum(1), index=pd.date_range(X["asof"] + pd.Timedelta(days=1), periods=90)).resample("W").sum().iloc[:-1]
    f3 = go.Figure([go.Scatter(x=w.index, y=w.values, name="Weekly demand (actual)", line=dict(color=C["gg"], width=2)), go.Scatter(x=w.index, y=w.rolling(4, min_periods=1).mean().values, name="4-week mean", line=dict(color=C["fg"], width=3)), go.Scatter(x=fw.index, y=fw.values, name="Forecast", line=dict(color=C["am"], width=3, dash="dot"))]); f3 = sty(f3, 330, "Demand trend and forecast (units per week)")
    so = P[P["rate"] > 0]["so"]; b = [int((so <= 7).sum()), int(((so > 7) & (so <= 14)).sum()), int(((so > 14) & (so <= 30)).sum()), int(((so > 30) & (so <= 90)).sum()), int(np.isinf(so).sum())]; f4 = sty(go.Figure(go.Bar(x=["≤ 7 days", "8-14 days", "15-30 days", "31-90 days", "none in 90 days"], y=b, marker_color=[C["co"], C["co"], C["am"], C["am"], C["sg"]])), 320, "Stockout risk overview: days until projected stockout"); return h, f1, f2, f3, f4
@safe(4)
def r_fc(c, s, l, p, st, hz, sku):
    X = G["X"]; fc = X["fc"]; P = getP(c, s, l, p, st); hz = int(hz or 30); h = banner() + note("<b>Method.</b> Four forecasters are backtested on the last 28 days (chronological split, never shuffled): seasonal naive (last week), 28-day moving average, damped-trend Holt-Winters with weekly seasonality, and a gradient-boosting model trained on past-only features (recent means, same-weekday mean, volatility, calendar of the target day, category). The lowest-WAPE method is chosen per SKU. Promotions are not forecast (future promo plans are unknown).", "info")
    if fc["msg"]: h += note(esc(" | ".join(fc["msg"])), "warn")
    if len(fc["perf"]): h += SEC("Backtest on the held-out last 28 days (WAPE = Σ|error| ÷ Σ actual)") + tbl(fc["perf"], [("method", "Method"), ("wape", "WAPE"), ("mae", "MAE (units/day/SKU)"), ("rmse", "RMSE"), ("chosen", "SKUs choosing it")], dict(wape=lambda v: pc(v, 1), mae=lambda v: fx(v, 2), rmse=lambda v: fx(v, 2), chosen=nf))
    def warn(r):
        w = []
        if r["hdays"] < 90: w.append("short history (baseline)")
        if not nul(r["wape"]) and r["wape"] > .6: w.append("high backtest error")
        if r["rate"] <= 0: w.append("no recent demand")
        return "; ".join(w) or "—"
    t = P.assign(w=P.apply(warn, axis=1)).sort_values("f30", ascending=False).head(40); h += SEC("SKU forecasts (units): top 40 by 30-day demand") + tbl(t, [("sku", "SKU"), ("pname", "Product"), ("f7", "7 d"), ("f14", "14 d"), ("f30", "30 d"), ("f90", "90 d"), ("method", "Selected method"), ("wape", "Backtest WAPE"), ("conf", "Reliability"), ("w", "Warnings")], dict(f7=nf, f14=nf, f30=nf, f90=nf, wape=lambda v: pc(v, 0)))
    sk = sku if sku in X["skus"] else t["sku"].iloc[0]; i = X["skus"].index(sk); f1 = empty("No backtest available")
    if fc["bt"]: dts_ = pd.date_range(X["asof"] - pd.Timedelta(days=27), X["asof"]); f1 = go.Figure([go.Scatter(x=dts_, y=fc["act"][:, i], name="Actual", line=dict(color=C["ch"], width=3))] + [go.Scatter(x=dts_, y=fc["bt"][n][:, i], name=n, line=dict(width=2, dash="dot")) for n in fc["names"]]); f1 = sty(f1, 340, "Actual vs predicted, held-out last 28 days: " + sk)
    hist = pd.Series(X["D"]["M"][-120:, i], index=X["D"]["dates"][-120:]); fd = pd.date_range(X["asof"] + pd.Timedelta(days=1), periods=hz); f2 = go.Figure([go.Scatter(x=hist.index, y=hist.values, name="Actual daily", line=dict(color=C["gg"])), go.Scatter(x=hist.index, y=hist.rolling(7, min_periods=1).mean().values, name="7-day mean", line=dict(color=C["fg"], width=3)), go.Scatter(x=fd, y=X["F"][:hz, i], name="Forecast (%s)" % fc["method"][i], line=dict(color=C["am"], width=3, dash="dot"))]); f2 = sty(f2, 340, "History and %d-day forecast: %s" % (hz, sk))
    key = "f%d" % hz if hz in (7, 14, 30, 90) else "f30"; g = P.groupby("cat")[key].sum().sort_values(); f3 = sty(go.Figure(go.Bar(y=g.index, x=g.values, orientation="h", marker_color=C["fg"])), 300, "Forecast demand by category, next %d days (units)" % hz); return h, f1, f2, f3
@safe(2)
def r_rep(c, s, l, p, st):
    X = G["X"]; S = X["S"]; P = getP(c, s, l, p, st); z = zval(S["sl"])
    h = banner() + note("<b>Formulas</b> (service level %s, z = %.2f; review period %d days). <b>Average daily demand</b> = mean forecast of the next 30 days. <b>Safety stock</b> = z × √(L·σ² + d²·σL²) (reduces to z × σ × √L when lead-time variability σL is 0 or unknown). <b>Reorder point</b> = forecast demand over the lead time + safety stock. <b>Inventory position</b> = on hand + confirmed incoming − allocated. <b>Target</b> = reorder point + review-period demand. <b>Order</b> = target − position when position ≤ reorder point (else 0), rounded up to case pack and supplier MOQ, never negative. Incoming stock counts in the position but is not available for sale before its arrival date: stockout timing uses the day-by-day projection." % (pc(S["sl"], 1), z, S["review"]), "info")
    t = P[(P["qty"] > 0) | P["state"].isin(ST[:4])].sort_values("pscore", ascending=False).head(100); h += tbl(t, [("sku", "SKU"), ("pname", "Product"), ("state", "Status"), ("rate", "Demand/day"), ("sd", "Demand SD"), ("lt", "Lead time d"), ("ltd", "Demand in lead time"), ("ss", "Safety stock"), ("rop", "Reorder pt"), ("pos", "Position"), ("tgt", "Target"), ("qty", "Order qty"), ("reorder_date", "Order on"), ("arrival", "Arrives"), ("est_cost", "Est. cost"), ("conf", "Confidence"), ("expl", "Reason")], dict(state=pill, rate=fx, sd=fx, lt=lambda v: fx(v, 0), ltd=nf, ss=nf, rop=nf, pos=nf, tgt=nf, qty=nf, reorder_date=dts, arrival=dts, est_cost=money))
    u = t.head(20); f1 = go.Figure([go.Bar(x=u["sku"], y=u["pos"], name="Inventory position", marker_color=C["fg"]), go.Bar(x=u["sku"], y=u["rop"], name="Reorder point", marker_color=C["am"])]); f1.update_layout(barmode="group"); return h, sty(f1, 340, "Position vs reorder point: 20 highest-priority SKUs")
@safe(3)
def r_risk(c, s, l, p, st):
    X = G["X"]; S = X["S"]; P = getP(c, s, l, p, st); h = banner() + note("Status comes from inventory position, demand, lead time and the day-by-day projection: <b>Critical</b> = no sellable stock or stockout within 3 days; <b>Stockout Risk</b> = projected to run out before a new order could arrive; <b>Reorder Now</b> = position ≤ reorder point; <b>Reorder Soon</b> = will reach it within the review period; <b>Overstocked</b> = cover above %d days; <b>Slow Moving</b> = demand below %s units/day." % (S["excess"], fx(S["slow"], 1)), "info")
    h += "<div class='kg'>" + card("Stockout exposure", money(P["expo"].sum()), "sales at risk before a new order can arrive", C["co"]) + card("Excess inventory value", money(P["excess_value"].sum()), "stock above the cover limit", C["li"]) + card("Slow-moving stock value", money(P[P["state"] == ST[6]]["inv_value"].sum()), "%d SKUs" % (P["state"] == ST[6]).sum()) + "</div>"
    order = {x: i for i, x in enumerate(ST)}; t = P[P["state"] != ST[4]].assign(_o=P["state"].map(order)).sort_values(["_o", "pscore"], ascending=[True, False]).head(60); h += SEC("Product-level risk table") + tbl(t, [("sku", "SKU"), ("pname", "Product"), ("cat", "Category"), ("state", "Status"), ("sell", "Available"), ("rate", "Demand/day"), ("lt", "Lead time"), ("dosp", "Cover (days)"), ("expo", "Stockout exposure"), ("excess_value", "Excess value"), ("expl", "Explanation")], dict(state=pill, sell=nf, rate=fx, lt=lambda v: fx(v, 0), dosp=lambda v: fx(v, 0), expo=money, excess_value=money))
    lo = P[P["rate"] > 0].nsmallest(8, "turn"); hv = P[P["rate"] > 0].nlargest(8, "cv"); ll = P.nlargest(8, "lt"); h += SEC("Unusually low turnover (annual turns)") + tbl(lo, [("sku", "SKU"), ("pname", "Product"), ("turn", "Turns/yr"), ("dosp", "Cover (days)")], dict(turn=lambda v: fx(v, 1), dosp=lambda v: fx(v, 0))) + SEC("Highest demand variability (CV)") + tbl(hv, [("sku", "SKU"), ("pname", "Product"), ("cv", "CV"), ("state", "Status")], dict(cv=lambda v: fx(v, 2), state=pill)) + SEC("Longest supplier lead times") + tbl(ll, [("sku", "SKU"), ("pname", "Product"), ("sname", "Supplier"), ("lt", "Lead time d"), ("state", "Status")], dict(lt=lambda v: fx(v, 0), state=pill))
    ct = pd.crosstab(P["cat"], P["state"]).reindex(columns=ST, fill_value=0); h += SEC("Category risk summary") + tbl(ct.reset_index(), [("cat", "Category")] + [(x, x) for x in ST]); f1 = go.Figure([go.Bar(x=ct.index, y=ct[x], name=GL[x] + " " + x, marker_color=STC[x]) for x in ST]); f1.update_layout(barmode="stack"); f1.update_xaxes(tickangle=-20)
    d = P[(P["rate"] > 0) & np.isfinite(P["dosp"])]; f2 = go.Figure(go.Scatter(x=d["lt"], y=np.minimum(d["dosp"], 365), mode="markers", text=d["sku"], marker=dict(size=9, color=[STC[x] for x in d["state"]], line=dict(width=1, color=C["ch"])))); mx = max(d["lt"].max() if len(d) else 1, 1); f2.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color=C["gg"], dash="dash")); f2 = sty(f2, 330, "Cover vs lead time (below the dashed line = stock shorter than lead time)"); f2.update_xaxes(title="Lead time (days)"); f2.update_yaxes(title="Cover (days, capped 365)"); return h, sty(f1, 340, "Status by category"), f2
@safe(3)
def r_sup(c, s, l, p, st, sku):
    X = G["X"]; P = getP(c, s, l, p, st); O = X["offers"]; ids = set(P["sku"]); O = O[O["sku"].isin(ids)]
    if O.empty: raise ValueError("No supplier offers in this view.")
    g = O.groupby(["supplier_id", "supplier_name"], dropna=False).agg(skus=("sku", "nunique"), lt=("lead_time_days", "mean"), sdl=("lead_time_sd", "mean"), moq=("moq", "mean"), ot=("on_time_rate", "mean"), ci=("cost_idx", "mean"), pref=("chosen", "sum")).reset_index(); risk = P[P["state"].isin(ST[:3])].groupby("sup").size()
    g["risk"] = g["supplier_id"].map(risk).fillna(0); chosen = O[O["chosen"]]; nch = int((~chosen["cheapest"]).sum()); h = banner() + note("<b>Scoring</b>: cost = cheapest ÷ offer cost; speed = fastest (lead time + variability) ÷ offer; reliability = on-time rate when supplied. Weights %.2f / %.2f / %.2f (editable in Settings); for SKUs that would stock out before the slowest offer could deliver, the speed weight is doubled. Offers with no reliability data are scored on cost and speed only and flagged. <b>The cheapest offer was not chosen for %d of %d SKUs</b> because lead time or reliability made it riskier. Lead times are the supplied values (no purchase-order history table)." % (X["S"]["wc"], X["S"]["ws"], X["S"]["wr"], nch, len(chosen)), "info")
    h += tbl(g.sort_values("risk", ascending=False), [("supplier_name", "Supplier"), ("skus", "SKUs offered"), ("lt", "Mean lead time d"), ("sdl", "Mean lead-time SD"), ("moq", "Mean MOQ"), ("ot", "On-time rate"), ("ci", "Cost index (1 = cheapest)"), ("pref", "Preferred for SKUs"), ("risk", "SKUs needing action")], dict(lt=fx, sdl=fx, moq=nf, ot=lambda v: pc(v, 0) if not nul(v) else "unavailable", ci=lambda v: fx(v, 2), pref=nf, risk=nf))
    sk = sku if sku in ids else P.sort_values("pscore", ascending=False)["sku"].iloc[0]; o = O[O["sku"] == sk].sort_values("score", ascending=False); h += SEC("Offer comparison for " + esc(sk)) + tbl(o, [("supplier_name", "Supplier"), ("unit_cost", "Unit cost"), ("lead_time_days", "Lead time d"), ("lead_time_sd", "Lead-time SD"), ("moq", "MOQ"), ("case_pack", "Case pack"), ("on_time_rate", "On-time rate"), ("score", "Score"), ("cheapest", "Cheapest"), ("chosen", "Preferred")], dict(unit_cost=money, lead_time_sd=lambda v: fx(v, 1) if not nul(v) else "unknown", moq=nf, case_pack=nf, on_time_rate=lambda v: pc(v, 0) if not nul(v) else "unavailable", score=lambda v: fx(v, 2), cheapest=lambda v: "yes" if v else "", chosen=lambda v: "✓ preferred" if v else ""))
    f1 = go.Figure(go.Scatter(x=g["lt"], y=g["ci"], mode="markers+text", text=g["supplier_name"], textposition="top center", marker=dict(size=10 + g["skus"] / 3, color=g["ot"].fillna(.5), colorscale=[[0, C["co"]], [.5, C["am"]], [1, C["sg"]]], showscale=True, colorbar=dict(title="On-time"), line=dict(width=1, color=C["ch"])))); f1 = sty(f1, 360, "Suppliers: lead time vs cost index (colour = on-time rate)"); f1.update_xaxes(title="Mean lead time (days)"); f1.update_yaxes(title="Cost index")
    f2 = sty(go.Figure(go.Bar(x=g["supplier_name"], y=g["pref"], marker_color=C["fg"])), 320, "SKUs for which each supplier is preferred"); f2.update_xaxes(tickangle=-20); return h, f1, f2
def cost_tbl(P, S):
    D_ = P["rate"].values * 365; H_ = P["cost"].values * P["hp_e"].values / 100; S_ = P["oc_e"].values; e = eoq(D_, S_, H_); suit = (P["cv"].fillna(9).values <= 1.0) & (e >= np.nan_to_num(P["moq"].values)) & (P["rate"].values > 0); Qc = np.where(suit, rnd(e, P["moq"].values, P["pack"].values), rnd(np.maximum(P["rate"].values * S["review"], 1e-9) * (P["rate"].values > 0), P["moq"].values, P["pack"].values))
    ao = np.where(Qc > 0, D_ / np.where(Qc > 0, Qc, 1) * S_, 0.0); ah = (Qc / 2 + P["ss"].values) * H_; ap = D_ * np.nan_to_num(P["pcost"].values); return P.assign(D=D_, eoq=e, suit=suit, Qc=Qc, ann_order=ao, ann_hold=ah, ann_proc=ap, ann_total=ao + ah + ap)
@safe(3)
def r_cost(c, s, l, p, st):
    X = G["X"]; S = X["S"]; P = getP(c, s, l, p, st); T = cost_tbl(P, S); Pd = plan(X, S0); Pd = Pd[Pd["sku"].isin(P["sku"])]; Td = cost_tbl(Pd, S0)
    h = banner() + note("<b>EOQ</b> = √(2 × annual demand × ordering cost ÷ annual holding cost per unit). It is used only where demand is steady (CV ≤ 1) and EOQ ≥ the supplier MOQ; otherwise the review-period quantity is used because variable demand and MOQs dominate the decision. Ordering cost, holding %% and annual demand (= 30-day forecast rate × 365) come from the data or Settings (multipliers ×%.2f ordering, ×%.2f holding)." % (S["ocm"], S["hpm"]), "info")
    h += "<div class='kg'>" + card("Annual ordering cost", money(T["ann_order"].sum()), "orders per year × ordering cost") + card("Annual holding cost (cycle + safety stock)", money(T["ann_hold"].sum()), "carrying cost at planned order sizes", C["am"]) + card("Annual procurement", money(T["ann_proc"].sum()), "forecast demand × supplier cost") + card("Working capital in stock", money(P["inv_value"].sum()), "on hand at cost", C["am"]) + card("Excess stock value", money(P["excess_value"].sum()), "above the cover limit", C["li"]) + card("Stockout exposure", money(P["expo"].sum()), "sales at risk", C["co"]) + card("Replenishment spend now", money(P["est_cost"].sum()), "recommended orders") + "</div>"
    h += SEC("Effect of your settings vs defaults (service 95%, review 14 d, ×1 costs)") + tbl(pd.DataFrame([("Safety stock (units)", Td["ss"].sum(), T["ss"].sum()), ("Safety stock value", (Td["ss"] * Td["cost"]).sum(), (T["ss"] * T["cost"]).sum()), ("Recommended order units now", Td["qty"].sum(), T["qty"].sum()), ("Recommended order value now", Td["est_cost"].sum(), T["est_cost"].sum()), ("Annual ordering cost", Td["ann_order"].sum(), T["ann_order"].sum()), ("Annual holding cost", Td["ann_hold"].sum(), T["ann_hold"].sum())], columns=["m", "d", "n"]), [("m", "Measure"), ("d", "Defaults"), ("n", "Your settings")], dict(d=lambda v: nf(v), n=lambda v: nf(v)))
    rows = []
    for sl in (.90, .95, .975, .99): q = plan(X, S, dict(sl=sl)); q = q[q["sku"].isin(P["sku"])]; rows.append(dict(sl=sl, ss=q["ss"].sum(), ssv=(q["ss"] * q["cost"]).sum(), hold=(q["ss"] * q["cost"] * q["hp_e"] / 100).sum(), qty=q["qty"].sum(), val=q["est_cost"].sum()))
    sv = pd.DataFrame(rows); h += SEC("Service level vs cost (design target: probability of no stockout in a lead-time cycle)") + tbl(sv, [("sl", "Service level"), ("ss", "Safety stock units"), ("ssv", "Safety stock value"), ("hold", "Annual holding on safety stock"), ("qty", "Order units now"), ("val", "Order value now")], dict(sl=lambda v: pc(v, 1), ss=nf, ssv=money, hold=money, qty=nf, val=money))
    t = T.sort_values("ann_total", ascending=False).head(30); h += SEC("EOQ and cost by SKU (top 30 by annual cost)") + tbl(t, [("sku", "SKU"), ("pname", "Product"), ("D", "Annual demand"), ("eoq", "EOQ"), ("suit", "EOQ suitable"), ("moq", "MOQ"), ("Qc", "Order size used"), ("ann_order", "Ordering"), ("ann_hold", "Holding"), ("ann_proc", "Procurement")], dict(D=nf, eoq=nf, suit=lambda v: "yes" if v else "no (variable demand or MOQ)", moq=nf, Qc=nf, ann_order=money, ann_hold=money, ann_proc=money))
    g = T.groupby("cat")[["ann_order", "ann_hold"]].sum(); f1 = go.Figure([go.Bar(x=g.index, y=g["ann_order"], name="Ordering", marker_color=C["li"]), go.Bar(x=g.index, y=g["ann_hold"], name="Holding", marker_color=C["am"])]); f1.update_layout(barmode="stack"); f1.update_xaxes(tickangle=-20); f2 = go.Figure(go.Scatter(x=sv["sl"] * 100, y=sv["hold"], mode="lines+markers", line=dict(color=C["fg"], width=3))); f2 = sty(f2, 320, "Annual holding cost of safety stock vs service level"); f2.update_xaxes(title="Service level (%)"); return h, sty(f1, 320, "Annual ordering vs holding cost by category (₹)"), f2
WBC = ["select", "sku", "product", "category", "status", "priority", "on_hand", "position", "forecast_30d", "reorder_point", "safety_stock", "rec_qty", "order_qty", "supplier", "expected_arrival", "unit_cost", "est_cost", "explanation"]; WBT = ["bool"] + ["str"] * 5 + ["number"] * 7 + ["str", "str", "number", "number", "str"]
def r_wb(c, s, l, p, st):
    if "X" not in G: return note("Load data first."), pd.DataFrame(columns=WBC)
    try:
        P = getP(c, s, l, p, st).sort_values("pscore", ascending=False).head(150); df = pd.DataFrame({"select": False, "sku": P["sku"].values, "product": P["pname"].values, "category": P["cat"].values, "status": P["state"].values, "priority": P["priority"].values, "on_hand": P["on_hand"].round(0).values, "position": P["pos"].round(0).values, "forecast_30d": P["f30"].round(0).values, "reorder_point": P["rop"].round(0).values, "safety_stock": P["ss"].round(0).values,
                                                                                                    "rec_qty": P["qty"].values, "order_qty": P["qty"].values, "supplier": P["sname"].values, "expected_arrival": P["arrival"].map(dts).values, "unit_cost": P["pcost"].round(2).values, "est_cost": P["est_cost"].round(0).values, "explanation": P["expl"].values})
        return note("Tick products, edit <b>order_qty</b> or <b>supplier</b> (use a supplier name that offers the SKU), then press <b>Build proposed purchase order</b>. This is planning only: nothing is sent to any supplier.", "ai"), df
    except Exception as e: return note("Could not build the workbench: " + esc(e), "err"), pd.DataFrame(columns=WBC)
def po_check(q, moq, pack):
    m = []; pk = max(pack, 1)
    if q <= 0: return "quantity must be above 0"
    if abs(q / pk - round(q / pk)) > 1e-9: m.append("not a multiple of case pack %d" % pk)
    if q < moq: m.append("below MOQ %d" % moq)
    return "; ".join(m) or "OK"
def build_po(df):
    if "X" not in G: return note("Load data first."), None
    X = G["X"]; O = X["offers"]; df = pd.DataFrame(df); rows = []
    if df.empty or "select" not in df.columns: return note("Nothing to build: load the workbench first.", "warn"), None
    for r in df.to_dict("records"):
        if str(r.get("select")).lower() not in ("true", "1"): continue
        sk = r["sku"]; q = pd.to_numeric(r["order_qty"], errors="coerce"); sup = str(r["supplier"]).strip().lower(); o = O[(O["sku"] == sk) & ((O["supplier_name"].astype(str).str.lower() == sup) | (O["supplier_id"].astype(str).str.lower() == sup))]
        if o.empty: rows.append(dict(sku=sk, product=r["product"], supplier=r["supplier"], qty=q, moq=np.nan, pack=np.nan, unit_cost=np.nan, value=np.nan, check="supplier does not offer this SKU")); continue
        o = o.iloc[0]; ck = "quantity is not a number" if nul(q) else po_check(q, o["moq"], o["case_pack"]); rows.append(dict(sku=sk, product=r["product"], supplier=o["supplier_name"], qty=q, moq=o["moq"], pack=o["case_pack"], unit_cost=o["unit_cost"], value=(q * o["unit_cost"]) if not nul(q) else np.nan, check=ck))
    if not rows: return note("No products ticked.", "warn"), None
    po = pd.DataFrame(rows); G["po"] = po; ok = po[po["check"] == "OK"]; bad = po[po["check"] != "OK"]; sm = ok.groupby("supplier").agg(lines=("sku", "size"), units=("qty", "sum"), value=("value", "sum")).reset_index()
    h = note("<b>Proposed purchase order (planning only; nothing is submitted).</b> %d valid lines worth <b>%s</b>; %d lines need fixing." % (len(ok), money(ok["value"].sum()), len(bad)), "ok" if bad.empty else "warn") + SEC("Order summary by supplier") + (tbl(sm, [("supplier", "Supplier"), ("lines", "Lines"), ("units", "Units"), ("value", "Value")], dict(units=nf, value=money)) if len(sm) else "") + SEC("Lines with validation") + tbl(po, [("sku", "SKU"), ("product", "Product"), ("supplier", "Supplier"), ("qty", "Qty"), ("moq", "MOQ"), ("pack", "Case pack"), ("unit_cost", "Unit cost"), ("value", "Value"), ("check", "Validation")], dict(qty=nf, moq=nf, pack=nf, unit_cost=money, value=money))
    po.to_csv("restock_proposed_po.csv", index=False); return h, "restock_proposed_po.csv"
@safe(3)
def r_sim(c, s, l, p, st, dm, ltadd, cm, sl, ssm, qm, moqm, psg, ps, pl, dis, hor, adapt, seed):
    X = G["X"]; S = X["S"]; ids = set(getP(c, s, l, p, st)["sku"]); keep = X["B"]["sku"].isin(ids).values; Xs = dict(X, B=X["B"][keep].reset_index(drop=True), F=X["F"][:, keep], ARR=X["ARR"][:, keep], ARRc=X["ARRc"][:, keep])
    w0 = dict(dm=1.0, ltm=1.0, ltadd=0.0, cm=0.0, dis=0.0, psg=0.0, ps=0, pl=0); w1 = dict(dm=1 + dm / 100, ltm=1.0, ltadd=float(ltadd), cm=float(cm), dis=float(dis), psg=float(psg), ps=int(ps), pl=int(pl)); pol = dict(sl=sl / 100, ssm=float(ssm), qm=float(qm), moqm=float(moqm)); H = int(hor); sd = int(seed or 7)
    A = simulate(Xs, S, {}, w0, H, sd); Bb = simulate(Xs, S, {}, w1, H, sd); Cc = simulate(Xs, S, pol, w1, H, sd, bool(adapt)); cases = [("Original policy, normal conditions", A), ("Original policy, scenario conditions", Bb), ("Modified policy, scenario conditions", Cc)]
    G["sim"] = pd.DataFrame([dict(case=n, fill_rate=r["fill"], skus_with_stockout=r["so_skus"], lost_revenue=r["lost_rev"], holding_cost=r["hold"], procurement_spend=r["spend"], excess_stock_value=r["excess"], orders_placed=r["orders"], avg_working_capital=r["wc"]) for n, r in cases]).assign(horizon_days=H, seed=sd, note="simulated estimate, not a guarantee")
    h = banner() + note("<b>Simulated estimates, not guarantees.</b> %d-day day-by-day simulation for the %d SKUs in view: demand is overdispersed Poisson around the forecast; orders follow (reorder point, order-up-to target) and arrive after a random lead time; incoming POs arrive on their dates (shifted by any delay). All three cases use the <b>same random draws</b>, so differences come only from policy and scenario. Lost sales are not back-ordered." % (H, len(Xs["B"])), "warn")
    rows = [("Service level (fill rate)", lambda r: pc(r["fill"], 1)), ("SKUs with a stockout", lambda r: nf(r["so_skus"])), ("Lost revenue", lambda r: money(r["lost_rev"])), ("Inventory holding cost", lambda r: money(r["hold"])), ("Procurement spend", lambda r: money(r["spend"])), ("Excess stock at end", lambda r: money(r["excess"])), ("Reorder frequency (orders placed)", lambda r: nf(r["orders"])), ("Average working capital in stock", lambda r: money(r["wc"]))]
    h += tbl(pd.DataFrame([[n] + [f(r) for _, r in cases] for n, f in rows], columns=["m", "a", "b", "c"]), [("m", "Measure"), ("a", cases[0][0]), ("b", cases[1][0]), ("c", cases[2][0])], dict(a=str, b=str, c=str)) + note("Simulation consistency check: %s. %s" % ("stock and order flows balance" if all(r["ok"] for _, r in cases) else "FAILED", "The modified policy re-plans with the new demand and lead times." if adapt else "The modified policy keeps today's demand and lead-time estimates (it does not know about the scenario)."), "info")
    nm = ["Original / normal", "Original / scenario", "Modified / scenario"]; f1 = go.Figure(go.Bar(x=nm, y=[r["fill"] * 100 for _, r in cases], marker_color=[C["gg"], C["co"], C["fg"]])); f1 = sty(f1, 320, "Service level (fill rate %)"); f1.update_yaxes(range=[0, 100])
    f2 = go.Figure([go.Bar(x=nm, y=[r[k] for _, r in cases], name=n, marker_color=cl) for k, n, cl in (("hold", "Holding cost", C["am"]), ("lost_rev", "Lost revenue", C["co"]), ("excess", "Excess stock value", C["li"]))]); f2.update_layout(barmode="group"); return h, f1, sty(f2, 330, "Cost, lost revenue and excess stock (₹)")
def ask(q, _=None):
    if "X" not in G: return "Load data first."
    X = G["X"]; P = X["P"]; S = X["S"]; q = (q or "").strip() or "Which products need ordering first?"; ql = q.lower(); parts = []; ids = re.findall(r"\bSKU\d{3}\b", q, re.I)
    if ids and ids[0].upper() in set(P["sku"]): parts.append(("sku_explanation", P[P["sku"] == ids[0].upper()]["expl"].iloc[0]))
    if re.search(r"first|order|immediate|urgent|priorit|replenish", ql): t = P[P["pscore"] > 0].sort_values("pscore", ascending=False).head(5); parts.append(("priority_orders", " | ".join("%s: %s (priority %s, score %.0f, %s)" % (r.sku, r.state, r.priority, r.pscore, ("order %s units from %s" % (nf(r.qty), r.sname)) if r.qty > 0 else "no order yet") for r in t.itertuples()) or "no SKU needs action"))
    if re.search(r"stockout|run out|risk", ql): t = P[P["state"].isin(ST[:2])].sort_values("expo", ascending=False).head(5); parts.append(("stockout_risk", "%d SKUs are critical or at stockout risk, about %s of sales exposed. Largest: " % (int(P["state"].isin(ST[:2]).sum()), money(P[P["state"].isin(ST[:2])]["expo"].sum())) + "; ".join("%s runs out in %s days, lead time %s d" % (r.sku, fx(r.so, 0), fx(r.lt, 0)) for r in t.itertuples())))
    if re.search(r"safety", ql): t = P[P["state"].isin(ST[:3])].nlargest(4, "ss"); parts.append(("safety_stock", "Safety stock uses z=%.2f for a %s service level. Largest need: " % (zval(S["sl"]), pc(S["sl"], 1)) + "; ".join("%s %s units (demand SD %s, lead time %s d, SD %s)" % (r.sku, nf(r.ss), fx(r.sd, 1), fx(r.lt, 0), fx(r.sdlt, 1)) for r in t.itertuples()) + ". Justified where demand variability or lead-time variability is high."))
    if re.search(r"over|excess|slow|capital|cash|tied", ql): t = P.nlargest(5, "excess_value"); parts.append(("working_capital", "Excess stock %s and slow-moving stock %s of %s total stock value. Largest excess: " % (money(P["excess_value"].sum()), money(P[P["state"] == ST[6]]["inv_value"].sum()), money(P["inv_value"].sum())) + "; ".join("%s %s (%s days cover)" % (r.sku, money(r.excess_value), fx(r.dosp, 0)) for r in t.itertuples()) + ". Pause reorders and review promotions or transfers."))
    if re.search(r"supplier|lead", ql): O = X["offers"]; g = O[O["chosen"]].groupby("supplier_name").agg(n=("sku", "size"), lt=("lead_time_days", "mean")); rk = P[P["state"].isin(ST[:3])].groupby("sname").size().sort_values(ascending=False); parts.append(("suppliers", "SKUs needing action by preferred supplier: " + "; ".join("%s %d" % (k, v) for k, v in rk.head(4).items()) + ". Longest mean lead time among preferred suppliers: " + "; ".join("%s %s d" % (k, fx(v, 0)) for k, v in g["lt"].sort_values(ascending=False).head(3).items()) + ". Reliability is only used where supplied."))
    if re.search(r"limit|caveat|trust", ql): parts.append(("limitations", "Forecasts are backtest-checked estimates, not guarantees; lead times are supplied values; demand is not corrected for past stockouts; replenishment is planned per SKU for the whole network; all orders need buyer approval."))
    if not parts: c = P["state"].value_counts(); parts = [("summary", "%d SKUs: %s. Inventory %s; recommended orders %s; excess %s." % (len(P), ", ".join("%s %d" % (k, v) for k, v in c.items()), money(P["inv_value"].sum()), money(P["est_cost"].sum()), money(P["excess_value"].sum())))]
    return "### Answer (deterministic, computed from the loaded data)\n" + "\n".join("- **%s**: %s" % (a, b) for a, b in parts) + "\n\n*Every figure is calculated from the active dataset. Buyers must review before ordering.*"
@safe(2)
def r_dq():
    D = G["D"]; X = G["X"]; h = banner() + note("Rows read: " + ", ".join("%s %s" % (TN[k], nf(v)) for k, v in D["rows_in"].items()) + ". Raw uploads stay unchanged; a validated copy is analysed. Excluded rows: <b>%d</b> (nothing is dropped or invented silently)." % sum(c for _, _, c in D["excluded"]), "ok")
    h += "".join(note(esc(m), "err" if l == "error" else "warn" if l == "warn" else "info") for l, m in D["issues"]) or note("No validation issues.", "ok")
    if X["B"]["overdue"].any(): h += note("%d SKUs have overdue purchase orders (expected date already passed): treated as arriving on day 1 (assumption)." % X["B"]["overdue"].sum(), "warn")
    if D["excluded"]: h += SEC("Excluded rows by reason") + tbl(pd.DataFrame(D["excluded"], columns=["t", "w", "c"]), [("t", "Table"), ("w", "Reason"), ("c", "Rows")], dict(c=nf))
    h += SEC("Preview of cleaned product master") + tbl(D["prod"].head(8), [("sku", "SKU"), ("product_name", "Product"), ("category", "Category"), ("unit_cost", "Unit cost"), ("unit_price", "Price")], dict(unit_cost=money, unit_price=money)); return h, SEC("Self-test results (run at start-up)") + "".join(note(("PASS " if ok else "FAIL ") + esc(g + ": " + n) + " — " + esc(d), "ok" if ok else "err") for g, n, ok, d in TESTS)
def export_all():
    if "X" not in G: return None, note("Load data first.")
    X = G["X"]; P = X["P"]; out = []; ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"); src = "SYNTHETIC demo data" if G.get("demo") else "user-uploaded data"
    def w(df, fn): df.assign(generated=ts, data_source=src).to_csv(fn, index=False); out.append(fn)
    w(P[["sku", "pname", "cat", "state", "priority", "sname", "rate", "lt", "ss", "rop", "pos", "tgt", "qty", "reorder_date", "arrival", "est_cost", "conf", "expl"]], "restock_recommendations.csv"); w(P[["sku", "pname", "f7", "f14", "f30", "f90", "method", "wape", "conf"]], "restock_forecasts.csv"); w(P[["sku", "pname", "cat", "state", "sell", "rate", "dosp", "so", "expo", "excess_value", "cv", "turn", "expl"]], "restock_inventory_risk.csv")
    w(X["offers"][["sku", "supplier_id", "supplier_name", "unit_cost", "lead_time_days", "lead_time_sd", "moq", "case_pack", "on_time_rate", "score", "cheapest", "chosen"]], "restock_supplier_comparison.csv"); w(P[["sku", "pname", "cat", "on_hand", "cost", "inv_value", "hold_cost", "excess_value"]], "restock_inventory_valuation.csv")
    if "po" in G: w(G["po"], "restock_proposed_po_report.csv")
    if "sim" in G: w(G["sim"], "restock_whatif_results.csv")
    c = P["state"].value_counts(); top = P[P["pscore"] > 0].sort_values("pscore", ascending=False).head(8); css = "body{font-family:Inter,Arial,sans-serif;background:#F6F3EA;color:#263C37;max-width:900px;margin:30px auto;padding:0 18px}h1{color:#183B35}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 10px}"
    body = "<h1>RESTOCK AI: Executive inventory summary</h1><p><i>Generated %s. Data source: %s. All recommendations are proposals for buyer review; forecasts and simulations are estimates.</i></p><ul><li>%d SKUs; inventory value %s; annual holding cost %s.</li><li>Status: %s.</li><li>Recommended orders: %d SKUs worth %s; excess stock %s.</li></ul><h2>Priority actions</h2><ul>%s</ul><h2>Definitions</h2><p>Safety stock = z × √(L·σ² + d²·σL²); reorder point = demand over lead time + safety stock; position = on hand + incoming − allocated; order = target − position, rounded to case pack and MOQ.</p>" % (ts, src, len(P), money(P["inv_value"].sum()), money(P["hold_cost"].sum()), ", ".join("%s %d" % (k, v) for k, v in c.items()), int((P["qty"] > 0).sum()), money(P["est_cost"].sum()), money(P["excess_value"].sum()), "".join("<li>%s</li>" % esc(r.expl) for r in top.itertuples()))
    open("restock_executive_summary.html", "w", encoding="utf-8").write("<html><head><meta charset='utf-8'><title>RESTOCK AI</title><style>" + css + "</style></head><body>" + body + "</body></html>"); out.append("restock_executive_summary.html"); return [f for f in out if os.path.exists(f)], note("Wrote %d report files (CSV and HTML)." % len(out), "ok")

# ------------------------------ self-tests ------------------------------
def mkX(rate=10., sd=2., oh=50., al=0., inc=None, lt=9., sdlt=0., moq=1., pack=1., cost=10., price=15., hp=20., oc=500., imp=2, hdays=700, wape=.2):
    B = pd.DataFrame([dict(sku="A", pname="a", cat="c", sup="S", sname="s", cost=cost, pcost=cost, price=price, moq=moq, pack=pack, lt=lt, sdlt=sdlt, lt_src="stated; SD supplied", rel=np.nan, on_hand=oh, alloc=al, inc=sum(q for _, q in (inc or [])), sd=sd, hdays=hdays, nz=500, last_sale=0, hp=hp, oc=oc, cost_src="x", imp=imp, wape=wape, method="m", miss=False, why="", overdue=False)])
    A, tot = mkarr(1, [(0, d, q) for d, q in (inc or [])]); return dict(B=B, F=np.full((90, 1), float(rate)), ARR=A, ARRc=np.cumsum(A, 0), asof=ASOF, skus=["A"], locs={})
def mini_raw():
    dt = pd.date_range("2026-01-01", periods=6); return dict(prod=pd.DataFrame(dict(sku=["A", "B"], unit_cost=[10.0, 20.0])), sales=pd.DataFrame(dict(date=list(dt[:3]) + list(dt[:3]) + [dt[4]], sku=["A"] * 3 + ["A"] * 3 + ["B"], store=["s1"] * 3 + ["s2"] * 3 + ["s1"], units=[1, 2, 3, 4, 5, 6, 7])), inv=pd.DataFrame(dict(sku=["A", "B"], on_hand=[10, 20])), sup=pd.DataFrame(dict(supplier_id=["S1", "S1"], sku=["A", "B"], lead_time_days=[5, 6], unit_cost=[9.0, 18.0])))
def run_selftests(verbose=True):
    if TESTS: return TESTS
    T = []; saved = dict(G)
    def chk(g, n, fn):
        try: ok, d = fn(); T.append((g, n, bool(ok), d))
        except Exception as e: T.append((g, n, False, type(e).__name__ + ": " + str(e)[:90]))
    row = lambda **k: plan(mkX(**k), S0).iloc[0]; z95 = 1.6448536
    def t1(): r = row(); return abs(r["ss"] - z95 * 2 * 3) < 1e-3 and abs(r["rop"] - (90 + r["ss"])) < 1e-6, "safety stock %.2f = z·σ·√L (%.2f); reorder point %.2f = 90 + safety stock" % (r["ss"], z95 * 6, r["rop"])
    def t1b(): r = row(sdlt=1.0); return abs(r["ss"] - z95 * math.sqrt(9 * 4 + 100 * 1)) < 1e-3, "lead-time variability widens safety stock to %.2f" % r["ss"]
    def t2(): r = row(oh=50, al=10, inc=[(5, 30)]); return r["pos"] == 70, "position = on hand 50 + incoming 30 − allocated 10 = %s" % nf(r["pos"])
    def t3(): r = row(oh=60); a = row(oh=1000); return abs(r["qty"] - math.ceil(r["tgt"] - r["pos"])) < 1e-6 and r["qty"] > 0 and a["qty"] == 0, "order = target − position = %s; zero when position is sufficient" % nf(r["qty"])
    def t4(): e = float(eoq(np.array([1200.]), np.array([50.]), np.array([2.]))[0]); return abs(e - math.sqrt(2 * 1200 * 50 / 2)) < 1e-9, "EOQ %.1f = √(2·1200·50/2)" % e
    def t5(): return float(rnd(13, 20, 6)) == 24 and float(rnd(25, 20, 6)) == 30 and float(rnd(0, 20, 6)) == 0 and float(rnd(-4, 20, 6)) == 0, "13 → 24 (MOQ 20, pack 6), 25 → 30, 0 → 0, negative → 0"
    def t6(): r = row(rate=5, oh=20, inc=[(10, 100)], lt=3); return r["so"] == 5 and r["pos"] == 120, "stockout on day %s even though 100 units arrive on day 10 (incoming not counted early); position %s" % (nf(r["so"]), nf(r["pos"]))
    def t7(): M = np.tile(np.arange(1., 8.), 8)[:, None]; sn = snaive(M, 56, 14); w0 = wape_cols(np.zeros((3, 1)), np.ones((3, 1))); return np.allclose(sn[:, 0], np.tile(np.arange(1., 8.), 2)) and np.allclose(M[-28:].mean(0), 4.0) and np.isnan(w0[0]), "seasonal naive repeats the weekly pattern exactly; moving average 4.0; WAPE with zero actuals is n/a (not infinity)"
    def t8(): r = np.random.default_rng(1); M = r.poisson(6, (420, 4)).astype(float) * (1 + .3 * (np.arange(420) % 7 >= 5))[:, None]; f = run_forecast(M, pd.date_range("2025-01-01", periods=420), np.arange(4) % 2); s = f["split"]; return s[0] < s[1] and f["F"].shape == (90, 4) and bool(len(f["perf"])), "latest ML training target day %s < first test day %s: no future data in training; backtest of %d methods ran" % (s[0], s[1], len(f["perf"]))
    def t9(): D = load_all(mini_raw()); m = D["M"]; return D["M"].shape == (5, 2) and m[0, 0] == 5 and m[2, 0] == 9 and m[3, 0] == 0 and any("treated as zero demand" in x for _, x in D["issues"]), "two stores summed per day (1+4 = 5, 3+6 = 9); missing days become documented zeros"
    def t10():
        r = mini_raw(); a = ""
        try: load_all({**r, "prod": r["prod"].drop(columns=["unit_cost"])})
        except ValueError as e: a = str(e)
        s = r["sales"].copy(); s.loc[0, "units"] = -3; D = load_all({**r, "sales": pd.concat([s, s.iloc[[1]], pd.DataFrame(dict(date=[pd.Timestamp("2026-01-02")], sku=["ZZ"], store=["s1"], units=[4]))], ignore_index=True)}); w = " | ".join(m for _, m in D["issues"]); sp = r["sup"].copy(); sp.loc[0, "lead_time_days"] = 0; D2 = load_all({**r, "sup": sp}); w2 = " | ".join(m for _, m in D2["issues"])
        return "unit_cost" in a and "negative units" in w and "duplicate" in w and "not found in the product master" in w and "lead time must be above 0" in w2, "missing column, negative quantity, duplicate, unknown SKU and invalid lead time are rejected with row references"
    def t11(): a = row(rate=0., sd=0., oh=10); b = row(rate=5., oh=0.); return a["qty"] == 0 and a["rop"] == 0 and not pd.isna(a["dos"]) and b["state"] == ST[0] and b["qty"] > 0, "zero demand: no order and no NaN; zero stock with demand: Critical, order %s" % nf(b["qty"])
    def t12(): s = [row(oh=20)["state"], row(oh=60)["state"], row(oh=95)["state"], row(oh=105)["state"], row(oh=500)["state"], row(rate=5., oh=2000.)["state"], row(rate=.3, oh=100.)["state"]]; return s == [ST[0], ST[1], ST[2], ST[3], ST[4], ST[5], ST[6]], "classification ladder: " + ", ".join(s)
    def t13(): g = pd.DataFrame(dict(supplier_id=["A", "B", "C"], lead_time_days=[30, 7, 30], lead_time_sd=[8., 1., 8.], unit_cost=[10., 11., 9.], on_time_rate=[.70, .95, np.nan])); o = score_offers(g, False, S0); return o.loc[o["score"].idxmax(), "supplier_id"] == "B" and not o["cheapest"].iloc[1] and np.isfinite(o["score"].iloc[2]), "the faster, more reliable supplier B wins although supplier C is cheapest; C (no reliability data) is scored on cost and speed only"
    def t14(): a = make_data(3); b = make_data(3); h = lambda d: hashlib.md5(pd.util.hash_pandas_object(d["sales"].head(5000)).values.tobytes()).hexdigest(); X = demo_X(); Xs = dict(X, B=X["B"].head(20).reset_index(drop=True), F=X["F"][:, :20], ARR=X["ARR"][:, :20], ARRc=X["ARRc"][:, :20]); w0 = dict(dm=1.0, ltm=1.0, ltadd=0.0, cm=0.0, dis=0.0, psg=0.0, ps=0, pl=0); s1 = simulate(Xs, S0, {}, w0, 60, 5); s2 = simulate(Xs, S0, {}, w0, 60, 5); return h(a) == h(b) and s1["fill"] == s2["fill"] and s1["ok"], "same seed: identical synthetic data and identical simulation (fill rate %s); stock flows balance" % pc(s1["fill"], 1)
    def t15():
        X = demo_X(); P = X["P"]; q = P["qty"]; ok = (q >= 0).all() and ((q == 0) | (q % P["pack"].clip(lower=1) == 0)).all() and ((q == 0) | (q >= P["moq"])).all() and (P["state"].isin(ST[:3]) | (q == 0) | P["state"].eq(ST[7])).all() and not P["sku"].duplicated().any(); c = P["state"].value_counts()
        return bool(ok) and len(P) >= 100 and c.get(ST[0], 0) + c.get(ST[1], 0) >= 3 and c.get(ST[5], 0) + c.get(ST[6], 0) >= 10, "%d SKUs; quantities never negative, respect case pack and MOQ; %d at stockout risk, %d overstocked or slow (both sides of the problem)" % (len(P), c.get(ST[0], 0) + c.get(ST[1], 0), c.get(ST[5], 0) + c.get(ST[6], 0))
    def t16(): return po_check(10, 24, 6).startswith("not a multiple") and "below MOQ" in po_check(12, 24, 6) and po_check(24, 24, 6) == "OK" and po_check(0, 1, 1) != "OK", "PO lines validate case pack and MOQ"
    def t17(): X = demo_X(); G.clear(); G.update(X=X, D=X["D"], demo=True); f, _ = export_all(); n = len(pd.read_csv("restock_recommendations.csv")); G.clear(); G.update(saved); return n == len(X["P"]) and "restock_executive_summary.html" in f, "%d files written; recommendations export has %d rows matching the plan" % (len(f), n)
    def t18(): X = demo_X(); G.clear(); G.update(X=X, D=X["D"], demo=True); a = ask("Which products need ordering first?"); G.clear(); G.update(saved); return "deterministic" in a and "SKU" in a, "advisor answers from calculated results without any API"
    def t19(): X = demo_X(); P0 = X["P"]; P1 = plan(X, S0, dict(dm=1.3)); P2 = plan(X, S0, dict(sl=.99)); return P1["rop"].sum() > P0["rop"].sum() and P2["ss"].sum() > plan(X, S0)["ss"].sum(), "scenario recalculation: +30%% demand raises reorder points; a 99%% service level raises safety stock"
    for g, n, f in [("replenishment", "safety stock and reorder point", t1), ("replenishment", "lead-time variability", t1b), ("replenishment", "inventory position", t2), ("replenishment", "suggested order quantity", t3), ("optimization", "EOQ formula", t4), ("replenishment", "MOQ and case-pack rounding", t5), ("replenishment", "incoming stock not counted before arrival", t6),
                    ("forecasting", "naive, moving average, WAPE edge case", t7), ("forecasting", "chronological split, no leakage", t8), ("validation", "demand aggregation", t9), ("validation", "invalid CSV uploads", t10), ("edge cases", "zero demand and zero stock", t11), ("classification", "stockout, reorder and overstock ladder", t12), ("supplier", "not simply the cheapest", t13),
                    ("reproducibility", "data and simulation", t14), ("recommendations", "consistency on demo data", t15), ("workbench", "PO validation", t16), ("reporting", "CSV export correctness", t17), ("advisor", "works without an API key", t18), ("scenario", "recalculation", t19)]: chk(g, n, f)
    G.clear(); G.update(saved); TESTS[:] = T
    if verbose:
        for g, n, ok, d in T: print(("PASS " if ok else "FAIL ") + g + ": " + n + " :: " + d)
        print("Result: %d of %d self-tests passed." % (sum(1 for t in T if t[2]), len(T)))
    return TESTS

# ------------------------------ project export ------------------------------
ALGO = """## How RESTOCK AI works: the 10-step algorithm
1. **Load and validate.** Read sales, product master, inventory, suppliers, incoming POs and costs (or the synthetic demo); reject missing, negative, duplicate, unknown-SKU and invalid values with row references; never silently fill gaps.
2. **Aggregate demand.** Sum store-level sales into daily SKU demand on a full calendar (days without rows become documented zeros).
3. **Backtest forecasters.** Hold out the last 28 days (chronological) and score seasonal naive, 28-day moving average, Holt-Winters and gradient boosting by WAPE, MAE and RMSE.
4. **Forecast.** Pick the lowest-error method per SKU (baseline if history is short) and forecast 7, 14, 30 and 90 days.
5. **Project stock.** Roll available stock forward day by day, adding incoming orders only on their arrival dates, to find stockout days and shortage within the lead time.
6. **Choose suppliers.** Score every offer on cost, speed (lead time plus variability) and reliability, with speed weighted more for urgent SKUs; never just the cheapest.
7. **Calculate policy.** Safety stock = z·√(L·σ²+d²·σL²); reorder point = lead-time demand + safety stock; position = on hand + incoming − allocated; order = target − position, rounded to case pack and MOQ.
8. **Classify and prioritise.** Critical, Stockout Risk, Reorder Now, Reorder Soon, Healthy, Overstocked, Slow Moving, ranked by risk, demand, lead time, importance and financial exposure.
9. **Optimise cost.** EOQ where demand is steady, ordering vs holding costs, working capital and service-level trade-offs; build and validate a proposed purchase order.
10. **Stress-test and report.** Seeded day-by-day what-if simulation (demand, delays, costs, disruptions, promotions) with common random numbers; exports and an executive summary for human review.
"""
METH = ALGO + "\n### Notes\n- Forecasts and simulations are estimates. Lost sales are not back-ordered and past stockouts are not corrected in demand history.\n- Replenishment is planned per SKU for the whole network; inventory by location is shown for filtering.\n- Not included: Streamlit (Gradio is used), PDF export, a hosted LLM, multi-echelon planning.\n"
LICENSE = "MIT License\n\nCopyright (c) 2026 RESTOCK AI contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
README = """# RESTOCK AI: Retail Inventory Replenishment Intelligence
Never run out. Never overstock.

Forecasts SKU demand, calculates safety stock and reorder points, compares suppliers, flags stockout and overstock risk, and builds validated proposed purchase orders. Bundled data is **fictional synthetic data**: 108 SKUs, 6 categories, 7 suppliers, 3 stores, 13 months of daily sales, with planted stockout-prone fast movers, overstocked slow movers and an unreliable supplier.

""" + ALGO + """
## Architecture
@@@mermaid
flowchart LR
  A[CSV or synthetic data] --> B[Validation]
  B --> C[Demand aggregation + backtested forecasts]
  C --> D[Day-by-day stock projection]
  D --> E[Supplier scoring]
  E --> F[Replenishment engine]
  F --> G[Status + priority]
  F --> H[Cost optimisation + EOQ]
  F --> I[What-if simulator]
  G --> J[Workbench, advisor, exports]
@@@

## Run
@@@
pip install -r requirements.txt
python app.py
@@@
In Google Colab paste `app.py` into one cell and run it; Gradio prints a public link. See the Methodology tab for formulas.

## Sample schema
`sales_history.csv` (date, sku, store, units, promo); `product_master.csv` (sku, product_name, category, unit_cost, unit_price, importance); `current_inventory.csv` (sku, location, on_hand, allocated); `supplier_info.csv` (supplier_id, supplier_name, sku, lead_time_days, lead_time_sd, moq, case_pack, unit_cost, on_time_rate); `incoming_purchase_orders.csv` (po_id, sku, supplier_id, qty, expected_date); `product_costs.csv` (sku, ordering_cost, holding_pct). Currency ₹.

## Testing, limits, responsible use
`pip install pytest && pytest tests` runs the built-in self-tests. Estimates are not guarantees; buyers must review every recommendation. Nothing is sent to suppliers. Add screenshots to `screenshots/` (none included).
""".replace("@@@", "`" * 3)
TFILE = "from app import run_selftests\n\n\ndef test_%s():\n    res = [r for r in run_selftests(verbose=False) if r[0] in %s]\n    assert res and all(r[2] for r in res), res\n"
def _src():
    try: return open(__file__, encoding="utf-8").read()
    except Exception: pass
    try:
        for s in list(get_ipython().user_ns.get("In") or [])[::-1]:
            if "RESTOCK AI — Retail Inventory Replenishment Intelligence" in s and "def main" in s: return s
    except Exception: pass
    return None
def write_samples():
    raw = demo_raw(); os.makedirs("sample_data", exist_ok=True); out = []; fn = dict(sales="sales_history", prod="product_master", inv="current_inventory", sup="supplier_info", inc="incoming_purchase_orders", cost="product_costs")
    for k, n in fn.items(): p = "sample_data/%s.csv" % n; raw[k].to_csv(p, index=False); out.append(p)
    return out
def build_project(files):
    root = "restock-ai-retail-replenishment"
    for d in ["data/sample", "tests", "docs", "screenshots"]: os.makedirs(os.path.join(root, d), exist_ok=True)
    src = _src()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    for f in files: open(os.path.join(root, "data/sample", os.path.basename(f)), "wb").write(open(f, "rb").read())
    W = {"requirements.txt": "gradio>=4.44,<6\nplotly\nscikit-learn\nstatsmodels\nscipy\npandas\nnumpy\n", "README.md": README, "LICENSE": LICENSE, ".gitignore": "__pycache__/\n.ipynb_checkpoints/\n*.zip\nrestock_*.csv\nrestock_*.html\n.env\n", "docs/methodology.md": METH, "docs/data_dictionary.md": "# Data dictionary\nSee README schemas. All sample data is synthetic and fictional.\n", "docs/assumptions_and_limitations.md": "# Assumptions and limitations\nSynthetic data with planted situations. Network-level planning. Lead times are supplied values. Demand history is not corrected for past stockouts. Simulations are estimates. Human review required.\n", "screenshots/README.md": "# Screenshots\nAdd your own screenshots here.\n"}
    for nm, g in [("forecasting", ("forecasting",)), ("replenishment", ("replenishment", "scenario")), ("inventory_optimization", ("optimization", "classification")), ("data_validation", ("validation", "edge cases")), ("supplier_analysis", ("supplier",)), ("reporting_reproducibility", ("reporting", "reproducibility", "recommendations", "workbench", "advisor"))]: W["tests/test_%s.py" % nm] = TFILE % (nm, repr(g))
    for fn, tx in W.items(): open(os.path.join(root, fn), "w", encoding="utf-8").write(tx)
    z = "restock_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

# ------------------------------ UI ------------------------------
CSS = """
.gradio-container{background:#F6F3EA!important;color:#263C37!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1400px!important}
.hdr{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;padding:24px 30px;border-radius:14px;background:#183B35;color:#F6F3EA;margin-bottom:12px}.brand{display:flex;gap:16px;align-items:center}
.logo{width:52px;height:52px;border-radius:12px;background:#C3E8B7;color:#183B35;font-weight:800;font-size:20px;display:flex;align-items:center;justify-content:center}.hdr h1{margin:0;font-size:28px;letter-spacing:2px;color:#F6F3EA}.hdr h1 span{color:#E7B96C}.hdr p{margin:4px 0 0;color:#D8C8F0;font-style:italic}
.badges span{display:inline-block;margin:3px;padding:4px 11px;border-radius:6px;background:#ffffff1a;color:#F6F3EA;font-size:12px}button[role=tab]{background:#183B35!important;color:#F6F3EA!important;border-radius:8px 8px 0 0!important;margin-right:2px}button[role=tab][aria-selected=true]{background:#E7B96C!important;color:#183B35!important;font-weight:700}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:10px 0}.kc{background:#fff;border:1px solid #183B3526;border-radius:10px;padding:14px 16px}.kl{color:#5d726c;font-size:11px;text-transform:uppercase;letter-spacing:1.1px}.kv{font-size:22px;font-weight:700;margin:6px 0 4px}.kd{color:#5d726c;font-size:12px}
.note{background:#fff;padding:10px 14px;border-radius:8px;margin:8px 0;color:#263C37;font-size:14px;line-height:1.5;border:1px solid #183B3518}.sec{font-size:15px;font-weight:700;color:#183B35;margin:18px 0 6px;text-transform:uppercase;letter-spacing:1px}
.pill{display:inline-block;border-radius:99px;padding:2px 10px;font-size:11px;font-weight:700;color:#183B35;white-space:nowrap}.tw{overflow-x:auto;border-radius:8px;border:1px solid #183B3526;margin:8px 0;background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#183B35;color:#F6F3EA;text-align:left;padding:9px 10px;font-weight:600}td{padding:8px 10px;border-top:1px solid #183B3512;color:#263C37;vertical-align:top}tr:hover td{background:#C3E8B722}button.primary{background:#183B35!important;color:#F6F3EA!important;border:none!important;font-weight:700!important}@media(max-width:640px){.hdr h1{font-size:21px}.kv{font-size:19px}}
"""
HEADER = "<div class='hdr'><div class='brand'><div class='logo'>RS</div><div><h1>RESTOCK <span>AI</span></h1><p>Never run out. Never overstock.</p></div></div><div class='badges'><span>Retail Inventory Replenishment Intelligence</span><span>Estimates, not guarantees</span><span>Buyer review required</span><span>No API key needed</span></div></div>"
def mkS(sl, rv, ocm, hpm, oc, hp, ex, sw, wc, ws, wr, lts): return dict(sl=float(sl) / 100, review=int(rv), ocm=float(ocm), hpm=float(hpm), oc=float(oc), hp=float(hp), excess=int(ex), slow=float(sw), wc=float(wc), ws=float(ws), wr=float(wr), ltscale=float(lts))
def run_all(*a):
    try:
        S = mkS(*a[:12]); files = a[12:18]; mapping = a[18]; seed = int(a[19] or 42); demo = all(f is None for f in files)
        if demo and not mapping: X = demo_X(seed, S)
        else: D = load_all(demo_raw(seed) if demo else dict(zip(TN.keys(), [read_csv_any(f) if f is not None else None for f in files])), mapping); X = analyze(D, S, get_fc(D))
        keep = {k: G[k] for k in ("po", "sim") if k in G}; G.clear(); G.update(X=X, D=X["D"], demo=demo, **keep); P = X["P"]
        return note(("Fictional demo retailer analysed. " if demo else "Your data analysed. ") + "%d SKUs, %d days of history, %d need action now, %d at stockout risk." % (len(P), X["D"]["M"].shape[0], int(P["state"].isin(ST[:3]).sum()), int(P["state"].isin(ST[:2]).sum())), "ok")
    except Exception as e: return note("Could not complete the analysis: " + esc(e), "err")
def refresh():
    if "X" not in G: return tuple(gr.update() for _ in range(5))
    P = G["X"]["P"]; locs = sorted({q for d in G["X"]["locs"].values() for q in d}); sk = list(P.sort_values("pscore", ascending=False)["sku"])
    return gr.update(choices=sorted(P["cat"].unique()), value=[]), gr.update(choices=[(n, i) for i, n in sorted(set(zip(P["sup"], P["sname"])))], value=[]), gr.update(choices=locs, value=[]), gr.update(choices=sk, value=sk[0]), gr.update(choices=sk, value=sk[0])
def build_ui(zip_path, sfiles):
    with gr.Blocks(css=CSS, title="RESTOCK AI", theme=gr.themes.Base(primary_hue="green", neutral_hue="stone")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("Settings: service level, review period, costs, supplier weights (press Recalculate after changing)", open=False):
            with gr.Row():
                s1 = gr.Slider(80, 99.9, 95, step=.1, label="Service-level target (%)"); s2 = gr.Slider(1, 60, 14, step=1, label="Review period (days)"); s3 = gr.Slider(.25, 3, 1, step=.05, label="Ordering-cost multiplier"); s4 = gr.Slider(.25, 3, 1, step=.05, label="Holding-cost multiplier"); s12 = gr.Slider(.5, 2, 1, step=.05, label="Supplier lead-time scale")
            with gr.Row():
                s5 = gr.Number(500, label="Default ordering cost per order (₹, if not supplied)"); s6 = gr.Number(22, label="Default annual holding cost (%, if not supplied)"); s7 = gr.Slider(30, 365, 120, step=10, label="Overstocked if cover exceeds (days)"); s8 = gr.Number(1.0, label="Slow-moving if demand below (units/day)")
            with gr.Row():
                s9 = gr.Slider(0, 1, .35, step=.05, label="Supplier weight: cost"); s10 = gr.Slider(0, 1, .35, step=.05, label="Supplier weight: speed"); s11 = gr.Slider(0, 1, .30, step=.05, label="Supplier weight: reliability"); seed = gr.Number(42, label="Demo seed"); run_b = gr.Button("Recalculate", variant="primary"); rg_b = gr.Button("Regenerate demo")
        with gr.Accordion("Filters (apply to every page)", open=True):
            with gr.Row():
                f_c = gr.Dropdown([], multiselect=True, label="Category"); f_s = gr.Dropdown([], multiselect=True, label="Supplier"); f_l = gr.Dropdown([], multiselect=True, label="Location (has stock)"); f_p = gr.Dropdown(["High", "Medium", "Low", "—"], multiselect=True, label="Priority"); f_st = gr.Dropdown(ST, multiselect=True, label="Status"); ap_b = gr.Button("Apply filters", variant="primary")
        status = gr.HTML(note("Loading the fictional retailer, backtesting forecasts and running self-tests (about 2 minutes)...", "info")); FIN = [f_c, f_s, f_l, f_p, f_st]
        with gr.Tabs():
            with gr.Tab("Overview"):
                ov_h = gr.HTML()
                with gr.Row():
                    ov1 = gr.Plot(); ov2 = gr.Plot()
                with gr.Row():
                    ov3 = gr.Plot(); ov4 = gr.Plot()
            with gr.Tab("Demand Forecasting"):
                with gr.Row():
                    fc_hz = gr.Dropdown(["7", "14", "30", "90"], value="30", label="Horizon (days)"); fc_sk = gr.Dropdown([], label="SKU for charts"); fc_b = gr.Button("Update", variant="primary")
                fc_h = gr.HTML()
                with gr.Row():
                    fc1 = gr.Plot(); fc2 = gr.Plot()
                fc3 = gr.Plot()
            with gr.Tab("Replenishment Engine"):
                rp_h = gr.HTML(); rp1 = gr.Plot()
            with gr.Tab("Inventory Risk"):
                rk_h = gr.HTML()
                with gr.Row():
                    rk1 = gr.Plot(); rk2 = gr.Plot()
            with gr.Tab("Suppliers"):
                su_sk = gr.Dropdown([], label="SKU offer comparison"); su_b = gr.Button("Compare", variant="primary"); su_h = gr.HTML()
                with gr.Row():
                    su1 = gr.Plot(); su2 = gr.Plot()
            with gr.Tab("Cost Optimisation"):
                co_h = gr.HTML()
                with gr.Row():
                    co1 = gr.Plot(); co2 = gr.Plot()
            with gr.Tab("Replenishment Workbench"):
                wb_h = gr.HTML(); wb_df = gr.Dataframe(headers=WBC, datatype=WBT, interactive=True, wrap=True, label="Order planning table (edit order_qty and supplier; tick select)"); po_b = gr.Button("Build proposed purchase order", variant="primary"); po_h = gr.HTML(); po_f = gr.File(label="Proposed PO (CSV)", interactive=False)
            with gr.Tab("What-If Simulator"):
                gr.Markdown("Compare today's policy with a modified one under changed conditions. Outputs are **simulated estimates**, not guarantees.")
                with gr.Row():
                    m1 = gr.Slider(-50, 100, 0, step=5, label="Demand change (%)"); m2 = gr.Slider(0, 30, 0, step=1, label="Supplier lead-time delay (days)"); m3 = gr.Slider(-20, 50, 0, step=1, label="Supplier cost change (%)"); m4 = gr.Slider(80, 99.9, 95, step=.1, label="Service level of modified policy (%)")
                with gr.Row():
                    m5 = gr.Slider(0, 3, 1, step=.1, label="Safety-stock multiplier"); m6 = gr.Slider(.5, 3, 1, step=.1, label="Order-quantity multiplier"); m7 = gr.Slider(.5, 3, 1, step=.1, label="MOQ multiplier"); m11 = gr.Slider(0, 80, 0, step=5, label="Supply disruption (% of delivered quantity lost)")
                with gr.Row():
                    m8 = gr.Slider(0, 200, 0, step=10, label="Promotional demand surge (%)"); m9 = gr.Slider(0, 80, 10, step=1, label="Promotion starts on day"); m10 = gr.Slider(1, 30, 7, step=1, label="Promotion length (days)"); m12 = gr.Slider(30, 90, 90, step=10, label="Simulation horizon (days)"); m13 = gr.Checkbox(False, label="Modified policy re-plans for the new demand and lead times"); m14 = gr.Number(7, label="Random seed")
                sm_b = gr.Button("Run simulation", variant="primary"); sm_h = gr.HTML()
                with gr.Row():
                    sm1 = gr.Plot(); sm2 = gr.Plot()
            with gr.Tab("AI Inventory Advisor"):
                gr.Markdown("Answers come from the calculated results. Try: *Which products need ordering first?* · *Which SKUs are at stockout risk?* · *Is more safety stock justified?* · *What is overstocked and where is cash tied up?* · *Which suppliers create risk?* · *Why is SKU001 a problem?*")
                ag_q = gr.Textbox(label="Ask a question"); ag_b = gr.Button("Ask", variant="primary"); ag_o = gr.Markdown()
            with gr.Tab("Data & Reports"):
                gr.Markdown("Upload CSVs: **sales_history, product_master, current_inventory, supplier_info** are required; **incoming_purchase_orders** and **product_costs** are optional. Leave empty for the fictional demo. Raw uploads are never modified.")
                up = []
                with gr.Row():
                    for k in TN: up.append(gr.File(label=TN[k] + ".csv" + ("" if k in REQ else " (optional)"), file_types=[".csv"]))
                mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="units=qty_sold, on_hand=stock_qty", lines=2); ld_b = gr.Button("Validate and load uploaded data", variant="primary")
                gr.File(value=sfiles, label="Sample CSV datasets (synthetic)", file_count="multiple", interactive=False); dq_h = gr.HTML(); t_h = gr.HTML()
                gr.Markdown("### Exports"); xp_b = gr.Button("Generate reports", variant="primary"); xp_m = gr.HTML(); xp_f = gr.File(label="Reports", file_count="multiple", interactive=False); gr.File(value=zip_path, label="Download GitHub-ready project (ZIP)", interactive=False)
            with gr.Tab("Methodology"):
                gr.Markdown(METH)
        RUN = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12] + up + [mp, seed]; SIM = FIN + [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14]
        def pages(e):
            e = e.then(r_ov, FIN, [ov_h, ov1, ov2, ov3, ov4]).then(r_fc, FIN + [fc_hz, fc_sk], [fc_h, fc1, fc2, fc3]).then(r_rep, FIN, [rp_h, rp1]).then(r_risk, FIN, [rk_h, rk1, rk2]).then(r_sup, FIN + [su_sk], [su_h, su1, su2]).then(r_cost, FIN, [co_h, co1, co2])
            return e.then(r_wb, FIN, [wb_h, wb_df]).then(r_sim, SIM, [sm_h, sm1, sm2]).then(r_dq, None, [dq_h, t_h])
        def full(ev): return pages(ev.then(refresh, None, [f_c, f_s, f_l, fc_sk, su_sk]))
        full(run_b.click(run_all, RUN, [status], show_progress="full")); full(ld_b.click(run_all, RUN, [status], show_progress="full")); full(demo.load(run_all, RUN, [status])); full(rg_b.click(lambda s: (s or 42) + 1, seed, seed).then(run_all, RUN, [status], show_progress="full"))
        ap_b.click(r_ov, FIN, [ov_h, ov1, ov2, ov3, ov4]).then(r_fc, FIN + [fc_hz, fc_sk], [fc_h, fc1, fc2, fc3]).then(r_rep, FIN, [rp_h, rp1]).then(r_risk, FIN, [rk_h, rk1, rk2]).then(r_sup, FIN + [su_sk], [su_h, su1, su2]).then(r_cost, FIN, [co_h, co1, co2]).then(r_wb, FIN, [wb_h, wb_df])
        fc_b.click(r_fc, FIN + [fc_hz, fc_sk], [fc_h, fc1, fc2, fc3]); su_b.click(r_sup, FIN + [su_sk], [su_h, su1, su2]); po_b.click(build_po, wb_df, [po_h, po_f]); sm_b.click(r_sim, SIM, [sm_h, sm1, sm2], show_progress="full"); xp_b.click(export_all, None, [xp_f, xp_m], show_progress="full"); ag_b.click(ask, ag_q, ag_o); ag_q.submit(ask, ag_q, ag_o)
    return demo
def main():
    sf = write_samples(); zp, saved = build_project(sf)
    print("=" * 78); print("RESTOCK AI starting. Demo data is FICTIONAL. No API key needed."); print("Backtesting forecasts and running self-tests (about 2 minutes):")
    demo_X(); run_selftests(verbose=True); demo = build_ui(zp, sf)
    print("Project ZIP:", zp, "| standalone app.py included:", "yes" if saved else "NO (create app.py yourself by pasting this cell into it)")
    print("Not included / not tested: Streamlit (Gradio used), PDF export, optional LLM, uploads of your own CSVs. Look for the line 'Running on public URL: https://....gradio.live' below: that is the link to share.")
    kw = dict(share=("SPACE_ID" not in os.environ), debug=False, show_error=True)
    if APP_PASSWORD: kw["auth"] = ("restock", APP_PASSWORD)
    print("=" * 78); demo.queue().launch(**kw)
if __name__ == "__main__" or "google.colab" in sys.modules:
    main()