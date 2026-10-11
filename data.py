"""Read-only election data for Election Lens.

Pulls election markets from prediction-market venues through the synpath SDK,
works out which state / office / district each market is about from its title,
and rolls the markets up into races with a Democratic and Republican win
probability. Nothing here can place an order.
"""
from __future__ import annotations

import math
import random
import re

import numpy as np
import pandas as pd

VENUES = {"kalshi": "Kalshi", "polymarket_us": "Polymarket US", "polymarket": "Polymarket (intl)"}

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
CODE_TO_STATE = {v: k for k, v in STATES.items()}
# Longest names first so "West Virginia" wins over "Virginia", "Arkansas" over "Kansas".
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATES), key=len, reverse=True)) + r")\b", re.I)
_DISTRICT_CODE_RE = re.compile(r"\b(" + "|".join(STATES.values()) + r")-?(\d{1,2}|AL)\b")
_DISTRICT_NUM_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:congressional\s+)?district|district\s+(\d{1,2})", re.I)

SEARCHES = [
    "Senate race", "Senate election 2026", "Governor race", "Governor election 2026",
    "gubernatorial", "House race", "congressional district", "Senate control",
    "House control", "midterms 2026",
]

RACE_COLS = ["race_id", "venue", "office", "state", "code", "district", "title", "dem_p", "rep_p",
             "leader", "leader_p", "chg_24h", "volume", "vol_24h", "spread", "close", "url", "markets"]


# ----------------------------------------------------------------------------
# Title parsing
# ----------------------------------------------------------------------------
def detect_state(text: str) -> tuple[str | None, str | None]:
    m = _STATE_RE.search(text or "")
    if m:
        name = next(k for k in STATES if k.lower() == m.group(1).lower())
        return name, STATES[name]
    m = _DISTRICT_CODE_RE.search(text or "")
    if m:
        return CODE_TO_STATE[m.group(1)], m.group(1)
    return None, None


def detect_office(text: str) -> str | None:
    t = (text or "").lower()
    if re.search(r"\b(control|balance of power|majority)\b", t) and re.search(r"\b(senate|house)\b", t) \
            and not _STATE_RE.search(text or ""):
        return "Senate control" if "senate" in t else "House control"
    if re.search(r"\bstate (senate|house|assembly|legislature)\b|\blieutenant governor\b|\bmayor", t):
        return None  # state-legislature / local races are out of scope
    if re.search(r"\bgovernor|gubernatorial\b", t):
        return "Governor"
    if re.search(r"\bsenate|senator\b", t):
        return "Senate"
    if (re.search(r"(?<!white )\bhouse\b|congressional|\bdistrict\b", t)) or _DISTRICT_CODE_RE.search(text or ""):
        return "House"
    return None


def detect_district(text: str) -> str | None:
    m = _DISTRICT_CODE_RE.search(text or "")
    if m:
        return m.group(2).lstrip("0") or "AL"
    m = _DISTRICT_NUM_RE.search(text or "")
    if m:
        return (m.group(1) or m.group(2)).lstrip("0")
    if re.search(r"at[- ]large", text or "", re.I):
        return "AL"
    return None


def detect_party(text: str) -> str | None:
    t = text or ""
    if re.search(r"\bdemocrat(ic|s)?\b|\bdems?\b|\(D\)|\bD\s*-\s*[A-Z]{2}\b", t, re.I):
        return "D"
    if re.search(r"\brepublican(s)?\b|\bGOP\b|\(R\)|\bR\s*-\s*[A-Z]{2}\b", t, re.I):
        return "R"
    if re.search(r"\bindependent\b|\(I\)", t, re.I):
        return "I"
    return None


def _name_tokens(name: str) -> list[str]:
    """Surname-ish tokens of a candidate name, for matching inside longer text."""
    toks = [t for t in re.split(r"[\s,]+", name or "") if len(t) > 2 and t[0].isupper()]
    return toks[-1:] or toks


def party_near_name(name: str | None, text: str | None) -> str | None:
    """Party written right next to the candidate's name, e.g. "El-Sayed (D)" or
    "Democrat Abdul El-Sayed". Whole-race text often names both parties, so only
    party words touching the name count."""
    if not name or not text:
        return None
    for tok in _name_tokens(name):
        n = re.escape(tok)
        pats = [
            (rf"{n}\s*\((D|Dem|Democrat|Democratic)\b", "D"), (rf"{n}\s*\((R|Rep|Republican|GOP)\b", "R"),
            (rf"{n}\s*\((I|Ind|Independent)\b", "I"),
            (rf"\bDemocrat(?:ic)?(?:\s+(?:candidate|nominee|challenger|incumbent|Senator|Sen\.|Rep\.|Gov\.))?\s+(?:[A-Z][\w.'-]+\s+){{0,2}}{n}\b", "D"),
            (rf"\bRepublican(?:\s+(?:candidate|nominee|challenger|incumbent|Senator|Sen\.|Rep\.|Gov\.))?\s+(?:[A-Z][\w.'-]+\s+){{0,2}}{n}\b", "R"),
            (rf"\b{n}\b[^.;\n]{{0,40}}\b(?:the\s+)?Democratic\s+(?:nominee|candidate|party)", "D"),
            (rf"\b{n}\b[^.;\n]{{0,40}}\b(?:the\s+)?Republican\s+(?:nominee|candidate|party)", "R"),
        ]
        for pat, party in pats:
            if re.search(pat, text):
                return party
    return None


def load_overrides(path: str = "parties.csv") -> dict[str, str]:
    """Optional name,party CSV for candidates the venues don't label."""
    try:
        df = pd.read_csv(path, comment="#")
        return {str(n).strip().lower(): str(p).strip().upper()[:1]
                for n, p in zip(df["name"], df["party"]) if str(n).strip()}
    except Exception:
        return {}


def single_party(text: str) -> str | None:
    """A party only when the text names exactly one (so "Dem vs GOP" titles don't count)."""
    t = text or ""
    found = {p for p, pat in (("D", r"\bdemocrat(ic|s)?\b|\bdems?\b|\(D\)"),
                              ("R", r"\brepublican(s)?\b|\bGOP\b|\(R\)")) if re.search(pat, t, re.I)}
    return found.pop() if len(found) == 1 else None


_NOT_WINNER = re.compile(
    r"\bprimar(y|ies)\b|\bnomin(ee|ation|ated)\b|\brunoff\b|\bmargin\b|\bby (more|less) than\b"
    r"|\bvote share\b|\bpopular vote\b|\bturnout\b|\bdebate\b|\bendorse|\bdrop(s)? out\b|\brecount\b"
    r"|\bconcede\b|\bpoll(s|ing)?\b|\bwin by\b|\bpercent(age)? of\b|\braise\b|\bfundrais"
    r"|\bcombo\b|\bparlay\b|\bsweep\b|\bboth\b|\ball (of|three|four|five)\b|\bany of\b", re.I)


def count_states(text: str) -> int:
    """How many different states a title names (combos/parlays name several)."""
    found = {m.lower() for m in _STATE_RE.findall(text or "")}
    found |= {c for c, _ in _DISTRICT_CODE_RE.findall(text or "")}
    return len(found)


def is_winner_question(text: str) -> bool:
    """General-election 'who wins' markets only; drops primaries, margins, turnout, debates, etc."""
    return not _NOT_WINNER.search(text or "")


def detect_cycle(text: str) -> bool:
    """True unless the title clearly names a different cycle than 2026."""
    years = re.findall(r"\b(20\d\d)\b", text or "")
    return not years or "2026" in years


# ----------------------------------------------------------------------------
# Live fetch
# ----------------------------------------------------------------------------
_EX: dict[str, object] = {}


def _venue(v):
    import synpath
    if v not in _EX:
        _EX[v] = synpath.exchange(v)
    return _EX[v]


def _prob(q):
    if q.mid is not None:
        return float(q.mid)
    if q.bid is not None and q.ask is not None:
        return (float(q.bid) + float(q.ask)) / 2
    if q.last is not None:
        return float(q.last)
    return None


def _row(m, event_title: str | None) -> dict:
    q = m.yes.quote
    label = m.outcome_label or (m.yes.label if m.yes.label and m.yes.label.lower() not in ("yes", "y") else None)
    return {
        "market_id": m.id, "venue": m.venue, "event_id": m.event_id or m.id,
        "event_title": event_title or m.title, "title": m.title, "label": label,
        "prob": _prob(q),
        "spread": (float(q.ask) - float(q.bid)) if q.ask is not None and q.bid is not None else None,
        "chg_24h": m.yes.price_change_24h,
        "vol_24h": m.stats.volume_24h, "volume": m.stats.volume_total,
        "close": pd.to_datetime(m.close_timestamp, unit="ms", utc=True) if m.close_timestamp else pd.NaT,
        "url": m.url,
        "desc": " ".join(str(x) for x in (m.description, (m.info or {}).get("subtitle"),
                                           (m.info or {}).get("yes_sub_title")) if x)[:3000],
    }


def fetch_markets(venue: str, pages: int = 3) -> pd.DataFrame:
    """Every election-looking market on one venue, deduplicated."""
    ex, rows, seen = _venue(venue), [], set()
    by_event = hasattr(ex, "search_events")
    for q in SEARCHES:
        cursor = None
        for _ in range(pages):
            if by_event:
                page = ex.search_events(q, limit=100, cursor=cursor)
                pairs = [(m, e.title) for e in page for m in e.markets]
            else:
                page = ex.search_markets(q, limit=100, cursor=cursor)
                pairs = [(m, None) for m in page]
            for m, etitle in pairs:
                if m.id not in seen:
                    seen.add(m.id)
                    rows.append(_row(m, etitle))
            cursor = page.next_cursor
            if not cursor:
                break
    return pd.DataFrame(rows)


def fetch_history(market_id: str, days: int = 60) -> pd.DataFrame:
    v = market_id.split(":", 1)[0]
    cs = _venue(v).fetch_ohlcv(market_id, timeframe="1d", limit=days)
    df = pd.DataFrame([{"time": pd.to_datetime(c.timestamp, unit="ms", utc=True), "close": c.close,
                        "volume": c.volume or 0.0} for c in cs])
    return df.dropna(subset=["close"]) if not df.empty else df


# ----------------------------------------------------------------------------
# Markets -> races
# ----------------------------------------------------------------------------
def assign_parties(m: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Work out each outcome's party, most reliable source first.

    1. Your parties.csv override  2. the outcome label  3. the market title
    4. party written beside the candidate's name in the rules/description
    5. inference: in a race where only one major party was found, the leading
       unlabeled candidate is taken to be the other major party.
    """
    overrides = overrides if overrides is not None else load_overrides()
    m = m.copy()
    if "desc" not in m:
        m["desc"] = ""
    parties, sources = [], []
    for lab, title, desc in zip(m["label"].fillna(""), m["title"].fillna(""), m["desc"].fillna("")):
        lab, title = lab.strip(), title.strip()
        checks = [
            ("override", overrides.get(lab.lower()) if lab else None),
            ("override", overrides.get(title.lower()) if not lab else None),
            ("label", detect_party(lab)),
            ("title", party_near_name(lab, title) if lab else None),
            ("title", single_party(title)),
            ("rules", party_near_name(lab, desc) if lab else None),
        ]
        hit = next(((src, val) for src, val in checks if val), (None, None))
        sources.append(hit[0]), parties.append(hit[1])
    m["party"], m["party_src"] = parties, sources

    for _, idx in m.groupby(["venue", "event_id"], sort=False).groups.items():
        g = m.loc[idx]
        if len(g) < 2:
            continue
        known = set(g["party"].dropna()) & {"D", "R"}
        unknown = g[g["party"].isna()].sort_values("prob", ascending=False)
        if len(known) == 1 and len(unknown):
            other = "R" if "D" in known else "D"
            top = unknown.index[0]
            m.loc[top, "party"], m.loc[top, "party_src"] = other, "inferred"
    return m


def build_races(markets: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Group markets into races and compute Democratic / Republican odds."""
    if markets is None or markets.empty:
        return pd.DataFrame(columns=RACE_COLS)
    m = markets.copy()
    m["text"] = m["event_title"].fillna("") + " | " + m["title"].fillna("")
    m = m[m["text"].map(detect_cycle)]
    m["office"] = m["text"].map(detect_office)
    m = m[m["office"].notna()]
    m = m[m["text"].map(is_winner_question)]
    m = m[m["text"].map(count_states) <= 1]   # drop multi-state combos/parlays
    st = m["text"].map(detect_state)
    m["state"] = [s[0] for s in st]
    m["code"] = [s[1] for s in st]
    m["district"] = m["text"].map(detect_district)
    m.loc[m["office"] != "House", "district"] = None
    # "Will Democrats win the House?" has no state: that's a chamber-control question.
    no_state = m["state"].isna() & m["office"].isin(["Senate", "House"])
    m.loc[no_state, "office"] = m.loc[no_state, "office"] + " control"
    m = m[(m["state"].notna()) | m["office"].str.endswith("control")]
    m = assign_parties(m, overrides)

    races = []
    for (venue, eid), g in m.groupby(["venue", "event_id"], sort=False):
        office = g["office"].mode().iloc[0]
        first = g.iloc[0]
        dem = g.loc[g.party == "D", "prob"].dropna()
        rep = g.loc[g.party == "R", "prob"].dropna()
        dem_p = float(dem.max()) if len(dem) else None
        rep_p = float(rep.max()) if len(rep) else None
        if dem_p is None and rep_p is not None and len(g) == 1:
            dem_p = 1 - rep_p  # single binary "Will Republicans win X?"
        if rep_p is None and dem_p is not None and len(g) == 1:
            rep_p = 1 - dem_p
        if dem_p is not None and rep_p is not None and dem_p + rep_p > 0:
            # Normalize the two-party share so spreads/overround don't distort the map.
            tot = dem_p + rep_p
            dem_n, rep_n = dem_p / tot, rep_p / tot
        else:
            dem_n, rep_n = dem_p, rep_p
        best = g.loc[g["prob"].idxmax()] if g["prob"].notna().any() else first
        leader = best["label"] or ({"D": "Democrat", "R": "Republican"}.get(best["party"]) or best["title"])
        dem_mkt = g.loc[g.party == "D", "market_id"]
        races.append({
            "race_id": f"{venue}:{eid}", "venue": venue, "office": office,
            "state": first["state"], "code": first["code"], "district": first["district"],
            "title": first["event_title"], "dem_p": dem_n, "rep_p": rep_n,
            "leader": leader, "leader_p": best["prob"],
            "chg_24h": (g.loc[g.party == "D", "chg_24h"].dropna().mean()
                        if (g.party == "D").any() else
                        -g.loc[g.party == "R", "chg_24h"].dropna().mean() if (g.party == "R").any() else None),
            "volume": g["volume"].sum(min_count=1), "vol_24h": g["vol_24h"].sum(min_count=1),
            "spread": g["spread"].mean(), "close": g["close"].max(), "url": first["url"],
            "markets": g[["market_id", "label", "title", "party", "party_src", "prob", "chg_24h", "vol_24h"]].to_dict("records"),
            "party_inferred": bool((g["party_src"] == "inferred").any()),
            "two_party": bool(len(dem) and len(rep)),
            "dem_market": dem_mkt.iloc[0] if len(dem_mkt) else None,
        })
    df = pd.DataFrame(races)
    if df.empty:
        df = pd.DataFrame(columns=RACE_COLS)
    return df


def filter_report(markets: pd.DataFrame) -> dict:
    """How many markets survive each filtering step (for the in-app data check)."""
    if markets is None or markets.empty:
        return {"fetched": 0}
    t = markets["event_title"].fillna("") + " | " + markets["title"].fillna("")
    out = {"fetched": len(t)}
    t = t[t.map(detect_cycle)]; out["2026 cycle"] = len(t)
    t = t[t.map(detect_office).notna()]; out["Senate/Gov/House"] = len(t)
    t = t[t.map(is_winner_question)]; out["winner questions"] = len(t)
    t = t[t.map(count_states) <= 1]; out["single state"] = len(t)
    t = t[t.map(lambda x: detect_state(x)[0] is not None or bool(re.search(r"control|majority", x, re.I)))]
    out["state found"] = len(t)
    return out


def pick_best(races: pd.DataFrame) -> pd.DataFrame:
    """One row per race: the venue with the most volume.

    Within a venue, a race listed twice keeps the market that has both major
    parties and the most volume. The venue gap compares only *different*
    venues, and the other venues' versions are kept so they can be checked.
    """
    if races.empty:
        return races
    r = races.copy()
    r["key"] = r["office"] + "|" + r["code"].fillna("US") + "|" + r["district"].fillna("")
    if "two_party" not in r:
        r["two_party"] = r["dem_p"].notna() & r["rep_p"].notna()
    r = r.sort_values(["two_party", "volume"], ascending=[False, False], na_position="last")
    per_venue = r.drop_duplicates(["key", "venue"])          # best market on each venue
    best = per_venue.drop_duplicates("key").copy()          # best venue overall

    def gap(g):
        d = g["dem_p"].dropna()
        return d.max() - d.min() if g["venue"].nunique() > 1 and len(d) > 1 else np.nan

    def others(g):
        return [{"venue": v, "title": t, "dem_p": d, "volume": vol}
                for v, t, d, vol in zip(g["venue"], g["title"], g["dem_p"], g["volume"])]

    best["venue_gap"] = best["key"].map(per_venue.groupby("key").apply(gap, include_groups=False))
    best["versions"] = best["key"].map(per_venue.groupby("key").apply(others, include_groups=False))
    return best


# ----------------------------------------------------------------------------
# Programmatic planning metrics
# ----------------------------------------------------------------------------
def add_planning(r: pd.DataFrame) -> pd.DataFrame:
    """Competitiveness, momentum and a 0-100 priority score per race."""
    if r.empty:
        return r
    r = r.copy()
    r["margin"] = (r["dem_p"] - 0.5).abs() * 2  # 0 = coin flip, 1 = certain
    r["competitiveness"] = (1 - r["margin"]).clip(0, 1)
    r["momentum"] = r["chg_24h"].abs().fillna(0)
    vol = np.log1p(r["volume"].fillna(0))
    r["attention"] = vol / vol.max() if vol.max() > 0 else 0.0
    office_w = r["office"].map({"Senate": 1.0, "Governor": 0.9, "House": 0.7}).fillna(0.5)
    r["priority"] = (100 * office_w * (0.65 * r["competitiveness"] + 0.2 * (r["momentum"] / 0.10).clip(0, 1)
                                        + 0.15 * r["attention"])).round(0)
    r["tier"] = pd.cut(r["competitiveness"], [-0.01, 0.3, 0.6, 0.8, 1.01],
                       labels=["Safe", "Likely", "Lean", "Toss-up"]).astype(str)
    return r


def rating(dem_p: float | None) -> str:
    if dem_p is None or pd.isna(dem_p):
        return "No data"
    if dem_p >= 0.85: return "Safe D"
    if dem_p >= 0.65: return "Likely D"
    if dem_p >= 0.55: return "Lean D"
    if dem_p > 0.45: return "Toss-up"
    if dem_p > 0.35: return "Lean R"
    if dem_p > 0.15: return "Likely R"
    return "Safe R"


# ----------------------------------------------------------------------------
# Demo data (synthetic, generic candidates)
# ----------------------------------------------------------------------------
_DEMO_SENATE = ["AL", "AK", "AR", "CO", "DE", "FL", "GA", "ID", "IL", "IA", "KS", "KY", "LA", "ME", "MA",
                "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM", "NC", "OH", "OK", "OR", "RI", "SC", "SD",
                "TN", "TX", "VA", "WV", "WY"]
_DEMO_GOV = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "FL", "GA", "HI", "ID", "IL", "IA", "KS", "ME", "MD",
             "MA", "MI", "MN", "NE", "NV", "NH", "NM", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
             "TX", "VT", "WI", "WY"]
# Rough partisan lean used only to make demo maps look plausible.
_LEAN = {"AL": -2, "AK": -1, "AZ": 0, "AR": -2, "CA": 2, "CO": 1, "CT": 1.5, "DE": 1.5, "FL": -1, "GA": -.2,
         "HI": 2, "ID": -2.5, "IL": 1.5, "IN": -1.5, "IA": -1, "KS": -1.3, "KY": -2, "LA": -1.8, "ME": .6,
         "MD": 2, "MA": 2.2, "MI": .1, "MN": .6, "MS": -1.6, "MO": -1.5, "MT": -1.4, "NE": -1.6, "NV": 0,
         "NH": .4, "NJ": 1, "NM": .8, "NY": 1.6, "NC": -.1, "ND": -2.4, "OH": -.7, "OK": -2.5, "OR": 1.2,
         "PA": 0, "RI": 1.8, "SC": -1.4, "SD": -2, "TN": -2, "TX": -.8, "UT": -2, "VT": 2, "VA": .5,
         "WA": 1.5, "WV": -2.5, "WI": 0, "WY": -3}


def demo_markets(venue: str) -> pd.DataFrame:
    r = random.Random(f"{venue}-elections")   # venue-specific noise
    base = random.Random("elections-base")     # race odds shared by all venues
    rows = []

    def add(eid, etitle, office_txt, p_dem, code):
        noise = r.gauss(0, 0.012) if venue != "kalshi" else 0
        p = min(0.98, max(0.02, p_dem + noise))
        vol = r.lognormvariate(10 + 2 * (1 - abs(p - .5) * 2), 1.0)
        for party, pp in (("Democratic", p), ("Republican", 1 - p)):
            rows.append({
                "market_id": f"{venue}:{eid}-{party[0]}", "venue": venue, "event_id": eid,
                "event_title": etitle, "title": f"{etitle}: {party} candidate", "label": f"{party} party",
                "prob": round(pp + r.choice([-.005, 0, .005]), 3), "spread": r.choice([.01, .01, .02, .03]),
                "chg_24h": round(r.gauss(0, .012 * (1 - abs(p - .5) * 1.6)) * (1 if party[0] == "D" else -1), 3),
                "vol_24h": vol / r.uniform(15, 40), "volume": vol,
                "close": pd.Timestamp("2026-11-04", tz="UTC"), "url": None,
            })

    for code in _DEMO_SENATE:
        p = 1 / (1 + math.exp(-(_LEAN[code] * 1.6 + base.gauss(0, .4))))
        add(f"SEN-{code}", f"{CODE_TO_STATE[code]} Senate race 2026 winner", "Senate", p, code)
    for code in _DEMO_GOV:
        p = 1 / (1 + math.exp(-(_LEAN[code] * 1.4 + base.gauss(0, .6))))
        add(f"GOV-{code}", f"{CODE_TO_STATE[code]} Governor race 2026 winner", "Governor", p, code)
    for code, n in [("PA", 7), ("PA", 8), ("MI", 7), ("MI", 10), ("NY", 17), ("CA", 13), ("CA", 22),
                    ("AZ", 1), ("AZ", 6), ("NE", 2), ("CO", 8), ("VA", 2), ("WI", 3), ("OH", 9), ("ME", 2)]:
        p = 1 / (1 + math.exp(-base.gauss(0, .5)))
        add(f"HOUSE-{code}{n}", f"{code}-{n:02d} House race 2026 winner", "House", p, code)
    for name in ("Senate", "House"):
        p = base.uniform(.3, .7)
        add(f"CTRL-{name}", f"Which party will control the {name} after the 2026 midterms?", name, p, None)
    return pd.DataFrame(rows)


def demo_history(market_id: str, days: int = 60) -> pd.DataFrame:
    r = random.Random(market_id)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("1D"), periods=days, freq="1D")
    x, xs = r.uniform(-1, 1), []
    for _ in idx:
        x += r.gauss(0, .08)
        xs.append(1 / (1 + math.exp(-x)))
    return pd.DataFrame({"time": idx, "close": np.round(xs, 3), "volume": [r.lognormvariate(7, 1) for _ in idx]})
