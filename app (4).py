"""Election Lens — 2026 midterm win odds by state, from prediction markets.

Run:  streamlit run app.py
Read-only: it reads market prices and never places orders.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D

st.set_page_config(page_title="Election Lens", page_icon="🗳️", layout="wide")

DEM, REP, NEUTRAL = "#2f6fb3", "#c8423b", "#b8bcc4"
ELECTION_DAY = date(2026, 11, 3)


def pct(x, signed=False, digits=0):
    if x is None or pd.isna(x):
        return "—"
    return (f"{x * 100:+.{digits}f}pt" if signed else f"{x * 100:.{digits}f}%")


def num(x):
    if x is None or pd.isna(x):
        return "—"
    for u, d in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.1f}{u}"
    return f"{x:,.0f}"


def race_name(r) -> str:
    if r["office"] == "House":
        return f"{r['code']}-{r['district'] or '?'} House"
    if r["office"].endswith("control"):
        return r["office"]
    return f"{r['state']} {r['office']}"


@st.cache_data(ttl=300, show_spinner=False)
def load(demo: bool, venues: tuple) -> tuple[pd.DataFrame, list[str]]:
    frames, errors = [], []
    for v in venues:
        try:
            m = D.demo_markets(v) if demo else D.fetch_markets(v)
            frames.append(D.build_races(m))
        except Exception as e:  # one venue failing shouldn't blank the app
            errors.append(f"{D.VENUES[v]}: {e}")
    allr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=D.RACE_COLS)
    return allr, errors


@st.cache_data(ttl=600, show_spinner=False)
def history(demo: bool, market_id: str):
    return D.demo_history(market_id) if demo else D.fetch_history(market_id)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🗳️ Election Lens")
st.sidebar.caption("2026 midterm win odds from prediction markets. Read-only.")
demo = st.sidebar.toggle("Demo data (offline)", value=False,
                         help="Synthetic races with generic candidates, for trying the app without network access.")
venues = st.sidebar.multiselect("Venues", list(D.VENUES), default=["kalshi", "polymarket_us"],
                                format_func=D.VENUES.get)
min_vol = st.sidebar.number_input("Hide races with total volume below", 0, value=0, step=1000,
                                  help="Thin markets give unreliable odds.")
if st.sidebar.button("↻ Refresh"):
    st.cache_data.clear()
days_left = (ELECTION_DAY - date.today()).days
st.sidebar.metric("Days to Election Day", max(days_left, 0), help="Tuesday, November 3, 2026")
if demo:
    st.sidebar.info("Demo mode: made-up odds, generic candidates.")

with st.spinner("Loading election markets…"):
    all_races, errs = load(demo, tuple(venues))
for e in errs:
    st.warning(f"Couldn't load {e}")
if all_races.empty:
    st.info("No election markets found. Check your venues, or switch on demo data to explore.")
    st.stop()

races = D.add_planning(D.pick_best(all_races))
if min_vol:
    races = races[races["volume"].fillna(0) >= min_vol]
races["race"] = races.apply(race_name, axis=1)
races["rating"] = races["dem_p"].map(D.rating)
state_races = races[~races["office"].str.endswith("control")]

tab_map, tab_toss, tab_ctrl, tab_ads, tab_all = st.tabs(
    ["🗺️ Map", "🎯 Toss-ups & movers", "🏛️ Chamber control", "📣 Ad planner", "📋 All races"])


# ----------------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------------
with tab_map:
    office = st.radio("Office", ["Senate", "Governor", "House"], horizontal=True, key="map_office")
    sub = state_races[state_races["office"] == office].copy()
    if sub.empty:
        st.info(f"No {office} markets found.")
    elif office == "House":
        st.caption("House races are by district, so they're listed rather than mapped. Only districts with a market appear.")
        sub = sub.sort_values("competitiveness", ascending=False)
        st.dataframe(sub[["race", "rating", "dem_p", "leader", "chg_24h", "volume", "venue"]],
                     hide_index=True, width="stretch", column_config={
                         "dem_p": st.column_config.ProgressColumn("Dem win odds", min_value=0, max_value=1, format="%.2f"),
                         "chg_24h": st.column_config.NumberColumn("24h Δ (Dem)", format="%+.3f"),
                         "volume": st.column_config.NumberColumn("Volume", format="%.0f")})
    else:
        sub["hover"] = sub.apply(lambda r: f"{r['state']} {office}<br>{r['rating']}<br>"
                                           f"Dem {pct(r['dem_p'])} · Rep {pct(r['rep_p'])}<br>"
                                           f"Volume {num(r['volume'])}", axis=1)
        fig = go.Figure(go.Choropleth(
            locations=sub["code"], locationmode="USA-states", z=sub["dem_p"].fillna(0.5),
            zmin=0, zmax=1, text=sub["hover"], hoverinfo="text",
            colorscale=[[0, REP], [0.35, "#e8a39d"], [0.5, "#f2f2f2"], [0.65, "#9dbfe3"], [1, DEM]],
            marker_line_color="white", marker_line_width=0.6,
            colorbar=dict(title="Dem odds", tickvals=[0, .25, .5, .75, 1],
                          ticktext=["R 100%", "R 75%", "50/50", "D 75%", "D 100%"]),
        ))
        fig.update_layout(geo=dict(scope="usa", projection_type="albers usa", bgcolor="rgba(0,0,0,0)",
                                   lakecolor="rgba(0,0,0,0)", showlakes=False),
                          height=500, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        counts = sub["rating"].value_counts()
        order = ["Safe D", "Likely D", "Lean D", "Toss-up", "Lean R", "Likely R", "Safe R"]
        st.caption("  ·  ".join(f"**{k}** {counts.get(k, 0)}" for k in order) +
                   f"  ·  states with no {office} market are grey")

        st.markdown("##### State detail")
        opts = sub.sort_values("state")
        pick = st.selectbox("State", range(len(opts)), key="map_state",
                            format_func=lambda i: f"{opts.iloc[i]['state']} — {opts.iloc[i]['rating']}")
        r = opts.iloc[pick]
        c = st.columns(4)
        c[0].metric("Democrat", pct(r["dem_p"]))
        c[1].metric("Republican", pct(r["rep_p"]))
        c[2].metric("24h move (Dem)", pct(r["chg_24h"], True, 1))
        c[3].metric("Total volume", num(r["volume"]))
        gap = r.get("venue_gap")
        if pd.notna(gap) and gap > 0.03:
            versions = r.get("versions") or []
            lines = "\n".join(f"- **{D.VENUES.get(v['venue'], v['venue'])}**: Dem {pct(v['dem_p'])} · "
                              f"volume {num(v['volume'])} · _{v['title']}_" for v in versions)
            if gap > 0.20:
                st.warning(f"Venues differ by {pct(gap)} on this race. A gap this big usually means they aren't "
                           f"the same question (or a party label is off), not real disagreement — compare the titles:\n{lines}\n\n"
                           "If a candidate's party is wrong, fix it in **parties.csv**. The map uses the venue with the most volume.")
            else:
                st.caption(f"Venues disagree by {pct(gap)} on this race — usually thin trading or fast-moving news.")
                with st.expander("Compare venues"):
                    st.markdown(lines)
        mk = pd.DataFrame(r["markets"]).sort_values("prob", ascending=False)
        mk["label"] = mk["label"].fillna(mk["title"])
        party_name = {"D": "Democrat", "R": "Republican", "I": "Independent"}
        src_note = {"inferred": " (inferred)", "override": " (your list)"}
        mk["Party"] = [(party_name.get(p, "Unknown") + src_note.get(s, "")) if p else "Unknown"
                       for p, s in zip(mk["party"], mk.get("party_src", [None] * len(mk)))]
        st.dataframe(mk[["label", "Party", "prob", "chg_24h", "vol_24h"]], hide_index=True, width="stretch",
                     column_config={"label": "Candidate",
                                    "prob": st.column_config.ProgressColumn("Odds", min_value=0, max_value=1, format="%.2f"),
                                    "chg_24h": st.column_config.NumberColumn("24h Δ", format="%+.3f"),
                                    "vol_24h": st.column_config.NumberColumn("24h volume", format="%.0f")})
        if (mk["Party"] == "Unknown").any() or r.get("party_inferred"):
            st.caption("Party is read from the market's title and rules. **(inferred)** means the venue didn't label "
                       "it, so the app assumed the leading unlabeled candidate is the other major party. "
                       "To pin or correct a party, add the name to **parties.csv**.")
        if r.get("dem_market"):
            try:
                h = history(demo, r["dem_market"])
                if h is not None and not h.empty:
                    f2 = go.Figure()
                    f2.add_trace(go.Scatter(x=h["time"], y=h["close"] * 100, name="Democrat", line=dict(color=DEM)))
                    f2.add_trace(go.Scatter(x=h["time"], y=(1 - h["close"]) * 100, name="Republican (implied)",
                                            line=dict(color=REP, dash="dot")))
                    f2.add_hline(y=50, line_color=NEUTRAL, line_dash="dash")
                    f2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Last 60 days",
                                     yaxis=dict(range=[0, 100], ticksuffix="%"), hovermode="x unified",
                                     legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(f2, width="stretch")
            except Exception as e:
                st.caption(f"History unavailable: {e}")
        if r.get("url"):
            st.markdown(f"[Open market on {D.VENUES.get(r['venue'], r['venue'])} ↗]({r['url']})")


# ----------------------------------------------------------------------------
# Toss-ups & movers
# ----------------------------------------------------------------------------
with tab_toss:
    st.subheader("The races that decide it")
    toss = state_races[state_races["dem_p"].between(0.35, 0.65)].sort_values("competitiveness", ascending=False)
    if toss.empty:
        st.info("No races between 35% and 65% right now.")
    else:
        fig = go.Figure(go.Bar(
            y=toss["race"], x=(toss["dem_p"] - 0.5) * 100, orientation="h",
            marker_color=[DEM if v > 0.5 else REP for v in toss["dem_p"]],
            customdata=toss["dem_p"] * 100,
            hovertemplate="Dem %{customdata:.0f}%<extra></extra>"))
        fig.update_layout(height=max(280, 26 * len(toss)), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="← leans Republican · leans Democrat → (points from 50/50)", zeroline=True),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Biggest 24-hour moves")
    mv = state_races.dropna(subset=["chg_24h"])
    mv = mv.reindex(mv["chg_24h"].abs().sort_values(ascending=False).index).head(10)
    for _, r in mv.iterrows():
        if abs(r["chg_24h"]) < 0.005:
            continue
        side = "Democrat" if r["chg_24h"] > 0 else "Republican"
        st.markdown(f"- **{r['race']}** moved {abs(r['chg_24h']) * 100:.1f}pt toward the **{side}** — "
                    f"now Dem {pct(r['dem_p'])} ({r['rating']})")


# ----------------------------------------------------------------------------
# Chamber control
# ----------------------------------------------------------------------------
with tab_ctrl:
    st.subheader("Who controls Congress")
    ctrl = races[races["office"].str.endswith("control")]
    cols = st.columns(2)
    for col, chamber in zip(cols, ["Senate", "House"]):
        with col:
            st.markdown(f"#### {chamber}")
            c = ctrl[ctrl["office"] == f"{chamber} control"]
            if c.empty:
                st.caption("No control market found.")
            else:
                r = c.iloc[0]
                fig = go.Figure(go.Bar(x=[r["dem_p"] * 100, r["rep_p"] * 100], y=["", ""], orientation="h",
                                       marker_color=[DEM, REP], text=[f"Dem {pct(r['dem_p'])}", f"Rep {pct(r['rep_p'])}"],
                                       textposition="inside"))
                fig.update_layout(barmode="stack", height=90, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                                  xaxis=dict(visible=False, range=[0, 100]), yaxis=dict(visible=False))
                st.plotly_chart(fig, width="stretch", key=f"ctrl_{chamber}")
                st.caption(f"Market volume {num(r['volume'])} · {D.VENUES.get(r['venue'], r['venue'])}")
            seats = state_races[state_races["office"] == chamber]
            if len(seats):
                exp_d = seats["dem_p"].fillna(0.5).sum()
                st.markdown(f"Across the **{len(seats)}** {chamber} races that have markets, the odds add up to about "
                            f"**{exp_d:.1f} D / {len(seats) - exp_d:.1f} R** expected wins.")
                st.caption("This counts only races with a market. It isn't a full seat projection; "
                           + ("seats not up this year are excluded." if chamber == "Senate" else
                              "most House districts have no market."))


# ----------------------------------------------------------------------------
# Ad planner (for programmatic political buyers)
# ----------------------------------------------------------------------------
with tab_ads:
    st.subheader("Where the next dollar matters")
    st.caption("Priority (0–100) blends how close the race is (65%), how fast odds are moving (20%) and how much "
               "money is trading on it (15%), weighted by office. Use it to tier geos, not as a spend formula.")
    plan = state_races.sort_values("priority", ascending=False).copy()
    offices = st.multiselect("Offices", ["Senate", "Governor", "House"], default=["Senate", "Governor", "House"])
    plan = plan[plan["office"].isin(offices)]

    k = st.columns(4)
    k[0].metric("Toss-up races", int((plan["tier"] == "Toss-up").sum()))
    k[1].metric("Lean races", int((plan["tier"] == "Lean").sum()))
    k[2].metric("Moved ≥3pt in 24h", int((plan["momentum"] >= 0.03).sum()))
    k[3].metric("Venues disagree 5–20pt", int(plan.get("venue_gap", pd.Series(dtype=float)).between(0.05, 0.20).sum()),
                help="Bigger gaps are usually mismatched questions, so they're left out of this count.")

    # State roll-up: a state with a close Senate AND a close Governor race is a bigger media market story.
    by_state = (plan.groupby(["state", "code"])
                .agg(races=("race", lambda s: ", ".join(s)), top_priority=("priority", "max"),
                     sum_priority=("priority", "sum"), closest=("competitiveness", "max"))
                .reset_index().sort_values("sum_priority", ascending=False))
    by_state["stacked"] = plan.groupby("code")["tier"].apply(lambda t: (t.isin(["Toss-up", "Lean"])).sum()).reindex(by_state["code"]).values

    notes = []
    top = plan.head(3)
    if len(top):
        notes.append("**Concentrate:** " + ", ".join(f"{r['race']} ({r['rating']})" for _, r in top.iterrows()) + ".")
    stacked = by_state[by_state["stacked"] >= 2]
    if len(stacked):
        notes.append("**Stacked ballots** (2+ competitive races, so local inventory gets crowded and CPMs climb first): "
                     + ", ".join(stacked["state"].head(6)) + ".")
    rising = plan[plan["momentum"] >= 0.03]
    if len(rising):
        notes.append("**Momentum triggers** (odds moved ≥3pt in 24h; revisit bids, creative and frequency caps): "
                     + ", ".join(rising["race"].head(6)) + ".")
    fading = plan[(plan["tier"] == "Safe") & (plan["office"].isin(["Senate", "Governor"]))]
    if len(fading):
        notes.append(f"**Likely over-served:** {len(fading)} Senate/Governor races are rated Safe, so persuasion "
                     "spend there is mostly wasted; fundraising and turnout audiences are the exception.")
    for n in notes:
        st.markdown("- " + n)

    st.markdown("##### Race priority")
    st.dataframe(plan[["race", "tier", "rating", "dem_p", "chg_24h", "volume", "priority"]], hide_index=True,
                 width="stretch", height=380, column_config={
                     "dem_p": st.column_config.ProgressColumn("Dem odds", min_value=0, max_value=1, format="%.2f"),
                     "chg_24h": st.column_config.NumberColumn("24h Δ (Dem)", format="%+.3f"),
                     "volume": st.column_config.NumberColumn("Mkt volume", format="%.0f"),
                     "priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%d")})

    st.markdown("##### State roll-up (for geo line items)")
    st.dataframe(by_state[["state", "code", "races", "stacked", "top_priority", "sum_priority"]], hide_index=True,
                 width="stretch", column_config={
                     "stacked": st.column_config.NumberColumn("Competitive races"),
                     "top_priority": st.column_config.NumberColumn("Top priority", format="%d"),
                     "sum_priority": st.column_config.NumberColumn("Combined", format="%d")})

    export = plan[["code", "state", "office", "district", "race", "tier", "rating", "dem_p", "priority"]].rename(
        columns={"code": "state_code", "dem_p": "dem_odds"})
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Race list (CSV)", export.to_csv(index=False), "election_priority_races.csv", "text/csv")
    c2.download_button("⬇️ State geo list (CSV)", by_state.to_csv(index=False), "election_priority_states.csv", "text/csv")


# ----------------------------------------------------------------------------
# All races
# ----------------------------------------------------------------------------
with tab_all:
    st.subheader("Every race with a market")
    unlabeled = races[races["dem_p"].isna() & ~races["office"].str.endswith("control")]
    if len(unlabeled):
        st.warning(f"{len(unlabeled)} race(s) show **No data** because no candidate's party could be found: "
                   + ", ".join(unlabeled["race"].head(10))
                   + ". Add those candidates to **parties.csv** (name,party) and refresh.")
    show = races.sort_values(["office", "state"])
    st.dataframe(show[["office", "race", "rating", "dem_p", "rep_p", "leader", "chg_24h", "volume", "spread",
                       "venue", "venue_gap", "title"]], hide_index=True, width="stretch", height=520,
                 column_config={
                     "dem_p": st.column_config.NumberColumn("Dem", format="%.2f"),
                     "rep_p": st.column_config.NumberColumn("Rep", format="%.2f"),
                     "chg_24h": st.column_config.NumberColumn("24h Δ", format="%+.3f"),
                     "volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                     "venue_gap": st.column_config.NumberColumn("Venue gap", format="%.2f"),
                     "title": st.column_config.TextColumn("Market title", width="large")})
    st.caption("Each race comes from the venue with the most volume. Titles are matched to states automatically, "
               "so check the market title if something looks off.")

st.sidebar.divider()
st.sidebar.caption("Odds are what traders are betting, not a forecast. Thin markets can be wrong. "
                   "Data via the open-source synpath SDK.")
