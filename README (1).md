# Election Lens

Win odds for the 2026 US midterms by state, read from prediction markets (Kalshi, Polymarket US, Polymarket). The app only reads market prices. It never places orders and needs no account or API keys.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Turn on **Demo data** in the sidebar to try it offline. Demo mode uses made-up odds and generic candidates.

## Tabs

- **🗺️ Map.** Senate and Governor races colored by Democratic vs Republican win odds, with ratings from Safe to Toss-up. Click into a state to see each outcome's odds, the 24h move, a 60-day chart, and whether the venues disagree. House races are listed by district.
- **🎯 Toss-ups & movers.** Every race between 35% and 65%, plus the biggest 24-hour swings.
- **🏛️ Chamber control.** The market's odds on which party controls the Senate and the House, plus expected wins summed across the races that have markets.
- **📣 Ad planner.** A 0–100 priority score per race, built from closeness (65%), momentum (20%) and market attention (15%), weighted by office. Also shows states with stacked competitive races, momentum triggers, and over-served safe races. Exports race and state CSVs for geo line items.
- **📋 All races.** The full table, with the source market title so you can check how each race was matched.

## How it works

The app searches each venue for election markets and reads each title to work out the state, office (Senate, Governor or House) and district. It then groups the markets into races and computes two-party odds. When a race is listed on more than one venue, it keeps the venue with the most volume and records how far the venues disagree.

## Caveats

- Odds reflect what traders are betting. They are not a forecast, and thin markets can be badly wrong. Use the sidebar's volume filter.
- Safe states and most House districts often have no market at all. Those show up grey on the map or are left out.
- Matching titles to races is automatic. If a race looks wrong, check its market title in **All races**.
