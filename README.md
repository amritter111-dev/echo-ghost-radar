Echo Ghost Commodity Cycle Radar
Systematic commodity cycle scanner that maps raw material signals → theme scores → equity leaders.
What it does
Scores 8 commodities (Copper, Oil, NatGas, Gold, Silver, Uranium, Corn, Wheat) using continuous momentum + MA slope
Rolls commodity signals up into 5 cycle themes (Steel, Energy, Uranium, Food Inflation, Precious Metals)
Scans equity universes for each active theme and ranks stocks by composite score
Relative strength vs SPY baked into every equity ranking
Leader filter — price above MA200, within 25% of 52-week high, minimum $5M daily dollar volume
Alert system — flags signal flips, acceleration, and global rank changes vs prior run
Persists history to CSV on every run, generates theme score charts
Requirements
pip install yfinance pandas numpy matplotlib
Usage
python echo_ghost_radar_v07.py
Output saves to radar_output/
Output files
commodity_history.csv — commodity signal log
theme_history.csv — theme score log
equity_history.csv — equity rankings log
global_history.csv — global top ideas log
theme_scores_history.csv — raw scores for charting
theme_chart.png — theme score history chart
