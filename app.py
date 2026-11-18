import math
import warnings
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
st.set_page_config(page_title="The Hedge Fund Lab", page_icon="📈", layout="wide")

# ----------------------------- Theme -----------------------------------------
st.markdown("""
<style>
  .stApp {background: #0b1220; color: #e8edf5;}
  [data-testid="stSidebar"] {background: #101a2b;}
  [data-testid="stMetric"] {background: #121f33; border: 1px solid #26364d; padding: 14px; border-radius: 12px;}
  .hero {padding: 22px 26px; border: 1px solid #26364d; border-radius: 16px;
         background: linear-gradient(120deg,#14243b,#101827); margin-bottom: 18px;}
  .muted {color:#9aabc1;}
  div[data-testid="stTabs"] button {font-weight:600;}
  .small-note {font-size: .85rem; color:#9aabc1;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Data ------------------------------------------
UNIVERSE = {
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "AMZN":"Amazon",
    "GOOGL":"Alphabet", "META":"Meta", "JPM":"JPMorgan Chase", "GS":"Goldman Sachs",
    "XOM":"Exxon Mobil", "CVX":"Chevron", "LLY":"Eli Lilly", "JNJ":"Johnson & Johnson",
    "WMT":"Walmart", "COST":"Costco", "KO":"Coca-Cola", "PG":"Procter & Gamble",
    "CAT":"Caterpillar", "GE":"GE Aerospace", "TSLA":"Tesla", "NFLX":"Netflix",
    "BRK-B":"Berkshire Hathaway", "SPY":"S&P 500 ETF", "QQQ":"Nasdaq 100 ETF",
    "TLT":"20+ Year Treasury ETF", "GLD":"Gold ETF", "HYG":"High Yield Bond ETF",
    "IWM":"Russell 2000 ETF", "EFA":"Developed Markets ETF"
}
SECTOR = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMZN":"Consumer",
    "GOOGL":"Technology","META":"Technology","JPM":"Financials","GS":"Financials",
    "XOM":"Energy","CVX":"Energy","LLY":"Healthcare","JNJ":"Healthcare",
    "WMT":"Consumer Staples","COST":"Consumer Staples","KO":"Consumer Staples",
    "PG":"Consumer Staples","CAT":"Industrials","GE":"Industrials","TSLA":"Consumer",
    "NFLX":"Communication","BRK-B":"Financials","SPY":"Broad Market","QQQ":"Technology",
    "TLT":"Fixed Income","GLD":"Commodities","HYG":"Fixed Income","IWM":"Broad Market","EFA":"International"
}
REGIMES = {
    "Global Financial Crisis (proxy)": ("2008-09-01","2009-03-31"),
    "Eurozone / China slowdown (proxy)": ("2011-07-01","2011-10-31"),
    "Q4 2018 selloff": ("2018-10-01","2018-12-31"),
    "COVID crash": ("2020-02-19","2020-03-23"),
    "2022 inflation / rate shock": ("2022-01-03","2022-10-14"),
    "2023 regional banking stress": ("2023-03-01","2023-03-31"),
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers, start, end):
    """Download adjusted daily prices; deterministic synthetic fallback keeps the lab usable offline."""
    try:
        import yfinance as yf
        raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                           progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices.dropna(how="all").ffill().dropna(axis=1, how="all")
        if len(prices) > 60 and prices.shape[1] > 0:
            return prices, "Yahoo Finance adjusted historical prices"
    except Exception:
        pass

    # Offline demonstration data. Not actual market observations.
    idx = pd.bdate_range(start=start, end=end)
    if len(idx) < 80:
        idx = pd.bdate_range(end=pd.Timestamp(end), periods=260)
    rng = np.random.default_rng(42)
    common = rng.normal(0.00025, 0.008, len(idx))
    out = {}
    for ticker in tickers:
        sector = SECTOR.get(ticker, "Broad Market")
        beta = {"Technology":1.25,"Financials":1.05,"Energy":0.95,"Healthcare":0.75,
                "Consumer Staples":0.65,"Industrials":1.0,"Consumer":1.15,
                "Communication":1.1,"Fixed Income":-0.15,"Commodities":0.2,
                "Broad Market":1.0,"International":0.95}.get(sector,1.0)
        idio = rng.normal(0, 0.010 + (0.003 if sector=="Technology" else 0), len(idx))
        returns = beta*common + idio + 0.00015
        # Add illustrative crisis-like volatility clusters to make regime tools demonstrable.
        for a,b,mult in [("2008-09-01","2009-03-31",2.8),("2020-02-19","2020-03-23",3.8),
                         ("2022-01-03","2022-10-14",1.8)]:
            mask = (idx >= a) & (idx <= b)
            returns[mask] = beta*common[mask] + rng.normal(-0.0005,0.018*mult,len(idx[mask]))
        out[ticker] = 100*np.cumprod(1+returns)
    return pd.DataFrame(out, index=idx), "Synthetic demonstration data (offline fallback)"

def metrics(r, rf=0.04, mar=0.0):
    r = r.dropna()
    if len(r) < 2:
        return {}
    ann = 252
    vol = r.std(ddof=1)*np.sqrt(ann)
    cagr = (1+r).prod()**(ann/len(r))-1
    excess = r - rf/ann
    sharpe = (excess.mean()*ann)/(r.std(ddof=1)*np.sqrt(ann)) if r.std(ddof=1)>0 else np.nan
    downside = np.minimum(r-mar/ann,0)
    dd = np.sqrt((downside**2).mean())*np.sqrt(ann)
    sortino = ((r.mean()-mar/ann)*ann)/dd if dd>0 else np.nan
    wealth = (1+r).cumprod()
    dd_series = wealth/wealth.cummax()-1
    maxdd = dd_series.min()
    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean() if (r <= var95).any() else var95
    return {"CAGR":cagr,"Annualized volatility":vol,"Sharpe":sharpe,"Sortino":sortino,
            "Max drawdown":maxdd,"Daily VaR 95%":var95,"Daily CVaR 95%":cvar95,
            "Downside deviation":dd}

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.2%}"
def fmt_num(x):
    return "—" if pd.isna(x) else f"{x:.2f}"

# ----------------------------- Header ----------------------------------------
st.markdown("""
<div class="hero">
  <div class="muted">INSTITUTIONAL PORTFOLIO SIMULATION • STUDENT EDITION</div>
  <h1 style="margin:4px 0 2px 0;">THE HEDGE FUND LAB</h1>
  <div class="muted">Build portfolios. Quantify risk. Survive different markets. Defend your decisions.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧭 Lab controls")
    st.caption("Configure the investment mandate")
    start = st.date_input("Historical start date", value=date(2016,1,1))
    end = st.date_input("Historical end date", value=date.today())
    risk_free = st.number_input("Annual risk-free rate", min_value=0.0, max_value=0.20,
                                value=0.04, step=0.0025, format="%.3f")
    mar = st.number_input("Minimum acceptable return (annual)", min_value=-0.20, max_value=0.20,
                          value=0.0, step=0.0025, format="%.3f")
    benchmark = st.selectbox("Benchmark", ["SPY","QQQ","IWM","EFA"], index=0)
    st.divider()
    st.markdown("### 🎯 Portfolio mandate")
    capital = st.number_input("Starting notional ($)", min_value=10000, max_value=100000000,
                              value=1000000, step=50000)
    max_weight = st.slider("Maximum absolute position weight", 5, 100, 40, 5)
    allow_short = st.toggle("Allow short positions", value=False)
    st.caption("Educational simulation only. Not investment advice.")

# portfolio controls
left, right = st.columns([1.05, 1.4])
with left:
    st.subheader("01 / Portfolio construction")
    tickers = st.multiselect("Select securities", list(UNIVERSE.keys()),
        default=["AAPL","MSFT","NVDA","JPM","XOM","LLY","SPY"],
        format_func=lambda x: f"{x} — {UNIVERSE[x]}")
    if not tickers:
        st.info("Select at least two securities to start the lab.")
        st.stop()
    st.caption("Set target weights. Long-only portfolios are normalized to 100%; short portfolios are gross-exposure normalized.")
    default_w = 100/len(tickers)
    weights = {}
    cols = st.columns(2)
    for i,t in enumerate(tickers):
        with cols[i%2]:
            weights[t] = st.number_input(f"{t} (%)", min_value=(-max_weight if allow_short else 0),
                max_value=max_weight, value=int(round(default_w)), step=1, key=f"w_{t}")
    if allow_short:
        gross = sum(abs(v) for v in weights.values())
        if gross == 0:
            st.warning("At least one position must have a non-zero weight.")
            st.stop()
        w = pd.Series(weights, dtype=float)/gross
        net = w.sum()
        st.caption(f"Gross exposure: 100% after scaling • Net exposure: {net:.1%}")
    else:
        total = sum(weights.values())
        if total <= 0:
            st.warning("Portfolio weights must sum to more than zero.")
            st.stop()
        w = pd.Series(weights, dtype=float)/total
        st.caption(f"Input weights normalized to 100% (input total: {total}%).")
    w = w.reindex(tickers).fillna(0)
    st.markdown("**Final portfolio weights**")
    alloc = pd.DataFrame({"Ticker":w.index,"Company":[UNIVERSE[x] for x in w.index],
                          "Weight":w.values,"Sector":[SECTOR.get(x,"Other") for x in w.index]})
    st.dataframe(alloc.style.format({"Weight":"{:.1%}"}), hide_index=True, use_container_width=True)

with right:
    st.subheader("02 / Market data & portfolio snapshot")
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))
    if start >= end:
        st.error("The start date must be earlier than the end date.")
        st.stop()
    with st.spinner("Loading market history..."):
        prices, data_source = load_prices(tuple(all_tickers), str(start), str(end))
    available = [t for t in tickers if t in prices.columns]
    if len(available) < 2:
        st.error("Not enough price history was returned. Try a different date range or tickers.")
        st.stop()
    tickers = available
    w = w.reindex(tickers).fillna(0)
    if not allow_short and w.sum() != 0:
        w = w/w.sum()
    elif allow_short and abs(w).sum() != 0:
        w = w/abs(w).sum()
    prices = prices.dropna(how="all").ffill()
    returns = prices.pct_change().dropna()
    portfolio_returns = returns[tickers].mul(w, axis=1).sum(axis=1)
    bench_returns = returns[benchmark] if benchmark in returns else pd.Series(index=returns.index,dtype=float)
    m = metrics(portfolio_returns, risk_free, mar)
    bm = metrics(bench_returns, risk_free, mar) if len(bench_returns.dropna()) else {}
    st.caption(f"Data source: {data_source} • {len(returns):,} daily observations")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Portfolio CAGR",fmt_pct(m.get("CAGR",np.nan)))
    k2.metric("Sharpe ratio",fmt_num(m.get("Sharpe",np.nan)))
    k3.metric("Sortino ratio",fmt_num(m.get("Sortino",np.nan)))
    k4.metric("Maximum drawdown",fmt_pct(m.get("Max drawdown",np.nan)))
    wealth = (1+portfolio_returns).cumprod()*capital
    bwealth = (1+bench_returns.fillna(0)).cumprod()*capital
    nav = pd.DataFrame({"Portfolio":wealth,"Benchmark":bwealth})
    fig = px.line(nav, title="Growth of $1 invested / portfolio notional", template="plotly_dark")
    fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_title_text="", margin=dict(l=10,r=10,t=45,b=10))
    st.plotly_chart(fig,use_container_width=True)

# ----------------------------- Tabs ------------------------------------------
tabs = st.tabs(["📊 Risk dashboard","🔗 Correlation lab","🌪️ Regime stress test",
                "⚙️ Optimizer","🏆 PM challenge","📚 Analyst handbook"])

with tabs[0]:
    st.subheader("Risk & performance attribution")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annualized volatility",fmt_pct(m.get("Annualized volatility",np.nan)))
    c2.metric("Daily VaR (95%)",fmt_pct(m.get("Daily VaR 95%",np.nan)))
    c3.metric("Daily CVaR (95%)",fmt_pct(m.get("Daily CVaR 95%",np.nan)))
    c4.metric("Downside deviation",fmt_pct(m.get("Downside deviation",np.nan)))
    metric_table = pd.DataFrame({
        "Metric":["CAGR","Annualized volatility","Sharpe ratio","Sortino ratio","Maximum drawdown",
                  "Daily VaR 95%","Daily CVaR 95%","Downside deviation"],
        "Portfolio":[m.get("CAGR"),m.get("Annualized volatility"),m.get("Sharpe"),m.get("Sortino"),
                     m.get("Max drawdown"),m.get("Daily VaR 95%"),m.get("Daily CVaR 95%"),
                     m.get("Downside deviation")],
        "Benchmark":[bm.get("CAGR"),bm.get("Annualized volatility"),bm.get("Sharpe"),bm.get("Sortino"),
                     bm.get("Max drawdown"),bm.get("Daily VaR 95%"),bm.get("Daily CVaR 95%"),
                     bm.get("Downside deviation")]
    })
    st.dataframe(metric_table.style.format({"Portfolio":"{:.2%}","Benchmark":"{:.2%}"},
                  na_rep="—"),hide_index=True,use_container_width=True)
    st.markdown("#### Drawdown profile")
    wealth_idx=(1+portfolio_returns).cumprod()
    dd=wealth_idx/wealth_idx.cummax()-1
    fig=px.area(dd,title="Underwater curve",template="plotly_dark")
    fig.update_layout(height=260,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      yaxis_tickformat=".0%")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Rolling 1-year Sharpe")
    roll=portfolio_returns.rolling(252).mean()*252-risk_free
    rollvol=portfolio_returns.rolling(252).std()*np.sqrt(252)
    rollsharpe=roll/rollvol.replace(0,np.nan)
    fig=px.line(rollsharpe,title="Rolling 252-trading-day Sharpe ratio",template="plotly_dark")
    fig.update_layout(height=260,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.subheader("Correlation is a regime-dependent relationship")
    corr=returns[tickers].corr()
    fig=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale="RdBu",
                  text_auto=".2f",aspect="auto",title="Daily return Pearson correlation matrix",
                  template="plotly_dark")
    fig.update_layout(height=max(420,35*len(tickers)),paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2)
    with a:
        x=st.selectbox("X-axis security",tickers,index=0,key="scatter_x")
    with b:
        y=st.selectbox("Y-axis security",tickers,index=min(1,len(tickers)-1),key="scatter_y")
    scatter=pd.DataFrame({"x":returns[x],"y":returns[y]}).dropna()
    r=scatter.corr().iloc[0,1]
    fig=px.scatter(scatter,x="x",y="y",trendline="ols",template="plotly_dark",
                   title=f"{x} vs {y} daily returns • Pearson ρ = {r:.3f}")
    fig.update_xaxes(tickformat=".1%"); fig.update_yaxes(tickformat=".1%")
    fig.update_layout(height=380,paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    rolling_corr=returns[x].rolling(63).corr(returns[y])
    fig=px.line(rolling_corr,title=f"Rolling 63-day correlation: {x} vs {y}",template="plotly_dark")
    fig.update_layout(height=260,paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.info("Correlation measures co-movement, not causation. Correlations can rise sharply during liquidity events and market selloffs.")

with tabs[2]:
    st.subheader("Historical & hypothetical regime stress testing")
    st.caption("Historical windows use the selected price history. If the selected dates do not cover a regime, that regime will be unavailable.")
    historical=[]
    for name,(a,b) in REGIMES.items():
        mask=(portfolio_returns.index>=a)&(portfolio_returns.index<=b)
        if mask.sum()>=2:
            rp=portfolio_returns.loc[mask]
            rb=bench_returns.reindex(rp.index).dropna()
            historical.append({"Regime":name,"Portfolio return":(1+rp).prod()-1,
                               "Benchmark return":(1+rb).prod()-1 if len(rb) else np.nan,
                               "Worst day":rp.min(),"Observations":len(rp)})
    if historical:
        hdf=pd.DataFrame(historical)
        st.dataframe(hdf.style.format({"Portfolio return":"{:.2%}","Benchmark return":"{:.2%}",
                    "Worst day":"{:.2%}"}),hide_index=True,use_container_width=True)
        fig=px.bar(hdf,x="Regime",y=["Portfolio return","Benchmark return"],barmode="group",
                   title="Portfolio vs benchmark across historical stress windows",template="plotly_dark")
        fig.update_layout(height=380,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.warning("No defined historical stress windows fall inside this date range. Expand the start date to include 2008 or 2020.")
    st.markdown("#### Custom instantaneous shock")
    st.caption("Apply a one-time shock by sector. This is a scenario analysis, not a forecast.")
    shock_defaults={"Technology":-25,"Financials":-20,"Energy":-15,"Healthcare":-10,
                    "Consumer Staples":-8,"Industrials":-18,"Consumer":-20,
                    "Communication":-18,"Fixed Income":4,"Commodities":5,
                    "Broad Market":-18,"International":-15}
    sectors=sorted(set(SECTOR.get(t,"Other") for t in tickers))
    shock_cols=st.columns(3)
    shocks={}
    for i,s in enumerate(sectors):
        with shock_cols[i%3]:
            shocks[s]=st.slider(f"{s} shock (%)",-60,40,shock_defaults.get(s,-15),1,key=f"shock_{s}")
    shock_return=sum(w[t]*shocks.get(SECTOR.get(t,"Other"),-15)/100 for t in tickers)
    loss=capital*shock_return
    st.metric("Custom scenario portfolio return",fmt_pct(shock_return),
              delta=f"${loss:,.0f} P&L on ${capital:,.0f}",delta_color="normal")
    st.caption("For long/short portfolios, this simple shock assumes linear exposure and does not model borrow costs, margin calls, options convexity, or liquidity.")

with tabs[3]:
    st.subheader("Portfolio optimizer & efficient frontier")
    st.caption("Mean-variance estimates are highly sensitive to the lookback window and expected-return assumptions. Treat outputs as model estimates, not prescriptions.")
    opt_method=st.selectbox("Optimization objective",["Maximum Sharpe","Minimum variance","Risk parity (inverse volatility)"])
    if st.button("Run optimizer",type="primary"):
        rr=returns[tickers].dropna()
        mu=rr.mean().values*252
        cov=rr.cov().values*252
        n=len(tickers)
        try:
            from scipy.optimize import minimize
            if opt_method=="Risk parity (inverse volatility)":
                inv=1/np.sqrt(np.diag(cov))
                ow=inv/inv.sum()
            else:
                def objective(x):
                    ret=x@mu
                    vol=np.sqrt(max(x@cov@x,1e-12))
                    if opt_method=="Maximum Sharpe":
                        return -((ret-risk_free)/vol)
                    return vol
                cons=[{"type":"eq","fun":lambda x:np.sum(x)-1}]
                bounds=[(0,1)]*n
                res=minimize(objective,np.repeat(1/n,n),bounds=bounds,constraints=cons,
                             method="SLSQP",options={"maxiter":1000})
                ow=res.x if res.success else np.repeat(1/n,n)
            opt=pd.DataFrame({"Ticker":tickers,"Current weight":w.values,"Optimized weight":ow})
            st.dataframe(opt.style.format({"Current weight":"{:.1%}","Optimized weight":"{:.1%}"}),
                         hide_index=True,use_container_width=True)
            st.metric("Optimizer estimated annual return",fmt_pct(ow@mu))
            st.metric("Optimizer estimated annual volatility",fmt_pct(np.sqrt(ow@cov@ow)))
        except Exception as e:
            st.error(f"Optimizer could not run: {e}")
    else:
        st.info("Click Run optimizer to calculate weights using the selected securities and historical lookback.")

with tabs[4]:
    st.subheader("Portfolio manager challenge")
    st.markdown("**Mandate:** Build a diversified portfolio that compounds capital while controlling downside risk.")
    target_sharpe=st.slider("Challenge minimum Sharpe",0.25,2.0,0.8,0.05)
    max_dd=st.slider("Maximum permitted drawdown (%)",5,60,25,1)
    min_holdings=st.slider("Minimum number of holdings",2,min(10,len(tickers)),min(5,len(tickers)),1)
    concentration=float(w.abs().max())
    score=0
    score += 35 if m.get("Sharpe",0)>=target_sharpe else max(0,35*m.get("Sharpe",0)/target_sharpe)
    score += 30 if abs(m.get("Max drawdown",0))*100<=max_dd else max(0,30*max_dd/(abs(m.get("Max drawdown",0))*100+1e-9))
    score += 20 if len(tickers)>=min_holdings else 20*len(tickers)/min_holdings
    score += 15 if concentration<=0.30 else max(0,15*0.30/concentration)
    score=int(round(min(100,score)))
    st.metric("Portfolio manager score",f"{score}/100",help="Illustrative teaching rubric; not an industry-standard rating.")
    st.progress(score/100)
    checks=[
        ("Risk-adjusted return",m.get("Sharpe",np.nan)>=target_sharpe,
         f"Sharpe {fmt_num(m.get('Sharpe',np.nan))} vs target {target_sharpe:.2f}"),
        ("Drawdown discipline",abs(m.get("Max drawdown",0))*100<=max_dd,
         f"Maximum drawdown {fmt_pct(m.get('Max drawdown',np.nan))} vs limit {max_dd}%"),
        ("Diversification",len(tickers)>=min_holdings,
         f"{len(tickers)} holdings vs minimum {min_holdings}"),
        ("Position sizing",concentration<=0.30,
         f"Largest absolute weight {concentration:.1%}; 30% guideline")
    ]
    for label,ok,detail in checks:
        st.write(("✅" if ok else "⚠️")+f" **{label}:** {detail}")
    st.markdown("#### Investment committee memo")
    thesis=st.text_area("State your investment thesis, key risk, catalyst, and what would make you change your mind.",
                        placeholder="Our portfolio is positioned for... The key risk is... We would reduce exposure if...")
    if st.button("Submit investment committee memo"):
        if len(thesis.strip())<80:
            st.warning("Write at least a few substantive sentences so the committee can assess your reasoning.")
        else:
            st.success("Memo submitted in this session. Discuss the thesis with your instructor or investment committee.")
            st.write(thesis)
    st.download_button("Download portfolio weights (CSV)",alloc.to_csv(index=False).encode(),
                       file_name="student_portfolio_weights.csv",mime="text/csv")

with tabs[5]:
    st.subheader("Analyst handbook: know what the numbers mean")
    handbook=[
        ("Sharpe ratio","Annualized excess return divided by annualized total volatility. Penalizes upside and downside volatility alike."),
        ("Sortino ratio","Excess return divided by downside deviation relative to a minimum acceptable return (MAR). Focuses on harmful volatility."),
        ("Correlation","Pearson correlation of periodic returns. +1 means perfectly aligned linear movement; 0 means no linear relationship; -1 means perfectly inverse."),
        ("Beta","Sensitivity of an asset or portfolio to benchmark returns. A beta above 1 historically indicates greater benchmark sensitivity."),
        ("Alpha","Return unexplained by the selected benchmark model. A simple benchmark-relative return is not the same as statistically estimated alpha."),
        ("Maximum drawdown","Largest peak-to-trough decline in the compounded wealth series during the sample."),
        ("VaR / CVaR","VaR is a loss quantile; CVaR is the average loss in the tail beyond that quantile. Historical estimates depend on sample size."),
        ("Diversification","Combining exposures whose returns are not perfectly correlated can reduce portfolio variance. Diversification is not guaranteed protection in a crisis.")
    ]
    for term,desc in handbook:
        with st.expander(term):
            st.write(desc)
    st.warning("Model limitations: daily historical data, no transaction costs, taxes, slippage, borrow fees, corporate-action modeling beyond adjusted prices, survivorship-bias correction, or intraday liquidity constraints. Historical regimes are descriptive, not predictive.")

st.divider()
st.markdown('<div class="small-note">THE HEDGE FUND LAB • Classroom simulation • Not investment advice • Market data availability depends on network access and Yahoo Finance.</div>',unsafe_allow_html=True)
