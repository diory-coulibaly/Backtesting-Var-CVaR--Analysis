import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, t
import math

st.set_page_config(page_title="Advanced VaR & CVaR Dashboard", layout="wide")
st.title("📊 Advanced VaR, CVaR & Backtesting Dashboard")
from PIL import Image

st.markdown("### 📋 Template Format Preview")
image = Image.open("CombinedPrices.JPG")
st.image(image, caption="Expected structure of CombinedPrices.csv (Date + Stocks + Factors)", use_container_width=True)


# Sidebar
st.sidebar.header("📁 Upload CombinedPrices.csv")
uploaded_file = st.sidebar.file_uploader("CSV with Date, FVX, SP500, stock prices", type="csv")

confLevel = st.sidebar.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.99, step=0.01)
rolling_window = st.sidebar.slider("Rolling Window (days)", 100, 300, 252, step=10)
alert_threshold = st.sidebar.slider("CVaR Alert Threshold (%)", 1.0, 10.0, 3.0)
use_student_t = st.sidebar.checkbox("Use Student-t Distribution (fat tails)", value=False)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "").str.replace(u'\xa0', '').astype(float)

    factorNames = st.sidebar.multiselect("Select Factor Columns", options=[c for c in df.columns if c != "Date"], default=["SP500", "FVX"])
    if "Intercept" not in factorNames:
        factorNames.append("Intercept")

    stockNames = [col for col in df.columns if col not in ["Date"] + factorNames]
    returns = df.drop(columns=["Date"]).pct_change().dropna()
    returns["Intercept"] = 1

    stockReturns = returns[stockNames]
    factorReturns = returns[factorNames]
    weights = np.array([1.0 / len(stockNames)] * len(stockNames))

    # --- Weekly Rebalancing Simulation ---
    rebalance_freq = 5  # every 5 trading days (weekly)
    weight_df = pd.DataFrame(index=stockReturns.index, columns=stockNames)
    for i in range(0, len(stockReturns), rebalance_freq):
        weight_df.iloc[i:i+rebalance_freq] = weights
    weight_df.fillna(method="ffill", inplace=True)
    dynamic_port_returns = (stockReturns * weight_df.astype(float)).sum(axis=1)
    port_returns = dynamic_port_returns

    def calculateVaR(risk, confLevel, principal=1, numMonths=1):
        z = t.ppf(1 - confLevel, df=len(port_returns)-1) if use_student_t else norm.ppf(1 - confLevel)
        vol = math.sqrt(risk)
        return abs(principal * z * vol * math.sqrt(numMonths))

    def calculateCVaR(port_returns, confLevel):
        var_threshold = np.percentile(port_returns, (1 - confLevel) * 100)
        return abs(port_returns[port_returns <= var_threshold].mean())

    # Historical Simulation
    hist_var = abs(np.percentile(port_returns, (1 - confLevel) * 100))
    hist_cvar = calculateCVaR(port_returns, confLevel)

    # Variance-Covariance
    vc_risk = np.dot(np.dot(weights, stockReturns.cov()), weights.T)
    vc_var = calculateVaR(vc_risk, confLevel)

    # --- Per-stock VaR Contributions ---
    port_vol = np.sqrt(vc_risk)
    cov_matrix = stockReturns.cov()
    marginal_var = cov_matrix.dot(weights) / port_vol
    component_var = weights * marginal_var
    percent_contrib_var = component_var / port_vol

    contrib_df = pd.DataFrame({
        "Stock": stockNames,
        "Weight": weights,
        "Marginal VaR": marginal_var,
        "Component VaR": component_var,
        "% Contribution": percent_contrib_var
    })

    st.subheader("🧮 Per-Stock VaR Contributions (Delta-Normal)")
    st.dataframe(contrib_df.style.format({
        "Weight": "{:.2%}", 
        "Marginal VaR": "{:.4f}", 
        "Component VaR": "{:.4f}", 
        "% Contribution": "{:.2%}"
    }))

    # Monte Carlo (fat tails if Student-t)
    mean_returns = stockReturns.mean()
    cov_matrix = stockReturns.cov()
    if use_student_t:
        sim_returns = np.random.standard_t(df=4, size=(10000, len(stockNames))) * np.sqrt(np.diag(cov_matrix))
        sim_returns += mean_returns.values
    else:
        sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=10000)
    sim_port_returns = np.dot(sim_returns, weights)
    mc_var = abs(np.percentile(sim_port_returns, (1 - confLevel) * 100))
    mc_cvar = calculateCVaR(sim_port_returns, confLevel)

    # Factor model
    xData = factorReturns
    modelCoeffs = []
    for oneStock in stockNames:
        yData = stockReturns[oneStock]
        model = sm.OLS(yData, xData).fit()
        coeffs = list(model.params)
        coeffs.append(np.std(model.resid, ddof=1))
        modelCoeffs.append(coeffs)
    modelCoeffs = pd.DataFrame(modelCoeffs, columns=factorNames + ["ResidVol"])
    modelCoeffs["Names"] = stockNames

    factorCov = factorReturns[[col for col in factorNames if col != "Intercept"]].cov()
    B_factors = modelCoeffs[[col for col in factorNames if col != "Intercept"]]
    reconstructedCov = np.dot(np.dot(B_factors, factorCov), B_factors.T)
    systemicRisk = np.dot(np.dot(weights, reconstructedCov), weights.T)
    idiosyncraticRisk = sum(modelCoeffs["ResidVol"] ** 2 * weights ** 2)
    factor_risk = systemicRisk + idiosyncraticRisk
    factor_var = calculateVaR(factor_risk, confLevel)

    # Table
    results_df = pd.DataFrame([
        ["Historical", hist_var, hist_cvar],
        ["Variance-Covariance", vc_var, "-"],
        ["Monte Carlo", mc_var, mc_cvar],
        ["Factor Model", factor_var, "-"]
    ], columns=["Method", f"VaR ({int(confLevel*100)}%)", f"CVaR ({int(confLevel*100)}%)"])
    st.subheader("📊 VaR & CVaR Summary")
    st.dataframe(results_df)

    # CVaR Tail Chart
    st.subheader("📉 CVaR Tail Loss Distribution")
    fig, ax = plt.subplots()
    ax.hist(port_returns, bins=100, alpha=0.7)
    ax.axvline(-hist_var, color="red", linestyle="--", label="VaR")
    ax.axvline(-hist_cvar, color="black", linestyle=":", label="CVaR")
    ax.set_title("Portfolio Return Distribution with VaR & CVaR")
    ax.legend()
    st.pyplot(fig)

    if hist_cvar * 100 > alert_threshold:
        st.error(f"⚠️ CVaR exceeds threshold: {hist_cvar:.2%}")

    # --- Dynamic Portfolio Rebalancing Simulation ---
    st.subheader("🔄 Simulated Rebalancing Strategy (Weekly Equal Weight)")
    rebalance_df = df.copy()
    rebalance_df["Portfolio Value"] = 1.0

    weekly_returns = stockReturns.copy()
    weekly_returns["Date"] = df["Date"].iloc[1:].values
    weekly_returns = weekly_returns.resample("W", on="Date").apply(lambda x: (1 + x).prod() - 1)

    weekly_weights = pd.DataFrame(1 / len(stockNames), index=weekly_returns.index, columns=stockNames)
    weekly_port_return = (weekly_weights * weekly_returns).sum(axis=1)
    rebalance_df = pd.DataFrame({
        "Week": weekly_returns.index,
        "Simulated Portfolio Return": weekly_port_return
    })
    rebalance_df["Cumulative Return"] = (1 + rebalance_df["Simulated Portfolio Return"]).cumprod()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(rebalance_df["Week"], rebalance_df["Cumulative Return"])
    ax3.set_title("Dynamic Weekly Rebalancing (Cumulative Return)")
    st.pyplot(fig3)

    # --- Backtesting ---
    st.subheader("📈 VaR Backtesting")

    if len(port_returns) < rolling_window:
        st.warning(f"⛔ Not enough data ({len(port_returns)} days) for rolling window of {rolling_window} days.")
    else:
        rolling_var = port_returns.rolling(window=rolling_window).apply(
            lambda x: np.percentile(x, (1 - confLevel) * 100)
        ).dropna()

        backtest_df = pd.DataFrame({
            "Date": df["Date"].iloc[-len(rolling_var):].reset_index(drop=True),
            "Portfolio Return": port_returns.iloc[-len(rolling_var):].reset_index(drop=True),
            "Rolling VaR": rolling_var.values
        })
        backtest_df["Breach"] = backtest_df["Portfolio Return"] < -backtest_df["Rolling VaR"]

        st.line_chart(backtest_df.set_index("Date")["Portfolio Return"])  # VaR line skipped for clarity
        num_breaches = backtest_df["Breach"].sum()
        expected_breaches = len(backtest_df) * (1 - confLevel)
        st.write(f"Observed Breaches: {num_breaches}, Expected: {expected_breaches:.1f}")

        # Kupiec Test
        pi = (1 - confLevel)
        T = len(backtest_df)
        x = num_breaches
        LR_uc = -2 * (np.log(((1 - pi) ** x) * (pi ** (T - x))) -
                      np.log((x / T) ** x * ((1 - x / T) ** (T - x))))
        p_value_uc = 1 - chi2.cdf(LR_uc, 1)
        st.write(f"Kupiec Test: LR={LR_uc:.3f}, p-value={p_value_uc:.4f}")

        # Christoffersen Test
        backtest_df["Lagged Breach"] = backtest_df["Breach"].shift(1).fillna(False)
        n00 = len(backtest_df[(~backtest_df["Lagged Breach"]) & (~backtest_df["Breach"])])
        n01 = len(backtest_df[(~backtest_df["Lagged Breach"]) & (backtest_df["Breach"])])
        n10 = len(backtest_df[(backtest_df["Lagged Breach"]) & (~backtest_df["Breach"])])
        n11 = len(backtest_df[(backtest_df["Lagged Breach"]) & (backtest_df["Breach"])])
        pi01 = n01 / (n00 + n01 + 1e-6)
        pi11 = n11 / (n10 + n11 + 1e-6)
        pi1 = (n01 + n11) / (n00 + n01 + n10 + n11 + 1e-6)
        logL0 = (n01 + n11) * np.log(pi1) + (n00 + n10) * np.log(1 - pi1)
        logL1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
        LR_ind = -2 * (logL0 - logL1)
        p_value_ind = 1 - chi2.cdf(LR_ind, 1)
        st.write(f"Christoffersen Test: LR={LR_ind:.3f}, p-value={p_value_ind:.4f}")
else:
    st.info("⬅️ Upload your CombinedPrices.csv file to begin analysis.")
