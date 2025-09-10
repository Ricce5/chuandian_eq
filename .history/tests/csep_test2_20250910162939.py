# Construct example observed catalog and multiple simulated catalogs (time-only + magnitude)
import json, random, os, math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

BASE_DIR = "/mnt"
os.makedirs(BASE_DIR, exist_ok=True)

# ------------------ Helper functions ------------------
def iso(dt): 
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def simulate_catalog(start, end, rate_per_day, b_value=1.0, mmin=3.0, mmax=5.5):
    """
    Simulate a time-only earthquake catalog using a homogeneous Poisson process for times,
    and Gutenberg-Richter magnitudes (truncated) via inverse CDF of 10^{-b(m-mmin)}.
    """
    days = (end - start).days
    expected = rate_per_day * days
    n = np.random.poisson(expected)
    # Event times uniformly over window (for simplicity)
    times = [start + timedelta(seconds=random.uniform(0, (end-start).total_seconds())) for _ in range(n)]
    times.sort()
    # Magnitudes from truncated GR
    # CDF: F(m) = (1 - 10^{-b (m-mmin)}) / (1 - 10^{-b (mmax-mmin)})
    # Inverse: m = mmin - (1/b) * log10(1 - u*(1 - 10^{-b (mmax-mmin)}))
    u = np.random.rand(n)
    denom = 1 - 10**(-b_value*(mmax-mmin))
    mags = mmin - (1.0/b_value) * np.log10(1 - u*denom)
    mags = np.clip(mags, mmin, mmax)
    return [{"time": iso(t), "magnitude": float(m)} for t, m in zip(times, mags)]

# ------------------ Construct example data ------------------
START = datetime(2019,7,4)
END   = datetime(2019,8,3)  # 30-day window
MMIN  = 3.0

# Observed catalog: make it slightly different from sims
obs_catalog = simulate_catalog(START, END, rate_per_day=2.2, b_value=1.0, mmin=MMIN, mmax=5.5)

# Simulated forecast: J catalogs with slightly uncertain rate
J = 1000
sim_catalogs = []
for j in range(J):
    # Sample rate per day from a lognormal around 2.0
    rate = float(np.random.lognormal(mean=np.log(2.0), sigma=0.25))
    cat = simulate_catalog(START, END, rate_per_day=rate, b_value=1.0, mmin=MMIN, mmax=5.5)
    sim_catalogs.append(cat)

# ------------------ Save to JSON files ------------------
obs_path = os.path.join(BASE_DIR, "observed_catalog.json")
with open(obs_path, "w") as f:
    json.dump({"start_time": iso(START), "end_time": iso(END), "min_magnitude": MMIN, "events": obs_catalog}, f, indent=2)

# Save simulated catalogs as a single JSON list
forecast_path = os.path.join(BASE_DIR, "simulated_catalogs.json")
with open(forecast_path, "w") as f:
    json.dump({
        "start_time": iso(START), "end_time": iso(END), "min_magnitude": MMIN,
        "catalogs": [{"events": cat} for cat in sim_catalogs]
    }, f, indent=2)

# ------------------ Do time-only evaluations (without PyCSEP): N-test, M-test (KS on magnitudes), PL-test-like ------------------
# N-test (empirical): compare N_obs with distribution of N_j from simulated catalogs
N_obs = len(obs_catalog)
N_js = np.array([len(cat) for cat in sim_catalogs])
p_ge = np.mean(N_js >= N_obs)
p_le = np.mean(N_js <= N_obs)

# M-test proxy: compare magnitude distributions via KS test (requires scipy; implement simple KS manually)
def ecdf(sample):
    x = np.sort(sample)
    n = len(x)
    def F(v): 
        return np.searchsorted(x, v, side='right')/n
    return F, x

obs_mags = np.array([e["magnitude"] for e in obs_catalog])
# Build union catalog magnitudes (CSEP做法)
union_mags = np.concatenate([np.array([e["magnitude"] for e in cat]) for cat in sim_catalogs]) if len(sim_catalogs)>0 else np.array([])

def ks_statistic(x, y):
    # x, y are 1D arrays
    Fx, _ = ecdf(x)
    Fy, _ = ecdf(y)
    grid = np.unique(np.concatenate([x, y]))
    d = np.max(np.abs([Fx(g) - Fy(g) for g in grid]))
    return float(d)

ks_mag = ks_statistic(obs_mags, union_mags) if len(union_mags)>0 and len(obs_mags)>0 else float('nan')

# PL-test (time-only approximation): bin to daily counts; use sim catalogs to estimate expected rate per bin, then compute Poisson log-likelihood for observed
def daily_counts(events, start, end):
    days = (end - start).days
    bins = [start + timedelta(days=d) for d in range(days+1)]
    # count per day
    ts = [datetime.fromisoformat(ev["time"]) for ev in events]
    counts = np.zeros(days, dtype=int)
    for t in ts:
        if start <= t < end:
            d = (t.date() - start.date()).days
            if 0 <= d < days:
                counts[d] += 1
    return counts

days = (END - START).days
obs_daily = daily_counts(obs_catalog, START, END)
sim_daily = np.stack([daily_counts(cat, START, END) for cat in sim_catalogs], axis=0)
lambda_hat = sim_daily.mean(axis=0) + 1e-9  # avoid zeros

# Poisson log-likelihood of observed under lambda_hat
def poisson_loglik(y, mu):
    return float(np.sum(y*np.log(mu) - mu - [math.lgamma(k+1) for k in y]))

pl_loglik = poisson_loglik(obs_daily, lambda_hat)

# Also compute simple deviance and PIT for diagnostics
deviance = 2*np.sum(obs_daily * np.log((obs_daily + 1e-9)/lambda_hat) - (obs_daily - lambda_hat))
# randomized PIT
from scipy.stats import poisson
F_y_minus = poisson.cdf(obs_daily-1, lambda_hat)
p_y = poisson.pmf(obs_daily, lambda_hat)
rng = np.random.default_rng(0)
pit = F_y_minus + rng.random(len(obs_daily)) * p_y

# ------------------ Produce simple plots ------------------
# 1) N-test histogram with observed marker
plt.figure()
plt.hist(N_js, bins=30)
plt.axvline(N_obs, linestyle='--')
plt.title("N-test: simulated counts vs observed")
plt.xlabel("Total events")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "n_test_hist.png"))
plt.close()

# 2) Daily counts vs expected
plt.figure()
plt.plot(range(days), obs_daily, label="Observed")
plt.plot(range(days), lambda_hat, label="Expected (from sims)")
plt.title("Daily counts vs expected (time-only)")
plt.xlabel("Day index")
plt.ylabel("Count / Expected rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "daily_counts_vs_expected.png"))
plt.close()

# 3) PIT histogram
plt.figure()
plt.hist(pit, bins=10, range=(0,1))
plt.title("Randomized PIT (daily Poisson)")
plt.xlabel("PIT")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "pit_hist.png"))
plt.close()

# ------------------ Summaries to show the user ------------------
summary = pd.DataFrame({
    "metric": ["N_obs", "median_N_sim", "p(N_sim >= N_obs)", "p(N_sim <= N_obs)", "KS(mag, union)", "PL_loglik", "Deviance"],
    "value": [N_obs, float(np.median(N_js)), float(p_ge), float(p_le), float(ks_mag), float(pl_loglik), float(deviance)]
})
print(summary)

print("Files created:")
print("Observed catalog:", obs_path)
print("Simulated catalogs:", forecast_path)
print("Plots:", os.path.join(BASE_DIR, "n_test_hist.png"), os.path.join(BASE_DIR, "daily_counts_vs_expected.png"), os.path.join(BASE_DIR, "pit_hist.png"))
