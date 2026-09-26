"""Analysis of a real randomized e-mail marketing experiment.

Data: Kevin Hillstrom's MineThatData E-Mail Analytics challenge. 64,000
customers who last purchased within 12 months were randomly assigned to three
arms: a men's-merchandise e-mail, a women's-merchandise e-mail, or no e-mail
(control). Outcomes were tracked for two weeks: site visit, purchase
(conversion) and dollars spent.

Decision: should the retailer send campaign e-mails, which version, and to whom?

Pre-specified plan (written before looking at outcomes):
  primary metric      conversion (purchased within 2 weeks)
  secondary metrics   visit rate, spend per customer (revenue)
  guardrail           none available: the data has no unsubscribe or complaint
                      field, a limitation noted in the memo
  tests               2 treatments x 3 metrics = 6 comparisons, Holm-corrected
                      at family-wise alpha = 0.05
  checks              SQL data validation, sample-ratio mismatch, covariate balance
"""
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "hillstrom_email_experiment.csv"
OUT = ROOT / "reports"
FIG = OUT / "figures"
SEED = 11
ARMS = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}
METRICS = ["conversion", "visit", "spend"]
COVS = ["recency", "history", "mens", "womens", "newbie"]


def sql(name):
    q = (ROOT / "sql" / name).read_text().replace("{path}", str(RAW))
    return duckdb.sql(q).df()


def load():
    df = pd.read_csv(RAW)
    df["arm"] = df.segment.map(ARMS)
    return df


# ---------------------------------------------------------------- validity checks
def validate():
    v = sql("validation.sql")
    assert v.violations.sum() == 0, f"data validation failed:\n{v}"
    return v


def srm(df):
    counts = df.arm.value_counts().reindex(["control", "mens", "womens"])
    chi2, p = stats.chisquare(counts, [len(df) / 3] * 3)
    return counts, chi2, p


def balance(df):
    X = pd.get_dummies(df[COVS + ["zip_code", "channel"]], columns=["zip_code", "channel"], dtype=float)
    rows = []
    for t in ["mens", "womens"]:
        a, b = X[df.arm == t], X[df.arm == "control"]
        smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        rows.append(smd.rename(t))
    return pd.concat(rows, axis=1)


def mde(p0, n, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(2 * p0 * (1 - p0) / n)


# ---------------------------------------------------------------- treatment effects
def effects(df, n_boot=2000):
    rng = np.random.default_rng(SEED)
    rows = []
    c = df[df.arm == "control"]
    for t in ["mens", "womens"]:
        d = df[df.arm == t]
        for m in METRICS:
            x, y = d[m].values, c[m].values
            diff = x.mean() - y.mean()
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            p = 2 * stats.norm.sf(abs(diff / se))
            row = {"treatment": t, "metric": m, "control_mean": y.mean(), "treated_mean": x.mean(),
                   "diff": diff, "rel_lift": diff / y.mean(), "ci_lo": diff - 1.96 * se,
                   "ci_hi": diff + 1.96 * se, "p_value": p}
            if m == "spend":   # heavy-tailed and zero-inflated: confirm with a bootstrap CI
                bs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n_boot)]
                row["boot_ci_lo"], row["boot_ci_hi"] = np.percentile(bs, [2.5, 97.5])
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out.p_value, alpha=0.05, method="holm")[1]
    out["significant_holm"] = out.p_holm < 0.05
    return out


def regression_adjusted(df):
    """Add pre-treatment covariates: same estimand, smaller standard errors (CUPED-style)."""
    rows = []
    for t in ["mens", "womens"]:
        d = df[df.arm.isin([t, "control"])].assign(treat=lambda x: (x.arm == t).astype(int))
        for m in METRICS:
            raw = smf.ols(f"{m} ~ treat", d).fit(cov_type="HC1")
            adj = smf.ols(f"{m} ~ treat + recency + np.log1p(history) + mens + womens + newbie"
                          " + C(zip_code) + C(channel)", d).fit(cov_type="HC1")
            rows.append({"treatment": t, "metric": m, "diff_raw": raw.params.treat,
                         "se_raw": raw.bse.treat, "diff_adj": adj.params.treat, "se_adj": adj.bse.treat,
                         "se_reduction": 1 - adj.bse.treat / raw.bse.treat,
                         "equiv_sample_gain": (raw.bse.treat / adj.bse.treat) ** 2 - 1})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- heterogeneity
def heterogeneity(df):
    """Does each e-mail work better for customers who previously bought that category?"""
    df = df.assign(past=np.select([(df.mens == 1) & (df.womens == 1), df.mens == 1],
                                  ["both", "mens only"], "womens only"))
    rows = []
    for t in ["mens", "womens"]:
        d = df[df.arm.isin([t, "control"])].assign(treat=lambda x: (x.arm == t).astype(int))
        for seg, g in d.groupby("past"):
            for m in ["visit", "spend"]:
                fit = smf.ols(f"{m} ~ treat", g).fit(cov_type="HC1")
                rows.append({"treatment": t, "past_purchases": seg, "metric": m, "n": len(g),
                             "effect": fit.params.treat, "ci_lo": fit.conf_int().loc["treat", 0],
                             "ci_hi": fit.conf_int().loc["treat", 1], "p_value": fit.pvalues.treat})
        # formal interaction test: is the effect different across purchase-history groups?
        for m in ["visit", "spend"]:
            full = smf.ols(f"{m} ~ treat * C(past)", d).fit(cov_type="HC1")
            names = [n for n in full.params.index if n.startswith("treat:")]
            w = full.wald_test(" = 0, ".join(names) + " = 0", scalar=True)
            rows.append({"treatment": t, "past_purchases": "INTERACTION TEST", "metric": m,
                         "n": len(d), "p_value": float(w.pvalue)})
    out = pd.DataFrame(rows)
    seg_rows = out.past_purchases != "INTERACTION TEST"
    out.loc[seg_rows, "p_holm"] = multipletests(out.loc[seg_rows, "p_value"], method="holm")[1]
    return out


# ---------------------------------------------------------------- targeting (uplift model)
def uplift_targeting(df, treat="mens"):
    """T-learner: predict visit probability with and without the e-mail, target by the difference.

    Trained on half the experiment and evaluated on the other half, where
    randomization lets us measure true incremental visits and spend in every
    targeting bucket.
    """
    d = df[df.arm.isin([treat, "control"])].copy()
    d["treat"] = (d.arm == treat).astype(int)
    X = pd.get_dummies(d[COVS + ["zip_code", "channel", "history_segment"]], dtype=float)
    Xtr, Xte, dtr, dte = train_test_split(X, d, test_size=0.5, random_state=SEED, stratify=d.treat)
    mk = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                min_samples_leaf=200, random_state=SEED)
    m1 = mk().fit(Xtr[dtr.treat == 1], dtr.visit[dtr.treat == 1])
    m0 = mk().fit(Xtr[dtr.treat == 0], dtr.visit[dtr.treat == 0])
    dte = dte.assign(uplift=m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1])
    dte["decile"] = pd.qcut(dte.uplift.rank(method="first", ascending=False), 10, labels=False) + 1

    def inc(g, m):
        return g.loc[g.treat == 1, m].mean() - g.loc[g.treat == 0, m].mean()

    by_dec = dte.groupby("decile").apply(lambda g: pd.Series({
        "n": len(g), "pred_uplift": g.uplift.mean(), "true_visit_uplift": inc(g, "visit"),
        "true_spend_uplift": inc(g, "spend")}), include_groups=False)
    # cumulative gain: share of total incremental visits captured by e-mailing the top k deciles
    total_visits = inc(dte, "visit") * len(dte)
    total_spend = inc(dte, "spend") * len(dte)
    cum = []
    for k in range(1, 11):
        g = dte[dte.decile <= k]
        cum.append({"top_deciles": k, "share_emailed": len(g) / len(dte),
                    "visit_capture": inc(g, "visit") * len(g) / total_visits,
                    "spend_capture": inc(g, "spend") * len(g) / total_spend})
    return by_dec, pd.DataFrame(cum)


def plot(eff, cum):
    FIG.mkdir(parents=True, exist_ok=True)
    blue, orange, ink2, grid = "#2a78d6", "#eb6834", "#52514e", "#e6e5e0"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    e = eff.copy()
    labels, y = [], []
    for i, (_, r) in enumerate(e.iterrows()):
        c = blue if r.treatment == "mens" else orange
        ax.errorbar(r.rel_lift * 100, i, xerr=[[(r["diff"] - r.ci_lo) / r.control_mean * 100],
                                               [(r.ci_hi - r["diff"]) / r.control_mean * 100]],
                    fmt="o", color=c, ms=6, capsize=3, lw=1.5)
        labels.append(f"{r.treatment} e-mail: {r.metric}")
    ax.set_yticks(range(len(e)), labels)
    ax.axvline(0, color=ink2, lw=0.8)
    ax.set_xlabel("Lift vs no e-mail (%, 95% CI)")
    ax.set_title("Both e-mails lift every metric; men's wins", loc="left", fontweight="bold")
    ax = axes[1]
    ax.plot([0] + list(cum.share_emailed * 100), [0] + list(cum.visit_capture * 100), color=blue, lw=2,
            marker="o", ms=4, label="Uplift-model targeting (incremental visits)")
    ax.plot([0, 100], [0, 100], color=ink2, lw=1, ls="--", label="Random targeting")
    ax.set_xlabel("Share of customers e-mailed (%)")
    ax.set_ylabel("Share of incremental visits captured (%)")
    ax.set_title("Uplift targeting did not beat random", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    for a in axes:
        a.grid(color=grid); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "email_experiment_results.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    v = validate()
    counts, chi2, p_srm = srm(df)
    bal = balance(df)
    metrics = sql("metrics.sql")
    n_arm = counts.min()
    design = pd.DataFrame([{"metric": m, "control_rate": df[df.arm == "control"][m].mean(),
                            "MDE_abs_80pct_power": mde(df[df.arm == "control"][m].mean(), n_arm)}
                           for m in ["conversion", "visit"]])
    eff = effects(df)
    adj = regression_adjusted(df)
    het = heterogeneity(df)
    by_dec, cum = uplift_targeting(df)

    for name, t in [("validation", v), ("arm_metrics_sql", metrics), ("balance_smd", bal),
                    ("design_mde", design), ("effects", eff), ("regression_adjusted", adj),
                    ("heterogeneity", het), ("uplift_deciles", by_dec), ("uplift_cumulative", cum)]:
        t.to_csv(OUT / f"{name}.csv")
    plot(eff, cum)

    pd.set_option("display.width", 200)
    print(f"SRM: counts={counts.to_dict()} chi2={chi2:.2f} p={p_srm:.3f}")
    print(f"balance: max |SMD| = {bal.abs().max().max():.4f}")
    print(design.round(5).to_string(index=False))
    print(eff.round(5).to_string(index=False))
    print(adj.round(5).to_string(index=False))
    print(het.round(4).to_string(index=False))
    print(by_dec.round(4).to_string())
    print(cum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
