"""Offline evidence screen for fixed-rule OOS results. Never writes execution approvals."""
import pandas as pd

from research_lab import standard_qualification
from strategy_audit_variants import OUT

RATIONALE = {
    "session_orb": "Opening auction imbalance can persist after a confirmed break.",
    "intraday_range_break": "Liquid-session range expansion can persist in an established trend.",
    "trend_momentum": "Trend acceleration can persist during volatility expansion.",
    "volatility_breakout_xgb": "Compression can precede expansion; prior labels filter breakouts.",
    "ema_trend_pullback": "A retracement can resume the prevailing trend.",
    "ema_rsi_pullback": "Momentum recovery can confirm a trend retracement.",
    "vwap_pullback": "Session VWAP can act as support or resistance on a reclaim.",
    "vwap_mean_reversion": "Balanced auctions can revert toward traded value.",
    "judas_sweep_mss": "A rejected liquidity sweep can reverse after a structure break.",
    "legacy_sweep_fvg": "Displacement after a rejected extreme can confirm a reversal.",
    "silver_bullet_fvg": "New York displacement can leave an imbalance during liquid hours.",
}

def audit(rows):
    data = rows.loc[rows.strategy != "portfolio_check"].copy().reset_index(drop=True)
    reasons = []
    for row in data.itertuples():
        def stats(trades, expectancy, pf):
            return dict(trades=trades, average_r=expectancy, profit_factor=pf,
                        max_drawdown_pct=row.max_drawdown_pct)
        gate = standard_qualification(stats(row.oos_trades,row.avg_net_r,row.baseline_pf),
                                      stats(row.stress_trades,row.stressed_avg_net_r,row.stressed_pf))
        failed = [reason for reason in gate["failures"] if "drawdown" not in reason]
        if any("drawdown" in reason for reason in gate["failures"]):
            failed.append("worst scenario drawdown exceeds limit or is unavailable")
        if row.folds_available != 2:
            failed.append("incomplete walk-forward coverage")
        if row.strategy not in RATIONALE:
            failed.append("missing structural hypothesis")
        reasons.append(failed)
    data["structural_reason"] = data.strategy.map(RATIONALE)
    data["source_pass"] = [not failure for failure in reasons]
    group = ["symbol","strategy","variant","session","max_trades_per_session"]
    equivalent = data.groupby(group+["exit_profile"],dropna=False)
    replicated = equivalent.source_pass.transform("all") & (equivalent.source.transform("nunique") >= 2)
    data["replicated"] = replicated
    alternatives = data.groupby(group,dropna=False).apply(
        lambda rows: rows.loc[rows.replicated,"exit_profile"].nunique(), include_groups=False)
    for i,key in enumerate(data[group].itertuples(index=False,name=None)):
        if not replicated.iat[i]:
            reasons[i].append("requires two passing source histories")
        if alternatives.loc[key] < 2:
            reasons[i].append("fewer than two replicated exit settings")
    data["failures"] = [" | ".join(items) for items in reasons]
    data["research_pass"] = data.failures.eq("")
    return data

if __name__ == "__main__":
    result = audit(pd.read_csv(OUT/"full_strategy_variants_report.csv"))
    result.to_csv(OUT/"risk_gate_audit.csv",index=False)
    print(result[["source_pass","replicated","research_pass"]].sum())
    print(result.loc[result.research_pass,["symbol","strategy","variant","session","exit_profile"]].drop_duplicates().to_string(index=False))
