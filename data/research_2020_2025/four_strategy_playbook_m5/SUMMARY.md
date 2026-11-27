# Broker M5 walk-forward replay

Closed UTC M5 bars for ORB and sweep. EMA/VWAP and Asian signals use complete M15 bars, H1 regimes use complete hourly bars. Fills use the following M5 open. Asian band flatness compares 0.25 and 0.50 ATR. Router uses the original strict Asian rule. No historical calendar CSV found, so the red-folder filter is inactive. Broker tick values are export-time estimates. Validation selects profiles on the preceding fifth of trading dates for each of the final two fifths. Commission and slippage are charged on partial exits proportionally. Incomplete sessions use their last observed close.

## Coverage (UTC)

- EURUSD_M5: 100,698 bars, 2025-07-22 12:15:00+00:00 to 2026-11-27 00:25:00+00:00
- GBPUSD_M5: 100,929 bars, 2025-07-21 17:05:00+00:00 to 2026-11-27 00:25:00+00:00
- Germany_40_M5: 99,748 bars, 2025-06-25 16:55:00+00:00 to 2026-11-27 00:25:00+00:00
- US_Tech_100_M5: 100,837 bars, 2025-06-25 15:00:00+00:00 to 2026-11-27 00:25:00+00:00
- USDJPY_M5: 100,929 bars, 2025-07-21 16:45:00+00:00 to 2026-11-27 00:25:00+00:00
- Wall_Street_30_M5: 100,837 bars, 2025-06-25 15:00:00+00:00 to 2026-11-27 00:25:00+00:00
- XAUUSD_M5: 100,013 bars, 2025-06-30 11:05:00+00:00 to 2026-11-27 00:25:00+00:00

## Selected profiles, pooled OOS

```text
        symbol    strategy  trades  win_pct  baseline_pf  stressed_pf  max_dd_pct  avg_net_r  passed
        EURUSD    Asian MR       0   0.0000       0.0000       0.0000      0.0000     0.0000   False
        EURUSD    EMA VWAP     121  41.3223       0.5573       0.4318     34.0730    -0.2291   False
        EURUSD      Router     241  37.3444       0.7700       0.5357     59.0528    -0.1467   False
        EURUSD Session ORB     306  28.1046       0.7282       0.5334     73.1262    -0.1825   False
        EURUSD   Sweep MSS     130  35.3846       0.7488       0.5442     37.9734    -0.1633   False
        GBPUSD    Asian MR       0   0.0000       0.0000       0.0000      0.0000     0.0000   False
        GBPUSD    EMA VWAP     133  41.3534       0.8245       0.6194     26.8706    -0.0836   False
        GBPUSD      Router     232  33.1897       0.6830       0.5292     55.8799    -0.2065   False
        GBPUSD Session ORB     332  25.0000       0.6269       0.4602     76.4486    -0.2612   False
        GBPUSD   Sweep MSS     133  34.5865       0.7158       0.5796     33.3065    -0.1774   False
    Germany 40    Asian MR       0   0.0000       0.0000       0.0000      0.0000     0.0000   False
    Germany 40    EMA VWAP     109  54.1284       1.2735       1.1056     11.7897     0.1136   False
    Germany 40      Router     219  36.0731       0.6408       0.5513     49.5277    -0.2203   False
    Germany 40 Session ORB     354  33.0508       0.7007       0.5875     65.8492    -0.1829   False
    Germany 40   Sweep MSS     126  37.3016       0.7345       0.6283     30.5987    -0.1525   False
   US Tech 100    Asian MR       0   0.0000       0.0000       0.0000      0.0000     0.0000   False
   US Tech 100    EMA VWAP     134  44.0299       0.8979       0.8041     29.1906    -0.0552   False
   US Tech 100      Router     231  34.6320       0.6965       0.6217     46.3741    -0.1924   False
   US Tech 100 Session ORB     148  29.7297       0.5469       0.4833     50.9922    -0.3565   False
   US Tech 100   Sweep MSS     164  35.9756       0.8089       0.7293     31.7981    -0.1208   False
        USDJPY    Asian MR       0   0.0000       0.0000       0.0000      0.0000     0.0000   False
        USDJPY    EMA VWAP     131  42.7481       0.6425       0.4693     35.0893    -0.1915   False
        USDJPY      Router     252  40.4762       0.7432       0.5553     53.1366    -0.1501   False
        USDJPY Session ORB     370  28.3784       0.6544       0.5237     72.5378    -0.2140   False
        USDJPY   Sweep MSS     129  37.9845       0.5894       0.4559     40.8020    -0.2468   False
Wall Street 30    Asian MR       0   0.0000       0.0000       0.0000      0.0000     0.0000   False
Wall Street 30    EMA VWAP     138  41.3043       0.9116       0.8183     19.8582    -0.0459   False
Wall Street 30      Router     233  38.1974       0.8765       0.7625     34.3154    -0.0748   False
Wall Street 30 Session ORB     126  34.1270       1.0236       0.8205     20.7967     0.0150   False
Wall Street 30   Sweep MSS     165  44.8485       0.7920       0.7152     28.9510    -0.1120   False
        XAUUSD    Asian MR       0   0.0000       0.0000       0.0000      0.0000     0.0000   False
        XAUUSD    EMA VWAP     130  50.7692       1.2297       1.2017      9.6162     0.0997    True
        XAUUSD      Router     235  37.8723       0.9390       0.8881     19.7753    -0.0350   False
        XAUUSD Session ORB     352  24.4318       0.8120       0.7861     41.7331    -0.1114   False
        XAUUSD   Sweep MSS     128  39.8438       0.9556       0.9163     17.1953    -0.0246   False
```
