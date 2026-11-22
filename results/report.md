# Order book imbalance - results

- Samples: 257,876 | span 21.13 h | recording segments: 5
- Median spread: 0.02 bps | fee assumption: 40 bps/side

## Predictive power

IC = rank correlation between signal and the future mid move. `IC_nonoverlap` and `t_stat` use non-overlapping samples (the honest numbers). `oos_R2` is fit on the first 70% of data, scored on the rest.

signal  horizon_s      n     IC  IC_nonoverlap  t_stat  hit_rate  oos_R2
imb_l1          1 257826 0.2605         0.2644 44.0173    0.6429 -0.1488
imb_l1          2 257776 0.2735         0.2758 32.5737    0.6337 -0.0921
imb_l1          5 257630 0.2333         0.2165 15.9185    0.6087  0.0022
imb_l1         10 257385 0.1766         0.1899  9.8105    0.5820  0.0008
imb_l1         30 256405 0.0878         0.0552  1.6154    0.5383 -0.0721
imb_l1         60 254932 0.0580         0.0215  0.4430    0.5266 -0.1853
imb_l5          1 257826 0.2449         0.2517 41.7538    0.6314 -0.0950
imb_l5          2 257776 0.2568         0.2552 29.9605    0.6252 -0.0414
imb_l5          5 257630 0.2204         0.2233 16.4395    0.6034  0.0476
imb_l5         10 257385 0.1654         0.1927  9.9615    0.5793  0.0392
imb_l5         30 256405 0.0856         0.0696  2.0389    0.5414 -0.0513
imb_l5         60 254932 0.0670         0.0650  1.3407    0.5336 -0.1640
ofi_1s          1 257826 0.2038         0.2119 34.8187    0.6240  0.0137
ofi_1s          2 257776 0.1982         0.2003 23.2071    0.6110  0.0165
ofi_1s          5 257630 0.1530         0.1528 11.0967    0.5831  0.0068
ofi_1s         10 257385 0.1081         0.1267  6.4755    0.5583 -0.0144
ofi_1s         30 256405 0.0573         0.0439  1.2829    0.5287 -0.0769
ofi_1s         60 254932 0.0428         0.0542  1.1157    0.5189 -0.1755

## Does it make money? (top-10% signals, cross spread, pay fees, test period only)

signal  horizon_s  trades  gross_bps  spread_bps  fees_bps  net_bps  win_rate_net
imb_l1          1     749       0.38        0.02        80   -79.66          0.00
imb_l1          2     497       0.59        0.02        80   -79.45          0.00
imb_l1          5     321       0.84        0.02        80   -79.20          0.00
imb_l1         10     249       0.83        0.02        80   -79.21          0.00
imb_l1         30     141       0.68        0.02        80   -79.37          0.00
imb_l1         60      95       0.66        0.02        80   -79.39          0.00
imb_l5          1     680       0.44        0.02        80   -79.58          0.00
imb_l5          2     436       0.69        0.02        80   -79.33          0.00
imb_l5          5     281       1.10        0.02        80   -78.93          0.00
imb_l5         10     204       1.25        0.02        80   -78.77          0.00
imb_l5         30     122       1.23        0.02        80   -78.80          0.00
imb_l5         60      83       0.20        0.02        80   -79.84          0.00
ofi_1s          1     215       0.43        0.02        80   -79.64          0.00
ofi_1s          2     179       0.58        0.02        80   -79.49          0.00
ofi_1s          5     150       0.79        0.02        80   -79.29          0.00
ofi_1s         10     124       0.77        0.02        80   -79.31          0.00
ofi_1s         30      87       0.78        0.02        80   -79.30          0.00
ofi_1s         60      64       0.38        0.02        80   -79.70          0.00
