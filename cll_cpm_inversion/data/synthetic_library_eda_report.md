# Synthetic library EDA

**Shape:** 1105 rows x 14 columns (7 CPM parameters + 6 morphology features + `sample_id`).

Built by replicate-averaging the final-MCS feature vector of each NaN-free
Saltelli sample (1105 of 1152 designed vectors; 47 dropped for empty /
NaN spheroid outputs), joined with the 7-parameter input vector that produced it.

- Rows: 1105
- Columns: 14

## CPM parameters

| variable | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| width | 1105 | 17.37 | 7.388 | 4.0 | 11.0 | 18.0 | 24.0 | 30.0 |
| temp | 1105 | 38.354 | 21.089 | 2.285 | 19.965 | 38.215 | 56.465 | 74.715 |
| lambda | 1105 | 3.888 | 4.917 | 0.102 | 0.417 | 1.568 | 5.659 | 19.59 |
| contact | 1105 | 25.083 | 14.117 | 1.191 | 12.676 | 24.926 | 37.176 | 49.809 |
| cm_adhesion | 1105 | 25.96 | 14.136 | 1.191 | 13.824 | 26.074 | 38.324 | 49.809 |
| contact_no | 1105 | 4.425 | 2.044 | 1.0 | 3.0 | 4.0 | 6.0 | 8.0 |
| neighbor | 1105 | 3.995 | 1.78 | 1.0 | 3.0 | 4.0 | 5.0 | 7.0 |

## Morphology features

| variable | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| total_area | 1105 | 178743.117 | 38065.619 | 137.0 | 183353.333 | 192675.222 | 195439.444 | 199129.778 |
| equivalent_diameter | 1105 | 471.594 | 72.019 | 13.055 | 483.169 | 495.3 | 498.84 | 503.527 |
| eccentricity | 1105 | 0.119 | 0.06 | 0.013 | 0.083 | 0.111 | 0.138 | 0.792 |
| solidity | 1105 | 0.924 | 0.126 | 0.006 | 0.944 | 0.975 | 0.987 | 0.997 |
| perimeter | 1105 | 9210.021 | 17615.434 | 58.99 | 1692.922 | 1861.651 | 2645.151 | 98026.724 |
| circularity | 1105 | 0.553 | 0.314 | 0.0 | 0.327 | 0.679 | 0.807 | 0.914 |
