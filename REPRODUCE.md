# Reproducing the APR article results

See Supplementary Note S1 for the role of each script. Requirements: Python >= 3.11 (requirements.txt).
Per-prediction files are in predictions/ (move them next to the scripts before re-running the bootstrap and table scripts).
Order: estimators.py, decomp.py, exp2_temporal.py, exp3.py, PGL_THR=0.75 exp3.py _thr75, exp2_spatial.py, exp2_spatial_b.py,
exp_nested.py, boot_nested.py, exp_sens.py (see q_apr.sh), analysis_apr.py, analysis3.py, analysis3.py _thr75, boot3.py, boot_spatial2.py, shap_apr.py, figs_rev.py, tables_rev.py, tables_rev2.py.
Physical constants in data/D_code/leadacid.py: see Supplementary Note S2 (compiled long-term monthly means for Bucharest; generic VRLA characteristics).
