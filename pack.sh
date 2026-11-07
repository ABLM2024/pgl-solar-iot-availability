set -e
cd /home/claude/apr
rm -rf pkg && mkdir -p pkg/code_and_data
cp -r repo/data pkg/code_and_data/
cp repo/README.md repo/LICENSE repo/CITATION.cff repo/requirements.txt repo/style.py repo/fig*.py repo/tables.py repo/select_structure.py repo/drivers.py repo/drivers.csv repo/structure_selection.csv pkg/code_and_data/ 2>/dev/null || true
cd work
cp *.py *.sh ../pkg/code_and_data/
cp res*.csv boot*.csv s_train.csv structure_selection.csv ../pkg/code_and_data/ 2>/dev/null || true
mkdir -p ../pkg/code_and_data/predictions && cp p2_*.csv p3_*.csv p_nested_*.csv p_sens_*.csv ../pkg/code_and_data/predictions/ 2>/dev/null || true
cp -r /home/claude/apr/res/figdata ../pkg/code_and_data/figdata
cat > ../pkg/code_and_data/REVISION_README.md <<'EOF'
# Code and data for the revised APR manuscript

See Supplementary Note S1 for the role of each script. Requirements: Python >= 3.11 (requirements.txt).
Per-prediction files are in predictions/ (move them next to the scripts before re-running the bootstrap and table scripts).
Order: estimators.py, decomp.py, exp2_temporal.py, exp3.py, PGL_THR=0.75 exp3.py _thr75, exp2_spatial.py, exp2_spatial_b.py,
exp_nested.py, exp_sens.py (see q_apr.sh), analysis_apr.py, analysis3.py, analysis3.py _thr75, boot3.py, boot_spatial2.py, shap_apr.py, figs_rev.py, tables_rev.py, tables_rev2.py.
Data sources of the physical constants in data/D_code/leadacid.py: [AUTHOR TO PROVIDE] (Supplementary Note S2).
EOF
cd ../pkg && rm -f ../out/Code_and_Data_APR.zip && zip -qr ../out/Code_and_Data_APR.zip code_and_data && ls -la ../out/Code_and_Data_APR.zip
