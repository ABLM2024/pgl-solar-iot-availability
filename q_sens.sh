PGL_FOLDS=fixed python3 exp_sens.py base
PGL_FOLDS=fixed PGL_TL=0 python3 exp_sens.py TL0
PGL_FOLDS=fixed PGL_TL=2.5 python3 exp_sens.py TL2.5
PGL_FOLDS=fixed PGL_TL=7.5 python3 exp_sens.py TL7.5
PGL_FOLDS=fixed PGL_TL=10 python3 exp_sens.py TL10
PGL_FOLDS=fixed PGL_RES=0.1 python3 exp_sens.py RES0.1
PGL_FOLDS=fixed PGL_RES=0.3 python3 exp_sens.py RES0.3
PGL_FOLDS=fixed PGL_PR=0.55 python3 exp_sens.py PR0.55
PGL_FOLDS=fixed PGL_PR=0.75 python3 exp_sens.py PR0.75
PGL_FOLDS=fixed PGL_NL=1 python3 exp_sens.py NL1
PGL_FOLDS=fixed PGL_NL=999 python3 exp_sens.py NLinf
PGL_FOLDS=fixed PGL_DOD=1.0 python3 exp_sens.py DOD1.0
PGL_FOLDS=fixed PGL_CM=off python3 exp_sens.py CMoff
PGL_FOLDS=fixed PGL_CM=dec python3 exp_sens.py CMdec
echo SENSDONE
