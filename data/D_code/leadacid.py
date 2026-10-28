# -*- coding: utf-8 -*-
"""Lead-acid (VRLA deep-cycle) energy-autonomy model for the 20-node network.

Replaces the lithium-parameterised model of the original manuscript.
Everything downstream of the battery description is regenerated here.
"""
import numpy as np

RNG = np.random.default_rng(20260729)
PR = 0.65                 # performance ratio: controller, low-light, soiling, cabling
ETA_RT = 0.80             # lead-acid round-trip energy efficiency
DOD = 0.50                # permitted depth of discharge in cyclic service
C_NOM_NODE = 84.0         # 12 V x 7 Ah
C_NOM_GW = 408.0          # 12 V x 34 Ah

# ---------------------------------------------------------------- climate
SITES = {
    '38N': dict(name='38° N (e.g. Athens)',
                H=[2.90, 3.40, 4.40, 5.30, 6.00, 6.60, 6.80, 6.50, 5.50, 4.20, 3.00, 2.60],
                T=[10, 10, 12, 16, 21, 26, 29, 29, 25, 20, 15, 11], sw=0.34),
    '44N': dict(name='44° N (Bucharest)',
                H=[2.00, 2.90, 3.80, 4.70, 5.40, 5.70, 6.00, 5.80, 4.90, 3.60, 2.20, 1.70],
                T=[-1, 1, 6, 12, 17, 21, 23, 22, 17, 11, 5, 1], sw=0.42),
    '52N': dict(name='52° N (e.g. Berlin)',
                H=[1.15, 1.95, 2.90, 4.10, 4.90, 5.10, 5.20, 4.90, 3.80, 2.40, 1.30, 0.85],
                T=[1, 2, 5, 10, 15, 18, 20, 19, 15, 10, 5, 2], sw=0.50),
    '60N': dict(name='60° N (e.g. Oslo)',
                H=[0.42, 1.15, 2.60, 3.90, 4.70, 4.90, 4.80, 4.10, 2.75, 1.25, 0.48, 0.22],
                T=[-4, -4, -1, 4, 10, 15, 17, 16, 11, 6, 0, -3], sw=0.58),
}
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_OF_DAY = np.repeat(np.arange(12), DAYS_IN_MONTH)     # 365
DJF = np.isin(MONTH_OF_DAY, [11, 0, 1])

# ------------------------------------------------- lead-acid characteristics
_CT = np.array([-30, -20, -10, 0, 10, 20, 25, 40])
_CF = np.array([0.63, 0.70, 0.78, 0.85, 0.92, 0.98, 1.00, 1.02])
_AT = np.array([-30, -20, -10, 0, 10, 25, 40])
_AF = np.array([0.22, 0.35, 0.50, 0.70, 0.88, 1.00, 1.00])


def cap_factor(T):
    """Usable-capacity availability factor of a VRLA cell, normalised to 25 degC."""
    return np.interp(T, _CT, _CF)


def acc_factor(T):
    """Charge-acceptance factor of a VRLA cell, normalised to 25 degC."""
    return np.interp(T, _AT, _AF)


def freeze_point(soc_frac):
    """Electrolyte freezing point (degC) as a function of state of charge."""
    return -8.0 - 52.0 * np.clip(soc_frac, 0.0, 1.0)


# ---------------------------------------------------------------- weather
# Enclosure thermal model (IP45 box, pole-mounted, battery of finite thermal mass):
# the cell runs warmer than air on average (solar gain on the enclosure), warmer still
# during the midday charging window, and does not reach the nocturnal air minimum.
DT_CELL_MEAN = 2.0     # daily-mean cell temperature above daily-mean air temperature
DT_CELL_CHARGE = 4.0   # cell temperature during the charging window
DT_CELL_MIN = 3.0      # nocturnal cell minimum above the air minimum


def synth_years(site, n_years, seed=0):
    """AR(1) clearness and temperature; returns (H, Tmean, Tmin) of shape (n_years,365)."""
    rng = np.random.default_rng(1000 + seed)
    s = SITES[site]
    Hm = np.array(s['H'])[MONTH_OF_DAY]
    Tm = np.array(s['T'])[MONTH_OF_DAY]
    # season-dependent relative SD of daily clearness: highest in winter
    sw, ss = s['sw'], 0.22
    seas = 0.5 * (1 + np.cos(2 * np.pi * (np.arange(365) - 350) / 365.0))   # 1 in winter
    sd = ss + (sw - ss) * seas

    rho = 0.55
    e = rng.normal(size=(n_years, 365))
    k = np.zeros((n_years, 365))
    k[:, 0] = 1.0 + sd[0] * e[:, 0]
    for d in range(1, 365):
        k[:, d] = 1.0 + rho * (k[:, d - 1] - 1.0) + sd[d] * np.sqrt(1 - rho ** 2) * e[:, d]
    k = np.clip(k, 0.10, 1.45)
    k /= k.mean(axis=1, keepdims=True) / 1.0        # preserve the long-term mean

    rt = 0.70
    et = rng.normal(size=(n_years, 365))
    dT = np.zeros((n_years, 365))
    for d in range(1, 365):
        dT[:, d] = rt * dT[:, d - 1] + 4.5 * np.sqrt(1 - rt ** 2) * et[:, d]
    Tmean = Tm[None, :] + dT
    Tmin = Tmean - 6.0
    return Hm[None, :] * k, Tmean, Tmin


# ---------------------------------------------------------------- core model
def simulate(H, Tmean, Tmin, panel_Wp, usable_Wh, load_W,
             fade=False, r_sulf=0.0015, freeze_step=0.15, tilt_gain=None):
    """Daily energy balance over an ensemble of synthetic years.

    H, Tmean, Tmin : (n_years, 365)
    panel_Wp, usable_Wh : scalars or arrays broadcastable against (n_grid, n_years, 1)
    Returns dict with annual/winter availability, SoC trace, capacity fade.
    """
    ny, nd = H.shape
    Hd = H if tilt_gain is None else H * tilt_gain[MONTH_OF_DAY][None, :]
    panel = np.atleast_1d(np.asarray(panel_Wp, float))
    ub = np.atleast_1d(np.asarray(usable_Wh, float))
    G = np.broadcast_shapes(panel.shape, ub.shape)
    panel = np.broadcast_to(panel, G).reshape(-1, 1)
    ub0 = np.broadcast_to(ub, G).reshape(-1, 1)
    ng = panel.shape[0]

    load_d = load_W * 24.0
    cf = cap_factor(Tmean + DT_CELL_MEAN)             # (ny,nd)
    af = acc_factor(Tmean + DT_CELL_CHARGE)
    cf25 = cap_factor(25.0)

    soc = np.repeat(ub0, ny, axis=1) * 1.0                 # (ng,ny) start full
    fade_f = np.zeros((ng, ny))
    up = np.zeros((ng, ny, nd), dtype=bool)
    soc_trace = np.zeros((ng, ny, nd))
    psoc_days = np.zeros((ng, ny))       # days below 30% of the usable reserve
    freeze_days = np.zeros((ng, ny))     # days on which the cell is below its freezing point
    freeze_events = np.zeros((ng, ny))   # distinct freezing episodes
    was_frozen = np.zeros((ng, ny), dtype=bool)

    for d in range(nd):
        # temperature- and ageing-corrected usable reserve for this day
        cap_d = ub0 * (cf[:, d][None, :] / cf25) * (1.0 - fade_f)
        soc = np.minimum(soc, cap_d)
        gen = panel * Hd[:, d][None, :] * PR
        net = gen - load_d
        chg = np.where(net > 0, net * ETA_RT * af[:, d][None, :], 0.0)
        soc_new = np.where(net > 0,
                           np.minimum(soc + chg, cap_d),
                           soc + net)
        ok = soc_new >= 0.0
        soc = np.where(ok, soc_new, 0.0)
        up[:, :, d] = ok
        soc_trace[:, :, d] = soc
        frac_use = np.divide(soc, np.maximum(cap_d, 1e-9))
        frac_nom = frac_use * DOD                         # relative to nominal capacity
        low = frac_use < 0.30
        frozen = (Tmin[:, d][None, :] + DT_CELL_MIN) < freeze_point(frac_nom)
        psoc_days += low
        freeze_days += frozen
        new_ep = frozen & ~was_frozen
        freeze_events += new_ep
        was_frozen = frozen
        if fade:
            # sulfation saturates: the coarsened sulfate fraction approaches a limit
            sulf = low * r_sulf * (1.0 - fade_f / 0.65)
            fade_f = np.clip(fade_f + sulf + new_ep * freeze_step, 0.0, 0.90)

    ann = up.mean(axis=2) * 100
    win = up[:, :, DJF].mean(axis=2) * 100
    return dict(ann_mean=ann.mean(axis=1), ann_sd=ann.std(axis=1),
                win_mean=win.mean(axis=1), win_sd=win.std(axis=1),
                up=up, soc=soc_trace, fade=fade_f, shape=G,
                psoc_days=psoc_days, freeze_days=freeze_days,
                freeze_events=freeze_events)


def panel_for_target(site, usable_Wh, load_W, target=99.0, season='win',
                     lo=3.0, hi=400.0, n_years=20, seed=0, tilt_gain=None):
    """Bisection on panel rating for a target ensemble-mean availability."""
    H, Tm, Tn = synth_years(site, n_years, seed)
    def f(p):
        r = simulate(H, Tm, Tn, p, usable_Wh, load_W, tilt_gain=tilt_gain)
        return (r['win_mean'] if season == 'win' else r['ann_mean'])[0] - target
    a, b = lo, hi
    if f(b) < 0:
        return np.nan
    for _ in range(40):
        m = 0.5 * (a + b)
        if f(m) < 0:
            a = m
        else:
            b = m
        if b - a < 0.05:
            break
    return 0.5 * (a + b)


# ============================================================ CHEMISTRIES
# Parameterisations used for the chemistry-comparison study (Sec. 3.10).
CHEM = {
    'VRLA': dict(
        label='VRLA deep-cycle lead-acid (deployed)',
        dod=0.50, eta_rt=0.80,
        cT=_CT, cF=_CF, aT=_AT, aF=_AF,
        wh_per_kg=33.0, eur_per_kWh=180.0, cycles50=500, cycles80=250,
        note='charges below 0 degC at reduced rate; sulfates under prolonged PSoC'),
    'LFP': dict(
        label='LiFePO4 with conventional BMS',
        dod=0.80, eta_rt=0.96,
        cT=np.array([-30, -20, -10, 0, 10, 20, 25, 40]),
        cF=np.array([0.70, 0.80, 0.88, 0.94, 0.98, 1.00, 1.00, 1.00]),
        aT=np.array([-30, -20, -0.5, 0.0, 5, 10, 25, 40]),
        aF=np.array([0.00, 0.00, 0.00, 0.05, 0.35, 0.60, 1.00, 1.00]),
        wh_per_kg=95.0, eur_per_kWh=340.0, cycles50=3500, cycles80=2500,
        note='charge below 0 degC prohibited by the BMS to avoid lithium plating'),
    'LFP_heat': dict(
        label='LiFePO4 with self-heating BMS',
        dod=0.80, eta_rt=0.96,
        cT=np.array([-30, -20, -10, 0, 10, 20, 25, 40]),
        cF=np.array([0.70, 0.80, 0.88, 0.94, 0.98, 1.00, 1.00, 1.00]),
        aT=np.array([-30, -20, -10, 0, 10, 25, 40]),
        aF=np.array([0.55, 0.65, 0.75, 0.85, 0.95, 1.00, 1.00]),
        wh_per_kg=88.0, eur_per_kWh=420.0, cycles50=3500, cycles80=2500,
        note='heater draws parasitic energy from the same budget (modelled as a load adder)'),
    'NMC': dict(
        label='Li-ion NMC',
        dod=0.80, eta_rt=0.95,
        cT=np.array([-30, -20, -10, 0, 10, 20, 25, 40]),
        cF=np.array([0.60, 0.72, 0.83, 0.91, 0.97, 1.00, 1.00, 0.99]),
        aT=np.array([-30, -20, -0.5, 0.0, 5, 10, 25, 40]),
        aF=np.array([0.00, 0.00, 0.00, 0.05, 0.30, 0.55, 1.00, 1.00]),
        wh_per_kg=180.0, eur_per_kWh=300.0, cycles50=1200, cycles80=800,
        note='charge below 0 degC prohibited; thermal-runaway risk in unattended enclosures'),
}
# parasitic heater load (W) drawn whenever T < 5 degC, LFP_heat only
HEATER_W = 0.35


def simulate_chem(H, Tmean, Tmin, panel_Wp, usable_Wh, load_W, chem='VRLA',
                  fade=False, r_sulf=0.0015, freeze_step=0.15, tilt_gain=None):
    """As simulate(), but with an explicit chemistry parameterisation."""
    c = CHEM[chem]
    ny, nd = H.shape
    Hd = H if tilt_gain is None else H * tilt_gain[MONTH_OF_DAY][None, :]
    panel = np.atleast_1d(np.asarray(panel_Wp, float))
    ub = np.atleast_1d(np.asarray(usable_Wh, float))
    G = np.broadcast_shapes(panel.shape, ub.shape)
    panel = np.broadcast_to(panel, G).reshape(-1, 1)
    ub0 = np.broadcast_to(ub, G).reshape(-1, 1)
    ng = panel.shape[0]

    cf = np.interp(Tmean + DT_CELL_MEAN, c['cT'], c['cF'])
    af = np.interp(Tmean + DT_CELL_CHARGE, c['aT'], c['aF'])
    cf25 = np.interp(25.0, c['cT'], c['cF'])
    heat = (HEATER_W * 24.0) * ((Tmean + DT_CELL_MEAN) < 5.0) if chem == 'LFP_heat' else 0.0

    soc = np.repeat(ub0, ny, axis=1) * 1.0
    fade_f = np.zeros((ng, ny))
    up = np.zeros((ng, ny, nd), dtype=bool)
    soc_trace = np.zeros((ng, ny, nd))
    for d in range(nd):
        cap_d = ub0 * (cf[:, d][None, :] / cf25) * (1.0 - fade_f)
        soc = np.minimum(soc, cap_d)
        gen = panel * Hd[:, d][None, :] * PR
        ld = load_W * 24.0 + (heat[:, d][None, :] if np.ndim(heat) else 0.0)
        net = gen - ld
        chg = np.where(net > 0, net * c['eta_rt'] * af[:, d][None, :], 0.0)
        soc_new = np.where(net > 0, np.minimum(soc + chg, cap_d), soc + net)
        ok = soc_new >= 0.0
        soc = np.where(ok, soc_new, 0.0)
        up[:, :, d] = ok
        soc_trace[:, :, d] = soc
        if fade and chem == 'VRLA':
            fu = np.divide(soc, np.maximum(cap_d, 1e-9))
            sulf = (fu < 0.30) * r_sulf
            frz = ((fu < 0.25) & (Tmin[:, d][None, :] + DT_CELL_MIN < freeze_point(fu * c['dod']))) * freeze_step
            fade_f = np.clip(fade_f + sulf + frz, 0.0, 0.90)
    ann = up.mean(axis=2) * 100
    win = up[:, :, DJF].mean(axis=2) * 100
    return dict(ann_mean=ann.mean(axis=1), ann_sd=ann.std(axis=1),
                win_mean=win.mean(axis=1), win_sd=win.std(axis=1),
                up=up, soc=soc_trace, fade=fade_f)


def panel_for_target_chem(site, usable_Wh, load_W, chem='VRLA', target=99.0,
                          season='win', n_years=20, seed=0, tilt_gain=None, hi=600.0):
    H, Tm, Tn = synth_years(site, n_years, seed)
    def f(p):
        r = simulate_chem(H, Tm, Tn, p, usable_Wh, load_W, chem, tilt_gain=tilt_gain)
        return (r['win_mean'] if season == 'win' else r['ann_mean'])[0] - target
    a, b = 3.0, hi
    if f(b) < 0:
        return np.nan
    for _ in range(45):
        m = 0.5 * (a + b)
        if f(m) < 0:
            a = m
        else:
            b = m
        if b - a < 0.05:
            break
    return 0.5 * (a + b)
