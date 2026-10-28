# Data

Study period: 13 June 2025 – 13 June 2026 (366 days). Network: 20 solar-powered air-quality nodes (Sensirion SEN55-based, 10 Wp panel, 12 V 7 Ah VRLA battery, LoRa radio), Pantelimon street canyon, Bucharest, Romania.

| Folder | File | Content |
|---|---|---|
| `A_telemetry/` | `Ambient_Temperature.csv` | Enclosure temperature reported by each node (°C). First line `sep=,`, then `Time` and 20 node columns (telemetry IDs `UC100-xxxxxx`); blank = node not reporting. |
| `B_availability/` | `node_availability_daily.csv` | Date × 20 nodes: daily fraction of expected particulate samples received. A node-day is *active* when the fraction is ≥ 0.25. |
| | `network_availability_daily.csv`, `network_availability_monthly.csv` | Network availability (mean over the 20 nodes) and active-node counts. |
| `C_reference/` | `reference_PM25_daily_EEA.csv` | Daily PM2.5 (µg m⁻³) at the European Environment Agency sampling points SPO-RO0065A (site A) and SPO-RO0229A (site B), with the measured network availability fraction. Source: European Environment Agency, Air Quality Download Service. |
| `F_siting/` | `node_siting.csv`, `node_id_mapping.csv` | Node number, telemetry ID, mounting type, site description, coordinates (where recorded), commissioning date, uptime and sky exposure. |
| `D_code/` | `leadacid.py` | Lead-acid battery model imported by the code. |

Licence: CC BY 4.0. The EEA reference data remain subject to the EEA terms of use.
