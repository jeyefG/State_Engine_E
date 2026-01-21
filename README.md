# State Engine Training (Console UI)

## Uso rápido

```bash
python scripts/train_state_engine.py \
  --symbol XAUUSD \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --model-out state_engine/models/xauusd_state_engine.pkl \
  --report-out reports/xauusd_state_engine.json \
  --class-weight-balanced
```

Opciones útiles:

- `--log-level INFO|DEBUG|WARNING`
- `--min-samples 2000`
- `--split-ratio 0.8`
- `--no-rich` (salida básica sin Rich)
- `--class-weight-balanced` (usa `class_weight="balanced"` en LightGBM)
- `--report-out` (guarda un reporte JSON con métricas, gating y metadatos)

## Event Scorer (M5) + Backtesting

Entrena el Event Scorer (M5) usando el contexto H1 ya desplazado y un triple-barrier con
`r_outcome` continuo (el score es telemetría; no genera señales):

```bash
python scripts/train_event_scorer.py \
  --symbol XAUUSD \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --model-dir state_engine/models \
  --k-bars 24 \
  --reward-r 1.0 \
  --sl-mult 1.0 \
  --r-thr 0.0 \
  --tie-break distance

Notas:
- Se reportan métricas de ranking en calibración: precision@K y lift@K por familia y por bins de `margin_H1`.
- También se guarda un CSV opcional en `state_engine/models/metrics_{symbol}_event_scorer.csv`.
```

### Telemetría Event Scorer (screen/triage/files)

El trainer ahora permite elegir el nivel de telemetría de consola (y manda el detalle a CSV/JSON):

```bash
python scripts/train_event_scorer.py \
  --symbol XAUUSD \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --telemetry screen
```

Ejemplo `--telemetry=screen` (máximo ~25 líneas):

```
RUN 20250712T120000Z_ab12cd34 | symbol=XAUUSD mode=production score_tf=M5 context_tf=H1 k_bars=24 start=2024-01-01 end=2025-12-31 cutoffs:ctx=2025-12-31 23:00:00 score=2025-12-31 23:55:00
Funnel: score_total=198720 after_merge=198702 after_ctx_dropna=198690 events_detected=8421 events_labeled=8210 events_post_state_filter=8012 events_post_meta=6420
VWAP validity: valid_pct_all=99.82% valid_pct_last_session=94.50% WARN
Baseline r_mean (post-filter): BALANCE n=3120 r_mean=0.0123 | TREND n=4010 r_mean=0.0189 | TRANSITION n=882 r_mean=-0.0041
Scorer vs baseline: delta_r_mean@20=0.0214 lift@20_ratio=1.18 spearman=0.0831 verdict=EDGE
Best regime: BALANCE|m0|LOOK_FOR_none|FAM1|E_NEAR_VWAP n=420 delta_r_mean@20=0.1123 flag=WIN
Worst regime: TREND|m2|LOOK_FOR_none|FAM3|E_REJECTION_VWAP n=260 delta_r_mean@20=-0.0542 flag=LOSE|RANK_INVERTED
```

Ejemplo `--telemetry=triage` (máximo ~60 líneas):

```
RUN 20250712T120000Z_ab12cd34 | symbol=XAUUSD mode=production score_tf=M5 context_tf=H1 k_bars=24 start=2024-01-01 end=2025-12-31 cutoffs:ctx=2025-12-31 23:00:00 score=2025-12-31 23:55:00
Top 5 regimes by delta_r_mean@20:
+ BALANCE|m0|LOOK_FOR_none|FAM1|E_NEAR_VWAP n=420 delta_r_mean@20=0.1123 flag=WIN
+ TREND|m1|LOOK_FOR_trend_pullback|FAM2|E_TOUCH_VWAP n=310 delta_r_mean@20=0.0844 flag=WIN
+ TRANSITION|m2|LOOK_FOR_none|FAM4|E_NEAR_VWAP n=220 delta_r_mean@20=0.0660 flag=WIN
+ BALANCE|m1|LOOK_FOR_balance_fade|FAM1|E_REJECTION_VWAP n=190 delta_r_mean@20=0.0552 flag=MIXED
+ TREND|m0|LOOK_FOR_none|FAM2|E_NEAR_VWAP n=175 delta_r_mean@20=0.0410 flag=MIXED
Bottom 5 regimes by delta_r_mean@20:
- TREND|m2|LOOK_FOR_none|FAM3|E_REJECTION_VWAP n=260 delta_r_mean@20=-0.0542 flag=LOSE|RANK_INVERTED
- BALANCE|m2|LOOK_FOR_balance_fade|FAM1|E_TOUCH_VWAP n=210 delta_r_mean@20=-0.0419 flag=LOSE
- TRANSITION|m1|LOOK_FOR_none|FAM5|E_NEAR_VWAP n=180 delta_r_mean@20=-0.0320 flag=LOSE
- BALANCE|m0|LOOK_FOR_none|FAM4|E_REJECTION_VWAP n=150 delta_r_mean@20=-0.0267 flag=LOSE
- TREND|m1|LOOK_FOR_trend_pullback|FAM2|E_REJECTION_VWAP n=140 delta_r_mean@20=-0.0188 flag=LOSE
Inverted ranks: 1/28
Stability (top 5 regimes):
* BALANCE|m0|LOOK_FOR_none|FAM1|E_NEAR_VWAP | 2025H1 n=150 r_mean@20=0.0981 | 2025H2 n=170 r_mean@20=0.1066 | 2026YTD n=100 r_mean@20=0.1210
* TREND|m1|LOOK_FOR_trend_pullback|FAM2|E_TOUCH_VWAP | 2025H1 n=110 r_mean@20=0.0701 | 2025H2 n=120 r_mean@20=0.0812 | 2026YTD n=80 r_mean@20=0.0884
* TRANSITION|m2|LOOK_FOR_none|FAM4|E_NEAR_VWAP | 2025H1 n=80 r_mean@20=0.0590 | 2025H2 n=90 r_mean@20=0.0623 | 2026YTD n=50 r_mean@20=0.0681
* BALANCE|m1|LOOK_FOR_balance_fade|FAM1|E_REJECTION_VWAP | 2025H1 n=70 r_mean@20=0.0449 | 2025H2 n=65 r_mean@20=0.0495 | 2026YTD n=55 r_mean@20=0.0512
* TREND|m0|LOOK_FOR_none|FAM2|E_NEAR_VWAP | 2025H1 n=60 r_mean@20=0.0380 | 2025H2 n=58 r_mean@20=0.0401 | 2026YTD n=40 r_mean@20=0.0433
Family training status: TRAINED=5 | SKIP_FAMILY_LOW_SAMPLES=2 | SKIP_FAMILY_SINGLE_CLASS=1
```

`--telemetry=files` no imprime extra en consola; todo el detalle se guarda en CSV + `summary_*.json`.

### Configuración por símbolo (Event Scorer)

Puedes definir overrides por símbolo en YAML/JSON sin romper la CLI actual. Si no pasas `--config`,
se usan los defaults de siempre.

Plantilla: `configs/symbols/_template.yaml`.

Ejemplo:

```bash
python scripts/train_event_scorer.py \
  --config configs/symbols/XAUUSD.mg.yaml \
  --mode research
```

Notas:
- `--mode` solo afecta los thresholds diagnósticos del reporte (no cambia el entrenamiento).
- El trainer del State Engine (H1) sigue usando solo argumentos CLI por ahora.

Ejecuta el pipeline de backtest (State Engine H1 → Events M5 → Scorer → Signals → Backtest):

```bash
python scripts/run_pipeline_backtest.py \
  --symbol XAUUSD \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --model-dir state_engine/models \
  --edge-threshold 0.6 \
  --max-holding-bars 24 \
  --reward-r 1.0 \
  --sl-mult 1.0 \
  --output-dir outputs
```

Eventos cubiertos: balance/transition/trend pullback y **trend continuation** (E_TREND_CONTINUATION) cuando `LOOK_FOR_trend_continuation` está activo en H1.

## Before (salida anterior, simplificada)

```
Training complete.
            LOOK_FOR_trend_pullback  LOOK_FOR_balance_fade  LOOK_FOR_transition_failure
2024-05-01                    0                  1                        0
2024-05-02                    0                  0                        0
...
```

## After (nueva UI de consola)

```
stage=descarga_h1
download_rows=22080 elapsed=3.14s
stage=build_dataset
features_raw=22080 labels_raw=22080 elapsed=0.82s
stage=align_and_clean
aligned_samples=20350 dropped_nan=1730 elapsed=0.04s
stage=split
n_train=16280 n_test=4070 split_ratio=0.80 elapsed=0.00s
stage=train_model
train_elapsed=2.10s
stage=evaluate
accuracy=0.6243 f1_macro=0.5981 elapsed=0.01s
stage=predict_outputs
outputs_rows=20350 elapsed=0.05s
stage=phase_d
look_for_coverage_rate=34.20% elapsed=0.01s
stage=save_model
model_path=models/xauusd_state_engine.pkl elapsed=0.00s

=== State Engine Training Summary ===
Symbol: XAUUSD
Period: 2024-01-01 -> 2025-12-31
Samples: 20350 (train=16280, test=4070)
Baseline: BALANCE (45.10%)
Accuracy: 0.6243 | F1 Macro: 0.5981
Look-for coverage rate: 34.20%
Last H1 bar used: 2025-06-01 12:00:00 | age_min=4.00
Server now (tick): 2025-06-01 12:04:00 | tick_age_min_vs_utc=0.25
Last bar decision: ALLOW=True | state_hat=BALANCE | margin=0.2134
Last bar rules fired: ['LOOK_FOR_balance_fade']
Model saved: models/xauusd_state_engine.pkl
Report saved: reports/xauusd_state_engine.json
```

## Diagnóstico RESCATE (antes vs después)

Antes (columnas truncadas y buckets con NaN):

```
BALANCE_RESCUE_GRID_EXTENDED
ax…  st…  di…    n  pct_state      ev     p10     p50     p90    delta  st…
...  NY   NaN  118     12.34  0.00012  -0.0003  0.0001  0.0005  0.00002  1
```

Después (columnas cortas, buckets sin NaN):

```
BALANCE_RESCUE_GRID_EXTENDED
axis    ses   age   dvwap        n  pct_state        ev       p10       p50       p90     delta  stable_n atr_bucket
ses,age NY    0-2   <=0.5      118   12.340000  0.000120 -0.000300  0.000100  0.000500  0.000020         2       1-1.25
```
