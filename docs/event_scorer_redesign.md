# Event Scorer Redesign (Fase E post-Fase D)

> **Objetivo**: redefinir el Event Scorer como un pipeline explícito, audit-able y “honesto”, que refine únicamente dentro de contextos ya definidos por Fase D (State + ALLOW + Family), sin crear edge cosmético ni competir con Fase D.

## Parte A — Diagnóstico técnico (forense)

### A1) Responsabilidades mezcladas hoy (por bloques)

**1) Configuración, precedencias y “modo”**
- CLI + defaults: `parse_args()` define un set grande de flags y defaults (score_tf, allow_tf, context_tf, meta_policy, thresholds, research, etc.).【F:scripts/train_event_scorer.py†L54-L141】
- Defaults/estructura YAML: `_default_symbol_config()` replica defaults de CLI dentro de `event_scorer` y `research` (doble fuente).【F:scripts/train_event_scorer.py†L144-L200】
- Resolución de thresholds y research: `_resolve_decision_thresholds()` y `_resolve_research_config()` fusionan base/mode y config legacy, con varias llaves posibles.【F:scripts/train_event_scorer.py†L203-L248】
- Merge de YAML y mutación de `args`: `main()` sobreescribe CLI con YAML (score_tf/allow_tf/context_tf, meta_policy, k_bars, etc.) y ajusta `args` nuevamente antes de correr la lógica principal.【F:scripts/train_event_scorer.py†L3634-L3723】

**2) Descarga, cortes temporales y normalización TF**
- `main()` descarga OHLCV en `score_tf` y `context_tf`, y aplica cutoff por timeframe (floor).【F:scripts/train_event_scorer.py†L3771-L3812】

**3) Construcción de contexto D→E (State + ALLOW + ctx_features)**
- `build_context()` calcula features, corre `state_model.predict_outputs`, aplica gating y luego intenta inyectar columnas de contexto para `phase_d.look_fors` (session/state_age/dist_vwap_atr).【F:scripts/train_event_scorer.py†L263-L319】
- `apply_phase_d.look_fors()` se invoca dentro de `build_context()` con el frame y config del símbolo.【F:scripts/train_event_scorer.py†L305-L313】

**4) Merge context TF → score TF**
- `merge_allow_score()` hace `merge_asof` de contexto a score, renombra `state_hat/margin` con sufijo de TF y rellena LOOK_FOR_* a 0.【F:scripts/train_event_scorer.py†L322-L344】

**5) Detección de eventos y diagnóstico VWAP**
- `detect_events()` se ejecuta después del merge (M5) con validaciones de VWAP y estadísticas asociadas.【F:scripts/train_event_scorer.py†L3885-L3899】

**6) Labeling/outcome (triple barrier)**
- `label_events()` etiqueta con `k_bars`, `reward_r`, `sl_mult`, `r_thr` dentro de `_run_training_for_k()`.【F:scripts/train_event_scorer.py†L2250-L2258】

**7) Filtering / gating del dataset**
- State filter (TRANSITION on/off), allow_id, margin bins y regime_id se construyen dentro de `_run_training_for_k()`.【F:scripts/train_event_scorer.py†L2375-L2406】
- Meta policy (margin + allow_active) filtra eventos y define el dataset final de training/calib.【F:scripts/train_event_scorer.py†L2433-L2503】

**8) Split train/calib + entrenamiento/reportes**
- Split temporal train/calib se hace manualmente sobre el dataset final (meta o no-meta) y se usa también para baseline/reportes.【F:scripts/train_event_scorer.py†L2692-L2729】

**9) Reporting y telemetría**
- `diagnostic_report`, `supply_funnel`, summary JSON, tablas de reporte y telemetría en pantalla se generan desde `_run_training_for_k()` y `main()` (modo fallback y normal).【F:scripts/train_event_scorer.py†L2504-L2555】

### A2) Dónde se rompe el objetivo “Fase E refina post-D” (root causes)

1) **ALLOWs pueden quedar “vacíos” o globales por falta de columnas de contexto**
   - `apply_phase_d.look_fors()` solo aplica filtros si existen columnas requeridas (session/state_age/dist_vwap_atr/etc.), y si faltan solo emite warnings y *no falla*; esto facilita un comportamiento silencioso con ALLOWs sin refinamiento contextual real (ALLOWs “globales”).【F:state_engine/gating.py†L489-L599】
   - `build_context()` intenta agregar algunas columnas, pero depende de `full_features` y no valida que el conjunto sea completo para los filtros declarados.【F:scripts/train_event_scorer.py†L263-L319】

2) **El funnel “post-D” no es explícito; E opera con population mix**
   - La construcción de `allow_id` y `regime_id` se hace *después* de etiquetar eventos, pero el pipeline no aplica un filtro explícito por (state + allow + family) antes de entrenar; el gating efectivo es solo `meta_policy` y `include_transition` (global).【F:scripts/train_event_scorer.py†L2375-L2503】

3) **Baselines no son equivalentes (pre-meta vs post-meta)**
   - `dataset_main` (post-meta) y `dataset_no_meta` se crean, y se calculan reportes comparando ambos en diferentes puntos; esto puede comparar métricas usando poblaciones distintas (baseline en no-meta vs scorer en meta).【F:scripts/train_event_scorer.py†L2692-L2729】

4) **LOOK_FOR_none domina por construcción**
   - Cuando no hay ALLOWs activos, `allow_id` se vuelve `LOOK_FOR_none`, lo que concentra eventos y empuja un scoring global (especialmente si los ALLOWs faltan o están desactivados).【F:scripts/train_event_scorer.py†L2375-L2385】

5) **E_GENERIC_VWAP puede dominar familias sin control contextual**
   - Existe una familia genérica (`E_GENERIC_VWAP`), potencialmente dominante si no se aplica scope por family/allow/state antes del ranking (no hay un filtro por familia en el pipeline).【F:state_engine/events.py†L65-L72】【F:scripts/train_event_scorer.py†L2375-L2395】

### A3) “Modo fantasma”: dobles fuentes de verdad + propuesta

**Dobles fuentes (observadas en el script):**
- `score_tf`, `allow_tf`, `context_tf`: CLI defaults + YAML override + metadata del state model (context_tf).【F:scripts/train_event_scorer.py†L54-L141】【F:scripts/train_event_scorer.py†L3634-L3723】【F:scripts/train_event_scorer.py†L427-L449】
- `meta_policy`: CLI (`--meta-policy`) + YAML (`event_scorer.meta_policy.enabled`) + override en research/phase_e por lógica interna.【F:scripts/train_event_scorer.py†L88-L107】【F:scripts/train_event_scorer.py†L3681-L3687】【F:scripts/train_event_scorer.py†L915-L931】
- `research.enabled`: top-level `research` vs `event_scorer.research` vs `event_scorer.modes.<mode>.*` fusionados dinámicamente.【F:scripts/train_event_scorer.py†L212-L248】
- `decision_thresholds`: base vs modes vs CLI fallback (`_resolve_baseline_thresholds`).【F:scripts/train_event_scorer.py†L203-L209】【F:scripts/train_event_scorer.py†L3615-L3631】
- `k_bars`: CLI vs YAML `event_scorer.k_bars` vs `k_bars_by_tf` vs `horizon_min` override.【F:scripts/train_event_scorer.py†L3615-L3733】【F:scripts/train_event_scorer.py†L480-L507】

**Propuesta: `EffectiveConfig` como “single source of truth”**
- Resolver una estructura única al inicio (`effective_config`) y **log completo** antes de cualquier descarga/processing.
- Precedencia explícita: **CLI → YAML symbol → metadata state_model (solo lectura)**, con decisión final registrada y sin re-mutaciones posteriores.
- Un solo lugar con:
  - `timeframes`: `{score_tf, context_tf, allow_tf}` (allow_tf solo metadata hasta que sea real).
  - `policy`: `{meta_policy.enabled/effective, include_transition, phase_e}`.
  - `labels`: `{k_bars, horizon_min, reward_r, sl_mult, r_thr}`.
  - `research`: `{enabled, features, k_bars_grid}`.
  - `thresholds`: `{decision_thresholds, baseline_thresholds}`.
  - `paths`: `{config_path, model_path, output_prefix}`.

---

## Parte B — Diseño propuesto (arquitectura objetivo)

### Pipeline explícito (mínimo pero correcto)

1) **ConfigResolver**
   - Construye `EffectiveConfig` único (single source of truth).
   - Log completo + hash de config y metadata del state model.

2) **ContextBuilder** (núcleo del acople D→E)
   - `full_features = FeatureEngineer.compute_features(ohlcv_ctx)`.
   - `outputs = state_model.predict_outputs(features)`.
   - `allows = gating.apply(outputs, features=full_features)`.
   - **Enriquecer ctx_features + alias** (session/state_age/dist_vwap_atr/etc.).
   - `apply_phase_d.look_fors()` **solo si columnas existen**.
   - Retorna `ctx_df` con `state_hat`, `margin`, `ctx_*`, `LOOK_FOR_*` alineado y `shift(1)`.

3) **EventDatasetBuilder**
   - `merge_asof` score TF con contexto TF → `df_score_ctx`.
   - `detect_events()` y `label_events()`.
   - Producción explícita de columnas: `state_hat`, `margin`, `allow_id`, `family_id`.

4) **FunnelFilter**
   - Aplicar scope **explícito**: `(state + allow + family)`.
   - `meta_policy` solo como filtro de riesgo opcional **con telemetría**.

5) **Trainer / Ranker**
   - Entrenamiento por family (como hoy), ranking y tablas.
   - Mantener outputs CSV/JSON actuales (compatibilidad) en PR-2.

### Regla central de diseño

> **Ningún filtro contextual puede ejecutarse si no existe su columna.**

- En `research` o cuando `phase_e=true`: **si `phase_d.look_fors.<X>.enabled=true` y faltan columnas requeridas → `raise ValueError` / abort explícito**.
- En `baseline/production`: **warning + auto-disable con telemetría**.

---

## Parte C — Plan incremental (3 PRs)

### PR-1 (desbloqueo mínimo)
**Objetivo**: “plumbing correcto” del ContextBuilder y phase_d.look_fors sin refactor masivo.

Checklist:
- [ ] **Las columnas `session`, `state_age`, `dist_vwap_atr` deben provenir de la misma fuente de verdad que usa `train_state_engine.py`** (reusar builder/módulo compartido o factorizar uno común; idealmente importando el builder desde `state_engine/context_features.py` o módulo equivalente). **No se permite calcular estas columnas “a mano” en el scorer**.
- [ ] Inyectar `ctx_features` correctos en `allow_context_frame` (session/state_age/dist_vwap_atr) y **alias explícitos** (p. ej. `ctx_dist_vwap_atr_abs → dist_vwap_atr`, `ctx_session_bucket → session`).
- [ ] Aplicar la política de validación estricta:
  - `research` o `--phase-e`: si `phase_d.look_fors.<X>.enabled=true` y faltan columnas requeridas → `raise ValueError` (abort explícito).
  - `baseline/production`: warning + auto-disable con telemetría.
- [ ] Logs pre/post `apply_phase_d.look_fors()` con conteo de LOOK_FOR_*, `allow_active_pct` y top 3 `allow_id`.

**Snippet sugerido (logging de EffectiveConfig)**
```python
# INFO: compacto
logger.info(
    "EFFECTIVE_CONFIG summary | symbol=%s score_tf=%s context_tf=%s mode=%s phase_e=%s",
    symbol,
    score_tf,
    context_tf,
    mode,
    phase_e,
)
# DEBUG: dump completo
logger.debug("EFFECTIVE_CONFIG=%s", effective_config)
```

### PR-2 (refactor estructural sin cambio funcional)
- Extraer `ConfigResolver`, `ContextBuilder`, `EventDatasetBuilder` en módulos nuevos.
- Mantener salidas CSV/JSON/logs existentes.
- Evitar cambios en modelos/labels/state engine.

### PR-3 (hardening)
- Unificar baseline pre/post meta (misma población comparada).
- Añadir conteos train/calib al funnel.
- Validaciones estrictas en research: si faltan columnas requeridas por un experimento → abort con mensaje claro.

---

## Parte D — Preguntas específicas

### D1) ¿Fuente de verdad de ALLOWs y en qué TF?
- La **fuente de verdad** de ALLOWs es el `ctx_h1` que produce `build_context()` (aplica gating + context filters), y luego se **mergea** a `score_tf` vía `merge_allow_score()` usando `context_tf`.【F:scripts/train_event_scorer.py†L263-L344】
- En la práctica, el TF real de ALLOWs es `context_tf` (no `allow_tf`), porque `context_tf` es el TF usado para descargar `ohlcv_ctx` y construir ALLOWs. El merge `merge_asof` usa ese `context_tf` para renombrar columnas y alinear los ALLOWs con score TF.【F:scripts/train_event_scorer.py†L322-L344】【F:scripts/train_event_scorer.py†L3787-L3823】

### D2) ¿El `shift(1)` actual produce leakage o pérdida de allow? ¿Está correcto para “t-1 gating”?
- `build_context()` aplica `ctx = ctx.shift(1)` al final. Esto **evita leakage** al usar el contexto del bar previo, lo cual es correcto para “t-1 gating”.【F:scripts/train_event_scorer.py†L316-L319】
- La pérdida posible es que la primera fila por ventana quede NaN y se caiga en el merge; esto es aceptable para evitar leakage y está alineado con un gating causal (≤ t-1).【F:scripts/train_event_scorer.py†L316-L344】

### D3) ¿`allow_tf` actualmente es decorativo? ¿Cómo debería funcionar en el diseño nuevo?
- **Sí, hoy es decorativo**: `allow_tf` se normaliza y se usa para metadata/logs/salida, pero el contexto real viene de `context_tf`, que es el TF usado para descargar `ohlcv_ctx` y construir ALLOWs.【F:scripts/train_event_scorer.py†L3616-L3812】【F:scripts/train_event_scorer.py†L389-L420】
- Diseño nuevo: eliminar `allow_tf` del flujo operativo o convertirlo en alias de `context_tf` (solo para retrocompatibilidad), y registrar su valor en `EffectiveConfig` sin ambigüedad.

### D4) ¿Cómo evitar que `E_GENERIC_VWAP` domine sin inventar reglas nuevas?
- **Telemetría + scope**: reportar share por `family_id` y aplicar scope explícito (state + allow + family) antes de ranking. No se propone tuning ni reglas nuevas: solo control de población y reporting honesto de dominancias.【F:state_engine/events.py†L65-L72】【F:scripts/train_event_scorer.py†L2375-L2395】

### D5) ¿Qué logs mínimos garantizan que “D llega vivo a E”?
- **EffectiveConfig summary en INFO** + dump completo en DEBUG.
- **Context columns disponibles** y conteos ALLOW pre/post `apply_phase_d.look_fors()`.
- **Distribución de ALLOWs** (`allow_active_pct`, top allow_id).【F:scripts/train_event_scorer.py†L305-L313】【F:scripts/train_event_scorer.py†L2438-L2445】
- **Supply funnel** con conteos por etapa (merge/dropna/events/meta).【F:scripts/train_event_scorer.py†L2504-L2514】

---

## PR-1 Acceptance Test

**Comando**
```bash
python scripts/train_event_scorer.py \
  --symbol XAUUSD.mg \
  --config on \
  --phase-e \
  --mode baseline \
  --score-tf M15 \
  --context-tf H2 \
  --k-bars 8 \
  --telemetry screen
```

**PASS**
- No aparecen warnings de columnas faltantes.
- Logs muestran: `has_session=True`, `has_state_age=True`, `has_dist_vwap_atr=True`.
- `LOOK_FOR_* pre_filter != post_filter` para al menos un allow.
- `allow_active_pct > 0`.

**FAIL**
- Sigue dominando `LOOK_FOR_none`.
- `post_filter == pre_filter` siempre.
- En research/phase-e se emiten warnings silenciosos en vez de abort explícito.

---

**Nota**: no se tocan modelos/labels del State Engine y no se introducen señales direccionales ni tuning. El objetivo es solo claridad, auditabilidad y acople correcto D→E.
