# AGENTS.md — State Engine / Quality / Contextual Layers  
Arquitectura por fases (descriptivo → contextual → exploratorio)

---

## Propósito general del repositorio

Este repositorio implementa un **State Engine** cuyo rol es describir la **estructura del mercado** de forma:

- robusta  
- causal (≤ t)  
- generalizable  
- independiente de resultado económico  

El sistema se organiza explícitamente en **fases conceptuales**, donde cada etapa tiene:
- un objetivo propio,
- un tipo de validación permitido,
- y límites epistemológicos claros.

El propósito del proyecto es **contextualizar decisiones de trading**, no generarlas prematuramente.

---

## Objetivo operativo del sistema (visión práctica)

El objetivo del sistema es **reducir y concentrar el espacio de interpretación del mercado en tiempo real**, acercándolo a un conjunto pequeño, explícito y auditable de **narrativas estructurales plausibles**, sin inferir outcome ni forzar decisiones.

El sistema no decide qué hacer.  
Decide **cómo entender dónde se está parado**.

---

## Visión general por fases

| Fase | Rol | Naturaleza |
|----|----|----|
| **A** | Representación temporal | descriptiva |
| **B** | Validación estructural | descriptiva |
| **C** | Calidad del contexto | descriptiva |
| **D** | Contextualización estructural avanzada | descriptiva-contextual |
| **E** | Scoring intradía condicionado | exploratoria |

> Las fases **no son intercambiables**.  
> Las métricas económicas **no son válidas** antes de Fase D,  
> y **no son requeridas ni centrales** en Fase D.

---

## Fase A — Representación temporal

**Objetivo**  
Encontrar, por símbolo, una **representación temporal razonable** del mercado.

**Qué se define**
- Timeframe base del State Engine (H1, H2, etc.)
- Tamaño de ventana (`window_hours`, `k_bars`)

**Criterios**
- coherencia estructural
- estabilidad temporal
- interpretabilidad humana

**Explícitamente fuera de scope**
- edge
- EV
- PnL
- decisiones de trading

---

## Fase B — Validación estructural

**Objetivo**  
Validar que el State Engine **clasifica estados de forma estable y no degenerada**.

**Qué se valida**
- distribución de estados
- persistencia temporal
- coherencia lógica
- estabilidad por splits temporales

**Resultado A+B**
> Cada símbolo queda asociado a un **TF + window_hours** razonable para describir su estructura,  
> sin exigir valor predictivo.

---

## Fase C — Quality Layer (calidad del contexto)

**Objetivo**  
Describir la **calidad interna** de un estado ya clasificado.

La Fase C **no busca edge**.  
Busca **reducir incertidumbre contextual**.

### Rol de la Quality Layer
- caracterizar estabilidad, coherencia, fricción, degradación
- siempre condicionada al estado base
- sin inferir dirección, timing ni outcome

### Principios no negociables
Las Quality Labels son:
- descriptivas, no predictivas  
- humanas y visualizables  
- independientes de métricas económicas  

**Preferencias explícitas**
- falsos negativos > falsos positivos  
- no clasificar > clasificar mal  
- “no clasificado” es un output válido  

---

## Fase D — LOOK_FORs  
**Contextualización estructural avanzada**

### Rol real de un LOOK_FOR

Un **LOOK_FOR** es una **etiqueta de contexto adicional**, definida sobre el output conjunto de:

- State Layer  
- Quality Layer  

Su función es **describir una sub-configuración estructural específica**,  
no evaluar su conveniencia económica.

### Qué es un LOOK_FOR
- una **descripción contextual más granular**
- una **condición estructural explícita**
- una **etiqueta interpretable**

### Qué NO es un LOOK_FOR
- NO es una señal
- NO es una decisión
- NO implica operabilidad
- NO habilita ni bloquea trades
- NO asume edge

Un LOOK_FOR **no responde a la pregunta**:  
> “¿Conviene tradear este contexto?”

Sino a:  
> “¿Se cumple esta configuración contextual específica?”

El término histórico ALLOW_* puede leerse semánticamente como LOOK_FOR_*:
indica un contexto donde tiene sentido observar activamente cierta configuración,
sin implicar decisión ni operabilidad.

### Especialización por símbolo

En Fase D, los LOOK_FORs:
- **sí pueden especializarse por símbolo**
- siguen siendo descriptivos
- no reentrenan modelos
- no optimizan performance

Cualquier uso posterior de LOOK_FORs como filtros operativos o gates económicos:
- es **una decisión de diseño futura**
- no está asumida
- no es parte de esta fase

LOOK_FORs nunca afectan filas ni métricas de ‘permitido’; son tags
---

## Fase E — Exploración de edge condicionado (Event Scorer)  
**Estado actual: exploratorio**

### Contexto histórico

El Event Scorer (M5) existe en el repositorio como resultado de exploraciones previas.

Sin embargo:
- fue implementado **antes** de una definición clara de las fases A–D
- su rol epistemológico no estaba correctamente delimitado
- su validez como “mejor enfoque” **no está demostrada**

> La Fase E **solo tiene sentido** si las Fases A–D están sólidas y bien definidas.

---

### Rol de la Fase E

La Fase E tiene como objetivo **explorar la existencia de edge estadístico relativo**,  
**condicionado estrictamente** a contextos estructurales ya definidos por las Fases A–D.

Esta fase **no crea contexto**.  
Solo interroga contextos **ya bien especificados**.

---

### Qué significa “edge” en Fase E

En esta fase, “edge” se entiende como:

- un **sesgo estadístico no trivial**
- observado **solo bajo ciertos contextos explícitos**
- estable a través del tiempo
- superior a un baseline comparable
- **sin asumir convertibilidad directa a señal operativa**

No se busca:
- maximizar PnL
- construir setups
- optimizar reglas de entrada/salida

---

### Qué hace la Fase E (permitido)

La Fase E **sí puede**:

- operar en temporalidad intradía (ej. M5)
- medir distribuciones condicionadas por:
  - State  
  - Quality  
  - ALLOW (como contexto)
- producir métricas de **edge relativo** (ej. `edge_score`)
- comparar contra:
  - contexto más amplio  
  - baseline incondicional  
- aceptar resultados nulos como outcome válido
- descartar hipótesis sin forzar conclusiones

El output de la Fase E es **telemetría**, no decisión.

---

### Qué NO hace la Fase E (límites duros)

La Fase E **NO debe**:

- generar BUY / SELL
- decidir timing
- definir SL / TP
- optimizar thresholds para performance
- reetiquetar o redefinir contexto
- justificar retroactivamente Fases A–D
- asumir que edge observado implica explotabilidad

Si alguna de estas condiciones se viola,  
la implementación **deja de ser Fase E** y debe reclasificarse.

---

### Criterios de validez de Fase E

Una implementación de Fase E se considera **válida** solo si cumple:

- **Condicionalidad estricta**  
  El edge se mide *solo* dentro de contextos definidos por A–D.

- **Separación semántica**  
  El scorer no introduce nuevas narrativas ni redefine estados.

- **Estabilidad temporal**  
  El edge observado persiste en splits temporales razonables.

- **Comparabilidad**  
  Existe un baseline claro contra el cual se mide el sesgo.

- **Tolerancia al resultado nulo**  
  “No hay edge aquí” es un outcome aceptado y explícito.

---

### Criterios de invalidez (red flags)

La Fase E se considera **epistemológicamente inválida** si:

- el edge solo existe globalmente, pero no condicionado
- el resultado depende críticamente de tuning fino
- pequeñas variaciones de parámetros destruyen el efecto
- el contexto se redefine para “rescatar” performance
- el scorer empieza a comportarse como señal encubierta

---

### Relación con fases posteriores

La Fase E **no habilita ejecución**.

Cualquier decisión de:
- convertir edge en señal
- definir reglas de entrada/salida
- automatizar decisiones

corresponde a **una fase posterior**, explícitamente distinta  
(y no asumida en este repositorio).

---

### Resultado esperado de la Fase E

Al completar la Fase E, el sistema puede:

- confirmar que ciertos contextos **no presentan edge**
- identificar contextos con **sesgo estadístico estable**
- descartar mecanismos intradía espurios
- informar decisiones futuras sin forzarlas

La Fase E **puede fallar**, y ese fallo es informativo.

---

> Si la Fase E necesita edge para justificar el contexto,  
> el problema está en las Fases A–D, no en el scorer.

---

## Arquitectura del repositorio

configs/
symbols/
_template.yaml
XAUUSD.yaml

scripts/
train_state_engine.py
train_event_scorer.py
run_pipeline_backtest.py
watchdog_state_engine.py

state_engine/
pipeline.py
features.py
labels.py
model.py
gating.py
scoring.py
session.py
mt5_connector.py

tests/
test_features.py
test_events.py
test_config_loader.py

La infraestructura se reutiliza entre fases.  
La **validez conceptual no se hereda automáticamente**.

---

## Fuera de scope (Fases A–C)

- optimización por performance  
- nuevos ALLOWs  
- reglas de entrada / salida  
- ejecución, SL/TP  
- automatización de decisiones  

---

## Objetivo acumulado del sistema

Consideradas en conjunto, las fases A–E buscan construir un sistema que:

- reduce la ambigüedad interpretativa del mercado en tiempo real
- acerca la lectura a narrativas estructurales específicas y auditables
- separa estrictamente descripción, contexto y exploración de edge
- evita decisiones prematuras basadas en interpretación débil
- preserva integridad epistemológica en todo el pipeline
