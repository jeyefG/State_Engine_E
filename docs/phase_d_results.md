# Phase D — Symbol Narratives Summary
_State Engine Project_

## Objetivo de Phase D

Phase D **no busca edge ni señales**.  
Su único objetivo es **explorar, identificar y validar narrativas estructurales por símbolo**, entendidas como:

- Sub-regímenes **raros, distinguibles y estables**
- Condicionados por:
  - `state_hat` (BALANCE / TRANSITION / TREND)
  - contexto (`ctx_*`)
  - (opcionalmente) quality labels
- Que **no redefinan el estado base**
- Con **baja colisión** (Jaccard bajo)
- Estables por split temporal

Si un símbolo **no tiene narrativas claras**, eso es un resultado válido y se documenta.

---

## Marco metodológico común

- Evaluación exclusiva de:
  - cobertura total
  - cobertura por split
  - Jaccard
- Prohibido:
  - p-hacking
  - tuning post-hoc
  - forzar simetría entre símbolos
- Los timeframes **no son homogéneos** entre símbolos:
  - Cada símbolo usa el TF/window mínimo que estabiliza su estructura

---

## Resumen global

| Símbolo | TF / Window | Resultado Phase D |
|------|-------------|------------------|
| XAUUSD | H2 / 24 | RICA |
| US500 | H2 / 24 | RICA |
| EURUSD | H2 / 24 | DELGADA |
| USDCLP | H1 / 12 | NO |
| USDMXN | H2 / 24 | MINIMA |
| GBPUSD | H2 / 24 | SELECTIVA |
| XAGUSD | H2 / 24 | CANONICA + AMPLIADA |
| AUDUSD | H2 / 24 | MINIMA |

---

# Narrativas por símbolo

---

## XAUUSD — Oro
**Phase D: RICA**

### Características estructurales
- Mercado profundo, macro-driven
- Aceptación clara de tendencias
- Balance funcional y estable
- Transiciones con resolución limpia

### Narrativas Phase D
XAUUSD mostró **múltiples narrativas válidas**, pequeñas, estables y ortogonales, que describen:
- compresiones reales de balance
- transiciones con aceptación
- pullbacks ordenados en tendencia

### Conclusión
XAUUSD es el **benchmark narrativo** del sistema:
- Rico
- Estructurado
- Con jerarquía clara entre estados

---

## US500 — Índice S&P 500
**Phase D: RICA**

### Características estructurales
- Mercado direccional
- TREND dominante
- Transiciones breves
- Balance menos relevante

### Narrativas Phase D
Narrativas sólidas en:
- pullbacks ordenados en TREND
- resoluciones limpias de TRANSITION
- compresiones previas a expansión

### Conclusión
US500 responde muy bien al enfoque narrativo:
- Phase D aporta contexto real
- No hay colisiones
- Buena base para Phase E

---

## EURUSD — Euro FX
**Phase D: DELGADA**

### Características estructurales
- FX core altamente eficiente
- Mucho comportamiento promedio
- Pocas rarezas reales

### Narrativas Phase D
- Algunas narrativas pequeñas sobreviven
- Ninguna domina
- La mayoría de comportamientos son STATE-level

### Conclusión
EURUSD tiene **poca personalidad narrativa**:
- Phase D existe, pero es delgada
- El edge (si existe) deberá venir más abajo (Phase E)

---

## USDCLP — Peso chileno
**Phase D: NO**

### TF / Window
- H1 / 12 (único símbolo con este setup)

### Características estructurales
- FX emergente
- Alta intervención indirecta
- Microestructura difusa
- TRANSITION colapsa semánticamente

### Resultados
- Todas las narrativas exploradas:
  - colapsaron
  - redefinieron el estado
  - o tuvieron colisión extrema

### Conclusión
USDCLP **no admite Phase D**:
- No hay sub-regímenes distinguibles
- El símbolo debe pasar a Phase E con:
  - STATE puro
  - Quality
  - Contexto macro

Este resultado es **estructural**, no un fallo metodológico.

---

## USDMXN — Peso mexicano
**Phase D: MINIMA**

### Características estructurales
- FX emergente con liquidez discontinua
- Shocks frecuentes
- TREND aparece por eventos

### Narrativas Phase D válidas
- `transition_early_escape`
- `balance_leaking`

Ambas:
- pequeñas
- estables
- sin colisión
- no redefinen estado

### Documentado (no Phase D)
- TREND suele ser shock-continuation (STATE-level)

### Conclusión
USDMXN tiene **Phase D mínima pero real**:
- Útil para contextualizar
- Sin forzar narrativa donde no existe

---

## GBPUSD — Libra FX
**Phase D: SELECTIVA**

### Características estructurales
- FX core menos eficiente que EURUSD
- Londres dominante
- Falsas rupturas frecuentes

### Narrativas Phase D válidas
- `transition_london_fake`
- `transition_mean_revert`
- `trend_delayed_acceptance` (secundaria)

Narrativas descartadas:
- acceptance temprana dominante
- expansión de balance (redefinía estado)

### Conclusión
GBPUSD presenta **pocas rarezas reales**, pero claras:
- London fake break es estructural
- Phase D aporta valor selectivo

---

## XAGUSD — Plata
**Phase D: CANONICA + AMPLIADA**

### Características estructurales
- Metal híbrido (macro + especulación)
- Menor profundidad que XAU
- Alta fragilidad estructural

### Narrativas Phase D canónicas
- `balance_unstable_compression`
  - balance existe pero no sostiene
  - explica por qué el balance en plata es difícil de operar

### Narrativas Phase D ampliadas (exploratorias)
- `transition_liquidity_snapback`
  - expansión violenta
  - falta de continuación
  - retorno rápido a referencia

### Mapa narrativo emergente
XAGUSD
│
├─ TRANSITION
│ ├─ Momentum burst (dominante, STATE-level)
│ └─ Liquidity snapback (frecuente, sub-regimen)
│
├─ TREND
│ └─ Failed continuation (STATE-level)
│
└─ BALANCE
└─ Unstable compression

markdown
Copiar código

### Conclusión
XAGUSD **no es XAU chico**:
- Es un mercado de vacíos de liquidez y retorno
- Phase D aporta **insight estructural profundo**
- Narrativas son clave para orientar Phase E

---

## AUDUSD — Dólar australiano
**Phase D: MINIMA**

### Características estructurales
- FX core–commodity
- Asia domina
- Comportamiento educado
- Poco explosivo

### Narrativas exploradas
- Asia drift sin aceptación (dominante, STATE-level)
- Balance funcional (STATE-level)
- Trend corto (STATE-level)

### Narrativa Phase D válida
- `transition_soft_revert`
  - mean-reversion persistente
  - rara
  - estable
  - sin colisión

### Conclusión
AUDUSD es el símbolo más “educado” del set:
- Pocas rarezas
- Phase D mínima
- El edge probable vive en mean-reversion fino (Phase E)

---

## Cierre de Phase D

Phase D queda **formalmente cerrada** con:

- Resultados no simétricos
- Decisiones honestas
- Narrativas solo donde existen
- Documentación explícita de NO-resultados

Esto deja al sistema en una posición sólida para **Phase E**, donde:
- No se fuerza edge
- Se parte de una comprensión estructural real por símbolo
