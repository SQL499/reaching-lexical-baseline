# PROTOCOLO EXPERIMENTAL
| Campo | Valor |
|---|---|
| Versión | 1.2 — **CONGELADO** (enmiendas pre-ejecución registradas en §13) |
| Fecha de congelamiento | 2026-07-18 |
| Autores | J. F. Céspedes Ticona, J. G. Melgar Puertas |
| Estado | Vigente. Toda desviación se registra en §13 ANTES de conocer resultados de evaluación |
| Rol del documento | Especificación de implementación (para el agente que programa) y fuente de la sección metodológica del informe de TP2 |

---

## 0. Regla de congelamiento

Este protocolo fija **antes de ejecutar cualquier experimento** todas las decisiones que podrían ajustarse oportunistamente al ver resultados: métricas, valores de K, baseline, preprocesamiento, hiperparámetros candidatos, criterio GO/NO-GO y análisis estadístico. Reglas:

1. El conjunto de evaluación (`ETIQUETADO_FINAL_VALIDATION_v2.csv`) **no se utiliza para ninguna decisión de desarrollo**: ni selección de modelo, ni ajuste de hiperparámetros, ni depuración de preprocesamiento. Se consulta únicamente en los experimentos E1, E2 y E4 definidos en §8, y sus resultados se registran de una sola vez por experimento.
2. Cualquier cambio posterior al congelamiento se anota en el registro de desviaciones (§13) con fecha, motivo y evidencia de que la decisión **no** se tomó mirando resultados de evaluación. Corregir un *bug* es legítimo y obliga a re-ejecutar todos los experimentos afectados.
3. Está prohibido agregar métricas, valores de K o baselines después de conocer resultados ("*metric shopping*").

---

## 1. Pregunta experimental e hipótesis

**Pregunta:** ¿El emparejamiento semántico basado en *embeddings* supera al emparejamiento por coincidencia léxica en la calidad del ranking de ofertas para perfiles laborales del contexto peruano (datos abiertos del MTPE)?

- **H1 (principal):** el modelo semántico *zero-shot* (`paraphrase-multilingual-MiniLM-L12-v2`, sin ajuste) supera a BM25 en NDCG@3, MRR y P@1 sobre los 32 perfiles utilizables de evaluación.
- **H2 (condicional):** el ajuste fino (*fine-tuning*) del modelo con los pares de entrenamiento mejora al *zero-shot*, medido por validación cruzada sobre el conjunto de entrenamiento (§9). Si H2 no se sostiene en CV, el modelo final es el *zero-shot* y el resultado negativo se reporta como hallazgo.
- **H3 (eficiencia, RNF-02):** el modelo final responde una consulta de emparejamiento contra el corpus completo de vacantes en un tiempo compatible con uso interactivo (medición en §10; no se fija umbral numérico a priori, en concordancia con el TI, que difiere las metas numéricas a la implementación; se reporta mediana y p95).

Correspondencia con el TI/ACP: OE4 e hito OE4-I1 (informe de validación con NDCG@K, Precision@K y MRR frente a método base de coincidencia léxica). El experimento E2 cumple además, sin costo adicional, la cláusula histórica de comparación "frente a un modelo de embeddings de referencia": el *zero-shot* ES ese modelo de referencia.

---

## 2. Datos: manifiesto canónico

### 2.1 Archivos y huellas SHA-256 (verificar antes de cada corrida)

| Archivo | SHA-256 |
|---|---|
| `ETIQUETADO_FINAL_TRAIN_v2.csv` | `476653352b2227f138fb42c01aa040b1da5e9607e12dd6b4a3107b2ab036fb9d` |
| `ETIQUETADO_FINAL_VALIDATION_v2.csv` | `a0f890eafda0e293c8539f09c02375b2346898f05bf967a25e54c8d3e2a57a2c` |
| `ofertas_consolidadas.csv` | `22e495ffc0562a36f9b27dae0bb3ab81365680c9ef1af3eead5bb8352b333e37` |
| `perfiles_consolidados.csv` | `c95322ca4416389dddc2e59c020f807cf0ea35d869ff28856d5de0b57e5364ee` |
| `JUSTIFICACIONES_FINAL_TRAIN_v2.csv` | `e96c84c755624ec673e70d5708ff3204e84a9af06800cdcca8054dc3dce6b6f8` |
| `JUSTIFICACIONES_FINAL_VALIDATION_v2.csv` | `e55871d062526fa8ef037c5406ecea5211c3be39246fd982dfe02c1ae8cfe1b2` |

Codificación de lectura: `utf-8-sig` (los archivos llevan BOM).

### 2.2 Aserciones de integridad (el código DEBE verificar y abortar si fallan)

- TRAIN: 800 filas; 160 `ID_POSTULANTE` únicos; exactamente 5 candidatas por perfil; 272 pares con `ETIQUETA=1`; 643 `AVISOID` únicos; 109 perfiles con ≥1 positivo.
- VALIDATION: 288 filas; 48 perfiles; exactamente 6 candidatas por perfil; 48 positivos; 285 `AVISOID` únicos; 32 perfiles utilizables; 16 con `PERFIL_SIN_RELEVANTE=1`.
- Intersección de `ID_POSTULANTE` entre particiones: **0** (perfiles disjuntos).
- `PERFIL_SIN_RELEVANTE` consistente con las etiquetas (0 inconsistencias).
- Sin pares (`ID_POSTULANTE`,`AVISOID`) duplicados.
- Todo `AVISOID` de los pares existe en `ofertas_consolidadas.csv`; todo `ID_POSTULANTE` existe en `perfiles_consolidados.csv`.

### 2.3 Textos canónicos (decisión ratificada)

- **Lado perfil:** `TEXTO_PERFIL` tomado de los **archivos de pares** (`ETIQUETADO_FINAL_*.csv`). Justificación: coherencia texto–etiqueta; es el texto que el proceso de etiquetado observó. Se documenta que 17 perfiles (13 train, 4 validation) tienen una versión más extensa en `perfiles_consolidados.csv` (el texto de pares es prefijo del consolidado; truncamiento de +58 a +952 caracteres ocurrido en la generación de pares).
- **Lado oferta:** `TEXTO_OFERTA` tomado de `ofertas_consolidadas.csv` mediante *join* por `AVISOID` (es el campo documentado como entrada al modelo en el diccionario de variables; las columnas de oferta presentes en los archivos de pares son informativas). En E0 se verifica por muestreo (n=20, semilla 42) que `TEXTO_OFERTA` corresponde a la concatenación título + sector + ocupación + competencias.
- La columna `TIPO_CANDIDATA` **no** se usa como entrada de ningún modelo (es metadato del muestreo).
- **Análisis de robustez R1 (obligatorio, en E4):** recomputar las métricas del modelo final usando el texto consolidado completo para los perfiles de evaluación utilizables afectados por truncamiento en origen (4 identificados sobre v1); reportar ambas versiones. Si difieren de forma material, discutirlo en el informe.

### 2.4 Exclusiones

Los 16 perfiles de evaluación con `PERFIL_SIN_RELEVANTE=1` se **excluyen** del cálculo de métricas de ranking (carecen de ítem relevante que ordenar), conforme a lo declarado en el TI §1.4.5. Las métricas se calculan sobre los **32 perfiles utilizables**.

### 2.5 Fuga conocida y documentada

20 vacantes aparecen en ambas particiones (7.0% de las vacantes de evaluación; afectan 22 pares de evaluación, 9 positivos). Los perfiles —las consultas del problema de ranking— son 100% disjuntos, que es la condición obligatoria; compartir documentos entre particiones es práctica aceptada en recuperación de información. Impacto: nulo para E1/E2 (sin entrenamiento); potencial y menor para E3/E4. **Análisis de sensibilidad S1 (obligatorio, en E4):** identificar los perfiles de evaluación cuyo conjunto de candidatas incluye ≥1 vacante compartida, recomputar las métricas de cabecera sobre el subconjunto complementario y reportar ambas cifras.

---

## 3. Estructura del problema de evaluación

Cada perfil utilizable de evaluación define una tarea de ranking: ordenar sus **6 vacantes candidatas** de mayor a menor afinidad. Distribución de relevantes por perfil utilizable: 20 perfiles con 1 positivo, 8 con 2 y 4 con 3 (media 1.50). En entrenamiento la tarea análoga tiene 5 candidatas por perfil.

Consecuencias de diseño (fijadas):

- Valores de K acotados por la longitud de lista (6). K=10 es imposible.
- El rango dinámico de varias métricas está comprimido (ver anclajes de azar, §5.3); toda tabla de resultados incluye la fila "ranking aleatorio".
- Precision@3 tiene techo estructural **0.500** (20 perfiles solo pueden alcanzar 1/3); se reporta como secundaria con el techo indicado al pie de tabla.

---

## 4. Preprocesamiento (dos tuberías, deliberadamente distintas)

### 4.1 Tubería léxica (solo para BM25)

Aplicada a `TEXTO_PERFIL` (consulta) y `TEXTO_OFERTA` (documentos):

1. Pasar a minúsculas.
2. Plegado de acentos/diacríticos (NFD → eliminar marcas combinantes; "formación" → "formacion"). Motivo: los textos mezclan grafías con y sin tilde; sin plegado, BM25 se penaliza artificialmente.
3. Tokenización por expresión regular `[a-z0-9ñ]+` (tras plegado; se preserva ñ).
4. Eliminación de *stopwords* del español: lista **embebida en el repositorio** (copiar la lista de NLTK `stopwords/spanish` a `src/resources/stopwords_es.txt` y versionarla; no depender de descargas en tiempo de ejecución).
5. *Stemming* con `SnowballStemmer("spanish")` (NLTK). Motivo: baseline léxico competitivo en un idioma morfológicamente rico; se evita el hombre de paja.

### 4.2 Tubería semántica (para el modelo de embeddings)

**Mínima por diseño:** los Transformers aprovechan el contexto completo; eliminar *stopwords* o aplicar *stemming* los degrada.

1. Normalización de espacios en blanco (colapsar múltiples espacios, `strip`).
2. Nada más. Se conservan mayúsculas, tildes y la estructura "Formacion: … Experiencia: …" / "TÍTULO. Sector: … Ocupacion: … Competencias: …", que aporta señal.
3. `model.max_seq_length = 256`. Justificación: p90 de perfiles = 81 palabras y p90 de ofertas usadas = 105 palabras; el límite por defecto (128 tokens) truncaría ~9–14% de los textos, y en las ofertas lo truncado serían las competencias (lo semánticamente rico). Con 256 se cubre ≈p99; el truncamiento residual (~1%) se declara como limitación.

---

## 5. Métricas, agregación y anclajes

### 5.1 Definiciones (relevancia binaria, `ETIQUETA`)

Para un perfil con lista rankeada de candidatas y vector de relevancias `rel[i] ∈ {0,1}` en el orden del ranking:

- **Precision@K** = (Σ_{i=1..K} rel[i]) / K, con K ∈ {1, 3}.
- **DCG@K** = Σ_{i=1..K} rel[i] / log2(i+1); **IDCG@K** = DCG@K del orden ideal; **NDCG@K** = DCG@K / IDCG@K, con K ∈ {3, 5}.
- **RR** = 1 / (posición del primer relevante); **MRR** = promedio de RR sobre perfiles.

Parrilla congelada: **primarias** NDCG@3, MRR, P@1; **secundarias** NDCG@5, P@3 (con techo 0.500 documentado). No se agregan otras.

### 5.2 Reglas operativas

- **Agregación:** macro-promedio sobre los 32 perfiles utilizables (cada perfil pesa igual).
- **Desempate determinista:** orden por (−score, `AVISOID` ascendente). Registrar en el log el número de empates por método (BM25 puede producir score 0 en candidatas sin solapamiento léxico; los empates existen y deben ser visibles).
- **Salidas por experimento:** `metrics.json` (agregados) y `per_profile.csv` (una fila por perfil con las 5 métricas y el ranking completo con scores). El CSV por perfil es insumo del análisis de errores.

### 5.3 Anclajes de interpretación (fijos, ya simulados con semilla 42, 20 000 permutaciones sobre la estructura real de evaluación)

| Métrica | Azar esperado | Techo |
|---|---|---|
| P@1 | 0.250 | 1.000 |
| P@3 | 0.250 | 0.500 |
| NDCG@3 | 0.393 | 1.000 |
| NDCG@5 | 0.544 | 1.000 |
| MRR | 0.489 | 1.000 |

Toda tabla de resultados del informe incluye la fila "Aleatorio (esperado)". Un MRR "alto" sin este anclaje induce a error: con listas de 6 y ~1.5 relevantes, el azar ya obtiene MRR ≈ 0.49.

### 5.4 Vectores de prueba unitaria (el harness DEBE pasarlos antes de E1; tolerancia ±0.0005)

`rel` en orden de ranking, 6 candidatas:

| Caso | rel | P@1 | P@3 | NDCG@3 | NDCG@5 | RR |
|---|---|---|---|---|---|---|
| A | [0,1,0,0,1,0] | 0.0 | 0.3333 | 0.3869 | 0.6240 | 0.5 |
| B | [1,0,0,0,0,0] | 1.0 | 0.3333 | 1.0000 | 1.0000 | 1.0 |
| C | [0,0,0,1,0,0] | 0.0 | 0.0000 | 0.0000 | 0.4307 | 0.25 |

Agregación macro de A+B+C: P@1 = 0.3333; P@3 = 0.2222; NDCG@3 = 0.4623; NDCG@5 = 0.6849; MRR = 0.5833. Además: prueba de error si un perfil llega sin relevantes (debe lanzar excepción, no devolver 0 silencioso) y prueba del desempate determinista.

---

## 6. Baseline léxico: BM25 (congelado)

- Implementación: `rank_bm25.BM25Okapi` con parámetros por defecto **k1 = 1.5, b = 0.75, epsilon = 0.25** (sin ajuste; ajustarlos sobre evaluación sería fuga y sobre train, innecesario para un baseline estándar — se documenta la decisión).
- **Índice/IDF sobre el corpus completo** de 53 233 ofertas (tubería léxica §4.1), no sobre las 6 candidatas del perfil: el IDF estimado con 6 documentos es degenerado. El ranking se restringe a las candidatas del perfil usando `get_batch_scores(query_tokens, indices_de_candidatas)`.
- Consulta = tokens léxicos del `TEXTO_PERFIL`.

---

## 7. Modelo semántico (congelado)

- Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (ganador del benchmarking del IB). Al descargarlo, **registrar el hash de revisión exacto** del repositorio del modelo en `results/env.txt` y cachearlo en `models/` para no depender de la red.
- `max_seq_length = 256` (§4.2). Embeddings con `normalize_embeddings=True`; afinidad = producto punto (equivale a similitud del coseno).
- *Zero-shot* (E2): el modelo tal cual, sin entrenamiento.
- Los embeddings de las 6 candidatas se calculan con el mismo modelo y configuración que el del perfil.

---

## 8. Secuencia de experimentos y compuertas

| ID | Experimento | Compuerta para avanzar |
|---|---|---|
| E0 | Preparación: verificación de hashes y aserciones (§2.2), construcción de textos canónicos, harness de métricas con pruebas §5.4, verificación por muestreo de `TEXTO_OFERTA` | 100% pruebas en verde + **Checkpoint C1 y C2** (§11) |
| E1 | BM25 sobre evaluación (32 perfiles) | Registro de resultados + **Checkpoint C3** |
| E2 | *Zero-shot* sobre evaluación; comparación E2 vs E1 con IC bootstrap y test de permutación (§9.3) | Registro. **Resultado mínimo defendible de la tesis alcanzado aquí** |
| E3 | *Fine-tuning* condicional con CV sobre TRAIN (§9); decisión GO/NO-GO | **Checkpoint C4** (ratificación humana del GO/NO-GO) |
| E4 | Evaluación final: **una sola pasada** del modelo final sobre evaluación; tabla completa (Aleatorio / BM25 / Zero-shot / Fine-tuned si GO) con IC y tests; robustez R1 (§2.3); sensibilidad S1 (§2.5); análisis cualitativo de errores (§9.4) | **Checkpoint C5** |
| E5 | Eficiencia (RNF-02) según §10 | — |

Prohibido reordenar: E1 antes que E2 evita calibrar el baseline conociendo al rival; E4 ocurre una única vez.

---

## 9. Fine-tuning condicional (E3)

### 9.1 Diseño de validación cruzada

- **GroupKFold con 5 particiones agrupando por `ID_POSTULANTE`** sobre los 160 perfiles de TRAIN (32 perfiles por partición). Ningún perfil aparece en entrenamiento y validación del mismo pliegue.
- Métrica de selección: **NDCG@3 macro** sobre los perfiles utilizables del pliegue retenido (ranking de sus 5 candidatas), desempate por MRR.
- Comparador: el *zero-shot* evaluado sobre **los mismos pliegues**.
- El conjunto de evaluación (validation split) **no interviene en E3 en absoluto**.

### 9.2 Rejilla de configuraciones (fijada a priori; no se amplía)

| Config | Pérdida | Datos de entrenamiento | Batch | LR | Épocas | Warmup |
|---|---|---|---|---|---|---|
| A1–A3 | `MultipleNegativesRankingLoss` | Solo pares positivos del pliegue (~219 por pliegue de 4/5); negativos in-batch | 32 | 2e-5 | 1, 2, 3 | 10% |
| B1–B3 | `CosineSimilarityLoss` | Todos los pares del pliegue (~640), etiqueta → 1.0/0.0 | 16 | 2e-5 | 1, 2, 3 | 10% |

Total: 6 configuraciones × 5 pliegues = 30 corridas (minutos en la RTX 4060). Semilla 42 en todas. Optimización adicional de hiperparámetros: **prohibida** (riesgo de sobreajuste de selección con 160 perfiles).

### 9.3 Criterio GO/NO-GO (congelado)

**GO** si la mejor configuración cumple simultáneamente: (a) NDCG@3 medio de CV **estrictamente mayor** que el del *zero-shot* en los mismos pliegues, y (b) mejora en **≥3 de 5 pliegues**. En caso contrario, **NO-GO**: el modelo final es el *zero-shot* y el fine-tuning se reporta como experimento con resultado negativo (hallazgo legítimo, coherente con la cláusula "condicional a la viabilidad de datos" del TI).

Si GO: reentrenar la configuración ganadora sobre los 160 perfiles completos de TRAIN (misma receta de datos de su pérdida, semilla 42) → ese es el modelo final para E4.

### 9.4 Análisis estadístico y de errores (E2 y E4)

- **IC 95% por bootstrap:** 10 000 remuestreos con reemplazo sobre los 33 perfiles (semilla 42), percentiles 2.5/97.5, para cada métrica de cada método **y para la diferencia pareada** (semántico − BM25).
- **Test de permutación pareado** (10 000 permutaciones de signo sobre las diferencias por perfil, semilla 42), bilateral, α = 0.05, para NDCG@3, MRR y P@1. Declarar la potencia limitada con n = 33.
- **Análisis cualitativo:** seleccionar los ≥5 perfiles con mayor |diferencia| de NDCG@3 entre métodos; contrastar textos, ranking y `JUSTIFICACIONES_FINAL_VALIDATION.csv`; documentar patrones (p. ej., sinonimia que BM25 pierde; términos raros que BM25 captura).

---

## 10. Eficiencia (E5, RNF-02)

Escenario realista de la aplicación: embeddings de las 53 233 ofertas **precalculados** (reportar tiempo total y por lote de ese precálculo). Medir, con 50 repeticiones tras 5 de calentamiento, y reportar mediana y p95 de:

1. Codificación de 1 perfil (GPU y CPU).
2. Similitud contra la matriz de 53 233 embeddings + extracción de top-10.
3. Extremo a extremo (1+2).

Registrar hardware exacto (CPU, RAM, GPU RTX 4060, versiones de driver/CUDA). Referencia opcional: latencia de una consulta BM25 sobre el corpus completo.

---

## 11. Puntos de control humano

| # | Momento | Qué se revisa |
|---|---|---|
| C0 | Antes de E0 | **Muestra de etiquetas positivas** (30 positivos aleatorios, semilla 42, estratificados train/validation) contra sus justificaciones. *Recomendado, no bloqueante*: la validez externa del etiquetado ya está cubierta por el juicio firmado de dos expertos; C0 queda como diligencia del autor (detección de errores mecánicos residuales y preparación para la defensa con ejemplos concretos). Cierra el pendiente 9.8 del resumen del proyecto |
| C1 | Fin de E0 | Harness: verificación a mano de los vectores §5.4 |
| C2 | Fin de E0 | 5 pares impresos con sus textos canónicos construidos (perfil + oferta) |
| C3 | Fin de E1 | Números de BM25 y conteo de empates |
| C4 | Fin de E3 | Ratificación del GO/NO-GO con la tabla de CV a la vista |
| C5 | Fin de E4 | Tabla final completa antes de redactar el informe |

**Modo de ejecución adoptado (v1.2):** por decisión del autor, la ejecución E0→E5 es **autónoma y de corrido**; los checkpoints C1–C5 se consolidan en una **revisión final asistida por checklist** (`CHECKLIST_REVISION_FINAL.md`). Las compuertas automáticas permanecen **bloqueantes** para el agente: verificación de hashes (§2.1), aserciones (§2.2) y pruebas del harness (§5.4); ante cualquier fallo, el agente aborta y reporta, sin improvisar correcciones de datos ni relajar aserciones. C0 permanece como diligencia recomendada del autor.

---

## 12. Reproducibilidad y repositorio

### 12.1 Semillas y determinismo

`PYTHONHASHSEED=42`; `random.seed(42)`; `numpy` semilla 42; `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`; `torch.backends.cudnn.deterministic=True`, `benchmark=False`. Nota honesta: el entrenamiento en GPU puede conservar no determinismo residual; la **evaluación** (métricas, ranking, bootstrap, permutaciones) debe ser bit-a-bit reproducible.

### 12.2 Entorno

- Python ≥ 3.10. Dependencias: `sentence-transformers`, `torch` (CUDA), `rank_bm25`, `nltk` (solo recursos copiados al repo), `pandas`, `numpy`, `scikit-learn`, `pyyaml`.
- Congelar `pip freeze > results/env.txt` en la primera corrida; fijar `requirements.txt` con versiones exactas a partir de ahí. Registrar hash de revisión del modelo HF (§7).

### 12.3 Estructura del repositorio

```
emparejamiento-semantico/
├── PROTOCOLO_EXPERIMENTAL.md      # este documento (fuente de verdad)
├── DECISIONES.md                  # registro de desviaciones (§13)
├── requirements.txt
├── config.yaml                    # rutas, semillas, K, hiperparámetros §9.2
├── data/raw/                      # los 6 CSV canónicos (hashes §2.1)
├── models/                        # caché del modelo HF (revisión registrada)
├── src/
│   ├── data.py                    # carga, aserciones §2.2, textos canónicos §2.3
│   ├── preprocess.py              # tuberías §4.1 y §4.2
│   ├── metrics.py                 # §5 (con desempate determinista)
│   ├── bm25_baseline.py           # §6
│   ├── semantic_model.py          # §7 (zero-shot y carga del fine-tuned)
│   ├── finetune.py                # §9 (CV, rejilla, GO/NO-GO)
│   ├── evaluate.py                # E1/E2/E4 (bootstrap, permutación, R1, S1)
│   └── efficiency.py              # §10
├── tests/test_metrics.py          # vectores §5.4 + casos de error
└── results/E1_bm25/ … E5_eficiencia/   # metrics.json, per_profile.csv, config y env por corrida
```

Cada corrida guarda: `metrics.json`, `per_profile.csv`, copia de `config.yaml`, hash de commit de git y marca temporal. Los notebooks, si existen, son exploratorios: **nada canónico vive en un notebook**.

Control de versiones: **git local desde el primer día** (`git init`), con `git config user.name` / `user.email` del autor, de modo que todo commit quede a su nombre. La publicación en un repositorio remoto (GitHub) puede diferirse al final del desarrollo sin afectar la trazabilidad; lo que no es negociable es el versionado local, que actúa como red de seguridad frente a ediciones del agente y sustenta el hash de commit registrado por corrida.

---

## 13. Registro de desviaciones

| Fecha | Sección afectada | Cambio | Motivo | ¿Se conocían resultados de evaluación? |
|---|---|---|---|---|
| 2026-07-18 | §11 (C0), §12.3, §14.1 | C0 pasa de obligatorio a recomendado; se documenta git local con publicación remota diferida; la mitigación de la amenaza 1 se actualiza a "dos expertos con cartas firmadas" | Recepción de las cartas de validación firmadas por dos expertos del dominio; decisión del autor sobre el flujo de repositorio | No (ningún experimento ejecutado) |
| 2026-07-19 | §0, §1, §2.1–§2.5, §3, §5.1–§5.3, §8, §11, §14 | **Dataset v2**: 6 etiquetas corregidas tras adjudicación del autor de las observaciones del juicio de expertos (pares 3, 19, 21, 31, 35a, 35b de la muestra); nuevos hashes; evaluación pasa a 48 positivos y 32 perfiles utilizables; anclajes de azar re-simulados; ejecución E0→E5 declarada autónoma con revisión final por checklist | Adjudicación documentada en `ADJUDICACION_PARES_DISPUTADOS.md` y `CHANGELOG_DATASET_v2.md` | No (ningún experimento ejecutado) |

---

## 14. Amenazas a la validez (declaradas desde el diseño)

1. **Etiquetas construidas** (no provienen de contrataciones reales; el MTPE no publica el vínculo postulante–vacante). Mitigación: criterio ocupacional explícito, justificación por par, auditoría con concordancia 100% sobre lote manual, y **validación mediante juicio de dos expertos del dominio, con cartas firmadas** (anexo de la tesis, con los datos de ambos expertos).
2. **n pequeño en evaluación** (32 perfiles). Mitigación: IC bootstrap, test de permutación pareado, declaración explícita de potencia limitada.
3. **Rango de métricas comprimido** (listas de 6, azar alto). Mitigación: fila de anclaje aleatorio en toda tabla; énfasis en la diferencia pareada, no en valores absolutos.
4. **20 vacantes compartidas entre particiones** (7% de eval). Mitigación: análisis de sensibilidad S1; perfiles disjuntos garantizados.
5. **Truncamiento residual** de textos largos (~1% con `max_seq_length=256`) y 17 perfiles con texto truncado en origen. Mitigación: análisis de robustez R1; documentación.
6. **Dominio único** (datos abiertos MTPE; español peruano). Se declara como alcance, no se generaliza más allá.

---

*Fin del protocolo. Versión 1.2 — congelada el 2026-07-19 (v1.0 del 2026-07-18 + enmiendas §13, todas previas a todo experimento). Cualquier edición posterior pasa por §13.*
