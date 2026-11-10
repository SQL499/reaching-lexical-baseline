# DECISIONES.md — Registro de desviaciones, interpretaciones y bugs

Complemento operativo del §13 del protocolo. Toda entrada se registra ANTES de conocer resultados de evaluación del experimento afectado. Formato: fecha, tipo (interpretación | bug | incidencia), descripción, motivo, experimentos afectados.

---

## 2026-07-20 — Interpretaciones conservadoras registradas ANTES de ejecutar E0 (ningún experimento ejecutado)

| # | Sección | Interpretación adoptada | Motivo |
|---|---|---|---|
| I1 | §9.4 | Bootstrap y test de permutación sobre los **32** perfiles utilizables (el texto dice "33 perfiles" / "n = 33") | Residuo de v1 no actualizado por la enmienda §13 del 2026-07-19: v2 fija 32 utilizables en §2.2, §2.4 y §5.2. Se adopta el valor coherente con el resto del documento congelado. |
| I2 | §8 (fila E2) | La referencia "(§9.3)" para IC bootstrap y test de permutación se lee como **§9.4** | §9.3 es el criterio GO/NO-GO; el contenido estadístico citado está en §9.4. Errata de referencia interna. |
| I3 | §9.4 | p-valor del test de permutación con la convención (1 + #{\|T_perm\| ≥ \|T_obs\|}) / (1 + N) | No especificado en el protocolo. Convención estándar (Phipson & Smyth 2010), ligeramente conservadora; evita p = 0. |
| I4 | §9.2 (rejilla A) | MultipleNegativesRankingLoss con *batching* aleatorio por defecto de la librería (semilla 42), sin sampler especial anti-duplicados | El protocolo solo especifica "negativos in-batch"; se adopta el comportamiento por defecto, sin añadir mecanismos no descritos. |
| I5 | §9.3 | "Mejora en ≥3 de 5 pliegues" = NDCG@3 del candidato **estrictamente mayor** que el del zero-shot en ese pliegue (empate no cuenta como mejora) | Lectura estricta/conservadora del criterio congelado. |
| I6 | §2.3 (R1) | Perfiles afectados por truncamiento detectados programáticamente (texto de pares ≠ texto consolidado, siendo prefijo de este). R1 recomputa las métricas de cabecera sobre los 32 perfiles utilizables sustituyendo el texto SOLO en los afectados; se reportan ambas versiones y los deltas por perfil afectado. Si el conteo hallado difiere de los "4 identificados sobre v1", se reporta el hallado | El protocolo describe los afectados como conteo sobre v1 ("4 identificados sobre v1"); la identificación programática es la única forma reproducible de fijarlos sobre v2. |
| I7 | §2.5 (S1) | Vacantes compartidas = AVISOID presentes a la vez en pares TRAIN y pares VALIDATION. Subconjunto complementario = perfiles utilizables sin ninguna vacante compartida entre sus 6 candidatas. Las cifras documentadas (20 vacantes, 22 pares, 9 positivos) se verifican en E0; una discrepancia se reporta como incidencia, no bloquea (no forma parte de §2.2) | Definición operativa mínima consistente con la redacción de §2.5. |
| I8 | §10 (E5) | El paso (1) se mide en GPU y CPU (explícito). Los pasos (2) similitud+top-10 y (3) extremo a extremo se miden en **ambas** variantes (GPU/torch y CPU/numpy) y se reportan ambas. La referencia BM25 (opcional en el protocolo) se incluye con el mismo esquema 5 calentamiento + 50 repeticiones | El protocolo no fija dispositivo para (2)-(3); reportar ambas variantes evita elegir a posteriori. |
| I9 | §12.3 / §7 | "Cachear el modelo en `models/`" se cumple guardando el snapshot HF en `models/` con su hash de revisión registrado en `results/env.txt` y `models/MODEL_INFO.txt` (versionado). Los binarios de pesos NO se versionan en git (`.gitignore`); los checkpoints intermedios de las 30 corridas de CV se descartan tras evaluarse (se conservan métricas y configuración). Si hay GO, el modelo final sí se conserva en `models/finetuned_final/` | Cachear ≠ versionar binarios (~470 MB por checkpoint; 30 corridas serían ~14 GB). La trazabilidad queda garantizada por el hash de revisión + semillas + receta congelada. |
| I10 | §5.3 | Los anclajes de azar son constantes congeladas y se usan tal cual en toda tabla. En E0 se re-simulan como control de calidad desde la estructura publicada en §3 (20/8/4 positivos, listas de 6, 20 000 permutaciones, semilla 42); una discrepancia material se reportaría como incidencia sin sustituir los valores congelados. Además se verifica como aserción que la distribución real de positivos por perfil utilizable coincide con §3 (si no coincide, se aborta: invalidaría anclajes y techo de P@3) | QA de constantes usadas en el informe; la aserción de distribución deriva de hechos publicados en §3. |
| I11 | §5.2 | Desempate determinista (−score, AVISOID ascendente) con AVISOID como entero (los AVISOID del dataset son numéricos) | Fija el orden total sin ambigüedad de representación. |
| I12 | §12.3 | Los 6 CSV canónicos se mueven de la raíz a `data/raw/` verificando SHA-256 antes y después del movimiento (verificado: idénticos) | Conformidad con la estructura de repositorio del protocolo. |
| I13 | §9.1 | En E3, "perfiles utilizables del pliegue" = perfiles de TRAIN con ≥1 positivo entre sus 5 candidatas (109 en total según §2.2). El zero-shot comparador se evalúa exactamente sobre los mismos pliegues y perfiles. Mejor configuración = mayor NDCG@3 CV medio; desempate por MRR CV medio | Transcripción operativa de §9.1; TRAIN no tiene columna de "sin relevante", se deriva de las etiquetas. |
| I14 | §9.2 | Entrenamiento con la API vigente de sentence-transformers (SentenceTransformerTrainer): optimizador AdamW (default de la librería), lr 2e-5, scheduler lineal con warmup_ratio = 0.10, fp32, semilla 42, batch y épocas según la rejilla. Todo parámetro no fijado por §9.2 se toma del default de la librería y queda congelado en `requirements.txt` + código | §9.2 fija pérdida, datos, batch, LR, épocas y warmup; el resto se delega al default documentado de la versión congelada. |
| I15 | Global | `PYTHONHASHSEED=42`, `PYTHONUTF8=1` y `PYTHONIOENCODING=utf-8` se fijan en el entorno del proceso al lanzar cada script (PYTHONHASHSEED debe existir antes de iniciar el intérprete para ser efectivo) | Cumplimiento real de §12.1 en Windows; UTF-8 evita errores de consola cp1252 con textos en español. |
| I16 | §5.2 / §12.1 | Codificación de textos para evaluación con batch fijo = 64, fp32, en GPU; la afinidad es producto punto sobre embeddings normalizados. El batch queda fijado en `config.yaml` para reproducibilidad bit-a-bit | El protocolo no fija batch de inferencia; se congela un valor único antes de ejecutar. |
| I17 | §9.4 (E4) | El análisis cualitativo y el test de permutación de E4 comparan **modelo final** vs BM25 ("semántico − BM25"). Si hay GO, la tabla incluye además la columna zero-shot (con sus IC) como referencia, sin tests adicionales no previstos | Evita añadir contrastes no congelados ("metric shopping"). |
| I18 | §2.3 (E0) | La verificación por muestreo de TEXTO_OFERTA (n=20, semilla 42) se muestrea sobre las 53 233 filas de `ofertas_consolidadas.csv` con `numpy.random.default_rng(42)` y compara contra el patrón observado del dataset: `"{NOMBREAVISO}. Sector: {SECTOR}. Ocupacion: {ESCO}. Competencias: {COMPETENCIAS}"` | La propiedad declarada ("título + sector + ocupación + competencias") es del archivo consolidado; el patrón exacto se fija antes de evaluar cualquier modelo. |

---

## Bugs e incidencias

### B1 — 2026-07-20 — Bug en el patrón de verificación de TEXTO_OFERTA (E0)

- **Síntoma:** la verificación por muestreo §2.3 (n=20, semilla 42) abortó E0 con 4/20 mismatches.
- **Causa:** el patrón esperado de I18 asumía las 4 secciones siempre presentes
  (`"{título}. Sector: …. Ocupacion: …. Competencias: …"`). Inspección de los datos:
  el consolidado **omite la sección completa cuando el campo está vacío o son solo
  espacios** (ESCO en blanco en 7 916/53 233 ofertas, 14.9%; NOMBREAVISO en blanco en 4).
  Bug del código de verificación, no de los datos: las etiquetas y textos canónicos no cambian.
- **Corrección:** `expected_oferta_text()` en `scripts/e0_prep.py` concatena con `". "`
  únicamente las secciones no vacías (título / `Sector: …` / `Ocupacion: …` / `Competencias: …`).
  Se actualiza así la definición operativa de I18. La comparación sigue siendo exacta.
- **Experimentos afectados:** solo E0 (ningún otro había corrido). Se re-ejecuta E0 completo.
- **¿Se conocían resultados de evaluación?** No: el fallo ocurrió antes de evaluar ningún método;
  solo se inspeccionaron campos de `ofertas_consolidadas.csv` (documentos, no etiquetas).
