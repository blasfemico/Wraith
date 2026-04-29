<div align="center">

<img src="assets/wraith-ghost-mark.png" alt="Wraith · Ghost Mark · WR-01 · Haunt Protocol" width="320" />

# Wraith


*`SUBSTRATE · VESSEL · SIGNAL`*

** LLM multipropósito entrenado desde scratch con cuantización de escalas de 9 niveles al límite de Shannon.**

<img src="https://img.shields.io/badge/clase-NPQN-FF003C?style=for-the-badge&labelColor=0A0A0A" />
<img src="https://img.shields.io/badge/pesos-Dualwire%209--level-39FF14?style=for-the-badge&labelColor=0B3D0B" />
<img src="https://img.shields.io/badge/params-186M-FF003C?style=for-the-badge&labelColor=0A0A0A" />
<img src="https://img.shields.io/badge/packed-74.9%20MB-39FF14?style=for-the-badge&labelColor=0B3D0B" />
<img src="https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-0A0A0A?style=for-the-badge&labelColor=FF003C" />

</div>

---

## ⚠️ Nota de estado (v1.1 — abril 2026)

El **5.73× de ventaja sobre fp16** reportado en este repo se obtuvo con una configuración cuyos **hiperparámetros y algunas elecciones arquitectónicas no estaban tuneados a las mejores prácticas modernas de LLMs** (warmup deshabilitado, weight decay 0.01, setup de capas que reflejaba iteraciones anteriores del codebase en vez de los defaults actuales de Pythia/LLaMA). Con un baseline fp16 tuneado a estándares modernos el margen esperable es menor.

**Por eso estamos ejecutando nuevas validaciones antes del arxiv final:**

1. **Baseline Pythia-style multi-seed** (n=5): arquitectura + hiperparámetros Pythia-160M exactos (LayerNorm + GELU + partial RoPE + warmup 1% + weight decay 0.1 + cosine schedule + grad clip 1.0) al mismo presupuesto de 1.6B tokens. Reportaremos `mean ± std` de val PPL.
2. **Wraith v2 modernizado**: mismo core NPQN pero emparejado con opciones modernas de best-practice — RMSNorm, SwiGLU, full RoPE, Grouped Query Attention (16:4), z-loss de Gemma2, hiperparámetros Pythia-matched. Comparación apples-to-apples más limpia.

El nuevo número probablemente será menor a 5.73× pero seguirá siendo positivo, y más importante, **defendible bajo baselines modernos**.

Los **números puramente medidos** no dependen del baseline comparativo y se mantienen:

- **74.9 MB** checkpoint empaquetado (98.2% del límite de Shannon, bit-exacto)
- **501 tok/s** decode en RTX 5070 @ **114 MB VRAM** @ **64 mJ/token**
- **52.1 tok/s** CPU inference (AMD Ryzen 7 5700G, AVX2)
- **Kernels CUDA propios 2.38–2.59×** sobre cuBLAS fp16
- **Integer-only training converge a 186M** (hecho empírico independiente del baseline)

La contribución conceptual (NPQN taxonomy, shadow int16 optimizer, DSSC failure mode + ASR fix, Dualwire 9-level) **no depende del headline comparativo**. El próximo update del README y paper reportará los nuevos baselines con transparencia total.

---

## ¿Qué es Wraith?

Wraith es la primera instancia concreta de una clase arquitectónica nueva: **Native Pure Quantized Network (NPQN)**. Un LLM que cumple **simultáneamente** tres propiedades que ningún trabajo previo combina a escala LLM y desde scratch:

- **Native** — entrenado con pesos cuantizados desde inicialización aleatoria (no post-hoc sobre un modelo fp16 preentrenado).
- **Pure** — el pipeline de pesos opera **enteramente en aritmética de punto fijo**. No hay bf16 masters, no hay optimizer Adam fp32. El optimizer es un **shadow int16 persistente** con redondeo estocástico.
- **Quantized** — pesos en **9 niveles discretos (Dualwire)**: `W = sc·wa + sf·wb`, con `sc` derivado determinísticamente de `mean(|a|)/127` y `sf = sc/3`. **3.17 bits/peso**, óptimo de Shannon para dos canales ternarios.

Todos los ingredientes individuales tienen prior art (WAGE 2018, NITI 2020, BitNet b1.58 2024, TRQ 2021). **La intersección de las tres propiedades a escala LLM, desde scratch, no la reclama ningún paper previo.**

---

## Resultados (medidos, no proyectados)

**Wraith-186M** entrenado en 1.6B tokens de SlimPajama (régimen sub-Chinchilla al 44% del óptimo):

| Métrica | Wraith-186M | LLaMA-186M fp16 (baseline idéntica) | Ventaja |
|---|---|---|---|
| Val PPL WikiText-103 (val split, durante training) | **107** | 614 | **5.73×** |
| WikiText-103 test PPL (post-hoc) | **223** | 636 | **2.86×** |
| SlimPajama held-out PPL | **83.34** | 185.84 | **2.23×** |
| SlimPajama train-chunk PPL (in-distribution, chunk_00000) | **74.46** | 170.85 | **2.29×** |
| Gap train/val (SlimPajama / WikiText-103) | **1.37×** | 3.59× | **2.62× menor** |
| Throughput decode (B=1) | **501 tok/s** | 388 tok/s | **+29%** |
| VRAM inferencia | **114 MB** | 1,031 MB | **−88.9%** |
| Energía por token | **64 mJ** | 84 mJ | **−24%** |
| Storage on-disk | **74.9 MB** | 372 MB | **4.97×** |

Hardware de inferencia: **NVIDIA RTX 5070 12 GB** (Blackwell sm_120), **AMD Ryzen 7 5700G** (CPU AVX2). Entrenamiento: **H100 80GB** vía Google Colab Pro.

El checkpoint packed se empaqueta a **74.9 MB** (5-trit/byte, 98.2% del límite de Shannon, lossless round-trip verificado).

### Zero-shot (5-shot LAMBADA, Winogrande, ARC-Easy)

Wraith-186M mantiene ventaja consistente sobre el baseline fp16 de arquitectura idéntica en las tres pruebas de zero-shot evaluadas. Detalles completos en el paper (Tabla 4).

---

## Contribuciones genuinamente originales

1. **Primer LLM multipropósito entrenado desde scratch con pipeline 100% integer** (cierra simultáneamente Native + Pure + Quantized).
2. **Dualwire 9-level con escalas deterministicamente derivadas** — sin parámetro α learnable (TRQ), sin learnable shift (TernaryLLM-DLT).
3. **Shadow int16 persistente** como optimizer state — distinto del int16 accumulator transient del matmul en NITI/Ghaffari; el nuestro vive entre training steps.
4. **Identificación y corrección del DSSC** (Derived-Scale Saturation Coupling) — failure mode específico al paradigma multi-channel + derived-scales, corregido vía **ASR (Adaptive Saturation Relief)**.

---

## Estructura del repo

```
Wraith/
├── paper/
│   ├── wraith_paper_es.md       ← paper completo en español (fuente de verdad)
│   ├── wraith_paper_en.md       ← versión inglesa (condensada)
│   └── charts/                  ← 21 figuras del paper (PNG)
├── funding/
│   └── pitch_deck.md            ← propuesta de financiamiento (Wraith 2B, $3,000 compute)
├── checkpoint/
│   └── wraith-186m-npqn-packed.pt   ← checkpoint público (74.9 MB, Git LFS)
├── LICENSE                      ← CC-BY-NC-SA 4.0
├── NOTICE                       ← alcance de la IP y lo que NO está incluido
└── README.md
```

---

## Qué **sí** está disponible en este repo

- ✅ **Paper completo** (ES + EN) con especificación arquitectónica, marco teórico PAC-Bayes y resultados reproducibles.
- ✅ **Checkpoint packed 186M** (74.9 MB) — suficiente para reproducir PPL, zero-shot, throughput y consumo energético reportados.
- ✅ **21 figuras** con todos los plots del paper.
- ✅ **Pitch deck** con la propuesta financiera para escalar a Wraith-2B × 100B tokens.

## Qué **no** está incluido (propiedad intelectual reservada)

- ❌ Motor de inferencia GPU (`WraithFastEngine` — kernels CUDA empaquetados, fused QKV/GateUp, embedding lookup packed, CUDA Graphs).
- ❌ Motor de inferencia CPU C++ (AVX2 + KV cache + activation quantization).
- ❌ Pipeline de entrenamiento NPQN (shadow int16 optimizer + SR + DSSC/ASR).
- ❌ Formato Dualwire 5-trit/byte codec.

Estos artefactos están disponibles bajo licencia para colaboraciones académicas o acuerdos comerciales. Ver sección **Contacto**.

---

## Cómo usar el checkpoint

El archivo `checkpoint/wraith-186m-npqn-packed.pt` está en formato **Dualwire 5-trit/byte packed**. Para reproducir los resultados necesitás una implementación fiel de la arquitectura Wraith (Sección 2 del paper). El paper especifica:

- Dimensiones exactas: 8 capas transformer, Peri-LN, dimensión oculta y heads documentados en Tabla 1.
- Tokenizer: GPT-2 BPE (vocab 50,257).
- Decodificación Dualwire: `W = sc·wa + sf·wb` con `sf = sc/3`, `wa ∈ {−1, 0, +1}`, `wb ∈ {−1, 0, +1}`, 9 niveles resultantes.
- Formato packed: 5 trits por byte (base-3 encoding, 98.2% del límite de Shannon).

Re-implementaciones independientes son bienvenidas — el paper está escrito para permitirlas.

---

## Roadmap

- **Wraith-186M (actual)** — publicado, validado, checkpoint abierto.
- **Wraith-2B** — **próximo objetivo**, requiere ~$3,000 en compute (100B tokens en H100). Busco co-financiamiento o sponsors académicos. Ver `funding/pitch_deck.md`.
- **Wraith-20B / 70B / 100B** — escalamiento a escala de producción, colaboración académica o industrial requerida.

---

## Cómo citar

```bibtex
@misc{villena2026wraith,
  author = {Villena, Dante},
  title  = {Wraith: A Native Pure Quantized Network (NPQN) with Shannon-Optimal 9-Level Dualwire Quantization},
  year   = {2026},
  note   = {Independent research, San Juan, Argentina},
  url    = {https://github.com/blasfemico/Wraith}
}
```

---

## Licencia

- **Paper, figuras y documentación**: [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — uso académico y no-comercial permitido con atribución y share-alike.
- **Checkpoint `wraith-186m-npqn-packed.pt`**: misma licencia (CC-BY-NC-SA 4.0). Usos comerciales derivados requieren acuerdo separado.
- **Código fuente (training + inferencia)**: **no incluido** en este repo. Reservado como propiedad intelectual del autor.

---

## Contacto

- 📧 **programmingblas@gmail.com**
- 💼 [LinkedIn — Dante Villena](https://www.linkedin.com/in/dante-villena/)
- 🐙 [GitHub — @blasfemico](https://github.com/blasfemico)

Interesado en:
- 🧑‍🤝‍🧑 **Co-founders / co-researchers** para seguir desarrollando Wraith.
- 💼 **Sponsors / inversores** que ayuden a la validación a escala 2B.
- 🎓 **Colaboración académica** para experimentos de escalamiento (20B / 100B).

---

<div align="center">

> _"The net is vast and infinite."_
> **— Motoko Kusanagi · Ghost in the Shell**

</div>
