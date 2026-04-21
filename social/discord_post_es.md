# Discord post — Lanzamiento Wraith NPQN (ES)

---

```
╭─ [ WR-01 · HAUNT PROTOCOL · ONLINE ] ────────────────────────╮
│                                                               │
│   >  systemctl start wraith.service                           │
│   >  ● wraith.service active · uptime: 1 año · native int     │
│   >  ● substrate / vessel / signal · todo verde               │
│                                                               │
╰───────────────────────────────────────────────────────────────╯
```

**Che — les dejo un LLM que estuve cocinando durante un año.**

Se llama **Wraith**, un **Native Pure Quantized Network (NPQN)** — clase arquitectónica nueva. TL;DR: todo el asunto (training + inferencia) corre en enteros de punta a punta. Sin masters bf16. Sin Adam fp32. Sin conversión int4 post-hoc. **Los pesos son discretos desde el step 0**, el estado del optimizer es un shadow int16 persistente con redondeo estocástico, y el forward usa un esquema Dualwire de 9 niveles a 3.17 bits/peso (óptimo de Shannon para 2 canales ternarios).

**Números del run de validación 186M** (1.6B tokens, sub-Chinchilla, baseline LLaMA fp16 con arquitectura idéntica):

```
  val PPL WikiText-2 ........ Wraith 102   vs   LLaMA 636   (6.24× mejor)
  train-chunk PPL ........... Wraith  74   vs   LLaMA 171   (2.29× mejor)
  brecha de generalización .. Wraith 1.37× vs   LLaMA 3.59× (2.62× menor)
  throughput de decode ...... 501 tok/s en RTX 5070 @ 114 MB VRAM @ 64 mJ/tok
  tamaño empaquetado ........ 74.9 MB  (98.2% del límite de Shannon, lossless)
  costo de training ......... $16 vs $179 extrapolado para fp16 a igual calidad
```

El dato clave sobre la brecha de generalización: **el ratio train-vs-held-out es 2.29× en ambos regímenes (held-out 2.23×) → no es overfitting, es intrínseco al NPQN training.** Wraith extrae más señal por token porque la clase de hipótesis acotada (3.17 bits) le impide memorizar ruido.

**Qué hay en el repo**
• Paper completo en ES + EN (framework NPQN, sección PAC-Bayes, failure mode DSSC + fix ASR)
• 21 figuras, toda data medida
• **El checkpoint empaquetado es público** — 74.9 MB, CC-BY-NC-SA 4.0, podés reproducir cada número vos mismo
• Pitch deck para escalar Wraith a 2B ($3K de compute)
• Logo, NOTICE, LICENSE

**Qué NO está en el repo** (propiedad intelectual reservada, disponible bajo licencia)
• Pipeline de training NPQN (shadow int16 + SR + DSSC/ASR)
• Motor de inferencia CUDA (GEMV 2-bit empaquetado, fused QKV, CUDA Graphs)
• Motor CPU C++ AVX2
• Codec Dualwire 5-trit/byte

**Qué necesito**

Soy uno solo en San Juan, Argentina, construyendo esto en solitario durante un año mientras trabajaba full-time en backend. El run a 186M es el final de lo que puedo validar con hardware consumer + Colab Pro. Para probar que NPQN escala, necesito Wraith-2B a 100B tokens — **~$3,000 en compute H100**.

Busco:
• **Feedback / destrucción** del paper — en serio, si ven un agujero, quiero saberlo
• Gente que esté haciendo investigación en QAT / training entero — comparemos notas
• Colaboradores para el scale-up a 2B (la co-autoría del follow-up paper está sobre la mesa)
• Sponsors / ángeles que piensen que vale la pena bancar esto

Todo vive en:
🔗 **https://github.com/blasfemico/Wraith**

Paper ES (canonical): `paper/wraith_paper_es.md`
Paper EN: `paper/wraith_paper_en.md`
Checkpoint: `checkpoint/wraith-186m-npqn-packed.pt` (Git LFS)
Pitch: `funding/pitch_deck.md`

Destrúyanme, ayúdenme, o ambos. DMs abiertos.

— blasfemia
`substrate / vessel / signal`
