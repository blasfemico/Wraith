# LinkedIn post — Lanzamiento Wraith NPQN (ES)

---

🚨 **Después de un año de investigación, Presento a Wraith — el primer Native Pure Quantized Network (NPQN).**

Una clase arquitectónica nueva de LLM: entrenado desde cero con un pipeline 100% entero — sin masters bf16, sin estados de Adam fp32, sin cuantización post-hoc. Cada etapa del entrenamiento y de la inferencia opera sobre pesos discretos.

**Qué lo hace distinto**

Tres propiedades que ningún LLM previo combina simultáneamente a escala:

• **Native** — pesos cuantizados desde la inicialización aleatoria, no parchados sobre un checkpoint fp16
• **Pure** — todo el pipeline de pesos es aritmética entera de punto fijo, incluyendo el optimizador
• **Quantized** — esquema Dualwire de 9 niveles a 3.17 bits/peso (óptimo de Shannon para dos canales ternarios)

El estado del arte tiene los ingredientes por separado — BitNet es native y quantized pero mantiene masters bf16; GPTQ/AWQ son pure y quantized pero post-hoc; NITI/WAGE son native y pure pero nunca escalaron a LLMs. **La intersección es lo nuevo.**

**Resultados medidos (186M parámetros, 1.6B tokens de training, régimen sub-Chinchilla)**

↳ Val PPL 107 en WikiText-103 (split val) vs 614 del baseline fp16 con arquitectura idéntica — una **mejora de 5.73×** al mismo presupuesto de tokens
↳ Brecha de generalización 1.37× vs 3.59× del LLaMA-fp16 — **2.62× menor**, consistente con la teoría PAC-Bayes para clases de hipótesis acotadas
↳ **501 tokens/segundo** de decode @ **114 MB de VRAM** @ **64 mJ/token** en una RTX 5070 consumer
↳ Modelo empaquetado a **74.9 MB** en disco (98.2% del límite de Shannon, bit-exacto sin pérdida)
↳ Costo de entrenamiento a calidad deployable: **$15.99 vs $178.75 extrapolado para fp16** — 11.2× más barato a igual calidad

El checkpoint está publicado abierto bajo licencia CC-BY-NC-SA 4.0. El paper, las 21 figuras y la propuesta financiera para escalar a 2B están en el repo.

**Por qué lo comparto ahora**

Soy investigador independiente en San Juan, Argentina. Lo construí solo durante el último año, compaginándolo con un trabajo full-time de backend. La validación a 186M está hecha — lo que sigue es escalar. Wraith-2B requiere ~$3,000 en compute para 100B tokens en H100.

Busco:

→ **Co-investigadores / co-founders** para llevar NPQN a escala de producción
→ **Sponsors / inversores ángeles** interesa dos en financiar la validación a 2B
→ **Colaboradores académicos** para los experimentos de escalado (20B / 70B)
→ **Reviewers** dispuestos a destruir el paper — prefiero conocer los agujeros ahora y no después del preprint

**Repo + paper + checkpoint + pitch deck**
→ https://github.com/blasfemico/Wraith

Abierto a un DM si algo de esto resuena.

---

#MachineLearning #LLM #Quantization #AIResearch #OpenSource #DeepLearning #NPQN
