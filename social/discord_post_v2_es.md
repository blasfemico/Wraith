# Discord post v2 — Lanzamiento Wraith NPQN (ES, presentación)

---

```
╭─ [ WR-01 · HAUNT PROTOCOL · ONLINE ] ────────────────────────╮
│                                                               │
│   >  wraith · 186M params · 74.9 MB · 501 tok/s               │
│   >  substrate / vessel / signal · todo verde                 │
│                                                               │
╰───────────────────────────────────────────────────────────────╯
```

👻 **Hola gente — les presento a Wraith.**

Un LLM que estuve construyendo solo durante un año en San Juan, Argentina. Hoy lo abro al público: paper completo, 21 figuras, checkpoint reproducible y pitch deck — todo en un solo repo.

**Lo corto:**

Wraith es un LLM de **186M parámetros** que pesa **74.9 MB** empaquetado, corre a **501 tokens/seg** en una RTX 5070 consumer y consume **64 mJ por token**. A ese tamaño tira **5.73× mejor perplejidad** en WikiText-103 val que un LLaMA fp16 con arquitectura idéntica entrenado con los mismos tokens.

**Lo interesante** es cómo se entrena: enteros de punta a punta, sin bf16 masters, sin Adam fp32, sin cuantización post-hoc. Es una **clase arquitectónica nueva** que llamo NPQN (Native Pure Quantized Network). El paper explica el porqué.

**Qué van a encontrar en el repo:**

📄 Paper completo (ES canonical + EN)
🖼️ 21 figuras, toda data medida (nada proyectado sin etiquetar)
💾 Checkpoint público de 74.9 MB — reproducí los números vos mismo
💰 Pitch deck para escalar a 2B (~$3K en H100)
📜 Licencia CC-BY-NC-SA 4.0

🔗 **https://github.com/blasfemico/Wraith**

**Qué busco:**

No vengo a vender nada — vengo a compartirlo y a invitar a que lo revienten. Si hacen research en QAT, integer training, small LLMs, o simplemente tienen curiosidad, pasen por el repo. Cualquier feedback, crítica, issue, o DM es bienvenido. Si alguien quiere colaborar en el scale-up a 2B, la puerta está abierta.

Gracias por leer 🙏

— blasfemia
`substrate / vessel / signal`
