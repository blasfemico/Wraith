# Discord post v2 — Wraith NPQN launch (EN, presentation)

---

```
╭─ [ WR-01 · HAUNT PROTOCOL · ONLINE ] ────────────────────────╮
│                                                               │
│   >  wraith · 186M params · 74.9 MB · 501 tok/s               │
│   >  substrate / vessel / signal · all green                  │
│                                                               │
╰───────────────────────────────────────────────────────────────╯
```

👻 **Hey everyone — let me introduce Wraith.**

An LLM I've been building solo for the past year from San Juan, Argentina. Today I'm opening it to the public: full paper, 21 figures, reproducible checkpoint and pitch deck — all in one repo.

**Short version:**

Wraith is a **186M-parameter** LLM that packs to **74.9 MB**, runs at **501 tokens/sec** on a consumer RTX 5070, and sips **64 mJ per token**. At that size it delivers **6.24× better perplexity** than an architecture-identical LLaMA fp16 trained on the same tokens.

**The interesting bit** is how it's trained: integer end-to-end, no bf16 masters, no fp32 Adam, no post-hoc quantization. It's a **new architectural class** I'm calling NPQN (Native Pure Quantized Network). The paper goes into the why.

**What you'll find in the repo:**

📄 Full paper (ES canonical + EN translation)
🖼️ 21 figures, all measured data (nothing projected without labeling)
💾 Public 74.9 MB checkpoint — reproduce every number yourself
💰 Pitch deck to scale to 2B (~$3K H100 compute)
📜 CC-BY-NC-SA 4.0 license

🔗 **https://github.com/blasfemico/Wraith**

**What I'm looking for:**

I'm not selling anything — I'm sharing it and inviting people to tear it apart. If you're researching QAT, integer training, small LLMs, or are just curious, drop by the repo. Feedback, critique, issues, DMs — all welcome. If anyone wants to collaborate on the 2B scale-up, the door is open.

Thanks for reading 🙏

— blasfemia
`substrate / vessel / signal`
