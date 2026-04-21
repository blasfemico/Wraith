# Discord post — Wraith NPQN launch

---

```
╭─ [ WR-01 · HAUNT PROTOCOL · ONLINE ] ────────────────────────╮
│                                                               │
│   >  systemctl start wraith.service                           │
│   >  ● wraith.service active · uptime: 1 year · native int    │
│   >  ● substrate / vessel / signal · all green                │
│                                                               │
╰───────────────────────────────────────────────────────────────╯
```

**Hey — dropping an LLM I've been cooking for a year.**

It's called **Wraith**, a **Native Pure Quantized Network (NPQN)** — new architectural class. TL;DR: the entire thing (training + inference) runs on integers end-to-end. No bf16 masters. No fp32 Adam. No post-hoc int4 conversion. **Weights are discrete from step 0**, the optimizer state is a persistent int16 shadow with stochastic rounding, and the forward uses a 9-level Dualwire scheme at 3.17 bits/weight (Shannon-optimal for 2 ternary channels).

**Numbers from the 186M validation run** (1.6B tokens, sub-Chinchilla, architecture-identical baseline LLaMA fp16):

```
  val PPL WikiText-103 ...... Wraith 107   vs   LLaMA 614   (5.73× better)
  train-chunk PPL ........... Wraith  74   vs   LLaMA 171   (2.29× better)
  generalization gap ........ Wraith 1.37× vs   LLaMA 3.59× (2.62× smaller)
  decode throughput ......... 501 tok/s on RTX 5070 @ 114 MB VRAM @ 64 mJ/tok
  packed model size ......... 74.9 MB  (98.2% of Shannon limit, lossless)
  training cost ............. $16 vs $179 extrapolated for fp16 at matched quality
```

The kicker on the generalization gap: **the ratio train-vs-held-out is 2.29× in both regimes (2.23× held-out) → it's not overfitting, it's intrinsic to NPQN training.** Wraith extracts more signal per token because the bounded hypothesis class (3.17 bits) prevents memorizing noise.

**What's in the repo**
• Full paper in ES + EN (NPQN framing, PAC-Bayes section, DSSC failure mode + ASR fix)
• 21 figures, all measured data
• **The packed checkpoint is public** — 74.9 MB, CC-BY-NC-SA 4.0, reproduce every number yourself
• Pitch deck for scaling Wraith to 2B ($3K compute ask)
• Logo, NOTICE, LICENSE

**What's NOT in the repo** (reserved IP, available under license)
• The NPQN training pipeline (shadow int16 + SR + DSSC/ASR)
• The CUDA inference engine (packed 2-bit GEMV, fused QKV, CUDA Graphs)
• The C++ AVX2 CPU engine
• The Dualwire 5-trit/byte codec

**What I need**

I'm one dude in San Juan, Argentina, been building this solo for a year while working backend full-time. The 186M run is the end of what I can validate with consumer hardware + Colab Pro. To prove NPQN scales, I need Wraith-2B at 100B tokens — **~$3,000 in H100 compute**.

Looking for:
• **Feedback / tear-down** of the paper — seriously, if you see a hole, I want to know
• Anyone doing QAT / integer training research — let's compare notes
• Collaborators for the 2B scale-up (co-authorship on the follow-up paper is on the table)
• Sponsors / angels who think this is worth underwriting

Everything lives at:
🔗 **https://github.com/blasfemico/Wraith**

Paper ES (canonical): `paper/wraith_paper_es.md`
Paper EN: `paper/wraith_paper_en.md`
Checkpoint: `checkpoint/wraith-186m-npqn-packed.pt` (Git LFS)
Pitch: `funding/pitch_deck.md`

Roast me, help me, or both. DMs open.

— blasfemia
`substrate / vessel / signal`
