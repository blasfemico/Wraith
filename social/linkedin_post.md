# LinkedIn post — Wraith NPQN launch

---

🚨 **After a year of research, I'm open-sourcing Wraith — the first Native Pure Quantized Network (NPQN).**

A new architectural class of LLM: trained from scratch with a pipeline 100% integer — no bf16 masters, no fp32 Adam states, no post-hoc quantization. Every stage of training and inference operates on discrete weights.

**What makes it different**

Three properties that no prior LLM combines simultaneously at scale:

• **Native** — quantized weights from random init, not retrofitted onto a fp16 checkpoint
• **Pure** — the entire weight pipeline is fixed-point integer, including the optimizer
• **Quantized** — 9-level Dualwire scheme at 3.17 bits/weight (Shannon-optimal for two ternary channels)

Prior art has each ingredient separately — BitNet is native and quantized but keeps bf16 masters; GPTQ/AWQ are pure and quantized but post-hoc; NITI/WAGE are native and pure but never scaled to LLMs. **The intersection is what's new.**

**Measured results (186M params, 1.6B training tokens, sub-Chinchilla)**

↳ Val PPL 107 on WikiText-103 (val split) vs 614 for an architecture-identical fp16 baseline — a **5.73× improvement** at the same token budget
↳ Generalization gap 1.37× vs LLaMA-fp16's 3.59× — **2.62× smaller**, consistent with PAC-Bayes theory for bounded hypothesis classes
↳ **501 tokens/sec** decode @ **114 MB VRAM** @ **64 mJ/token** on a consumer RTX 5070
↳ Full model packed to **74.9 MB** on disk (98.2% of the Shannon limit, bit-exact lossless)
↳ Training cost to reach deployable quality: **$15.99 vs $178.75 extrapolated for fp16** — 11.2× cheaper at matched quality

The checkpoint is published openly under CC-BY-NC-SA 4.0. The paper, all 21 figures, and the funding proposal for scaling to 2B are all in the repo.

**Why I'm sharing now**

I'm an independent researcher in San Juan, Argentina. I've been building this alone for the past year while holding a full-time backend job. The 186M validation is done — what comes next is scaling. Wraith-2B requires ~$3,000 in compute for 100B tokens on an H100.

I'm looking for:

→ **Co-researchers / co-founders** to help push NPQN to production scale
→ **Sponsors / angel investors** interested in underwriting the 2B validation
→ **Academic collaborators** for 20B / 70B scaling experiments
→ **Reviewers** willing to tear the paper apart — I'd rather learn the flaws now than after a preprint

**Repo + paper + checkpoint + pitch deck**
→ https://github.com/blasfemico/Wraith

Open to a DM if any of this resonates.

---

#MachineLearning #LLM #Quantization #AIResearch #OpenSource #DeepLearning #NPQN
