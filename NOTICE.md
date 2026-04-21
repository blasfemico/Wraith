Wraith — NPQN research release
==============================

This repository is a research release of the Wraith project (Native Pure
Quantized Network, NPQN). It contains the materials necessary to read,
verify and build upon the research described in the paper.

WHAT IS INCLUDED
----------------

  - The full paper (Spanish and English) with architecture specification,
    PAC-Bayes theoretical framework, experimental methodology and results.
  - All 21 figures from the paper.
  - The pitch deck and funding proposal for scaling Wraith to 2B parameters.
  - The packed Wraith-186M checkpoint (74.9 MB, Dualwire 5-trit/byte format,
    lossless round-trip verified).

WHAT IS NOT INCLUDED
--------------------

The following components are NOT part of this public release. They are
reserved as intellectual property of the author and available only under
separate academic or commercial licensing terms:

  1. NPQN Training pipeline
     - Shadow int16 persistent optimizer with stochastic rounding
     - DSSC (Derived-Scale Saturation Coupling) detection and ASR correction
     - Asymmetric weight decay for latents vs derived scales
     - Full training loop (~7,400 lines of Python)

  2. GPU inference engine (WraithFastEngine / WraithFastGraphed)
     - Packed 2-bit GEMV CUDA kernels
     - Fused QKV and fused GateUp kernels
     - Packed embedding lookup kernel
     - CUDA Graphs integration for decode

  3. CPU inference engine
     - C++ AVX2 implementation
     - KV cache management
     - Activation quantization and operator fusion

  4. Dualwire 5-trit/byte packed codec
     - Encoder and decoder specifications
     - Round-trip verification utilities

RE-IMPLEMENTATIONS
------------------

Independent re-implementations of the Wraith architecture, training pipeline
or inference engine from the public paper specification are welcome and fall
outside the scope of these reserved IP rights, as long as no reserved code
is used.

The checkpoint itself is licensed under CC-BY-NC-SA 4.0 (see LICENSE). Using
the checkpoint weights to reproduce the results reported in the paper — or
to perform further non-commercial research — is explicitly permitted.

CONTACT
-------

Dante Villena
San Juan, Argentina
programmingblas@gmail.com
https://github.com/blasfemico
