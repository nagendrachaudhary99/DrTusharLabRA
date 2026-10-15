# iTRIALSPACE Research Notes

## Landscape

Virtual imaging trials (VITs) replace patients, scanners and readers with computational models that have perfect ground truth (Abadi et al., JMI 2020). The regulatory precedent is FDA's **VICTRE** (JAMA Netw Open 2018): 2,986 virtual patients, Monte Carlo imaging (MC-GPU) and a computational reader, which reproduced the direction of a human DBT-vs-DM trial. FDA's **M-SYNTH** (NeurIPS 2023) applied the idea to *stress-testing AI*. Duke's **XCAT** phantoms and **DukeSim** are the physics-based route for CT. AAPM TG387 (Med Phys 2024) requires VITs to show credibility, reproducibility and accessibility.

In parallel, generative AI now produces realistic 3D CT. Examples: **NVIDIA MAISI** (latent diffusion + 3D ControlNet, mask-conditioned on up to 127 structures, with "lung tumor" and the five lung lobes in its label dictionary), **MedSyn** (text-guided lung CT), and lesion synthesis such as **Hu et al. CVPR 2023**, **DiffTumor** (CVPR 2024) and **LeFusion** (ICLR 2025, lung nodules on LIDC). iTRIALSPACE sits where these two lines meet: it uses generative models instead of, or alongside, physics simulation to build programmable virtual cohorts.

## Most relevant open-source components for a real implementation

| Need | Component | Why |
|---|---|---|
| Mask-conditioned CT generation | MAISI (MONAI / NV-Generate-CTMR) | Only open 3D CT generator with ControlNet mask control including lung tumor + lobe labels. Apache-2.0 code; check weight license; needs 16-50+ GB GPU |
| Lesion inpainting into real scans | DiffTumor / LeFusion approach, MONAI Generative | Keeps real background anatomy and scanner noise while controlling nodule mask size and location |
| Lobe placement and plausibility checks | TotalSegmentator (v2, 117 classes; lobe classes; lung_nodules task) | Places nodules inside a requested lobe; maps detections back to lobes |
| AI under test | MONAI Model Zoo lung nodule detector, nnU-Net | Realistic, reproducible baselines |
| Protocol variation (dose, kernel) | LEAP / ASTRA / ODL (projection + FBP with Poisson noise); TorchIO for MVP-level approximations | Physically correct, kernel-dependent noise texture vs. white-noise shortcuts |
| I/O and geometry | pydicom, SimpleITK | Correct mm-space measurement; vendor/kernel/dose DICOM tags |
| Reference data | LIDC-IDRI / LUNA16, NLST, LNDb, MSD Lung, DeepLesion | Training, realism checks, real-world size distributions, external domains |
| Statistics | FROC/CPM, Wilson CIs, patient-level bootstrap, iMRMC | Standard, defensible reporting |
| Orchestration | LangGraph.js + RAG | Turns natural-language trial specs into validated parameters; grounds design in guidelines; reports with citations |

## What an MVP can credibly demonstrate

1. **Programmable trial specification.** Natural language is parsed into a structured, validated design (diameter in mm, lobe, solid/part-solid/GGN, vendor/kernel/dose, cohort size), with guardrails grounded in Lung-RADS v2022 (6/8/15 mm) and Fleischner 2017 thresholds.
2. **Factorial cohort design and stratified evaluation.** FROC/CPM and per-stratum sensitivity with Wilson or bootstrap CIs and n. This addresses hidden stratification (Oakden-Rayner et al. 2020) and GMLP principle 8.
3. **Robustness profiles.** Sensitivity vs nodule size, lobe, type and protocol, plus measurement bias near Lung-RADS thresholds.
4. **Traceability.** Every case carries ground truth and generation parameters (seed, model version). This supports TG387 reproducibility and the CM&S credibility framing (context of use, model risk).

**Be honest about limits.** An MVP running in a Next.js app will likely use precomputed or simulated volumes and image-domain approximations for scanner effects, not live MAISI inference or projection-domain physics. It should not claim regulatory-grade realism. The FDA 2023 CM&S guidance excludes standalone ML models from its scope, and no generative VIT tool was found to be MDDT-qualified. Known open gaps the MVP can name rather than solve: emulating vendors and kernels with validated NPS/MTF matching; realism of small (<10 mm), subsolid and juxtapleural synthetic nodules; memorization checks; and validating synthetic-vs-real performance agreement.
