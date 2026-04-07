# EcoRewind: Portfolio & Research Publication Plan
 
## Overview
 
EcoRewind is a deep learning system for counterfactual ecosystem forecasting —
predicting what a wetland **would have looked like** without a hurricane. This
document covers: (1) a dynamic web portfolio, (2) a conference/journal research
paper, and (3) a self-contained conference poster. Together these make the
project a strong portfolio piece and submission-ready research contribution.
 
---
 
## Part 1: Interactive Web Portfolio
 
### Stack
 
| Layer     | Technology                  | Why                                           |
|-----------|-----------------------------|-----------------------------------------------|
| Frontend  | Next.js 14 (App Router)     | SSR, easy Vercel deploy, TypeScript           |
| Styling   | Tailwind CSS + shadcn/ui    | Fast, clean scientific aesthetic              |
| Maps      | Mapbox GL JS                | High-perf WebGL tile rendering, custom layers |
| Charts    | Recharts                    | Composable, React-native, good animations     |
| 3D globe  | deck.gl (optional)          | Stunning flyover for hero section             |
| Hosting   | Vercel (free hobby tier)    | CI/CD from GitHub, global CDN                 |
 
### Site Structure
 
```
/                    → Hero + project pitch (30-second explainer)
/demo                → Interactive counterfactual explorer (main feature)
/methodology         → Technical write-up with model diagrams
/results             → Model comparison table + metrics plots
/about               → Author info, links to paper/code
```
 
### `/demo` — Interactive Counterfactual Explorer
 
This is the centrepiece. The page has three panels:
 
**Left panel — Map**
- Mapbox satellite basemap centred on either Everglades or Barataria Bay
- Toggle ecosystem switcher (Everglades / Mississippi)
- Band selector: NDVI | NDWI | RGB | SAR_VV
- Quarter slider: 2016-Q1 → 2024-Q4
- Layer toggle: "Actual" / "Counterfactual" / "Split"
  - Split mode: vertical drag divider — left=actual, right=counterfactual
- Colour ramp: green (high NDVI) → yellow → red (damaged)
- Impact overlay: semi-transparent heatmap of ΔNDVI > 0.05 threshold
 
**Right panel — Charts**
- **NDVI Timeline**: line chart showing actual vs counterfactual NDVI over time
  - Shaded 95% CI band (from MC Dropout)
  - Vertical dashed line at hurricane event
  - Annotated "Hurricane Irma / Ida" label
- **Recovery curve**: % pixels within 0.05 NDVI of pre-event baseline vs quarter
- **Impact summary cards**:
  - "X km² impacted" (pixels with ΔNDVI > threshold)
  - "~Y tCO₂ equivalent loss" (carbon_factor × ΔNDVI × area)
  - "Estimated recovery: Q [Z]" (when CF and actual converge)
 
**Bottom panel — Model comparison**
- Tab switcher: ConvLSTM | UNet-Temporal | EcoTransformer
- Per-model: NDVI R², SSIM score, temporal coherence, inference time
- Side-by-side prediction frames for the hurricane quarter
 
### Data Preparation Pipeline (run once after training)
 
Create these scripts to export web-ready assets:
 
```
scripts/
  05_generate_counterfactual_maps.py   # run inference → save NDVI/NDWI arrays
  06_compute_recovery_metrics.py       # TTR, carbon loss, impact extent
  07_export_for_web.py                 # convert to JSON + COG tiles
```
 
**`05_generate_counterfactual_maps.py`** — key outputs:
```python
# For each ecosystem + model combination, save:
# outputs/web_assets/{ecosystem}_{model}/
#   actual_ndvi.npy         (T, H, W) float32
#   counterfactual_ndvi.npy (T, H, W) float32
#   cf_uncertainty.npy      (T, H, W) float32 — MC Dropout std
#   impact_mask.npy         (T, H, W) bool    — ΔNDVI > threshold
```
 
**`07_export_for_web.py`** — produces:
```
web/public/data/
  everglades_metadata.json     # bounds, dates, hurricane info
  everglades_ndvi_actual.json  # compressed NDVI timeseries (mean per quarter)
  everglades_ndvi_cf.json      # counterfactual timeseries
  everglades_ci.json           # 95% confidence intervals
  tiles/                       # Mapbox-compatible tile pyramid (optional)
```
 
For the tile pyramid, use `rio-cogeo` to produce Cloud-Optimised GeoTIFFs and
`tippecanoe` (or `rio-tiler`) to serve them. Alternatively, encode the (H, W)
arrays as PNG images (one per quarter) and serve directly.
 
### Building the Web App
 
```bash
# Scaffold
npx create-next-app@latest web --typescript --tailwind --app
cd web
npm install mapbox-gl @mapbox/mapbox-gl-geocoder recharts lucide-react
 
# Key component files to create:
# web/components/CounterfactualMap.tsx    — Mapbox map with NDVI layer
# web/components/NdviTimeline.tsx         — Recharts line chart
# web/components/ImpactSummary.tsx        — stat cards
# web/components/ModelComparison.tsx      — results table
# web/app/demo/page.tsx                   — assembles panels
```
 
Set `NEXT_PUBLIC_MAPBOX_TOKEN` in `.env.local`.
 
### Deployment
 
```bash
# Push web/ folder to GitHub
# Connect repo to vercel.com → auto-deploys on every push
vercel --prod
```
 
Domain: `eco-rewind.vercel.app` (free) or custom domain.
 
---
 
## Part 2: Research Paper (RP)
 
### Target Venues (ordered by fit)
 
| Venue                            | Type        | Pages | Deadline (approx)    |
|----------------------------------|-------------|-------|----------------------|
| IEEE IGARSS 2026                 | Conference  | 4     | Jan 2026             |
| ISPRS Annals                     | Conference  | 8     | Rolling              |
| Remote Sensing (MDPI)            | Journal     | any   | Rolling, open access |
| Climate Change AI (NeurIPS WS)   | Workshop    | 4–8   | Sep 2025             |
| EarthVision (CVPR WS)            | Workshop    | 4–8   | Mar 2026             |
 
**Recommended first target**: *Remote Sensing* (MDPI) — open access, strong
impact factor (5.0), rolling submissions, accepts ML-heavy papers, quick review.
 
### Paper Outline
 
**Title**: "EcoRewind: Counterfactual Ecosystem Forecasting Using Deep
Spatiotemporal Learning for Hurricane Impact Attribution"
 
**Abstract** (~250 words):
We present EcoRewind, a framework for counterfactual prediction of wetland
ecosystems — reconstructing what NDVI/NDWI conditions would have been in the
absence of a hurricane disturbance. We train on multi-spectral Sentinel-2 and
Sentinel-1 SAR time series (7 bands, 32 quarters, 10 m resolution) for two
case studies: Everglades (Hurricane Irma, 2017) and Barataria Bay, Louisiana
(Hurricane Ida, 2021). We compare three architectures: ConvLSTM, U-Net Temporal,
and our proposed EcoTransformer. Counterfactual trajectories enable direct
attribution of ecosystem damage to the storm event, estimation of carbon
sequestration losses, and prediction of recovery timelines. The EcoTransformer
achieves NDVI R²=X.XX on held-out spatially separated patches...
 
**1. Introduction** (~1 page)
- Hurricanes cause persistent ecosystem disruption beyond the initial event
- Challenge: confound — season, climate variability → hard to isolate storm impact
- Counterfactual prediction: what would have happened without the storm?
- Prior work: manual baseline extrapolation, statistical anomaly detection
- Our approach: learned spatiotemporal dynamics → neural counterfactual
 
**2. Related Work** (~0.5 page)
- ConvLSTM (Shi et al. 2015), PredRNN (Wang et al.)
- Vegetation dynamics: NDVI-based recovery (Parker et al., Roberts et al.)
- Hurricane damage remote sensing (Chambers et al., Cuevas et al.)
- Transformer-based video prediction: VideoGPT, Earthformer
 
**3. Data** (~0.5 page)
- GEE export: Sentinel-2 SR (Blue, Green, Red, NIR, NDVI, NDWI) + Sentinel-1 SAR
- Quarterly composites: median aggregation, cloud masking
- Two sites: Everglades (Florida), Barataria Bay (Louisiana)
- 128×128 patch extraction, 50% overlap, 85% NaN rejection
- Train/val/test split: 70/15/15 by spatial block
 
**4. Method** (~1.5 pages)
- 4.1 Tensor construction + normalization
- 4.2 Loss function: reconstruction + temporal smoothness + ecological constraints + SSIM
- 4.3 ConvLSTM baseline
- 4.4 U-Net Temporal baseline
- 4.5 EcoTransformer (proposed): factorized spatiotemporal attention
- 4.6 Counterfactual generation: MC Dropout uncertainty, climatology anchoring
 
**5. Experiments** (~1 page)
- Metrics: NDVI R², SSIM, temporal autocorrelation error, carbon estimate error
- Quantitative comparison table (Table 1)
- Ablation: loss components, temporal attention, MC samples
 
**6. Results** (~0.5 page)
- Everglades Irma: pre-event NDVI=X, event ΔNDVI=−0.113, recovery by Q[Z]
- Barataria Ida: pre-event NDVI=X, event ΔNDVI=−Y, recovery by Q[Z]
- Carbon loss estimates with uncertainty
 
**7. Discussion + Conclusion** (~0.5 page)
 
### Key Figures for the Paper
 
1. **Figure 1** — Study area map: Everglades + Barataria Bay on US map with GEE AOI polygons
2. **Figure 2** — Model architecture diagram: EcoTransformer with attention blocks
3. **Figure 3** — Qualitative comparison: 3×3 grid (pre/event/post, actual/pred/CF) per ecosystem
4. **Figure 4** — NDVI recovery curves: actual vs counterfactual with 95% CI
5. **Table 1** — Quantitative comparison: ConvLSTM / UNet / EcoTransformer per ecosystem
 
### Writing Tools
- **Overleaf**: LaTeX collaboration — use IEEE or MDPI template
- **draw.io / Figma**: architecture diagrams
- **matplotlib**: all plots should use `seaborn` style for publication quality
 
---
 
## Part 3: Conference Poster
 
**Size**: 36×48 inches (portrait) or 48×36 (landscape, better for widescreen)
 
**Layout** (6 sections):
 
```
┌─────────────────────────────────────────────────────────┐
│  TITLE + AUTHORS + INSTITUTION + QR CODE (to web demo)  │
├────────────┬────────────┬────────────┬──────────────────┤
│ Motivation │   Data &   │   Model    │     Results      │
│            │  Pipeline  │ (diagram)  │  (NDVI curves +  │
│ Hurricane  │            │            │   CF maps)       │
│ damage,    │ GEE export │ EcoTrans-  │                  │
│ why CF?    │ → tensors  │ former     │ Table + figures  │
├────────────┴────────────┴────────────┴──────────────────┤
│  Impact: Carbon loss estimate │ Recovery timeline        │
│  "Irma caused ~X tCO₂ loss,  │  [recovery curve image] │
│  recovery in Q[Z] 2020"       │                         │
└─────────────────────────────────────────────────────────┘
```
 
**Poster tools**: Canva (free, easy), Adobe Illustrator, or PowerPoint.
Export at 300 DPI for print.
 
---
 
## Part 4: Execution Checklist
 
### Phase 1 — Finish Training (now)
- [ ] Run `python scripts/03_train_unet.py` to completion (losses.py is fixed)
- [ ] Run `python scripts/03_train_unet.py --model eco_transformer` (new model)
- [ ] Collect metrics: NDVI R², SSIM, temporal coherence
 
### Phase 2 — Generate Assets (1–2 days after training)
- [ ] `python scripts/05_generate_counterfactual_maps.py`
- [ ] `python scripts/06_compute_recovery_metrics.py`
- [ ] `python scripts/07_export_for_web.py`
 
### Phase 3 — Build Web Demo (3–5 days)
- [ ] Scaffold Next.js app in `web/`
- [ ] Build CounterfactualMap component (Mapbox NDVI layer)
- [ ] Build NdviTimeline chart (Recharts)
- [ ] Wire up data JSON files
- [ ] Deploy to Vercel
 
### Phase 4 — Write Paper (2–4 weeks)
- [ ] Create Overleaf project (use MDPI template)
- [ ] Write Methods section first (easiest, already implemented)
- [ ] Create all 5 figures
- [ ] Write Results with actual numbers from training
- [ ] Write Introduction + Related Work
- [ ] Proofread + submit
 
### Phase 5 — Poster (1–2 days before conference)
- [ ] Export key figures from paper
- [ ] Lay out in Canva or Illustrator
- [ ] Add QR code linking to web demo
- [ ] Print at 300 DPI
 
---
 
## Repository Polish for Portfolio
 
Add to `README.md`:
```markdown
## Live Demo
[eco-rewind.vercel.app](https://eco-rewind.vercel.app) — interactive counterfactual explorer
 
## Key Results
| Model | Everglades NDVI R² | Barataria Bay NDVI R² |
|-------|--------------------|-----------------------|
| ConvLSTM | X.XX | X.XX |
| UNet-Temporal | X.XX | X.XX |
| EcoTransformer | X.XX | X.XX |
 
## Citation
[BibTeX entry once published]
```
 
GitHub profile README badge:
```markdown
[![EcoRewind Demo](https://img.shields.io/badge/Live_Demo-EcoRewind-green?style=flat-square)](https://eco-rewind.vercel.app)
```