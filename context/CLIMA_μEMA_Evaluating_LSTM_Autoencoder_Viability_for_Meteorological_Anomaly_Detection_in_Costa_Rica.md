# CLIMA-µEMA: Evaluating LSTM-Autoencoder Viability for Meteorological Anomaly Detection in Costa Rica

**Isaac Palma-Medina**\
isaac.palma.001@student.uni.lu\
Université du Luxembourg\
Esch-sur-Alzette, Luxembourg

## Abstract

Monitoring localized meteorological risk across decentralized weather
station networks demands scalable, label-efficient anomaly detection.
CLIMA-µEMA evaluates LSTM-Autoencoder reconstruction error against
verified historical emergency alerts via an end-to-end Medallion
pipeline, ablating four bottleneck dimensions across five seeds (80
training runs total). At (d = 16), local models achieve PA-F1 ≥ 0.938 ±
0.006 and PA-Recall ≈ 1.000 at four viable stations; denoising and
global variants yield no consistent gain. Failures map directly to data,
sensor, and label deficiencies, consistently across all ablation
conditions.

## 1 Introduction

The University of Costa Rica (UCR) operates 12 decentralized campuses
and precincts across diverse microclimates \[5, 15\]. Managing localized
disaster risks across this footprint requires continuous, site-specific
monitoring via UCR's 10-station Automatic Micro Weather Stations (µEMA;
Micro Estaciones Meteorológicas Automáticas) network (see Appendices,
Figure 1) \[7\].

Processing this high-frequency atmospheric data presents a technical
bottleneck. Traditional numerical weather prediction (NWP) models do not
scale efficiently for real-time anomaly detection in decentralized
Internet of Things (IoT) networks \[19\]. Furthermore, classical
statistical methods and rigid rule-based thresholds fail to capture the
nonlinear, chaotic dynamics of contemporary climate systems \[16\].

To bridge this gap, I introduce the Climate Localized Incident
Monitoring with Autoencoders (CLIMA; Control Localizado de Incidentes
Meteorológicos con Autoencoders) framework. Rather than simulating
atmospheric physics, CLIMA utilizes unsupervised learning on continuous
µEMA data---including precipitation, pressure, and luminous intensity
\[8\]---to detect meteorological deviations. By evaluating autoencoder
reconstruction errors, it automatically flags sequences that fail to
reconstruct normative atmospheric states.

This detection boundary is calibrated against verified historical
emergency alerts from the Costa Rican National Commission for Risk
Prevention and Emergency Response (CNE) \[4\].

This report presents CLIMA's foundational architecture and preliminary
validation. My contributions are: (1) an end-to-end data orchestration
pipeline for the µEMA network, and (2) experiments demonstrating the
viability of autoencoder reconstruction error as a robust anomaly
detection mechanism for localized environmental incidents.

## 2 Related Work

CLIMA operates at the intersection of multivariate time-series anomaly
detection and data-driven meteorology. Modern anomaly detection systems
increasingly rely on sliding windows and models that capture both
temporal and inter-metric dependencies \[6, 16\]. This is crucial for
meteorology, where weather signals are nonlinear and standard NWP is
computationally prohibitive for high-frequency local deployment \[2, 12,
19\]. While deep learning for environmental prediction remains limited
in Latin America \[11\], related slope-disaster and weather-forecasting
systems demonstrate that fusing heterogeneous sensor streams through
deep, layered architectures significantly improves real-time warning
capabilities \[2, 12, 14\].

For anomaly detection, reconstruction-based models trained exclusively
on normal data are highly effective. However, standard autoencoders
treat observations in isolation and struggle with temporal structures.
These limitations render Long Short-Term Memory Autoencoder (LSTM-AE)
models preferable; their recurrent architecture captures sequential
dependencies and seasonal contexts, directly reducing false alarms in
meteorological quality control \[13, 16, 18\]. CLIMA leverages these
temporal capabilities by training LSTM-AEs on fixed-length local weather
sequences rather than isolated data points. Finally, combining targeted
feature design with dynamic thresholding ensures localized
interpretability for operators \[1\], while robust evaluation requires
tracking multi-sensor dependencies rather than independent readings
\[6\].

## 3 Experiments

### 3.1 Data Processing Pipeline

To transform raw telemetry into model-ready arrays, the framework
implements a Medallion architecture (Figure 2). A Raw Layer extracts
sensor readings alongside emergency alerts digitized via LLM-based OCR.
The Silver Layer resamples to 10-minute intervals, removes failure
periods, interpolates pressure gaps and zero-fills precipitation and
lux, and appends cyclical time features. The Gold Layer dilates alert
periods into an asymmetric anomaly mask (48h pre-alert, 120h post-alert
\[10\]), scales features on normal data only, and segments the series
into 144-step (24h) sliding windows. Arrays follow an 80/20 temporal
train--test split; anomalous windows are reserved for testing and
boundary-adjacent normal windows form a calibration set.

### 3.2 LSTM-AE Architecture

The LSTM-AE (Figure 3) processes 24-hour sliding windows (144 timesteps,
10-minute resolution) over 7 features: three continuous meteorological
variables (pressure, precipitation, luminous intensity) and four
cyclical temporal embeddings (hour and day-of-year sine/cosine). The
encoder compresses sequences through a three-layer LSTM funnel (128 → 64
→ (d) hidden units), retaining only the final hidden state (h_n) as a
(d)-dimensional bottleneck. The decoder mirrors this symmetrically ((d)
→ 64 → 128 units), receiving the bottleneck repeated 144 times before a
linear projection to three continuous outputs. Four bottleneck sizes are
ablated: (d `\in `{=tex}{8, 16, 32, 64}), yielding compression ratios
from 126:1 to 16:1 over the 144×7 input.

Dropout ((p = 0.2)) follows the first two layers of both sub-networks;
the bottleneck and final pre-projection activations are intentionally
uncorrupted.

To support data-limited stations, experiments run under standard and
augmented regimes. Augmentation applies Gaussian noise to simulate
calibration drift and temporal masking---zeroing random short spans to
force reconstruction from broader temporal context \[17\]. Training
minimizes a feature-weighted MSE over continuous features only (1.5 /
2.0 / 1.0 for pressure, precipitation, lux), with precipitation weighted
highest as CNE alerts are predominantly rainfall-driven. Cyclical
embeddings are excluded from both loss and decoder output. Anomaly
scores apply the same weighting with 3-step rolling mean smoothing.

### 3.3 Experimental Setup

Four configurations---Local and Global, both Baseline and
Augmented---are evaluated, excluding the faulty recinto-guapiles
station. The Global setup applies chronological partitioning per-station
before concatenation, preserving temporal ordering. Models are trained
for up to 100 epochs using AdamW ((lr = 1 `\times 10`{=tex}\^{-3})),
Cosine Annealing with Warm Restarts, gradient clipping, and early
stopping on a held-out 10% validation partition. Each configuration is
evaluated across five fixed seeds and four bottleneck dimensions ((d
`\in `{=tex}{8, 16, 32, 64})), totalling 80 training runs; all reported
metrics are mean ± standard deviation across seeds.

Detection thresholds are calibrated independently of test labels: local
models sweep the 85th, 90th, and 95th percentiles of validation
reconstruction errors, while the global model uses the 95th percentile
of the boundary-normal calibration set. Performance is evaluated via
pointwise and Point Adjustment (PA) metrics. Precision, Recall, and F1
are standard binary classification metrics; FPR is the fraction of
normal windows incorrectly flagged. Under PA, a ground-truth anomaly is
credited as detected only if ≥ 10% of its windows are flagged
(approximately 17 hours of the dilated block); PA-Recall and PA-F1 are
Recall and F1 under this criterion.

### 3.4 Results

Across the 80-run ablation, the four viable stations yield stable PA-F1
at (P = 85) (Tables 1 and 3). At (d = 16), Local Baseline achieves PA-F1
of 0.947 ± 0.005, 0.960 ± 0.006, 0.980 ± 0.005, and 0.938 ± 0.005 at
recinto-esparza, recinto-santa-cruz, sede-atlantico_turrialba, and
sede-sur_golfito respectively; three of four stations attain PA-Recall =
1.000 across all seeds. Results are consistent across (d `\in `{=tex}{8,
16, 32}); at (d = 64), sede-atlantico_turrialba shows bimodal behaviour
((`\sigma`{=tex}\_{PA-F1} = 0.387)), indicating seed-sensitive learning
at larger bottlenecks on this data-scarce station.

Augmentation yields no consistent gain ((\|`\Delta `{=tex}PA-F1\|
`\leq 0.006`{=tex}) at (d = 16)) and destabilises
sede-atlantico_turrialba at (d `\in `{=tex}{8, 32}) in some seeds. These
four stations share at least 4,440 training windows and continuously
functioning sensors. Threshold sensitivity (Table 2) confirms PA-Recall
≈ 1.000 is stable for three viable stations across (P `\in `{=tex}{85,
90, 95}); the ablation reveals that for sede-atlantico_turrialba and
recinto-esparza Baseline, higher percentiles introduce inter-seed
variance, making (P = 85) the safest operating point for the former and
(P = 90) for the latter.

The five remaining stations fail consistently across all 80 runs:
finca-2 achieves PA-F1 = 0.936 ± 0.006 but FPR ≈ 0.757 due to luminous
intensity sensor inactivity prior to 2025-05-20; liberia shows
consistently low PA-Recall (0.086 ± 0.001 across all (d)), indicating
persistent event-label misalignment; finca-3 local models fail at (d
`\in `{=tex}{8, 32, 64}) (PA-Recall ≤ 0.055) with a single seed
achieving PA-Recall = 1.000 at (d = 16) (mean = 0.241 ± 0.424),
revealing borderline local detectability; caribe_limon and finca-1
produce all-zero output across all seeds and dimensions because fewer
than 1,650 training windows cannot stabilise the calibration
distribution.

The Global model robustly recovers finca-3 with PA-F1 = 0.926 ± 0.001
and PA-Recall = 1.000 across all (d), confirming global context as a
reliable compensator for label misalignment---but produces no
operational recovery at data-limited stations.

## 4 Discussion

The pipeline's symmetric LSTM-AE architecture, feature-matched loss
function, independent per-station splitting, and label-free threshold
calibration collectively prevent metric inflation and objective
mismatch. The 5-seed, 4-dimension ablation confirms results are stable
at (d `\in `{=tex}{8, 16, 32}) across viable stations; at (d = 64),
sede-atlantico_turrialba---the most data-scarce viable station---shows
bimodal failure under both configurations, suggesting larger bottlenecks
amplify initialisation sensitivity when training data is limited.

The denoising objective offers no consistent operational advantage;
augmentation slightly increases FPR without improving mean PA-Recall,
and destabilises some seeds at atypical bottleneck sizes.

PA-F1 must be contextualised alongside test set anomaly rates, as
dilated blocks expand the anomalous window count.

Station heterogeneity reveals four operational states: viable (≥1.2
years data, yielding PA-Recall = 1.000 at (P = 90)), sensor-limited
(requiring hardware recalibration due to abrupt regime shifts),
event-limited (requiring cross-sensor label verification due to
label--event misalignment), and data-limited (requiring 2--3 more years
of data to stabilise calibration). Operational deployment requires
defining per-station FPR budgets, as the labeling scheme assumes
continuous sensor operation.

### 4.1 Conclusion

CLIMA-µEMA demonstrates conditional viability of LSTM-AE reconstruction
error as a label-efficient mechanism for localized anomaly detection:
across a 5-seed, 4-dimension ablation, four viable stations achieve
PA-F1 ≥ 0.938 ± 0.006 at (d = 16) with PA-Recall ≈ 1.000, confirming
result stability. Neither denoising nor global architectures improved
local performance, except at finca-3, where global context robustly
recovers event detection (PA-F1 = 0.926 ± 0.001 across all (d)). System
failures map to actionable interventions: continued data collection for
data-limited sites, label verification for event-limited sites, and
hardware inspection for sensor-limited sites.

## Acknowledgments

The author thanks UCR for providing access to the µEMA telemetry
infrastructure and the expert human capital that made this work
possible, and the CNE for maintaining the historical alert registry used
for model evaluation. The author also acknowledges the Climatological
Observation Laboratory (LOSIC-UCR) for the design and deployment of the
station hardware underlying this dataset.

## Data Availability

The source code and datasets supporting this work are publicly available
at <https://github.com/isaac-pm/clima-uema/tree/main>.

## Data Ownership and Usage Notice

The meteorological datasets in this repository are the exclusive
property of UCR, collected via µEMA stations designed by LOSIC-UCR. The
AGPLv3 license governing the source code does not extend to these
datasets; public visibility does not constitute an Open Data release.
This software is an academic proof-of-concept only and must not be used
as a substitute for official emergency alerts issued by the CNE or
equivalent national authorities. All related parties assume no liability
for decisions based on its outputs.

## Statement on the Use of Generative Artificial Intelligence

The author declares that generative AI was used exclusively for
spell-checking and proofreading of this manuscript (Gemini 3.1 Pro).
Regarding the code repository, AI assistance (Claude Sonnet 4.6 via
Claude Code and GPT-5.2-Codex via OpenCode) was limited to general
scaffolding; all design choices and implementation specifics were made
by the author based on the research analysis and intent presented
herein.

## References

\[1\] Tolulope Ale, Vandana P. Janeja, and Nicole-Jeanne Schlegel. 2024.
*Harnessing Feature Clustering For Enhanced Anomaly Detection With
Variational Autoencoder And Dynamic Threshold*. In IGARSS 2024 - 2024
IEEE International Geoscience and Remote Sensing Symposium. IEEE,
Athens, Greece, 1,2. doi:10.1109/IGARSS53475.2024.10640794

\[2\] Maira Aracne, Tommaso Ruga, Camilla Lops, Deborah Federico,
Luciano Caroprese, Ester Zumpano, Sergio Montelpare, Mariano
Pierantozzi, Francesco Dattola, Pasquale Iaquinta, Miriam Iusi, Raffaele
Greco, Marco Talerico, Valentina Coscarella, Luca Legato, Ivana
Pellegrino, Sonia Bergamaschi, Mirko Orsini, Riccardo Martoglia, Andrea
Livaldi, Abeer Jelali, and Simone Sbreglia. 2025. *Comparing Deep
Learning Approaches for Weather Forecasting: Insights from the PRECEDE
Project*. EDBT/ICDT Workshops (2025), 2,4.

\[3\] Hylke E. Beck, Niklaus E. Zimmermann, Tim R. McVicar, Noemi
Vergopolan, Alexis Berg, and Eric F. Wood. 2018. *Present and future
Köppen-Geiger climate classification maps at 1-km resolution*.
Scientific Data 5, 1 (Oct. 2018), 180214. doi:10.1038/sdata.2018.214

\[4\] Comisión Nacional de Prevención de Riesgos y Atención de
Emergencias (CNE). 2026. *Histórico de Alertas (Alert History)*.
<https://www.cne.go.cr/preparativos_respuestas/alertas/historicoalertas.aspx>

\[5\] Consejo Universitario de la Universidad de Costa Rica. 1974.
*Estatuto Orgánico de la Universidad de Costa Rica (Organic Statute of
the University of Costa Rica)*.
<https://www.cu.ucr.ac.cr/normativ/estatuto_organico.pdf>

\[6\] Kyle DeMedeiros, Abdeltawab Hendawi, and Marco Alvarez. 2023. *A
Survey of AI-Based Anomaly Detection in IoT and Sensor Networks*.
Sensors 23, 3 (Jan. 2023), 2,3,6. doi:10.3390/s23031352

\[7\] Marcial Garbanzo-Salas and Kattia Medina-Arias. 2024. *Red de
Monitoreo Atmosférico - UCR (Atmospheric Monitoring Network - UCR)*.

\[8\] Marcial Garbanzo-Salas, Kattia Medina-Arias, Rubén
Madrigal-Cordero, and Alberto Salazar-Murillo. 2025. *Sistemas De Alerta
Temprana En Monitoreo Atmosférico Para La Gestion Del Riesgo: Guia Para
La Persona Usuaria (Early Warning Systems in Atmospheric Monitoring for
Risk Management: User Guide)*.

\[9\] Payal R. Makhasana, Joseph A. Santanello, Patricia M.
Lawston-Parker, and Joshua K. Roundy. 2026. *Understanding
Land--Atmosphere Interactions during Coupling Whiplash Events*. Journal
of Hydrometeorology 27, 4 (March 2026), 437--452.
doi:10.1175/JHM-D-24-0171.1

\[10\] Kaighin A. McColl, Qing He, Hui Lu, and Dara Entekhabi. 2019.
*Short-Term and Long-Term Surface Soil Moisture Memory Time Scales Are
Spatially Anticorrelated at Global Scales*. Journal of Hydrometeorology
20, 6 (June 2019), 1165--1182. doi:10.1175/JHM-D-18-0141.1

\[11\] Aldo Márquez-Grajales, Ramiro Villegas-Vega, Fernando
Salas-Martínez, Héctor-Gabriel Acosta-Mesa, and Efrén Mezura-Montes.
2024. *Characterizing drought prediction with deep learning: A
literature review*. MethodsX 13 (Dec. 2024), 2,12--13.
doi:10.1016/j.mex.2024.102800

\[12\] M. S. Pavithran, B. Sreeram, Adwait V. Pillai, and R. Jothi.
2025. *Multi-Scale Weather Forecasting Using Deep Learning Architectures
With Chennai Climate Data*. IEEE Access 13 (2025), 207303,207317.
doi:10.1109/ACCESS.2025.3640667

\[13\] Teresa Kristine Spohn, Eoin Walsh, Kevin Horan, John O'Donoghue,
Tim Charnecki, Merlin Haslam, and Sarah Gallagher. 2026. *A machine
learning approach using autoencoders to perform quality control on
meteorological data*. Environmental Data Science 5 (2026), 2,3,5,7.
doi:10.1017/eds.2026.10030

\[14\] Wang Ting and Ying Wang. 2025. *Utilizing deep learning for
intelligent monitoring and early warning of slope disasters in public
space design*. Frontiers in Environmental Science 13 (May 2025), 2--3.
doi:10.3389/fenvs.2025.1536481

\[15\] Universidad de Costa Rica. 2026. *Sedes y Recintos (Campuses and
Precincts)*. <https://www.ucr.ac.cr/acerca-u/sedes-recintos.html>

\[16\] Fengling Wang, Yiyue Jiang, Rongjie Zhang, Aimin Wei, Jingming
Xie, and Xiongwen Pang. 2025. *A Survey of Deep Anomaly Detection in
Multivariate Time Series: Taxonomy, Applications, and Directions*.
Sensors 25, 1 (Jan. 2025), 1,2,8--11. doi:10.3390/s25010190

\[17\] Qingsong Wen, Liang Sun, Fan Yang, Xiaomin Song, Jingkun Gao, Xue
Wang, and Huan Xu. 2021. *Time Series Data Augmentation for Deep
Learning: A Survey*. In Proceedings of the Thirtieth International Joint
Conference on Artificial Intelligence, 4653--4660.
doi:10.24963/ijcai.2021/631. arXiv:2002.12478 \[cs.LG\].

\[18\] G K Wijaya, T C S Nova, A Anggraeni, M A Yusuf, and I Kharisudin.
2025. *Comparative Study of Autoencoder and LSTM-AE for Extreme
Temperature Anomaly Detection in Semarang*. Proceedings of 2025
International Conference on Data Science and Official Statistics
(ICDSOS) 2025 (2025), 156,157,162.

\[19\] Yuting Wu and Wei Xue. 2024. *Data-Driven Weather Forecasting and
Climate Modeling from the Perspective of Development*. Atmosphere 15, 6
(June 2024), 1,2,11. doi:10.3390/atmos15060689

## Appendices

### Figure 1: Geographic distribution and operational timeline of µEMA meteorological stations in Costa Rica

Stations are mapped against high-resolution Köppen-Geiger climate zones
\[3\]. The timeline specifies data availability and highlights the
subset of stations utilized for model training.

![Figure 1 --- Geographic distribution and operational timeline of µEMA
meteorological stations in Costa Rica](page-4.png)

### Figure 2: The µEMA Medallion Pipeline

A three-stage architecture (Raw, Silver, Gold) transforming sensor
telemetry and emergency alerts into model-ready NPY datasets.

### Figure 3: LSTM-Autoencoder Architecture

Schematic diagram of the LSTM-Autoencoder architecture, detailing the
multi-layer encoder-decoder structure, hidden state extraction ((h_n)),
and final linear projection for anomaly inference.

![Figures 2 and 3 --- Medallion pipeline and LSTM-Autoencoder
architecture](page-5.png)

## Tables

### Table 1: Local and Global configurations

Local configurations at (P = 85) and Global configurations at (P = 95),
all nine stations; (d = 16), 5 seeds (mean ± std). Prec = Precision; FPR
= False Positive Rate; PA = Point Adjustment protocol; BL = Baseline;
Aug = Augmented. † Luminous intensity sensor inactivity prior to
2025-05-20. ‡ All-zero output across all seeds and dimensions.

  ---------------------------------------------------------------------------------------------------------
  Station                    Config            Prec            F1           FPR         PA-F1     PA-Recall
  -------------------------- -------- ------------- ------------- ------------- ------------- -------------
  recinto-esparza            Local BL   0.691±0.066   0.379±0.077   0.396±0.042   0.947±0.005   1.000±0.000

  recinto-esparza            Local      0.721±0.015   0.415±0.014   0.395±0.026   0.947±0.003   1.000±0.000
                             Aug                                                              

  recinto-santa-cruz         Local BL   0.689±0.043   0.288±0.015   0.376±0.064   0.960±0.006   1.000±0.000

  recinto-santa-cruz         Local      0.685±0.025   0.283±0.006   0.373±0.038   0.961±0.004   1.000±0.000
                             Aug                                                              

  sede-atlantico_turrialba   Local BL   0.884±0.043   0.430±0.123   0.204±0.023   0.980±0.005   0.994±0.013

  sede-atlantico_turrialba   Local      0.895±0.028   0.430±0.073   0.195±0.042   0.978±0.010   0.989±0.015
                             Aug                                                              

  sede-sur_golfito           Local BL   0.786±0.015   0.598±0.007   0.497±0.043   0.938±0.005   1.000±0.000

  sede-sur_golfito           Local      0.788±0.019   0.601±0.010   0.493±0.051   0.939±0.006   1.000±0.000
                             Aug                                                              

  sede-central_finca-2†      Local BL   0.840±0.021   0.778±0.104   0.757±0.080   0.936±0.006   1.000±0.000

  sede-central_finca-2†      Local      0.831±0.024   0.726±0.145   0.727±0.175   0.938±0.014   1.000±0.000
                             Aug                                                              

  sede-central_finca-3       Local BL   0.379±0.107   0.101±0.081   0.370±0.102   0.261±0.382   0.241±0.424

  sede-central_finca-3       Local      0.387±0.097   0.102±0.074   0.371±0.098   0.259±0.383   0.239±0.425
                             Aug                                                              

  sede-central_finca-3       Global             ---   0.444±0.005   0.724±0.003   0.926±0.000   1.000±0.000
                             BL                                                               

  sede-central_finca-3       Global             ---   0.439±0.003   0.721±0.002   0.926±0.000   1.000±0.000
                             Aug                                                              

  sede-guanacaste_liberia    Local BL   0.422±0.014   0.143±0.002   0.575±0.037   0.143±0.002   0.086±0.001

  sede-guanacaste_liberia    Local      0.416±0.023   0.141±0.004   0.586±0.048   0.142±0.005   0.086±0.003
                             Aug                                                              

  sede-caribe_limon‡         Local BL         0.000         0.000         0.000         0.000         0.000

  sede-caribe_limon‡         Local            0.000         0.000         0.000         0.000         0.000
                             Aug                                                              

  sede-central_finca-1‡      Local BL         0.000         0.000         0.000         0.000         0.000

  sede-central_finca-1‡      Local            0.000         0.000         0.000         0.000         0.000
                             Aug                                                              
  ---------------------------------------------------------------------------------------------------------

### Table 2: Threshold sensitivity

Threshold sensitivity for the four operationally viable stations; (d =
16), 5 seeds (mean ± std). Bold marks the recommended operating point:
highest (P) where PA-Recall ≈ 1.000 is stable across seeds while
minimising FPR. ★ PA-Recall \< 1.000 or high variance in some seeds.

  --------------------------------------------------------------------------------------------------------
  Station                    Config            P            F1           FPR          PA-F1      PA-Recall
  -------------------------- -------- ---------- ------------- ------------- -------------- --------------
  recinto-esparza            BL               85   0.379±0.077   0.396±0.042    0.947±0.005    1.000±0.000

  recinto-esparza            BL               90   0.341±0.079   0.339±0.097    0.954±0.013    1.000±0.000

  recinto-esparza            BL               95   0.269±0.092   0.266±0.121   0.795±0.360★   0.817±0.410★

  recinto-esparza            Aug              85   0.415±0.014   0.395±0.026    0.947±0.003    1.000±0.000

  recinto-esparza            Aug              90   0.380±0.011   0.351±0.027    0.952±0.003    1.000±0.000

  recinto-esparza            Aug              95   0.324±0.008   0.280±0.023    0.962±0.003    1.000±0.000

  recinto-santa-cruz         BL               85   0.288±0.015   0.376±0.064    0.960±0.006    1.000±0.000

  recinto-santa-cruz         BL               90   0.284±0.015   0.363±0.053    0.962±0.005    1.000±0.000

  recinto-santa-cruz         BL               95   0.277±0.015   0.341±0.047    0.964±0.005    1.000±0.000

  recinto-santa-cruz         Aug              85   0.283±0.006   0.373±0.038    0.961±0.004    1.000±0.000

  recinto-santa-cruz         Aug              90   0.279±0.006   0.359±0.039    0.962±0.004    1.000±0.000

  recinto-santa-cruz         Aug              95   0.275±0.005   0.343±0.043    0.964±0.004    1.000±0.000

  sede-atlantico_turrialba   BL               85   0.430±0.123   0.204±0.023    0.980±0.005    0.994±0.013

  sede-atlantico_turrialba   BL               90   0.306±0.081   0.114±0.023   0.825±0.372★   0.818±0.407★

  sede-atlantico_turrialba   BL               95   0.187±0.028   0.067±0.035   0.824±0.384★   0.815±0.413★

  sede-atlantico_turrialba   Aug              85   0.430±0.073   0.195±0.042    0.978±0.010    0.989±0.015

  sede-atlantico_turrialba   Aug              90   0.332±0.079   0.130±0.065    0.984±0.013    0.989±0.015

  sede-atlantico_turrialba   Aug              95   0.167±0.086   0.061±0.061   0.792±0.434★   0.790±0.437★

  sede-sur_golfito           BL               85   0.598±0.007   0.497±0.043    0.938±0.005    1.000±0.000

  sede-sur_golfito           BL               90   0.586±0.006   0.461±0.037    0.942±0.004    1.000±0.000

  sede-sur_golfito           BL               95   0.553±0.006   0.325±0.055    0.959±0.007    1.000±0.000

  sede-sur_golfito           Aug              85   0.601±0.010   0.493±0.051    0.939±0.006    1.000±0.000

  sede-sur_golfito           Aug              90   0.591±0.009   0.466±0.051    0.942±0.006    1.000±0.000

  sede-sur_golfito           Aug              95   0.559±0.007   0.332±0.065    0.958±0.008    1.000±0.000
  --------------------------------------------------------------------------------------------------------

### Table 3: Bottleneck ablation

Local Baseline, (P = 85), 5 seeds (mean ± std). Bold marks the most
stable configuration per station. ★ High variance: bimodal across seeds.

  -------------------------------------------------------------------------------------------------
  Station                               d            F1           FPR          PA-F1      PA-Recall
  -------------------------- ------------ ------------- ------------- -------------- --------------
  recinto-esparza                       8   0.395±0.089   0.429±0.025    0.942±0.003    1.000±0.000

  recinto-esparza                      16   0.379±0.077   0.396±0.042    0.947±0.005    1.000±0.000

  recinto-esparza                      32   0.438±0.077   0.381±0.048    0.948±0.006    1.000±0.000

  recinto-esparza                      64   0.433±0.036   0.415±0.042    0.944±0.005    1.000±0.000

  recinto-santa-cruz                    8   0.290±0.004   0.338±0.011    0.964±0.001    1.000±0.000

  recinto-santa-cruz                   16   0.288±0.015   0.376±0.064    0.960±0.006    1.000±0.000

  recinto-santa-cruz                   32   0.278±0.019   0.342±0.008    0.964±0.001    1.000±0.000

  recinto-santa-cruz                   64   0.268±0.021   0.372±0.048    0.961±0.005    1.000±0.000

  sede-atlantico_turrialba              8   0.453±0.058   0.206±0.015    0.980±0.007    0.994±0.013

  sede-atlantico_turrialba             16   0.430±0.123   0.204±0.023    0.980±0.005    0.994±0.013

  sede-atlantico_turrialba             32   0.399±0.081   0.205±0.048    0.978±0.011    0.989±0.014

  sede-atlantico_turrialba             64   0.338±0.144   0.206±0.054   0.807±0.387★   0.807±0.416★

  sede-sur_golfito                      8   0.570±0.073   0.540±0.068    0.933±0.008    1.000±0.000

  sede-sur_golfito                     16   0.598±0.007   0.497±0.043    0.938±0.005    1.000±0.000

  sede-sur_golfito                     32   0.602±0.006   0.495±0.038    0.938±0.004    1.000±0.000

  sede-sur_golfito                     64   0.603±0.006   0.495±0.016    0.938±0.002    1.000±0.000
  -------------------------------------------------------------------------------------------------
