# Operational Disaster Standards, Proven Formulas, and Feature Store Architecture

**Academic Context:** 7th Semester Major Project Phase 1 Evaluation  
**Institution:** National Institute of Technology Karnataka (NITK), Surathkal  
**Department:** Information Technology  
**Project:** AI-Driven Spatio-Temporal Framework for Multimodal Disaster Intelligence and Early Warning  
**Guide:** Dr. Sowmya Kamath S.  
**Team:** Tejas Bajaj (231IT080), Garv Mandloi (231IT023), Swaroop Raghavendra Rao (231IT078)

---

## 1. Executive Summary & Core Distinction

To maintain strict scientific and academic rigor, we draw a hard, unambiguous boundary between two concepts:

1. **Operational Early Warning Thresholds (Proven Standards):**  
   Government-calibrated, empirical decision rules published by operational agencies (GSI, IMD, CWC) that trigger statutory disaster alerts (Green / Yellow / Orange / Red).
2. **Machine Learning Feature Crosses (Derived Features):**  
   Mathematical interaction terms engineered into the **86-dimensional WorldState vector** (`FeatureSpec.v3`) to provide downstream models (Phase 2 GNN / STGCN) with explicit non-linear terms without forcing neural networks to approximate multiplication from scarce disaster data.

> [!CAUTION]
> **Defensive Rule for Evaluators:**  
> Never claim that `compound_saturation = soil_moisture * precipitation_72h` is an official early warning score.  
> It is an **interaction feature** inside a feature store. The actual warning classifications are either determined by **published GSI/IMD thresholds** (rule-based) or **trained Phase 2 ML models** (data-driven).

---

## 2. Official Proven Standards in India

### 2.1. Geological Survey of India (GSI) — Regional Landslide Early Warning System (LEWS)

GSI is the nodal agency for landslide hazard assessment in India. In the Western Ghats (Wayanad, Nilgiris, Uttara Kannada), GSI’s operational Landslide Early Warning System relies on an **empirical bilinear rainfall threshold** coupled with static hazard zonation:

#### Mathematical Formulation:
$$\text{Status} = f(R_{24}, R_{72}, \text{Susceptibility})$$

Where:
* **$R_{24}$ (Daily Trigger Rainfall):** Cumulative rainfall in the last 24 hours ($\text{mm}$).
* **$R_{72}$ (Antecedent Moisture Loading):** Cumulative rainfall in the last 72 hours ($\text{mm}$).
* **Susceptibility:** National Landslide Susceptibility Mapping (NLSM) rating (Low, Moderate, High, Very High).

#### Operational Decision Matrix (Western Ghats Calibrated):
* **Green (Normal / No Alert):**  
  $R_{24} < 30\text{ mm}$ AND $R_{72} < 70\text{ mm}$
* **Yellow (Watch / Advisory):**  
  $R_{24} \ge 30\text{ mm}$ OR $R_{72} \ge 70\text{ mm}$ in High/Very High susceptibility zones.
* **Orange (Alert / Prepare):**  
  $R_{24} \ge 50\text{ mm}$ AND $R_{72} \ge 100\text{ mm}$ in Moderate-to-High susceptibility zones.
* **Red (Warning / Take Action):**  
  $R_{24} \ge 80\text{ mm}$ AND $R_{72} \ge 150\text{ mm}$ in High/Very High susceptibility zones.

---

### 2.2. India Meteorological Department (IMD) — Standard Precipitation Categorization

IMD defines standardized 24-hour rainfall classification used across all national disaster management bulletins:

| Classification | 24-Hour Rainfall Interval ($R_{24}$) | Operational Color Code |
| :--- | :--- | :--- |
| **Very Light to Light** | $0.1 - 15.5\text{ mm}$ | Green |
| **Moderate** | $15.6 - 64.4\text{ mm}$ | Green |
| **Heavy Rainfall** | $64.5 - 115.5\text{ mm}$ | Yellow (Be Aware) |
| **Very Heavy Rainfall** | $115.6 - 204.4\text{ mm}$ | Orange (Be Prepared) |
| **Extremely Heavy Rainfall** | $> 204.4\text{ mm}$ | Red (Take Action) |

---

### 2.3. Classical Hydrology — Antecedent Precipitation Index (API)

Established by Kohler & Linsley (1951), API is the international standard for estimating soil saturation in the absence of in-situ physical probes:

$$\text{API}_t = \sum_{i=1}^{N} k^i \cdot P_{t-i}$$

* **$P_{t-i}$:** Daily rainfall $i$ days prior to day $t$.
* **$k$:** Recession coefficient (typically $0.80 \le k \le 0.88$ for humid tropical regions like the Western Ghats).
* **$N$:** Lookback window (typically 15 to 30 days).

---

## 3. What Is `compound_saturation` in `FeatureSpec.v3`?

In our codebase (`src/disaster_intel/state/spec.py`), line 241 defines:

```python
DerivedFeature(
    name="compound_saturation",
    left="soil_moisture_rootzone__24h",
    right="precipitation__72h",
)
```

### The Computer Science / ML Justification:
1. **Polynomial Interaction Term (Feature Cross):**  
   In feature store design (e.g., Feast, Google Vertex, Uber Michelangelo), creating pairwise multiplicative terms ($x_i \cdot x_j$) is a standard feature engineering technique.
2. **Physics-Informed Feature Engineering:**  
   Geotechnical literature (Bogaard & Greco, 2018; Terzaghi Effective Stress) proves that slope failure requires *both* subsurface saturation *and* triggering storm downpours.
3. **Overcoming Sample Scarcity:**  
   Catastrophic slope failures (like Shirur or Wayanad) occur rarely in a single taluk. Deep neural networks struggle to approximate non-linear multiplications from limited training examples. By providing `compound_saturation` as a pre-computed feature, we inject domain knowledge directly into the GNN node feature vector.

---

## 4. Benchmark Validation Against Historical Ground Truth

During the July 2024 Shirur catastrophic landslide, our feature store recorded the following live values at the focal H3 cell (`8760b526dffffff`):

1. **IMD Standard:**  
   - $R_{24} = 185.2\text{ mm}$ $\rightarrow$ **VERY HEAVY RAINFALL (Orange/Red Threshold)**
2. **GSI Standard:**  
   - $R_{24} = 185.2\text{ mm} > 80\text{ mm}$  
   - $R_{72} = 418.6\text{ mm} > 150\text{ mm}$  
   - Terrain Slope: $28.4^\circ$, GSI Susceptibility: $0.85$ (High Hazard)  
   - $\rightarrow$ **GSI CRITICAL RED WARNING BREACHED**
3. **Feature Vector Representation:**  
   - `precipitation__72h`: $418.6\text{ mm}$
   - `soil_moisture_rootzone__24h`: $0.468\text{ m}^3/\text{m}^3$ (near porosity limit)
   - `compound_saturation`: $195.9\text{ m}^3/\text{m}^3 \cdot \text{mm}$ (Peak non-linear feature spike)

Both the official government thresholds and our feature store vectors confirm extreme catastrophe.

---

## 5. Viva Voce Defense Script (Q&A)

**Q: Where did you get the formula for compound saturation? Did GSI publish it?**  
> *"No, Sir. GSI did not publish `compound_saturation` as a warning index, and we do not claim it is an early warning score. In Phase 1, our deliverable is an 86-dimensional Feature Store. In standard machine learning practice, `compound_saturation` is an engineered **interaction term (feature cross)** between 24-hour rootzone soil moisture and 72-hour cumulative precipitation.*  
> *For operational alert baselines, we evaluate against GSI's official Bilinear Rainfall Threshold ($R_{24} \ge 80\text{ mm}$, $R_{72} \ge 150\text{ mm}$) and IMD's standard precipitation tiers. In Phase 2, our Spatio-Temporal GNN will learn the predictive weights across all 86 features directly from historical landslide event catalogs."*

**Q: Why not just use the GSI rule instead of building a whole GNN system?**  
> *"Because static GSI rainfall thresholds suffer from high false-alarm rates. A regional threshold of 150 mm/72h flags an entire district of 5,000 square kilometers, triggering warning fatigue. Our H3-based framework localizes data to individual 5 km² hexagons and fuses satellite SAR microwave backscatter, INSAT-3D convective cooling rates, and river hydrographs, giving granular, compound-hazard intelligence that a simple 2D rainfall threshold cannot provide."*

