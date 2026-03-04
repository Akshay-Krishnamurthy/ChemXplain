# 🧬 ChemXplain: Explainable AI for Autonomous Lead Optimization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-GPU_Accelerated-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/RDKit-Cheminformatics-09A655?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Version-V13-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Publication_Ready-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <b>A closed-loop, human-in-the-loop computational chemistry platform bridging Explainable AI (XAI) and Generative Drug Design.</b><br>
  <i>From liability identification to synthesizable bioisostere generation — all in one pipeline.</i>
</p>

---

## 📌 Table of Contents
1. [What is ChemXplain?](#what-is-chemxplain)
2. [Key Features](#key-features)
3. [Pipeline Overview](#pipeline-overview)
4. [Algorithms & Mathematical Foundations](#algorithms--mathematical-foundations)
5. [Installation (GPU)](#installation--reproducibility-gpu)
6. [Usage](#usage)
7. [Adding Custom Properties](#adding-custom-properties-67th-property)
8. [Acknowledgments & References](#acknowledgments--references)
9. [License](#license)

---

## 🔬 What is ChemXplain?

ChemXplain is a **research-grade, autonomous lead optimization platform** that answers the most critical question in early-stage drug discovery:

> *"My lead compound has poor ADMET properties — which exact atoms are responsible, and how do I fix them?"*

Traditional black-box GNN models predict toxicity but cannot explain *why*. ChemXplain solves this by applying **Occlusion Sensitivity Analysis** (adapted from computer vision) directly to molecular graphs, pinpointing liability-driving atoms at the node, bond, and ring-motif level — then uses **CReM (Chemically Reasonable Mutations)** to exhaustively replace those atoms with validated pharmaceutical fragments from ChEMBL.

The result: a scientifically rigorous, fully explainable path from a toxic lead compound to a synthesizable, optimized bioisostere — in minutes.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **pH 7.4 Correction** | Protonates/deprotonates using `dimorphite_dl` before any prediction — mimicking physiological conditions |
| **66 ADMET Properties** | Deep learning predictions via `admet_ai` (hERG, DILI, Ames, BBB, Solubility, CYP450s, and 60 more) |
| **Applicability Domain** | Tanimoto similarity check against reference pharmaceutical space — alerts on "alien" chemotypes |
| **PAINS Screening** | Automatic Pan-Assay Interference Compound detection (Baell & Holloway, 2010) |
| **3D Occlusion XAI** | Interactive py3Dmol rendering — atom spheres sized/colored by occlusion impact |
| **2D Sharp Map** | RDKit SVG heat map — atom-level color gradient (green → red) of causal impact |
| **Similarity Contour Map** | Publication-quality Gaussian probability cloud over the 2D structure (Riniker & Landrum, 2013) |
| **Bond Ablation Map** | Removes individual bonds to identify critical connectivity — not just atoms |
| **Ring/Motif Ablation** | Masks entire ring systems simultaneously to reveal macro-structural contributions |
| **Multi-Property Trade-Off Quadrant** | Scatter plot identifying atoms that simultaneously drive multiple liabilities |
| **Point Mutation Array** | Heatmap of N/O/F/S bioisosteric swaps across top-5 liability atoms |
| **⚛️ Gasteiger Charge Map** | *(V13)* Atom-level partial charge contour overlay — electron-poor (red) vs electron-rich (blue) atoms (Gasteiger & Marsili, 1980) |
| **⚛️ Crippen LogP Map** | *(V13)* Per-atom logP contribution contour overlay — lipophilic hotspots (pink) vs hydrophilic regions (green) (Wildman & Crippen, 1999) |
| **CReM Exhaustive Generation** | Mines ChEMBL for chemically valid, synthesizable Matched Molecular Pairs |
| **Lipinski Ro5 Filter** | Optional drug-likeness gating (deactivatable for PROTACs/macrocycles) |
| **SA Score** | Ertl & Schuffenhauer (2009) synthetic accessibility scoring on all generated molecules |
| **Gemini 2.5 Flash Reports** | LLM-generated pharmacological rationale and comparative structural analysis |
| **Human-in-the-Loop Gate** | Interactive atom selector — override AI targets with expert chemical intuition |
| **GPU Acceleration** | CUDA auto-detection — all 66 models run on GPU if available |

---

## ⚙️ Pipeline Overview

```
SMILES Input
    │
    ▼
┌─────────────────────────────┐
│  1. pH 7.4 Correction       │  dimorphite_dl
│     (Physiological Species) │
└─────────────┬───────────────┘
              │
    ▼
┌─────────────────────────────┐
│  2. Safety Pre-Screening    │  Applicability Domain (Tanimoto)
│     PAINS Filter            │  RDKit FilterCatalog
└─────────────┬───────────────┘
              │
    ▼
┌─────────────────────────────┐
│  3. ADMET Fingerprinting    │  admet_ai (66 GNN models)
│     66 Properties Predicted │  GPU Accelerated
└─────────────┬───────────────┘
              │
    ▼
┌─────────────────────────────┐
│  4. Occlusion Sensitivity   │  Node / Bond / Motif Ablation
│     XAI Analysis            │  3D View, 2D Map, Contour, Quadrant
└─────────────┬───────────────┘
              │
    ▼
┌─────────────────────────────┐
│  4b. Physicochemical Maps   │  Gasteiger Charges + Crippen LogP    ← V13
│      (Cross-validation)     │  RDKit SimilarityMaps C++ backend
└─────────────┬───────────────┘
              │
    ▼  [Human Decision Gate]
┌─────────────────────────────┐
│  5. CReM Bioisostere        │  ChEMBL-sourced fragment library
│     Generation              │  Exhaustive, radius=1 context
└─────────────┬───────────────┘
              │
    ▼
┌─────────────────────────────┐
│  6. Re-scoring & Ranking    │  ADMET-AI + SA Score on all mutants
│     Lipinski Ro5 Filter     │  Download CSV of all molecules
└─────────────┬───────────────┘
              │
    ▼
┌─────────────────────────────┐
│  7. Gemini LLM Report       │  Structural change rationale
│     Pharmacological Summary │  Synthesis feasibility commentary
└─────────────────────────────┘
```

---

## 🧮 Algorithms & Mathematical Foundations

### 1. Occlusion Sensitivity Analysis (Feature Ablation)

ChemXplain adapts the occlusion sensitivity technique of Zeiler & Fergus (2014) — originally developed for convolutional neural networks in computer vision — to molecular graphs. Each heavy atom is systematically masked (atomic number set to 0), the ADMET property is re-predicted, and the causal impact is computed:

```
W(i) = f_P( G_baseline ) - f_P( G \ {v_i} )
```

Where:
- `f_P` is the ADMET-AI graph neural network predicting property `P`
- `G_baseline` is the original molecular graph
- `G \ {v_i}` is the graph with atom `i` replaced by a wildcard node (atomic number 0)
- `W(i)` is the **directional causal weight** of atom `i`

**Interpretation:**
- `W(i) > 0` → removing atom `i` **lowers** the score → atom is a **liability driver**
- `W(i) < 0` → removing atom `i` **raises** the score → atom is a **protective suppressor**

Normalized absolute weights are used for visualization:

```
W_norm(i) = ( |W(i)| - min|W| ) / ( max|W| - min|W| )    ∈ [0, 1]
```

This normalized score drives the 2D Sharp Map color gradient and 3D sphere sizing.

---

### 2. Bond Ablation Analysis

Each bond is removed entirely (not just an atom) to measure connectivity-level impact:

```
B(k) = | f_P( G_baseline ) - f_P( G \ {e_k} ) |
```

Where `e_k` is bond `k`. Bond weights are rendered as a YlOrRd heat map over the 2D structure, identifying critical bonds (e.g., linker bonds between pharmacophore fragments).

---

### 3. Ring / Motif Ablation

All atoms in a ring system `R_j` are simultaneously masked:

```
M(j) = | f_P( G_baseline ) - f_P( G \ R_j ) |
```

This reveals macro-structural contributions that are invisible to single-atom occlusion (e.g., "the entire piperidine ring drives hERG, not any single atom").

---

### 4. Multi-Property Trade-Off Quadrant

Two properties `P1` and `P2` are simultaneously ablated per atom, producing a 2D vector per atom:

```
v(i) = ( W_P1(i),  W_P2(i) )
```

Atoms plotted in the **bottom-left quadrant** have `W_P1(i) < 0` AND `W_P2(i) < 0`, meaning their removal suppresses **both** liabilities simultaneously — these are the highest-priority medicinal chemistry targets.

---

### 5. Matched Molecular Pair (MMP) Generation via CReM

ChemXplain uses the CReM (Chemically Reasonable Mutations) framework rather than hallucination-prone VAEs or GANs. For each target atom `v_t` identified by XAI, CReM performs a database lookup against a pre-fragmented ChEMBL library:

```
Mutants = { G' | G' = G[ v_t ← f_c ],  f_c ∈ F_ChEMBL,  context(f_c, G, v_t, radius=1) }
```

The `radius=1` constraint enforces that only fragments chemically compatible with the **immediate local environment** of the target atom are accepted — preserving exact valency, hybridization, and ring geometry.

---

### 6. Synthetic Accessibility (SA) Score

Every generated molecule is evaluated using the Ertl & Schuffenhauer SA Score:

```
SA = fragment_score - complexity_penalty
```

Where `fragment_score` is based on the frequency of molecular fragments in a 1M-molecule reference set from PubChem, and `complexity_penalty` accounts for ring complexity, stereocenters, and macrocycle structure. Scores range from 1 (trivially synthesizable) to 10 (virtually impossible to synthesize).

---

### 7. Applicability Domain (Tanimoto Similarity)

Morgan fingerprints (ECFP4, radius=2, 2048 bits) are computed. The maximum Tanimoto similarity against a curated reference set of known pharmaceuticals is reported:

```
AD_score = max  T( FP_query,  FP_r )
           r∈R
```

- `AD > 0.3`  → High confidence (scaffold is pharmaceutical-like)
- `0.15 < AD ≤ 0.3` → Moderate confidence
- `AD ≤ 0.15` → Alien chemotype — predictions may be unreliable

---

## 💻 Installation & Reproducibility (GPU)

These exact steps have been validated for **NVIDIA GPUs with CUDA 12.1**. Follow them sequentially to avoid dependency conflicts.

### Prerequisites
- **OS:** Linux (Ubuntu 20.04+ recommended) or Windows with WSL2
- **GPU:** NVIDIA GPU with CUDA 12.1 compatible driver
- **RAM:** ≥ 16 GB recommended
- **Disk:** ≥ 5 GB free (for conda env + ChEMBL database)

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ChemXplain.git
cd ChemXplain
```

---

### Step 2: Create the Conda Environment

**Option A — Automatic (Recommended):**
```bash
conda env create -f environment.yml
conda activate chemxplain
```

**Option B — Manual step-by-step:**
```bash
# Create Python 3.11 environment
conda create -n chemxplain python=3.11 -y
conda activate chemxplain

# Install RDKit via conda (ensures correct C++ bindings)
conda install -c conda-forge rdkit=2024.03.1 -y

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all remaining Python dependencies
pip install \
  admet-ai==1.0.0 \
  crem==0.2.10 \
  dimorphite-dl==2.1.2 \
  streamlit==1.36.0 \
  py3Dmol \
  stmol \
  seaborn \
  matplotlib \
  pandas \
  numpy \
  google-genai
```

---

### Step 3: Download the ChEMBL Fragment Database

ChemXplain requires the curated CReM ChEMBL fragment database. Run the setup script:

```bash
chmod +x setup.sh
bash setup.sh
```

Or manually:
```bash
wget "https://zenodo.org/records/16909329/files/chembl22_sa2_hac12.db.gz?download=1" \
     -O chembl22_sa2_hac12.db.gz
gunzip chembl22_sa2_hac12.db.gz
```

The database (`chembl22_sa2_hac12.db`, ~150 MB) contains **pre-fragmented ChEMBL22 structures** filtered to synthetic accessibility ≤ 2 and max 12 heavy atoms per fragment — the highest-quality fragment space available for medicinal chemistry.

---

### Step 4: Obtain a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a free API key
3. Paste it into the sidebar when running the app

---

### Exact Dependency Versions

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core runtime |
| RDKit | 2024.03.1 (conda) | Cheminformatics engine |
| PyTorch | Latest CUDA 12.1 | GPU acceleration |
| admet-ai | 1.0.0 | 66 ADMET GNN models |
| crem | 0.2.10 | MMP generation |
| dimorphite-dl | 2.1.2 | pH correction |
| streamlit | 1.36.0 | Web UI |
| py3Dmol | Latest | 3D molecular visualization |
| stmol | Latest | Streamlit-py3Dmol bridge |
| google-genai | Latest | Gemini 2.5 Flash API |
| seaborn | Latest | Statistical visualization |
| matplotlib | Latest | Plotting engine |
| pandas | Latest | DataFrame operations |
| numpy | Latest | Numerical computing |

---

## 🖥️ Usage

### Launch the Application
```bash
conda activate chemxplain
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`

### Workflow

**Step 1 — Diagnostics:**
1. Enter your **Gemini API Key** in the sidebar
2. Verify the CReM database path (default: `chembl22_sa2_hac12.db`)
3. Input your **target SMILES** (Terfenadine is pre-loaded as a demonstration)
4. Select your **Primary property** (XAI optimization target) and **Secondary property** (trade-off axis)
5. Click **🔬 Step 1: Run Full Diagnostics**

The app will:
- Correct the molecule to its dominant species at pH 7.4
- Screen for PAINS structural alerts
- Predict all 66 ADMET properties
- Run Occlusion XAI across all 8 visualization tabs
- Generate a Gemini pharmacological report

**Step 2 — Human Decision Gate:**
- Review the XAI maps and AI report
- Decide whether to proceed with generative optimization

**Step 3 — Generation:**
1. Select atoms to mutate (AI suggests top-3; you can override)
2. Choose optimization direction (Minimize / Maximize)
3. Toggle Lipinski Ro5 filter (uncheck for PROTACs/macrocycles)
4. Click **🚀 Step 3: Launch Exhaustive Generative Loop**
5. Download the full CSV of ranked molecules + top-5 structure grid

---

## 🔧 Adding Custom Properties (67th Property)

Because ChemXplain dynamically reads column names from the predictions DataFrame, you can inject a custom property (a QED score, a binding affinity model, or any `sklearn`/`PyTorch` model output) and it will **automatically appear in all dropdowns, XAI maps, and generative loops** without any additional code.

### Step 1: Add the wrapper function (in Section 2 of `app.py`)
```python
from rdkit.Chem import QED

def get_extended_predictions(smiles_list, model):
    # Get the standard 66 properties
    preds_df = model.predict(smiles=smiles_list)
    
    # Calculate your 67th custom property
    custom_scores = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            # --- REPLACE THIS LINE with your custom model ---
            score = QED.default(mol)
            custom_scores.append(score)
        except Exception:
            custom_scores.append(float('nan'))
    
    # Inject as new column — app auto-discovers it
    preds_df["QED_DrugLikeness_67"] = custom_scores
    return preds_df
```

### Step 2: Replace `admet_model.predict(...)` with `get_extended_predictions(...)` at all 6 call sites in `app.py`.

Your 67th property will now appear in the Selectbox dropdowns, be ablated by the Occlusion XAI engine, and drive the CReM generative loop.

---

## 📜 Acknowledgments & References

ChemXplain stands entirely on the shoulders of open-source science. If you use this tool in your research, please cite the following foundational works:

### Core Methodology
1. **Occlusion Sensitivity (XAI backbone):**
   Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *European Conference on Computer Vision (ECCV)*. https://doi.org/10.1007/978-3-319-10590-1_53

2. **ADMET-AI (property prediction engine):**
   Swanson, K., et al. (2023). ADMET-AI: A machine learning ADMET platform. *Journal of Chemical Information and Modeling*. https://doi.org/10.1021/acs.jcim.3c01064

3. **CReM (bioisostere generation):**
   Polishchuk, P. (2020). CReM: chemically reasonable mutations framework for structure generation. *Journal of Cheminformatics*, 12, 28. https://doi.org/10.1186/s13321-020-00431-w

4. **Dimorphite-DL (pH correction):**
   Ropp, P. J., et al. (2019). Dimorphite-DL: An open-source program for enumerating the ionization states of drug-like small molecules. *Journal of Cheminformatics*, 11, 14. https://doi.org/10.1186/s13321-019-0336-9

### Visualization
5. **Similarity Contour Maps:**
   Riniker, S., & Landrum, G. A. (2013). Similarity maps — a visualization strategy for molecular fingerprints. *Journal of Cheminformatics*, 5, 43. https://doi.org/10.1186/1758-2946-5-43

6. **RDKit (cheminformatics engine):**
   Landrum, G., et al. RDKit: Open-source cheminformatics. https://www.rdkit.org

### Safety Filters
7. **PAINS filters:**
   Baell, J. B., & Holloway, G. A. (2010). New substructure filters for removal of pan assay interference compounds (PAINS). *Journal of Medicinal Chemistry*, 53(7), 2719–2740. https://doi.org/10.1021/jm901137j

### Scoring
8. **SA Score:**
   Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *Journal of Cheminformatics*, 1, 8. https://doi.org/10.1186/1758-2946-1-8

9. **Lipinski Ro5:**
   Lipinski, C. A., et al. (2001). Experimental and computational approaches to estimate solubility and permeability in drug discovery. *Advanced Drug Delivery Reviews*, 46(1-3), 3–26. https://doi.org/10.1016/S0169-409X(00)00129-0

### AI Report Generation
10. **Google Gemini 2.5 Flash:**
    Google DeepMind. (2024). Gemini: A family of highly capable multimodal models. https://deepmind.google/technologies/gemini/

### Physicochemical Maps (V13)
11. **Gasteiger-Marsili Partial Charges:**
    Gasteiger, J., & Marsili, M. (1980). Iterative partial equalization of orbital electronegativity — a rapid access to atomic charges. *Tetrahedron*, 36(22), 3219–3228. https://doi.org/10.1016/0040-4020(80)80168-2

12. **Crippen logP Atom Contributions:**
    Wildman, S. A., & Crippen, G. M. (1999). Prediction of physicochemical parameters by atomic contributions. *Journal of Chemical Information and Computer Sciences*, 39(5), 868–873. https://doi.org/10.1021/ci990307l

13. **RDKit SimilarityMaps C++ Backend:**
    Landrum, G. (2020). Similarity maps with the new drawing code. *RDKit Blog*. https://greglandrum.github.io/rdkit-blog/posts/2020-01-03-similarity-maps-with-new-drawing-code.html

---

## ⚠️ Checklist Before Publishing

- [ ] Remove any hardcoded API keys from `app.py` (currently safe — key is taken via sidebar)
- [ ] Add screenshots to `assets/` folder and link them in this README
- [ ] Run `bash setup.sh` on a fresh machine to verify database download works
- [ ] Test with `conda env create -f environment.yml` on a second machine
- [ ] Add your institution/author name to the license

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

<p align="center">
  <i>Built with ❤️ for the computational drug discovery community.</i><br>
  <i>If this tool helped your research, please ⭐ the repository.</i>
</p>
