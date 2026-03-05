# 🧬 ChemXplain: Explainable AI for Autonomous Lead Optimization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-GPU_Accelerated-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/RDKit-2024.03.1-09A655?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/REINVENT4-4.7.15-9B59B6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Version-V16-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>End-to-end platform for library screening, explainable ADMET profiling, and generative lead optimisation.</b><br>
  <i>Powered by Occlusion XAI · ADMET-AI · CReM · REINVENT4 · Gemini 2.5 Flash</i>
</p>

---

## 📌 Table of Contents

1. [What is ChemXplain?](#-what-is-chemxplain)
2. [Key Features](#-key-features)
3. [Two Workflow Modes](#-two-workflow-modes)
4. [Pipeline Overview](#-pipeline-overview)
5. [Algorithms & Mathematical Foundations](#-algorithms--mathematical-foundations)
6. [Installation — Main App (GPU)](#-installation--main-app-gpu)
7. [REINVENT4 Setup — Separate Environment](#-reinvent4-setup--separate-environment)
8. [Troubleshooting](#-troubleshooting)
9. [Usage](#-usage)
10. [Adding Custom Properties](#-adding-custom-properties)
11. [Acknowledgments & References](#-acknowledgments--references)
12. [Author](#-author)
13. [License](#-license)

---

## 🔬 What is ChemXplain?

ChemXplain is a **research-grade, autonomous drug discovery platform** that answers the two most critical questions in early-stage drug discovery:

> *"Which molecules in my library are the best starting points?"*
> *"My lead compound is toxic — which exact atoms are responsible, and how do I fix them?"*

Traditional black-box GNN models predict toxicity but cannot explain *why*. ChemXplain solves this by applying **Occlusion Sensitivity Analysis** (adapted from computer vision) directly to molecular graphs, pinpointing liability-driving atoms at the node, bond, and ring-motif level. It then drives two state-of-the-art generative engines — **CReM** (fragment-based, deterministic) and **REINVENT4** (transformer-based, AstraZeneca's production tool) — to replace those atoms with validated pharmaceutical replacements.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **Library EDA (Mode A)** | Upload CSV/Excel → PCA, UMAP, physicochemical distributions, scaffold diversity, Tanimoto heatmap, LogP-MW landscape, ADMET heatmap |
| **Batch ADMET Profiling** | Score entire libraries with all 66 ADMET models → master CSV download |
| **pH 7.4 Correction** | Protonates/deprotonates using `dimorphite_dl` before any prediction |
| **66 ADMET Properties** | Deep learning predictions via `admet_ai` (hERG, DILI, Ames, BBB, Solubility, CYP450s, and 60 more) |
| **Applicability Domain** | Tanimoto similarity check — alerts on out-of-distribution chemotypes |
| **PAINS Screening** | Automatic Pan-Assay Interference Compound detection (Baell & Holloway, 2010) |
| **3D Occlusion XAI** | Interactive py3Dmol — atom spheres sized/colored by causal impact |
| **2D Sharp Map** | RDKit SVG heat map — atom-level YlOrRd gradient |
| **Similarity Contour Map** | Publication-quality Gaussian probability cloud (Riniker & Landrum, 2013) |
| **Bond Ablation Map** | Removes individual bonds to identify critical connectivity |
| **Ring/Motif Ablation** | Masks entire ring systems — reveals macro-structural contributions |
| **Trade-Off Quadrant** | Identifies atoms that simultaneously drive multiple liabilities |
| **Point Mutation Array** | Heatmap of N/O/F/S bioisosteric swaps across top-5 liability atoms |
| **Gasteiger Charge Map** | Atom-level electron density overlay (Gasteiger & Marsili, 1980) |
| **Crippen LogP Map** | Per-atom logP contribution overlay (Wildman & Crippen, 1999) |
| **CReM Generation** | Exhaustive ChEMBL-sourced Matched Molecular Pairs, radius=1 context |
| **REINVENT4 Generation** | AstraZeneca's Mol2Mol transformer with property conditioning tokens |
| **Bioisostere Fallback** | Rule-based cascade when CReM finds no replacements |
| **Lipinski Ro5 Filter** | Optional drug-likeness gating (deactivatable for PROTACs/macrocycles) |
| **SA Score** | Ertl & Schuffenhauer (2009) synthetic accessibility on all generated molecules |
| **Gemini 2.5 Flash Reports** | LLM-generated pharmacological rationale and comparative structural analysis |
| **Human-in-the-Loop Gate** | Interactive atom selector — override AI targets with expert intuition |
| **GPU Acceleration** | CUDA auto-detection — all 66 models run on GPU if available |

---

## 🔀 Two Workflow Modes

```
MODE A — Library Analysis
┌─────────────────────────────────────────────────────────┐
│  Upload CSV/Excel with SMILES column                    │
│  → Validate & compute physicochemical properties        │
│  → PCA / UMAP chemical space visualisation              │
│  → Scaffold diversity & Bemis-Murcko analysis           │
│  → Tanimoto similarity heatmap                          │
│  → Batch 66-property ADMET profiling → CSV download     │
│  → Gemini AI narrative library summary                  │
│  → Select one molecule → hand off to Mode B             │
└─────────────────────────────────────────────────────────┘

MODE B — Single Molecule Deep Dive
┌─────────────────────────────────────────────────────────┐
│  Input SMILES (or receive from Mode A)                  │
│  → pH 7.4 correction + AD check + PAINS filter          │
│  → 66 ADMET predictions + Radar chart                   │
│  → Occlusion XAI: 9 visualisation tabs                  │
│  → Gemini mechanistic diagnostic report                  │
│  → Engine selector: CReM+Fallback  OR  REINVENT4        │
│  → ADMET re-scoring + Top-5 grid + Gemini analysis      │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ Pipeline Overview

```
SMILES Input / Library Upload
          │
          ▼
┌─────────────────────────────┐
│  1. pH 7.4 Correction       │  dimorphite_dl
│     Applicability Domain    │  Tanimoto vs. reference space
│     PAINS Filter            │  RDKit FilterCatalog
└─────────────┬───────────────┘
              │
          ▼
┌─────────────────────────────┐
│  2. ADMET Fingerprinting    │  admet_ai (66 GNN models)
│     66 Properties           │  GPU Accelerated
└─────────────┬───────────────┘
              │
          ▼
┌─────────────────────────────┐
│  3. Occlusion XAI           │  Node / Bond / Motif Ablation
│     9 Visualisation Tabs    │  3D · 2D · Contour · Quadrant
└─────────────┬───────────────┘
              │
          ▼
┌─────────────────────────────┐
│  4. Physicochemical Maps    │  Gasteiger Charges (RdBu_r)
│     Cross-validation Layer  │  Crippen LogP (PiYG_r)
└─────────────┬───────────────┘
              │
          ▼  [Human Decision Gate]
┌─────────────────────────────┐
│  5. Generative Engine       │  ── CReM (ChEMBL, radius=1)
│                             │  ── REINVENT4 (Mol2Mol transformer)
│                             │  ── Bioisostere Fallback cascade
└─────────────┬───────────────┘
              │
          ▼
┌─────────────────────────────┐
│  6. Re-scoring & Ranking    │  ADMET-AI + SA Score
│     Lipinski Ro5 Filter     │  Download full CSV
└─────────────┬───────────────┘
              │
          ▼
┌─────────────────────────────┐
│  7. Gemini LLM Report       │  Structural change rationale
│     3-Part Analysis         │  Synthesis feasibility
└─────────────────────────────┘
```

---

## 🧮 Algorithms & Mathematical Foundations

### 1. Occlusion Sensitivity Analysis (Feature Ablation)

ChemXplain adapts the occlusion sensitivity technique of Zeiler & Fergus (2014) to molecular graphs. Each heavy atom is systematically masked (atomic number set to 0), the ADMET property is re-predicted, and the causal impact is computed:

```
W(i) = f_P( G_baseline ) - f_P( G \ {v_i} )
```

- `W(i) > 0` → atom i is a **liability driver** (removing it lowers toxicity)
- `W(i) < 0` → atom i is a **protective suppressor** (removing it raises toxicity)

Normalized absolute weights for visualization:

```
W_norm(i) = ( |W(i)| - min|W| ) / ( max|W| - min|W| )    ∈ [0, 1]
```

### 2. Bond Ablation

```
B(k) = | f_P( G_baseline ) - f_P( G \ {e_k} ) |
```

### 3. Ring / Motif Ablation

```
M(j) = | f_P( G_baseline ) - f_P( G \ R_j ) |
```

### 4. Multi-Property Trade-Off Quadrant

```
v(i) = ( W_P1(i),  W_P2(i) )
```

Atoms in the **bottom-left quadrant** (W_P1 < 0 AND W_P2 < 0) simultaneously suppress both liabilities — highest-priority targets.

### 5. MMP Generation via CReM

```
Mutants = { G' | G' = G[ v_t <- f_c ],  f_c in F_ChEMBL,  context(f_c, G, v_t, radius=1) }
```

### 6. SA Score

```
SA = fragment_score - complexity_penalty
```

Scores 1 (trivially synthesizable) → 10 (virtually impossible).

### 7. Applicability Domain (Tanimoto)

```
AD_score = max  T( FP_query,  FP_r )
           r∈R
```

- `AD > 0.30` → High confidence
- `0.15 < AD ≤ 0.30` → Moderate confidence
- `AD ≤ 0.15` → Alien chemotype — predictions unreliable

### 8. PCA Chemical Space

Morgan fingerprints (ECFP4, radius=2, 2048 bits) are computed for all library molecules. StandardScaler normalizes the feature matrix, then PCA reduces to 2 components. Optionally UMAP provides non-linear manifold projection.

---

## 💻 Installation — Main App (GPU)

> **Tested on:** Ubuntu 20.04/22.04, NVIDIA GPU, CUDA 12.1, Conda 23+

### Step 1: Clone the Repository

```bash
git clone https://github.com/Akshay-Krishnamurthy/ChemXplain.git
cd ChemXplain
```

### Step 2: Create the Conda Environment

```bash
conda create -n chemxplain python=3.11 -y
conda activate chemxplain
```

### Step 3: Install RDKit (via conda — critical for C++ bindings)

```bash
conda install -c conda-forge rdkit=2024.03.1 -y
```

> **Why conda and not pip?** RDKit requires compiled C++ extensions. The conda-forge build is pre-compiled and avoids runtime crashes.

### Step 4: Install PyTorch with CUDA

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Install all Python dependencies

```bash
python -m pip install \
  admet-ai \
  crem \
  dimorphite-dl \
  streamlit==1.36.0 \
  py3Dmol \
  stmol \
  seaborn \
  matplotlib \
  pandas \
  numpy \
  google-genai \
  scikit-learn \
  umap-learn \
  ipython_genutils \
  ipyspeck \
  ipywidgets
```

> **Important:** Always use `python -m pip` (not just `pip`) to ensure packages install into the active conda environment.

### Step 6: Download the ChEMBL Fragment Database

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

### Step 7: Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a free API key
3. Paste it into the sidebar when running the app

### Step 8: Launch the App

```bash
conda activate chemxplain
python -m streamlit run app.py
```

Opens at `http://localhost:8501`

---

### Exact Dependency Versions

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core runtime |
| RDKit | 2024.03.1 (conda-forge) | Cheminformatics engine |
| PyTorch | Latest (CUDA 12.1) | GPU acceleration |
| admet-ai | Latest | 66 ADMET GNN models |
| crem | Latest | MMP generation |
| dimorphite-dl | Latest | pH correction |
| streamlit | 1.36.0 | Web UI |
| scikit-learn | Latest | PCA / StandardScaler |
| umap-learn | Latest | UMAP chemical space |
| py3Dmol | Latest | 3D molecular visualization |
| stmol | Latest | Streamlit-py3Dmol bridge |
| google-genai | Latest | Gemini 2.5 Flash API |
| seaborn | Latest | Statistical visualization |
| matplotlib | Latest | Plotting engine |
| pandas | Latest | DataFrame operations |
| numpy | Latest | Numerical computing |

---

## 🤖 REINVENT4 Setup — Separate Environment

> **Why a separate environment?**
> REINVENT4 requires `chemprop < 1.6` (old API), while ADMET-AI requires `chemprop >= 2.2` (new API). These two versions are **fundamentally incompatible** — they cannot share the same Python environment. ChemXplain solves this by running REINVENT4 as an external subprocess, so each tool uses its own isolated environment with no conflicts.

### Step 1: Create a dedicated REINVENT4 environment

```bash
conda create -n reinvent4 python=3.10 -y
conda activate reinvent4
```

> Python 3.10 is required — REINVENT4 does not support 3.11+ due to dependency constraints.

### Step 2: Clone and install REINVENT4

```bash
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4
pip install -e .
```

### Step 3: Verify installation

```bash
reinvent --help
```

You should see the REINVENT4 help output. If successful:

### Step 4: Get the executable path

```bash
which reinvent
```

Copy the full path — it will look like:
```
/home/YOUR_USERNAME/anaconda3/envs/reinvent4/bin/reinvent
```

### Step 5: Locate the Mol2Mol prior files

```bash
ls REINVENT4/priors/
```

You will see several `.prior` files. Choose one based on your task:

| Prior file | Best for |
|---|---|
| `mol2mol_similarity.prior` | Standard lead optimization (recommended) |
| `mol2mol_mmp.prior` | Strict Matched Molecular Pair generation |
| `mol2mol_scaffold_generic.prior` | Scaffold hopping — change the core ring system |
| `mol2mol_high_similarity.prior` | Conservative changes — keep close to parent |
| `mol2mol_medium_similarity.prior` | Moderate structural variation |

Copy the full path to your chosen prior:
```
/home/YOUR_USERNAME/AkshayH/chemXplain/REINVENT4/priors/mol2mol_similarity.prior
```

### Step 6: Configure in the ChemXplain UI

1. Switch back to your main environment and run the app:
```bash
conda activate chemxplain
python -m streamlit run app.py
```

2. In the app, scroll to **Step 3 — Generative Engine**
3. Select **🤖 REINVENT4**
4. Open **⚙️ REINVENT4 Settings**
5. Paste your paths:
   - `reinvent executable`: `/home/YOUR_USERNAME/anaconda3/envs/reinvent4/bin/reinvent`
   - `Mol2Mol prior (.prior)`: `/home/YOUR_USERNAME/.../priors/mol2mol_similarity.prior`

### Step 7: Configure Property Conditioning Tokens (optional)

REINVENT4's Mol2Mol transformer supports **property condition tokens** that steer generation toward desired physicochemical profiles. In the UI you can set:

| Token | Controls | Options |
|---|---|---|
| Solubility | Aqueous solubility direction | `no_change`, `low->high`, `high->low` |
| Clearance | Hepatic clearance direction | `no_change`, `low->high`, `high->low` |
| LogD bin | Target lipophilicity range | 80+ discrete bins (e.g. `LogD_(1.9, 2.1]`) |

---

## 🔧 Troubleshooting

### `ImportError: cannot import name 'load_model' from 'chemprop.models'`

REINVENT4 was accidentally installed into your `chemxplain` environment and downgraded chemprop. Fix:

```bash
conda activate chemxplain
pip uninstall reinvent -y
python -m pip install --upgrade admet-ai
```

---

### `❌ Cannot find Mol2Mol prior file`

The prior path in the UI is wrong. Run this to find the correct path:

```bash
conda activate reinvent4
ls /home/$(whoami)/*/REINVENT4/priors/
```

Paste the full absolute path into the REINVENT4 Settings box.

---

### `ValidationError: run_type — Field required`

The TOML config generated for REINVENT4 has a structural issue. This is auto-handled in V16. If it persists, ensure you are running the latest `app.py` from this repository.

---

### `UMAP not available — install umap-learn`

```bash
conda activate chemxplain
python -m pip install umap-learn
```

---

### `stmol` / `py3Dmol` not rendering in browser

```bash
python -m pip install --upgrade stmol py3Dmol ipython_genutils ipyspeck ipywidgets
```

---

### App crashes on CPU-only machine

The app fully supports CPU mode. ADMET-AI will auto-detect no GPU and run on CPU — it will be slower but produce identical results.

---

## 🖥️ Usage

### Mode A — Library Analysis

1. Select **📚 Library Analysis** in the sidebar
2. Upload a CSV or Excel file with a `SMILES` column
3. The app will auto-compute physicochemical properties and render all EDA plots
4. Click **Run Batch ADMET** to score all molecules with all 66 models
5. Download the master CSV
6. Click any molecule to hand it off to Mode B for deep dive

### Mode B — Single Molecule Deep Dive

1. Select **🔬 Single Molecule Deep Dive** in the sidebar
2. Enter a SMILES string (Terfenadine is pre-loaded as a demonstration)
3. Select Primary property (XAI target) and Secondary property (trade-off axis)
4. Click **🔬 Step 1: Run Full Diagnostics**
5. Review the 9 XAI visualisation tabs and AI report
6. Choose **CReM + Fallback** or **REINVENT4** as generative engine
7. Configure atom targets and filters
8. Launch generation and download ranked molecules

---

## 🔧 Adding Custom Properties

Because ChemXplain dynamically reads column names from the predictions DataFrame, you can inject any custom property and it will automatically appear in all dropdowns, XAI maps, and generative loops.

```python
from rdkit.Chem import QED

def get_extended_predictions(smiles_list, model):
    preds_df = model.predict(smiles=smiles_list)
    
    custom_scores = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            score = QED.default(mol)   # replace with your model
            custom_scores.append(score)
        except Exception:
            custom_scores.append(float('nan'))
    
    preds_df["QED_DrugLikeness"] = custom_scores
    return preds_df
```

Replace all `admet_model.predict(smiles=...)` calls in `app.py` with `get_extended_predictions(...)`.

---

## 📜 Acknowledgments & References

ChemXplain stands entirely on the shoulders of open-source science. If you use this tool in your research, please cite the following foundational works:

### Core Methodology

1. **Occlusion Sensitivity (XAI backbone):**
   Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*. https://doi.org/10.1007/978-3-319-10590-1_53

2. **ADMET-AI (property prediction):**
   Swanson, K., et al. (2023). ADMET-AI: A machine learning ADMET platform. *J. Chem. Inf. Model.* https://doi.org/10.1021/acs.jcim.3c01064

3. **CReM (fragment-based generation):**
   Polishchuk, P. (2020). CReM: chemically reasonable mutations framework. *J. Cheminform.*, 12, 28. https://doi.org/10.1186/s13321-020-00431-w

4. **REINVENT4 (transformer generation):**
   Loeffler, H. H., et al. (2024). Reinvent 4: Modern AI–driven generative molecule design. *J. Cheminform.*, 16, 20. https://doi.org/10.1186/s13321-024-00812-5

5. **Dimorphite-DL (pH correction):**
   Ropp, P. J., et al. (2019). Dimorphite-DL. *J. Cheminform.*, 11, 14. https://doi.org/10.1186/s13321-019-0336-9

### Visualization

6. **Similarity Contour Maps:**
   Riniker, S., & Landrum, G. A. (2013). Similarity maps. *J. Cheminform.*, 5, 43. https://doi.org/10.1186/1758-2946-5-43

7. **RDKit SimilarityMaps C++ Backend:**
   Landrum, G. (2020). Similarity maps with the new drawing code. *RDKit Blog*. https://greglandrum.github.io/rdkit-blog/posts/2020-01-03-similarity-maps-with-new-drawing-code.html

8. **RDKit (cheminformatics engine):**
   Landrum, G., et al. RDKit: Open-source cheminformatics. https://www.rdkit.org

### Safety Filters

9. **PAINS filters:**
   Baell, J. B., & Holloway, G. A. (2010). *J. Med. Chem.*, 53(7), 2719–2740. https://doi.org/10.1021/jm901137j

### Scoring & Drug-likeness

10. **SA Score:**
    Ertl, P., & Schuffenhauer, A. (2009). *J. Cheminform.*, 1, 8. https://doi.org/10.1186/1758-2946-1-8

11. **Lipinski Ro5:**
    Lipinski, C. A., et al. (2001). *Adv. Drug Deliv. Rev.*, 46(1-3), 3–26. https://doi.org/10.1016/S0169-409X(00)00129-0

12. **Bemis-Murcko Scaffolds:**
    Bemis, G. W., & Murcko, M. A. (1996). *J. Med. Chem.*, 39(15), 2887–2893. https://doi.org/10.1021/jm9602928

### Physicochemical Maps

13. **Gasteiger-Marsili Partial Charges:**
    Gasteiger, J., & Marsili, M. (1980). *Tetrahedron*, 36(22), 3219–3228. https://doi.org/10.1016/0040-4020(80)80168-2

14. **Crippen logP Atom Contributions:**
    Wildman, S. A., & Crippen, G. M. (1999). *J. Chem. Inf. Comput. Sci.*, 39(5), 868–873. https://doi.org/10.1021/ci990307l

### AI Report Generation

15. **Google Gemini 2.5 Flash:**
    Google DeepMind. (2024). Gemini: A family of highly capable multimodal models. https://deepmind.google/technologies/gemini/

---

## 👤 Author

**Akshay Krishnamurthy Hegde**

- 🔬 **Field:** Computational Drug Discovery · Machine Learning · Cheminformatics
- 🛠️ **Tools:** RDKit · ADMET-AI · PyTorch · scikit-learn · XGBoost · SHAP · PBPK Modelling
- 🐙 **GitHub:** [Akshay-Krishnamurthy](https://github.com/Akshay-Krishnamurthy)

> *"Bridging the gap between AI explainability and real-world drug discovery."*

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

<p align="center">
  <i>Built for the computational drug discovery community.</i><br>
  <i>If this tool helped your research, please ⭐ the repository.</i>
</p>
