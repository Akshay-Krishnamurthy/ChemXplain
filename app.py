"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ChemXplain — AI-Powered Drug Discovery Platform                    ║
║          Library EDA  •  ADMET Profiling  •  XAI  •  Lead Optimisation     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Workflow:
  MODE A — Library Analysis (CSV/Excel Upload)
    Step 1 · Upload & validate molecular library
    Step 2 · Publication-quality EDA (PCA, UMAP, physicochemical distributions,
             scaffold diversity, Ro5 compliance, Tanimoto similarity heatmap,
             LogP-MW landscape, ADMET library heatmap)
    Step 3 · Gemini AI narrative summary of the library
    Step 4 · Batch ADMET profiling (all 66 properties) → master CSV download
    Step 5 · Select one molecule → hand off to Deep Dive (Mode B)

  MODE B — Single Molecule Deep Dive
    Step 1 · pH 7.4 correction, Applicability Domain, PAINS filter
    Step 2 · ADMET Radar + 66-property table
    Step 3 · Occlusion XAI: 3D, sharp map, contour, node plot, bond map,
             motif ablation, trade-off quadrant, mutation heatmap, physico maps
    Step 4 · Gemini mechanistic diagnostic report
    Step 5 · Engine selector — CReM+Fallback  OR  REINVENT4
    Step 6 · ADMET re-scoring + Top-5 grid + Gemini comparative analysis

Scientific references:
  Occlusion XAI      : Zeiler & Fergus (2014)
  ADMET predictions  : ADMET-AI (Swanson et al.), 66 neural network models
  CReM               : Polishchuk (2020), J. Chem. Inf. Model.
  REINVENT4          : Loeffler et al. (2024), J. Cheminform. 16:20
  PCA chemical space : Morgan FP-based, Ertl et al. (2020) methodology
  SA Score           : Ertl & Schuffenhauer (2009)
  Gasteiger charges  : Gasteiger & Marsili (1980)
  Crippen logP       : Wildman & Crippen (1999)
  PAINS              : Baell & Holloway (2010)
  Bemis-Murcko       : Bemis & Murcko (1996)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io, os, sys, random, subprocess, tempfile, shutil, warnings
warnings.filterwarnings("ignore")

# ── Scientific ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import seaborn as sns

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ── ADMET-AI ──────────────────────────────────────────────────────────────────
from admet_ai import ADMETModel

# ── CReM ──────────────────────────────────────────────────────────────────────
from crem.crem import mutate_mol

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
from stmol import showmol

# ── Gemini ────────────────────────────────────────────────────────────────────
try:
    from google import genai
    LEGACY_GEMINI = False
except ImportError:
    import google.generativeai as genai
    LEGACY_GEMINI = True

# ── SA Scorer ─────────────────────────────────────────────────────────────────
SA_AVAILABLE = False
for _p in [os.path.join(RDConfig.RDContribDir, "SA_Score"),
           os.path.join(os.path.dirname(RDConfig.__file__), "..", "Contrib", "SA_Score")]:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        try:
            import sascorer; SA_AVAILABLE = True; break
        except ImportError:
            pass

CMAP_WR = LinearSegmentedColormap.from_list("WhiteRed", ["#FFFFFF", "#D62728"])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(layout="wide", page_title="ChemXplain · AI Drug Discovery", page_icon="🧬")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .block-container { padding-top: 0 !important; padding-left: 1.5rem; padding-right: 1.5rem; max-width:100%; }

  /* ── Global section header ── */
  .section-hdr {
    font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
    margin:1.2rem 0 .5rem; padding:6px 12px; border-radius:6px;
    background:linear-gradient(90deg,#e8f0ff,transparent);
    border-left:3px solid #3B5BDB; color:#1a1a3e;
  }
  .engine-card{border-radius:12px;padding:16px;min-height:155px;line-height:1.65}
  div[data-testid="metric-container"]{
    background:linear-gradient(135deg,#f0f4ff,#fff);
    border:1px solid #d8e4ff; border-radius:10px; padding:10px 14px;
  }

  /* ── Hero strip ── */
  .hero-strip {
    background: linear-gradient(135deg, #0c0f2e 0%, #131740 40%, #0a2352 70%, #0c1a45 100%);
    padding: 52px 40px 28px; border-radius: 0 0 24px 24px;
    margin: -1rem -1.5rem 24px; position:relative; overflow:hidden;
  }
  .hero-strip::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:340px; height:340px; border-radius:50%;
    background:radial-gradient(circle, rgba(91,130,255,.18) 0%, transparent 70%);
  }
  .hero-strip::after {
    content:''; position:absolute; bottom:-80px; left:30%;
    width:260px; height:260px; border-radius:50%;
    background:radial-gradient(circle, rgba(157,78,221,.12) 0%, transparent 70%);
  }
  .hero-tag {
    display:inline-block; background:rgba(91,130,255,.15); border:1px solid rgba(91,130,255,.35);
    border-radius:20px; padding:3px 14px; font-size:.72rem; color:#8ca3ff;
    letter-spacing:.8px; text-transform:uppercase; font-weight:600; margin-bottom:12px;
  }
  .hero-title {
    font-family:'Syne',sans-serif; font-size:2.5rem; font-weight:800;
    color:#ffffff; line-height:1.15; margin:0 0 10px;
  }
  .hero-title span { color:#5b82ff; }
  .hero-sub {
    color:#8a9abb; font-size:.92rem; font-weight:300; margin:0; max-width:520px; line-height:1.6;
  }
  .hero-pills { display:flex; gap:10px; margin-top:18px; flex-wrap:wrap; }
  .hero-pill {
    background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
    border-radius:20px; padding:5px 14px; font-size:.75rem; color:#c0cde8;
    font-weight:500; letter-spacing:.3px;
  }
  .hero-pill.accent { background:rgba(91,130,255,.2); border-color:rgba(91,130,255,.4); color:#8ca3ff; }
  .hero-stat { text-align:center; }
  .hero-stat-num { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:#5b82ff; }
  .hero-stat-label { font-size:.7rem; color:#6a7a9b; text-transform:uppercase; letter-spacing:.6px; }

  /* ── Sidebar global ── */
  section[data-testid="stSidebar"] > div:first-child { padding-top:0 !important; }
  section[data-testid="stSidebar"] { background:#07090f !important; }
  section[data-testid="stSidebar"] * { color:#c0cde8; }
  section[data-testid="stSidebar"] .stTextInput input,
  section[data-testid="stSidebar"] .stTextArea textarea {
    background:#111425 !important; border:1px solid #1e2340 !important;
    color:#c0cde8 !important; border-radius:7px !important; font-size:.8rem !important;
  }
  section[data-testid="stSidebar"] .stRadio label { font-size:.82rem !important; }
  section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color:#c0cde8; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-strip">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;position:relative;z-index:1;gap:10px">
    <div style="font-family:'Syne',sans-serif;font-size:3.2rem;font-weight:800;color:#ffffff;letter-spacing:1px;line-height:1.2;margin-bottom:2px;padding-top:8px">Chem<span style="color:#5b82ff">Xplain</span></div>
    <div class="hero-tag">&#x1F9EC; AI-Powered &middot; Research-Grade</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:600;color:#c8d4f5;line-height:1.4;margin:4px 0">Democratize <span style="color:#5b82ff">Computational</span> Drug Discovery</div>
    <p class="hero-sub" style="max-width:600px;margin:0 auto 6px">End-to-end platform for library screening, explainable ADMET profiling, and generative lead optimisation &mdash; powered by Occlusion XAI, ADMET-AI, CReM &amp; REINVENT4.</p>
    <div class="hero-pills" style="justify-content:center">
      <span class="hero-pill accent">66 ADMET Models</span>
      <span class="hero-pill">Occlusion XAI</span>
      <span class="hero-pill">PCA &middot; UMAP &middot; EDA</span>
      <span class="hero-pill">CReM + REINVENT4</span>
      <span class="hero-pill">Gemini AI Reports</span>
    </div>
    <div style="display:flex;gap:32px;align-items:center;justify-content:center;flex-wrap:wrap;margin-top:10px">
      <div class="hero-stat"><div class="hero-stat-num">66</div><div class="hero-stat-label">ADMET Models</div></div>
      <div style="width:1px;height:50px;background:rgba(255,255,255,.1)"></div>
      <div class="hero-stat"><div class="hero-stat-num">9</div><div class="hero-stat-label">XAI Views</div></div>
      <div style="width:1px;height:50px;background:rgba(255,255,255,.1)"></div>
      <div class="hero-stat"><div class="hero-stat-num">3</div><div class="hero-stat-label">Gen Engines</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:18px 14px 10px;border-bottom:1px solid #111425'>
      <div style='display:flex;align-items:center;gap:10px'>
        <div style='font-size:1.5rem'>🧬</div>
        <div>
          <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;
                      color:#5b82ff;letter-spacing:.4px'>ChemXplain</div>
          <div style='font-size:.65rem;color:#3a4a6b;letter-spacing:.5px;text-transform:uppercase'>
            AI Drug Discovery Platform
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Workflow selector ─────────────────────────────────────────────────────
    st.markdown("<div style='padding:10px 14px 4px;font-size:.67rem;color:#3a4a6b;text-transform:uppercase;letter-spacing:.8px;font-weight:600'>Workflow Mode</div>", unsafe_allow_html=True)
    app_mode = st.radio("mode", ["📚 Library Analysis", "🔬 Single Molecule Deep Dive"],
                        label_visibility="collapsed")

    # ── BioRender-style Pipeline Diagram ─────────────────────────────────────
    st.markdown("""
    <div style='margin:0 -1rem;padding:14px 14px 6px;background:#07090f;border-top:1px solid #0d1020'>
      <div style='font-size:.67rem;color:#3a4a6b;text-transform:uppercase;letter-spacing:.8px;
                  font-weight:600;margin-bottom:10px'>Discovery Pipeline</div>

      <!-- Step 1 -->
      <div style='background:linear-gradient(135deg,#0d1535,#0a1228);border:1px solid #1a2550;
                  border-radius:10px;padding:9px 11px;margin-bottom:3px;display:flex;align-items:center;gap:9px'>
        <div style='width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#1a3a8a,#2952cc);
                    display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0'>📂</div>
        <div>
          <div style='font-size:.73rem;font-weight:600;color:#8ca3ff'>Input Library</div>
          <div style='font-size:.62rem;color:#3a4a7b'>CSV / Excel · SMILES validation</div>
        </div>
      </div>

      <!-- Arrow -->
      <div style='text-align:center;color:#1e2a50;font-size:.8rem;line-height:1.2'>▼</div>

      <!-- Step 2 -->
      <div style='background:linear-gradient(135deg,#0d1535,#0a1228);border:1px solid #1a2550;
                  border-radius:10px;padding:9px 11px;margin-bottom:3px;display:flex;align-items:center;gap:9px'>
        <div style='width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#0e4a3a,#0d8a5e);
                    display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0'>📊</div>
        <div>
          <div style='font-size:.73rem;font-weight:600;color:#4ecca3'>Chemical Space EDA</div>
          <div style='font-size:.62rem;color:#3a4a7b'>PCA · UMAP · Scaffold · Ro5 · Tanimoto</div>
        </div>
      </div>

      <div style='text-align:center;color:#1e2a50;font-size:.8rem;line-height:1.2'>▼</div>

      <!-- Step 3 -->
      <div style='background:linear-gradient(135deg,#0d1535,#0a1228);border:1px solid #1a2550;
                  border-radius:10px;padding:9px 11px;margin-bottom:3px;display:flex;align-items:center;gap:9px'>
        <div style='width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#4a1a8a,#7c3dcc);
                    display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0'>🧠</div>
        <div>
          <div style='font-size:.73rem;font-weight:600;color:#c084fc'>ADMET Profiling</div>
          <div style='font-size:.62rem;color:#3a4a7b'>66 deep learning models · hERG · DILI · BBB</div>
        </div>
      </div>

      <div style='text-align:center;color:#1e2a50;font-size:.8rem;line-height:1.2'>▼</div>

      <!-- Step 4 -->
      <div style='background:linear-gradient(135deg,#0d1535,#0a1228);border:1px solid #1a2550;
                  border-radius:10px;padding:9px 11px;margin-bottom:3px;display:flex;align-items:center;gap:9px'>
        <div style='width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#7a2a0a,#cc5522);
                    display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0'>🔥</div>
        <div>
          <div style='font-size:.73rem;font-weight:600;color:#fb923c'>Occlusion XAI</div>
          <div style='font-size:.62rem;color:#3a4a7b'>3D · Contour · Bond · Motif · Quadrant</div>
        </div>
      </div>

      <div style='text-align:center;color:#1e2a50;font-size:.8rem;line-height:1.2'>▼</div>

      <!-- Step 5 — Fork -->
      <div style='background:linear-gradient(135deg,#0a1a0a,#0d2010);border:1px solid #0f3020;
                  border-radius:10px;padding:8px 10px;margin-bottom:3px'>
        <div style='font-size:.67rem;font-weight:600;color:#4ade80;margin-bottom:6px;text-align:center'>
          ⚗️ Generative Engine
        </div>
        <div style='display:flex;gap:6px'>
          <div style='flex:1;background:#0a1f12;border:1px solid #14402a;border-radius:7px;padding:6px 8px;text-align:center'>
            <div style='font-size:.75rem'>🧩</div>
            <div style='font-size:.6rem;color:#4ade80;font-weight:600'>CReM</div>
            <div style='font-size:.55rem;color:#1e4a30'>ChEMBL DB<br>+ Bioisostere</div>
          </div>
          <div style='flex:1;background:#0a1f12;border:1px solid #14402a;border-radius:7px;padding:6px 8px;text-align:center'>
            <div style='font-size:.75rem'>🤖</div>
            <div style='font-size:.6rem;color:#4ade80;font-weight:600'>REINVENT4</div>
            <div style='font-size:.55rem;color:#1e4a30'>RL + Transformer<br>Mol2Mol</div>
          </div>
        </div>
      </div>

      <div style='text-align:center;color:#1e2a50;font-size:.8rem;line-height:1.2'>▼</div>

      <!-- Step 6 -->
      <div style='background:linear-gradient(135deg,#1a1500,#2a2000);border:1px solid #2e2a00;
                  border-radius:10px;padding:9px 11px;display:flex;align-items:center;gap:9px'>
        <div style='width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#3a2a00,#8a6a00);
                    display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0'>🏆</div>
        <div>
          <div style='font-size:.73rem;font-weight:600;color:#fbbf24'>Lead Candidates</div>
          <div style='font-size:.62rem;color:#3a3010'>SA Score · Gemini AI Report · CSV export</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Molecule Visualizer ───────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:12px 14px 4px;border-top:1px solid #0d1020;
                font-size:.67rem;color:#3a4a6b;text-transform:uppercase;letter-spacing:.8px;font-weight:600'>
      🔭 Molecule Visualizer
    </div>
    """, unsafe_allow_html=True)

    viz_smi = st.text_area("SMILES:", height=60,
                           value=st.session_state.get("deep_dive_smi",
                               "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4"),
                           label_visibility="collapsed",
                           placeholder="Paste any SMILES…",
                           key="sidebar_viz_smi")

    _viz_mol = Chem.MolFromSmiles(viz_smi.strip()) if viz_smi.strip() else None
    if _viz_mol:
        AllChem.Compute2DCoords(_viz_mol)
        _dr = rdMolDraw2D.MolDraw2DSVG(268, 190)
        _dr.drawOptions().useBWAtomPalette()
        _dr.drawOptions().setBackgroundColour((0.028, 0.035, 0.062, 1.0))
        rdMolDraw2D.PrepareAndDrawMolecule(_dr, _viz_mol)
        _dr.FinishDrawing()
        _svg = _dr.GetDrawingText()
        components.html(f"""
        <div style='background:#07090f;border:1px solid #1a2040;border-radius:10px;
                    overflow:hidden;margin:2px 0 8px'>
          {_svg}
        </div>""", height=200)

        _mw   = round(Descriptors.MolWt(_viz_mol), 1)
        _lp   = round(Descriptors.MolLogP(_viz_mol), 2)
        _hbd  = rdMolDescriptors.CalcNumHBD(_viz_mol)
        _hba  = rdMolDescriptors.CalcNumHBA(_viz_mol)
        _tpsa = round(rdMolDescriptors.CalcTPSA(_viz_mol), 1)
        _qed  = round(Descriptors.qed(_viz_mol), 3)
        _rot  = rdMolDescriptors.CalcNumRotatableBonds(_viz_mol)
        _har  = _viz_mol.GetNumHeavyAtoms()
        _ro5  = _mw<=500 and _lp<=5 and _hbd<=5 and _hba<=10
        _rc   = "#22c55e" if _ro5 else "#ef4444"
        _rl   = "Ro5 ✓ Compliant" if _ro5 else "Ro5 ✗ Violation"

        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:.7rem;margin-bottom:5px'>
          <div style='background:#0d1020;border:1px solid #151a35;border-radius:7px;padding:6px 8px'>
            <div style='color:#5b82ff;font-weight:600;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px'>MW</div>
            <div style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{_mw} <span style='color:#3a4a7b;font-size:.6rem'>Da</span></div>
          </div>
          <div style='background:#0d1020;border:1px solid #151a35;border-radius:7px;padding:6px 8px'>
            <div style='color:#5b82ff;font-weight:600;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px'>LogP</div>
            <div style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{_lp}</div>
          </div>
          <div style='background:#0d1020;border:1px solid #151a35;border-radius:7px;padding:6px 8px'>
            <div style='color:#5b82ff;font-weight:600;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px'>HBD / HBA</div>
            <div style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{_hbd} / {_hba}</div>
          </div>
          <div style='background:#0d1020;border:1px solid #151a35;border-radius:7px;padding:6px 8px'>
            <div style='color:#5b82ff;font-weight:600;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px'>TPSA</div>
            <div style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{_tpsa} <span style='color:#3a4a7b;font-size:.6rem'>Å²</span></div>
          </div>
          <div style='background:#0d1020;border:1px solid #151a35;border-radius:7px;padding:6px 8px'>
            <div style='color:#5b82ff;font-weight:600;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px'>QED</div>
            <div style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{_qed}</div>
          </div>
          <div style='background:#0d1020;border:1px solid #151a35;border-radius:7px;padding:6px 8px'>
            <div style='color:#5b82ff;font-weight:600;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px'>RotB / HvyAt</div>
            <div style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{_rot} / {_har}</div>
          </div>
        </div>
        <div style='text-align:center;background:{_rc}18;border:1px solid {_rc}55;
                    border-radius:7px;padding:5px;font-size:.72rem;color:{_rc};font-weight:600'>
          {_rl} &nbsp;·&nbsp; {_har} heavy atoms
        </div>
        """, unsafe_allow_html=True)
    else:
        components.html("""
        <div style='background:#07090f;border:1px dashed #1a2040;border-radius:10px;
                    height:175px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;color:#1e2a50;font-size:.78rem;gap:6px'>
          <div style='font-size:1.6rem;opacity:.4'>⬡</div>
          <div>Enter a valid SMILES above</div>
        </div>""", height=185)

    # ── Config ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:10px 14px 4px;border-top:1px solid #0d1020;
                font-size:.67rem;color:#3a4a6b;text-transform:uppercase;letter-spacing:.8px;font-weight:600'>
      ⚙️ Configuration
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input("Gemini API Key:", type="password", placeholder="AIza…")
    CREM_DB = st.text_input("CReM DB path:", value="chembl22_sa2_hac12.db")

    _gpu_color = "#22c55e" if DEVICE=="cuda" else "#f59e0b"
    _gpu_label = f"🖥️ GPU · {torch.cuda.get_device_name(0)}" if DEVICE=="cuda" else "⚠️ CPU mode"
    _db_color  = "#22c55e" if os.path.exists(CREM_DB) else "#ef4444"
    _db_label  = "✅ CReM DB ready" if os.path.exists(CREM_DB) else "❌ CReM DB not found"

    st.markdown(f"""
    <div style='display:flex;flex-direction:column;gap:4px;padding:4px 0 10px'>
      <div style='background:{_gpu_color}12;border:1px solid {_gpu_color}40;border-radius:6px;
                  padding:5px 10px;font-size:.7rem;color:{_gpu_color}'>{_gpu_label}</div>
      <div style='background:{_db_color}12;border:1px solid {_db_color}40;border-radius:6px;
                  padding:5px 10px;font-size:.7rem;color:{_db_color}'>{_db_label}</div>
    </div>
    <div style='text-align:center;padding:8px 0;font-size:.6rem;color:#1e2a50;border-top:1px solid #0d1020'>
      ChemXplain · Occlusion XAI · ADMET-AI · CReM · REINVENT4
    </div>
    """, unsafe_allow_html=True)

# ── Load ADMET models ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_admet():
    return ADMETModel()

with st.spinner("⏳ Loading 66 ADMET deep learning models…"):
    admet_model = load_admet()

if "all_properties" not in st.session_state:
    _tmp = admet_model.predict(smiles=["C"])
    st.session_state.all_properties = sorted(list(_tmp.columns))
all_props = st.session_state.all_properties

if "stage" not in st.session_state:
    st.session_state.stage = 0

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def correct_ph(smi):
    try:
        from dimorphite_dl import protonate_smiles
        res = protonate_smiles(smi, ph_min=7.4, ph_max=7.4)
        if res and res[0]!=smi: return res[0], True
        return smi, False
    except ImportError:
        try:
            from dimorphite_dl import DimorphiteDL
            dl = DimorphiteDL(min_ph=7.4, max_ph=7.4, max_variants=1, quiet=True)
            res = dl.protonate(smi)
            if res and res[0]!=smi: return res[0], True
        except Exception: pass
    except Exception: pass
    return smi, False

def check_pains(mol):
    p = FilterCatalogParams(); p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return [e.GetDescription() for e in FilterCatalog(p).GetMatches(mol)]

_AD_REF = ["CC(=O)OC1=CC=CC=C1C(=O)O","CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
           "CN1C=NC2=C1C(=O)N(C(=O)N2C)C","CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"]
def check_ad(smi):
    mol=Chem.MolFromSmiles(smi)
    if not mol: return 0.0
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
    return max((DataStructs.TanimotoSimilarity(fp,AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(r),2,2048)) for r in _AD_REF if Chem.MolFromSmiles(r)),default=0.0)

def passes_lipinski(smi):
    mol=Chem.MolFromSmiles(smi)
    if not mol: return False
    v=0
    if Descriptors.MolWt(mol)>500: v+=1
    if Descriptors.MolLogP(mol)>5: v+=1
    if rdMolDescriptors.CalcNumHBD(mol)>5: v+=1
    if rdMolDescriptors.CalcNumHBA(mol)>10: v+=1
    return v<=1

def get_sa_score(smi):
    if not SA_AVAILABLE: return float("nan")
    mol=Chem.MolFromSmiles(smi)
    if not mol: return float("nan")
    return sascorer.calculateScore(mol)

def compute_descriptors(smi):
    mol=Chem.MolFromSmiles(smi)
    if not mol: return {}
    return {"MW":round(Descriptors.MolWt(mol),2),
            "LogP":round(Descriptors.MolLogP(mol),3),
            "HBD":rdMolDescriptors.CalcNumHBD(mol),
            "HBA":rdMolDescriptors.CalcNumHBA(mol),
            "TPSA":round(rdMolDescriptors.CalcTPSA(mol),2),
            "RotBonds":rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Rings":rdMolDescriptors.CalcNumRings(mol),
            "ArRings":rdMolDescriptors.CalcNumAromaticRings(mol),
            "HeavyAtoms":mol.GetNumHeavyAtoms(),
            "FractionCSP3":round(rdMolDescriptors.CalcFractionCSP3(mol),3),
            "QED":round(Descriptors.qed(mol),3)}

def get_scaffold(smi):
    mol=Chem.MolFromSmiles(smi)
    if not mol: return ""
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception: return ""

def get_fp(smi, radius=2, nbits=2048):
    mol=Chem.MolFromSmiles(smi)
    if not mol: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol,radius,nbits)

def call_gemini(prompt, key):
    if LEGACY_GEMINI:
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text
    client=genai.Client(api_key=key)
    return client.models.generate_content(model="gemini-2.5-flash",contents=prompt).text

def img_dl(buf, caption, dl_label, fname):
    if buf:
        buf.seek(0); st.image(buf, use_container_width=True, caption=caption)
        buf.seek(0); st.download_button(dl_label, data=buf, file_name=fname, mime="image/png", use_container_width=True)

def admet_widget(label, key, pd_, thresh, low_good=True):
    val=pd_.get(key,float("nan"))
    if pd.isna(val): st.info(f"ℹ️ **{label}:** N/A"); return
    if low_good:
        fn = st.error if val>thresh else (st.warning if val>thresh-.2 else st.success)
        tag = "HIGH RISK" if val>thresh else ("MODERATE" if val>thresh-.2 else "SAFE")
        icon = "🔴" if val>thresh else ("🟡" if val>thresh-.2 else "🟢")
    else:
        fn = st.success if val>thresh else (st.warning if val>thresh-.2 else st.error)
        tag = "HIGH" if val>thresh else ("MODERATE" if val>thresh-.2 else "LOW")
        icon = "🟢" if val>thresh else ("🟡" if val>thresh-.2 else "🔴")
    fn(f"{icon} **{label}:** {val:.3f} — {tag}")

# ── ADMET Radar ───────────────────────────────────────────────────────────────
def radar_chart(pd_):
    keys=["hERG","DILI","Ames","BBB_Martins","Pgp_Broccatelli"]
    labs=["hERG","DILI","Ames","BBB","Pgp"]
    vals=[float(pd_.get(k,0) or 0) for k in keys]+[float(pd_.get(keys[0],0) or 0)]
    angs=np.linspace(0,2*np.pi,5,endpoint=False).tolist()+[0]
    fig,ax=plt.subplots(figsize=(4,4),subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.fill(angs,vals,color="#e94560",alpha=0.25); ax.plot(angs,vals,color="#e94560",lw=2)
    ax.set_yticklabels([]); ax.set_xticks(angs[:-1]); ax.set_xticklabels(labs,fontsize=10)
    ax.set_ylim(0,1); ax.set_title("Toxicity Risk Radar",y=1.1,fontweight="bold",fontsize=11)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); return buf

# ── Physicochemical maps ──────────────────────────────────────────────────────
def gasteiger_map(mol):
    from rdkit.Chem import rdPartialCharges
    try:
        mh=Chem.AddHs(mol); rdPartialCharges.ComputeGasteigerCharges(mh)
        chgs=[mh.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in range(mol.GetNumAtoms())]
        chgs=[c if np.isfinite(c) else 0.0 for c in chgs]
        d=Draw.MolDraw2DCairo(500,500); d.drawOptions().useBWAtomPalette()
        SimilarityMaps.GetSimilarityMapFromWeights(mol,chgs,d,colorMap="RdBu_r",contourLines=10,alpha=0.30)
        d.FinishDrawing(); return io.BytesIO(d.GetDrawingText())
    except Exception: return None

def logp_map(mol):
    try:
        w=[float(x[0]) for x in rdMolDescriptors._CalcCrippenContribs(mol)]
        d=Draw.MolDraw2DCairo(500,500); d.drawOptions().useBWAtomPalette()
        SimilarityMaps.GetSimilarityMapFromWeights(mol,w,d,colorMap="PiYG_r",contourLines=10,alpha=0.30)
        d.FinishDrawing(); return io.BytesIO(d.GetDrawingText())
    except Exception: return None

# ── XAI functions ─────────────────────────────────────────────────────────────
def occlusion_xai(smi,prop,base,model):
    bm=Chem.MolFromSmiles(smi)
    if not bm: return [],[],[]
    try: Chem.Kekulize(bm,clearAromaticFlags=True)
    except Exception: pass
    n=bm.GetNumAtoms(); pert,vidx=[],[]
    for i in range(n):
        rw=Chem.RWMol(bm); rw.GetAtomWithIdx(i).SetAtomicNum(0); rw.GetAtomWithIdx(i).SetFormalCharge(0)
        try: pert.append(Chem.MolToSmiles(rw)); vidx.append(i)
        except Exception: pass
    aw,dw=[0.0]*n,[0.0]*n
    if pert:
        sc=model.predict(smiles=pert)[prop].values
        for idx,s in zip(vidx,sc):
            if not pd.isna(s): aw[idx]=abs(base-s); dw[idx]=s-base
    mn,mx=min(aw),max(aw)
    nw=[(w-mn)/(mx-mn) if mx!=mn else 0.0 for w in aw]
    return nw,aw,dw

def edge_ablation_svg(smi,prop,base,model):
    mol=Chem.MolFromSmiles(smi); pert,vb=[],[]
    for b in mol.GetBonds():
        rw=Chem.RWMol(mol); rw.RemoveBond(b.GetBeginAtomIdx(),b.GetEndAtomIdx())
        try: pert.append(Chem.MolToSmiles(rw)); vb.append(b.GetIdx())
        except Exception: pass
    bw={b.GetIdx():0.0 for b in mol.GetBonds()}
    if pert:
        for bidx,sc in zip(vb,model.predict(smiles=pert)[prop].values):
            if not pd.isna(sc): bw[bidx]=abs(base-sc)
    AllChem.Compute2DCoords(mol)
    dr=rdMolDraw2D.MolDraw2DSVG(500,500); dr.drawOptions().useBWAtomPalette()
    mx=max(bw.values()) if bw else 0; cmap=plt.cm.YlOrRd; hb,hc=[],{}
    for bidx,w in bw.items():
        nw=(w/mx) if mx>0 else 0
        if nw>0.05: hb.append(bidx); hc[bidx]=cmap(nw)[:3]
    rdMolDraw2D.PrepareAndDrawMolecule(dr,mol,highlightBonds=hb,highlightBondColors=hc)
    dr.FinishDrawing(); return dr.GetDrawingText()

def motif_ablation(smi,prop,base,model):
    mol=Chem.MolFromSmiles(smi); rings=mol.GetRingInfo().AtomRings()
    if not rings: return None
    pert,labels=[],[]
    for i,ring in enumerate(rings):
        rw=Chem.RWMol(mol)
        for ai in ring: rw.GetAtomWithIdx(ai).SetAtomicNum(0)
        try: pert.append(Chem.MolToSmiles(rw)); labels.append(f"Ring {i+1} ({len(ring)} at.)")
        except Exception: pass
    if not pert: return None
    impacts=[abs(base-sc) if not pd.isna(sc) else 0.0 for sc in model.predict(smiles=pert)[prop].values]
    fig,ax=plt.subplots(figsize=(6,max(3,len(labels)*.6))); fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    sns.barplot(x=impacts,y=labels,palette="Reds_d",ax=ax,edgecolor=".2")
    ax.set_xlabel(f"Occlusion Impact on {prop}",fontsize=10); ax.set_title("Ring/Motif Ablation",fontweight="bold")
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150); plt.close(fig); return buf

def mutation_heatmap(smi,prop,base,aw,model):
    mol=Chem.MolFromSmiles(smi)
    ranking=sorted([(i,w,mol.GetAtomWithIdx(i).GetSymbol()) for i,w in enumerate(aw)],key=lambda x:x[1],reverse=True)
    targets=[(i,sym) for i,w,sym in ranking if w>0.01 and sym!="H"][:5]
    if not targets: return None
    muts={"N":7,"O":8,"F":9,"S":16}; matrix=np.full((len(targets),4),np.nan)
    pert,tracking=[],[]
    for row,(ai,_) in enumerate(targets):
        for col,(sym,anum) in enumerate(muts.items()):
            rw=Chem.RWMol(mol); at=rw.GetAtomWithIdx(ai)
            if at.GetAtomicNum()==anum: matrix[row,col]=base; continue
            at.SetAtomicNum(anum); at.SetFormalCharge(0)
            try: Chem.SanitizeMol(rw); pert.append(Chem.MolToSmiles(rw)); tracking.append((row,col))
            except Exception: pass
    if pert:
        for (r,c),sc in zip(tracking,model.predict(smiles=pert)[prop].values): matrix[r,c]=sc
    fig,ax=plt.subplots(figsize=(6,max(3,len(targets)*.7))); fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    sns.heatmap(matrix,annot=True,fmt=".3f",cmap="coolwarm",xticklabels=list(muts.keys()),
                yticklabels=[f"{sym} (N{i})" for i,sym in targets],ax=ax,linewidths=.5,linecolor="grey")
    ax.set_title(f"Point Mutation Array — Base {prop}: {base:.3f}",fontsize=11)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150); plt.close(fig); return buf

def tradeoff_scatter(smi,p1,p2,s1,s2,model,dw1):
    _,_,dw2=occlusion_xai(smi,p2,s2,model)
    mol=Chem.MolFromSmiles(smi); labs=[f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(len(dw1))]
    aw1=np.abs(dw1)
    fig,ax=plt.subplots(figsize=(8,6)); fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhline(0,color="gray",lw=1,ls="--"); ax.axvline(0,color="gray",lw=1,ls="--")
    sc=ax.scatter(dw1,dw2,c=aw1,cmap="coolwarm",s=100,edgecolor="black",alpha=.8)
    plt.colorbar(sc,ax=ax,label=f"|Impact on {p1}|")
    for i in np.argsort(aw1)[-5:]:
        if aw1[i]>0.01: ax.annotate(labs[i],(dw1[i],dw2[i]),xytext=(5,5),textcoords="offset points",fontsize=8,fontweight="bold")
    kw=dict(fontsize=9,alpha=.55,transform=ax.transAxes)
    ax.text(.51,.97,"Suppresses P1, drives P2",va="top",**kw)
    ax.text(.01,.97,"Suppresses BOTH",va="top",**kw)
    ax.text(.01,.02,"DRIVES BOTH → best target",va="bottom",color="red",fontweight="bold",**kw)
    ax.text(.51,.02,"Drives P1, suppresses P2",va="bottom",**kw)
    ax.set_xlabel(f"Δ {p1} when atom removed"); ax.set_ylabel(f"Δ {p2} when atom removed")
    ax.set_title("Occlusion Quadrant Analysis",fontsize=12); ax.grid(True,ls=":",alpha=.4); plt.tight_layout()
    buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150); plt.close(fig); return buf

# ═══════════════════════════════════════════════════════════════════════════════
# EDA PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_physico(df):
    props=["MW","LogP","HBD","HBA","TPSA","RotBonds","QED","FractionCSP3"]
    titles=["Molecular Weight (Da)","Lipophilicity (LogP)","H-Bond Donors","H-Bond Acceptors",
            "Topological PSA (Å²)","Rotatable Bonds","Drug-likeness (QED)","Fraction sp³ C"]
    refs={"MW":500,"LogP":5,"HBD":5,"HBA":10,"TPSA":140,"RotBonds":10}
    fig,axes=plt.subplots(2,4,figsize=(16,8)); fig.patch.set_facecolor("white")
    plt.suptitle("Physicochemical Property Distributions",fontsize=14,fontweight="bold",y=1.01)
    pal=sns.color_palette("husl",8)
    for ax,prop,title,color in zip(axes.flat,props,titles,pal):
        if prop not in df.columns: ax.set_visible(False); continue
        data=df[prop].dropna(); ax.set_facecolor("#f8f9fa")
        sns.histplot(data,ax=ax,color=color,edgecolor="white",lw=.5,bins=25,stat="density",alpha=.85)
        sns.kdeplot(data,ax=ax,color="black",lw=1.5)
        if prop in refs: ax.axvline(refs[prop],color="#e94560",ls="--",lw=1.5,label=f"Ro5 ({refs[prop]})"); ax.legend(fontsize=7,framealpha=.7)
        ax.axvline(data.median(),color="#2196F3",ls=":",lw=1.5)
        ax.set_title(title,fontsize=9,fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("Density",fontsize=8)
        ax.text(.97,.97,f"n={len(data)}\nmed={data.median():.1f}",transform=ax.transAxes,ha="right",va="top",fontsize=7,color="#444")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); return buf

def plot_ro5(df):
    v=pd.Series(0,index=df.index)
    if "MW" in df.columns: v+=(df["MW"]>500).astype(int)
    if "LogP" in df.columns: v+=(df["LogP"]>5).astype(int)
    if "HBD" in df.columns: v+=(df["HBD"]>5).astype(int)
    if "HBA" in df.columns: v+=(df["HBA"]>10).astype(int)
    counts=v.value_counts().sort_index()
    colors=["#27AE60","#F39C12","#E74C3C","#8E44AD","#2C3E50"]
    fig,axes=plt.subplots(1,2,figsize=(12,4)); fig.patch.set_facecolor("white")
    labels=[f"{x} violation{'s' if x!=1 else ''}" for x in counts.index]
    axes[0].pie(counts.values,labels=labels,colors=colors[:len(counts)],autopct="%1.1f%%",startangle=90,textprops={"fontsize":9})
    axes[0].set_title("Lipinski Ro5 Violations",fontweight="bold",fontsize=11)
    axes[1].set_facecolor("#f8f9fa")
    bars=axes[1].bar(counts.index.astype(str),counts.values,color=colors[:len(counts)],edgecolor="white")
    for b,val in zip(bars,counts.values): axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+.3,str(val),ha="center",fontsize=9,fontweight="bold")
    axes[1].set_xlabel("Violations",fontsize=10); axes[1].set_ylabel("Count",fontsize=10)
    axes[1].set_title("Violation Distribution",fontweight="bold",fontsize=11)
    for sp in ["top","right"]: axes[1].spines[sp].set_visible(False)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); return buf

def plot_pca(smiles_list, highlight=None):
    fps,vidx=[],[]
    for i,smi in enumerate(smiles_list):
        fp=get_fp(smi)
        if fp is not None: fps.append(list(fp)); vidx.append(i)
    if len(fps)<3:
        fig,ax=plt.subplots(figsize=(5,4)); ax.text(.5,.5,"Not enough molecules for PCA",ha="center"); buf=io.BytesIO(); plt.savefig(buf,format="png"); plt.close(fig); return buf
    X=StandardScaler().fit_transform(np.array(fps,dtype=np.float32))
    pca=PCA(n_components=2,random_state=42); coords=pca.fit_transform(X)
    fig,ax=plt.subplots(figsize=(9,7)); fig.patch.set_facecolor("white"); ax.set_facecolor("#f8f9fa")
    sc=ax.scatter(coords[:,0],coords[:,1],c=range(len(coords)),cmap="viridis",s=55,alpha=.75,edgecolor="white",lw=.4,zorder=3)
    plt.colorbar(sc,ax=ax,label="Compound index",shrink=.8)
    if highlight is not None and highlight in vidx:
        hi=vidx.index(highlight)
        ax.scatter(coords[hi,0],coords[hi,1],c="#e94560",s=220,marker="*",edgecolor="black",lw=1.2,zorder=5,label="Selected")
        ax.legend(fontsize=9)
    pct=pca.explained_variance_ratio_*100
    ax.set_xlabel(f"PC1 ({pct[0]:.1f}% variance)",fontsize=11); ax.set_ylabel(f"PC2 ({pct[1]:.1f}% variance)",fontsize=11)
    ax.set_title(f"Chemical Space — PCA of Morgan Fingerprints\n(n={len(coords)}, total variance: {pct.sum():.1f}%)",fontsize=12,fontweight="bold")
    ax.grid(True,ls=":",alpha=.4); [ax.spines[sp].set_visible(False) for sp in ["top","right"]]
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); return buf

def plot_umap(smiles_list, highlight=None):
    if not UMAP_AVAILABLE: return None
    fps,vidx=[],[]
    for i,smi in enumerate(smiles_list):
        fp=get_fp(smi)
        if fp is not None: fps.append(list(fp)); vidx.append(i)
    if len(fps)<10: return None
    X=np.array(fps,dtype=np.float32)
    reducer=umap_lib.UMAP(n_neighbors=min(15,len(fps)-1),min_dist=.3,metric="jaccard",random_state=42,verbose=False)
    coords=reducer.fit_transform(X)
    fig,ax=plt.subplots(figsize=(9,7)); fig.patch.set_facecolor("white"); ax.set_facecolor("#f8f9fa")
    sc=ax.scatter(coords[:,0],coords[:,1],c=range(len(coords)),cmap="plasma",s=55,alpha=.75,edgecolor="white",lw=.4,zorder=3)
    plt.colorbar(sc,ax=ax,label="Compound index",shrink=.8)
    if highlight is not None and highlight in vidx:
        hi=vidx.index(highlight)
        ax.scatter(coords[hi,0],coords[hi,1],c="#e94560",s=220,marker="*",edgecolor="black",lw=1.2,zorder=5,label="Selected")
        ax.legend(fontsize=9)
    ax.set_xlabel("UMAP-1",fontsize=11); ax.set_ylabel("UMAP-2",fontsize=11)
    ax.set_title("Chemical Space — UMAP (Jaccard, Morgan FP)\nCaptures non-linear scaffold clustering",fontsize=12,fontweight="bold")
    ax.grid(True,ls=":",alpha=.4); [ax.spines[sp].set_visible(False) for sp in ["top","right"]]
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); return buf

def plot_scaffold(smiles_list):
    scaffolds=[get_scaffold(s) for s in smiles_list]
    sc=pd.Series([s for s in scaffolds if s]).value_counts()
    uniq_frac=len(sc)/max(len(scaffolds),1)
    top=sc.head(20)
    fig,axes=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor("white")
    axes[0].set_facecolor("#f8f9fa")
    colors=plt.cm.tab20(np.linspace(0,1,len(top)))
    bars=axes[0].barh(range(len(top)),top.values,color=colors,edgecolor="white")
    axes[0].set_yticks(range(len(top))); axes[0].set_yticklabels([f"Scaffold {i+1}" for i in range(len(top))],fontsize=8)
    axes[0].set_xlabel("Count",fontsize=10)
    axes[0].set_title(f"Top-{len(top)} Bemis-Murcko Scaffolds\n({len(sc)} unique/{len(scaffolds)} total · {uniq_frac*100:.1f}% diversity)",fontsize=10,fontweight="bold")
    for b,v in zip(bars,top.values): axes[0].text(v+.1,b.get_y()+b.get_height()/2,str(v),va="center",fontsize=8)
    [axes[0].spines[sp].set_visible(False) for sp in ["top","right"]]
    cum=np.cumsum(sc.values)/len(scaffolds)
    axes[1].set_facecolor("#f8f9fa"); axes[1].plot(range(1,len(sc)+1),cum,color="#4A90D9",lw=2)
    axes[1].fill_between(range(1,len(sc)+1),cum,alpha=.15,color="#4A90D9")
    axes[1].axhline(.5,color="#e94560",ls="--",lw=1.5,label="50% coverage")
    axes[1].axhline(.9,color="#F39C12",ls="--",lw=1.5,label="90% coverage")
    axes[1].set_xlabel("Scaffolds (ranked)",fontsize=10); axes[1].set_ylabel("Cumulative Coverage",fontsize=10)
    axes[1].set_title("Scaffold Coverage Curve (Cyclic System Recovery)",fontsize=10,fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True,ls=":",alpha=.4)
    [axes[1].spines[sp].set_visible(False) for sp in ["top","right"]]
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); return buf

def plot_tanimoto(smiles_list, max_mols=50):
    sub=smiles_list[:max_mols]; fps=[get_fp(s) for s in sub]; fps=[f for f in fps if f]
    n=len(fps)
    if n<3:
        fig,ax=plt.subplots(figsize=(4,3)); ax.text(.5,.5,"Too few molecules",ha="center"); buf=io.BytesIO(); plt.savefig(buf,format="png"); plt.close(fig); return buf
    mat=np.array([[DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in range(n)] for i in range(n)])
    fig,ax=plt.subplots(figsize=(min(n*.35+2,14),min(n*.35+2,14))); fig.patch.set_facecolor("white")
    sns.heatmap(mat,ax=ax,cmap="YlOrRd",vmin=0,vmax=1,
                xticklabels=[f"M{i+1}" for i in range(n)],yticklabels=[f"M{i+1}" for i in range(n)],
                linewidths=.3,linecolor="white",cbar_kws={"shrink":.7,"label":"Tanimoto Similarity"})
    ax.set_title(f"Pairwise Tanimoto Similarity Heatmap (Morgan FP, r=2)\nFirst {n} compounds",fontsize=11,fontweight="bold")
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=130,bbox_inches="tight"); plt.close(fig); return buf

def plot_logp_mw(df):
    if "MW" not in df.columns or "LogP" not in df.columns:
        fig,ax=plt.subplots(figsize=(5,4)); ax.text(.5,.5,"MW/LogP unavailable",ha="center"); buf=io.BytesIO(); plt.savefig(buf,format="png"); plt.close(fig); return buf
    fig,ax=plt.subplots(figsize=(9,6)); fig.patch.set_facecolor("white"); ax.set_facecolor("#f8f9fa")
    ax.add_patch(Rectangle((0,-10),500,15,lw=2,edgecolor="#27AE60",facecolor="#27AE60",alpha=.06,label="Ro5 zone"))
    qed=df["QED"] if "QED" in df.columns else None
    sc=ax.scatter(df["MW"],df["LogP"],c=qed,cmap="viridis",s=50,alpha=.75,edgecolor="white",lw=.4,zorder=3)
    if qed is not None: plt.colorbar(sc,ax=ax,label="QED",shrink=.8)
    ax.axvline(500,color="#e94560",ls="--",lw=1.5,alpha=.8,label="MW=500")
    ax.axhline(5,color="#e94560",ls=":",lw=1.5,alpha=.8,label="LogP=5")
    ax.set_xlabel("Molecular Weight (Da)",fontsize=11); ax.set_ylabel("Lipophilicity (LogP)",fontsize=11)
    ax.set_title("Lipophilicity–MW Landscape (colour = QED)",fontsize=12,fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True,ls=":",alpha=.4); [ax.spines[sp].set_visible(False) for sp in ["top","right"]]
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); return buf

def plot_admet_heatmap(adf, keys):
    avail=[p for p in keys if p in adf.columns]
    if not avail:
        fig,ax=plt.subplots(); ax.text(.5,.5,"No ADMET props",ha="center"); buf=io.BytesIO(); plt.savefig(buf,format="png"); plt.close(fig); return buf
    sub=adf[avail].copy()
    for col in avail:
        mn,mx=sub[col].min(),sub[col].max(); sub[col]=(sub[col]-mn)/(mx-mn+1e-9)
    fig,ax=plt.subplots(figsize=(max(10,len(avail)*.8),max(6,len(sub)*.22+2))); fig.patch.set_facecolor("white")
    sns.heatmap(sub,ax=ax,cmap="RdYlGn_r",vmin=0,vmax=1,xticklabels=avail,yticklabels=False,
                cbar_kws={"shrink":.6,"label":"Normalised score (0=best, 1=worst)"},linewidths=0)
    ax.set_title("ADMET Library Heatmap — Key Properties\n(Normalised; red = high risk)",fontsize=12,fontweight="bold")
    ax.set_xlabel("ADMET Property",fontsize=10); ax.set_ylabel(f"Compounds (n={len(sub)})",fontsize=10)
    plt.xticks(rotation=45,ha="right",fontsize=8)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=130,bbox_inches="tight"); plt.close(fig); return buf

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION ENGINES
# ═══════════════════════════════════════════════════════════════════════════════
_BIO_PAIRS=[
    ("c1ccccc1","c1ccncc1"),("c1ccncc1","c1ccccc1"),("c1ccccc1","c1ccnc(n1)"),
    ("[OH]","[NH2]"),("[NH2]","[OH]"),("C(=O)[OH]","c1nnn[nH]1"),
    ("C(=O)O[CX4]","C(=O)N"),("[Cl]","[F]"),("[F]","[Cl]"),
    ("[Cl]","C(F)(F)F"),("[CH3]","C(F)(F)F"),("[nH]","o"),
    ("C(=O)","S(=O)(=O)"),("[CH2][CH2]","[CH2]O"),("[CH3]","[H]"),("C1CCCCC1","C1CCCC1"),
]

def run_crem(smi,targets,db):
    mol=Chem.MolFromSmiles(smi); out=[]
    for ai in targets:
        for r,tag in [(1,"CReM-r1"),(2,"CReM-r2")]:
            try:
                hits=list(mutate_mol(mol,db_name=db,radius=r,max_replacements=None,replace_ids=[ai],return_mol=False))
                for m in hits: out.append({"smiles":m,"target_atom":ai,"method":tag})
                if hits: break
            except Exception: pass
    seen={}
    for d in out: seen[d["smiles"]]=d
    return list(seen.values())

def run_bioisostere(smi,targets):
    mol=Chem.MolFromSmiles(smi)
    if not mol: return []
    results,seen=[],set()
    for sf,st_smi in _BIO_PAIRS:
        try:
            patt=Chem.MolFromSmarts(sf); repl=Chem.MolFromSmiles(st_smi)
            if not patt or not repl: continue
            for prod in AllChem.ReplaceSubstructs(mol,patt,repl,replaceAll=False):
                try:
                    Chem.SanitizeMol(prod); ps=Chem.MolToSmiles(prod)
                    if ps and ps not in seen and ps!=smi:
                        seen.add(ps); match=mol.GetSubstructMatch(patt)
                        ai=match[0] if match else (targets[0] if targets else 0)
                        results.append({"smiles":ps,"target_atom":ai,"method":"Bioisostere"})
                except Exception: pass
        except Exception: pass
    return results

def run_simwalk(smi,targets,n_rounds=12,n_perturb=30,min_sim=.35):
    mol=Chem.MolFromSmiles(smi)
    if not mol: return []
    pfp=AllChem.GetMorganFingerprintAsBitVect(mol,2,2048); PAL=[6,7,8,9,16,17,35]
    results,seen=[],{smi}; atoms=targets if targets else list(range(mol.GetNumAtoms()))
    for _ in range(n_rounds):
        for _ in range(n_perturb):
            try:
                rw=Chem.RWMol(mol); chosen=random.sample(atoms,min(random.choice([1,2]),len(atoms)))
                for ai in chosen:
                    rw.GetAtomWithIdx(ai).SetAtomicNum(random.choice(PAL))
                    rw.GetAtomWithIdx(ai).SetFormalCharge(0); rw.GetAtomWithIdx(ai).SetNumExplicitHs(0)
                Chem.SanitizeMol(rw); ps=Chem.MolToSmiles(rw)
                if ps in seen: continue
                fp=AllChem.GetMorganFingerprintAsBitVect(rw,2,2048)
                if DataStructs.TanimotoSimilarity(pfp,fp)>=min_sim:
                    seen.add(ps); results.append({"smiles":ps,"target_atom":chosen[0],"method":"SimWalk"})
            except Exception: pass
    return results

def run_cascade(smi,targets,db,crem_ok):
    results,msg=[],""
    if crem_ok:
        results=run_crem(smi,targets,db)
        if results: msg=f"✅ **CReM:** {len(results)} ChEMBL fragments found."
    if len(results)<20:
        bio=run_bioisostere(smi,targets); new=[h for h in bio if h["smiles"] not in {r["smiles"] for r in results}]
        results+=new; msg+=f"\n🔬 **Bioisostere rules:** +{len(new)} transforms."
    if len(results)<20:
        walk=run_simwalk(smi,targets); new=[h for h in walk if h["smiles"] not in {r["smiles"] for r in results}]
        results+=new; msg+=f"\n🎲 **Similarity Walk:** +{len(new)} analogues (Tanimoto ≥ 0.35)."
    if not msg: msg="⚠️ No molecules generated. Try different atoms."
    return results,msg

_R4_TOML="""\
run_type = "sampling"
device   = "{device}"
[parameters]
model_file       = "{model_file}"
smiles_file      = "{smiles_file}"
output_file      = "{output_csv}"
num_smiles       = {num_smiles}
temperature      = {temperature}
unique_molecules = true
randomize_smiles = true
"""

def find_r4_exe():
    found=shutil.which("reinvent")
    if found: return found
    for c in [os.path.expanduser("~/reinvent_env/bin/reinvent"),
              os.path.expanduser("~/.local/bin/reinvent"),
              os.path.join(sys.prefix,"bin","reinvent")]:
        if os.path.isfile(c) and os.access(c,os.X_OK): return c
    return None

def find_r4_prior():
    import importlib.util
    spec=importlib.util.find_spec("reinvent")
    if spec and spec.origin:
        for root,_,files in os.walk(os.path.dirname(spec.origin)):
            for f in files:
                if f.endswith(".prior") and "mol2mol" in f.lower(): return os.path.join(root,f)
    for c in [os.path.expanduser("~/REINVENT4/priors/mol2mol.prior"),
              os.path.expanduser("~/reinvent4/priors/mol2mol.prior"),
              "./priors/mol2mol.prior","./mol2mol.prior"]:
        if os.path.isfile(c): return c
    return None

def run_reinvent4(smiles,exe,prior,num_smiles=200,temperature=1.2,
                  cond="Solubility_no_change Clint_no_change LogD_(1.9, 2.1]"):
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    tmpdir=tempfile.mkdtemp(prefix="chemxplain_r4_")
    try:
        smi_file=os.path.join(tmpdir,"input.smi"); toml_file=os.path.join(tmpdir,"run.toml"); out_csv=os.path.join(tmpdir,"output.csv")
        mol=Chem.MolFromSmiles(smiles)
        if not mol: return [], "❌ Invalid input SMILES."
        # Build SMILES safe for the Mol2Mol prior vocab.
        # The prior tokeniser knows stereochemistry ([C@@H] etc.) but NOT isotope
        # labels ([2H], [13C]) or ring closures >= %10.  Try isomeric first; if any
        # bracket token is absent from the known vocab, fall back to non-isomeric.
        _R4_VOCAB = {
            'C','c','N','n','O','o','S','s','P','p','F','Cl','Br','I',
            '[C@@H]','[C@H]','[C@@]','[C@]','[N@@+]','[N@+]','[n+]','[n-]',
            '[nH]','[NH+]','[NH2+]','[O-]','[O]','[S@@]','[S@]','[s+]',
            '[se]','[te]','[11C]','[11CH3]','[18F]','[19F]','[123I]',
            '=','#','-','(',')','/','\\\\','.','*','^','$','%',
            '1','2','3','4','5','6','7','8','9',
        }
        import re as _re
        def _smiles_ok(s):
            brackets = _re.findall(r'\[[^\]]+\]', s)
            return all(b in _R4_VOCAB for b in brackets)
        clean_iso = Chem.MolToSmiles(mol, isomericSmiles=True).strip()
        clean = clean_iso if _smiles_ok(clean_iso) else Chem.MolToSmiles(mol, isomericSmiles=False).strip()
        with open(smi_file,"w") as f: f.write(f"{clean}\t{cond}\n")
        with open(toml_file,"w") as f: f.write(_R4_TOML.format(device=device,model_file=prior,smiles_file=smi_file,output_csv=out_csv,num_smiles=num_smiles,temperature=temperature))
        res=subprocess.run([exe,toml_file],capture_output=True,text=True,timeout=600)
        if res.returncode!=0: return [],f"❌ REINVENT4 error (code {res.returncode}):\n```\n{res.stderr[-1500:]}\n```"
        if not os.path.isfile(out_csv): return [],"❌ REINVENT4 produced no output CSV."
        df=pd.read_csv(out_csv); col=next((c for c in df.columns if c.strip().upper() in ("SMILES","SMILE")),None)
        if not col: return [],f"❌ No SMILES column. Columns: {list(df.columns)}"
        out,seen=[],set()
        for s in df[col].dropna().astype(str):
            if s in seen or s==smiles: continue
            m=Chem.MolFromSmiles(s)
            if m:
                try:
                    Chem.SanitizeMol(m); canon=Chem.MolToSmiles(m)
                    if canon not in seen: seen.add(canon); out.append({"smiles":canon,"target_atom":0,"method":"REINVENT4"})
                except Exception: pass
        return out,f"✅ **REINVENT4:** {len(out)} valid molecules generated."
    except subprocess.TimeoutExpired: return [],"❌ REINVENT4 timed out (10 min)."
    except Exception as e: return [],f"❌ REINVENT4 error: {e}"
    finally: shutil.rmtree(tmpdir,ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ████  MODE A — LIBRARY ANALYSIS  ████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode.startswith("📚"):
    st.markdown('<p class="section-hdr">📚 Library Analysis — Batch Mode</p>', unsafe_allow_html=True)
    st.markdown("Upload a **CSV or Excel** file with SMILES strings. "
                "The platform performs publication-quality EDA, computes all 66 ADMET properties, "
                "and lets you select any molecule for single-compound XAI deep dive.")

    uploaded=st.file_uploader("Upload compound library (CSV / XLSX / XLS):",type=["csv","xlsx","xls"])

    if uploaded:
        try:
            raw_df=pd.read_excel(uploaded) if uploaded.name.endswith((".xlsx",".xls")) else pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Cannot read file: {e}"); st.stop()

        st.success(f"✅ Loaded {len(raw_df)} rows × {len(raw_df.columns)} columns")
        with st.expander("Preview uploaded data"):
            st.dataframe(raw_df.head(10),use_container_width=True)

        smi_cands=[c for c in raw_df.columns if any(x in c.lower() for x in ["smiles","smi","mol","structure"])]
        smiles_col=st.selectbox("Select the SMILES column:",raw_df.columns.tolist(),
                                index=raw_df.columns.tolist().index(smi_cands[0]) if smi_cands else 0)

        smiles_raw=raw_df[smiles_col].dropna().astype(str).tolist()
        valid_smiles,invalid_count=[],0
        for s in smiles_raw:
            mol=Chem.MolFromSmiles(s)
            if mol: valid_smiles.append(Chem.MolToSmiles(mol))
            else: invalid_count+=1

        c1,c2,c3=st.columns(3)
        c1.metric("Uploaded",len(smiles_raw)); c2.metric("Valid",len(valid_smiles),f"-{invalid_count} invalid" if invalid_count else None)
        c3.metric("Unique Scaffolds",len(set(get_scaffold(s) for s in valid_smiles if get_scaffold(s))))

        if not valid_smiles: st.error("❌ No valid SMILES found."); st.stop()
        if invalid_count>0: st.warning(f"⚠️ {invalid_count} unparseable SMILES excluded.")

        st.markdown("---")
        if st.button("🔬 Run EDA + Batch ADMET Profiling",type="primary",use_container_width=True):
            with st.spinner("⚗️ Computing RDKit descriptors…"):
                desc_rows=[compute_descriptors(s) for s in valid_smiles]
                desc_df=pd.DataFrame(desc_rows); desc_df.insert(0,"SMILES",valid_smiles)
                st.session_state["lib_desc"]=desc_df; st.session_state["lib_smi"]=valid_smiles

            with st.spinner("🧠 Running all 66 ADMET models on entire library…"):
                adf=admet_model.predict(smiles=valid_smiles); adf.insert(0,"SMILES",valid_smiles)
                st.session_state["lib_admet"]=adf

            with st.spinner("📊 Generating publication-quality plots…"):
                KEY_ADMET=["hERG","DILI","Ames","BBB_Martins","Pgp_Broccatelli","Solubility","Clearance_Hepatocyte","Half_Life","CYP3A4_Substrate"]
                st.session_state.update({
                    "plt_physico": plot_physico(desc_df),
                    "plt_ro5":     plot_ro5(desc_df),
                    "plt_pca":     plot_pca(valid_smiles),
                    "plt_scaffold":plot_scaffold(valid_smiles),
                    "plt_landscape":plot_logp_mw(desc_df),
                    "plt_tanimoto":plot_tanimoto(valid_smiles),
                    "plt_admet_hm":plot_admet_heatmap(adf,KEY_ADMET),
                    "plt_umap":    plot_umap(valid_smiles) if UMAP_AVAILABLE else None,
                    "eda_done":    True,
                })
            st.rerun()

        if st.session_state.get("eda_done"):
            desc_df   = st.session_state["lib_desc"]
            adf       = st.session_state["lib_admet"]
            smiles_lib= st.session_state["lib_smi"]

            # Summary row
            st.markdown('<p class="section-hdr">📊 Library Summary</p>',unsafe_allow_html=True)
            ro5_pass=sum(passes_lipinski(s) for s in smiles_lib)
            uniq_sc=len(set(get_scaffold(s) for s in smiles_lib if get_scaffold(s)))
            m1,m2,m3,m4,m5,m6=st.columns(6)
            m1.metric("Compounds",len(smiles_lib)); m2.metric("Ro5 Compliant",f"{ro5_pass} ({ro5_pass/len(smiles_lib)*100:.0f}%)")
            m3.metric("Median MW",f"{desc_df['MW'].median():.0f} Da" if "MW" in desc_df.columns else "N/A")
            m4.metric("Median LogP",f"{desc_df['LogP'].median():.2f}" if "LogP" in desc_df.columns else "N/A")
            m5.metric("Median QED",f"{desc_df['QED'].median():.3f}" if "QED" in desc_df.columns else "N/A")
            m6.metric("Unique Scaffolds",f"{uniq_sc} ({uniq_sc/len(smiles_lib)*100:.0f}%)")

            # EDA tabs
            st.markdown('<p class="section-hdr">🎨 Exploratory Data Analysis</p>',unsafe_allow_html=True)
            ptabs=st.tabs(["📐 Physicochemical","🟢 Ro5 Compliance","🗺️ PCA Space","🌐 UMAP",
                           "🏗️ Scaffolds","🔥 Tanimoto","🌄 LogP–MW","🧪 ADMET Heatmap"])

            for tab, key, cap, fname in [
                (ptabs[0],"plt_physico","Property distributions — blue dotted = median, red dashed = Ro5 limit","physico.png"),
                (ptabs[1],"plt_ro5","Lipinski Ro5 violation counts","ro5.png"),
                (ptabs[2],"plt_pca","PCA of 2048-bit Morgan fingerprints (r=2)","pca.png"),
                (ptabs[4],"plt_scaffold","Bemis-Murcko scaffold frequency + cumulative coverage curve","scaffold.png"),
                (ptabs[5],"plt_tanimoto","Pairwise Tanimoto similarity matrix (first 50 compounds)","tanimoto.png"),
                (ptabs[6],"plt_landscape","LogP vs MW landscape coloured by QED — green = Ro5 zone","landscape.png"),
                (ptabs[7],"plt_admet_hm","Key ADMET properties normalised 0–1. Red = high risk.","admet_hm.png"),
            ]:
                with tab:
                    buf=st.session_state.get(key)
                    img_dl(buf,cap,f"📥 Download PNG",fname)

            with ptabs[3]:
                if UMAP_AVAILABLE and st.session_state.get("plt_umap"):
                    img_dl(st.session_state["plt_umap"],"UMAP — Jaccard distance, Morgan FP","📥 Download PNG","umap.png")
                else:
                    st.info("Install `umap-learn` to enable UMAP: `pip install umap-learn`")

            # Gemini narrative
            st.markdown('<p class="section-hdr">🤖 AI Library Report</p>',unsafe_allow_html=True)
            if not api_key:
                st.warning("Enter your Gemini API Key in the sidebar to generate the AI report.")
            else:
                if st.button("✍️ Generate AI Library Report",use_container_width=True):
                    with st.spinner("Gemini analysing library…"):
                        try:
                            prompt=f"""
You are a senior computational medicinal chemist reviewing a chemical library EDA.
Library: {len(smiles_lib)} compounds
Median MW: {desc_df['MW'].median():.1f} Da | Median LogP: {desc_df['LogP'].median():.2f}
Median QED: {desc_df['QED'].median():.3f} | Median TPSA: {desc_df['TPSA'].median():.1f} Å²
Lipinski compliant: {ro5_pass}/{len(smiles_lib)} ({ro5_pass/len(smiles_lib)*100:.0f}%)
Unique Bemis-Murcko scaffolds: {uniq_sc} ({uniq_sc/len(smiles_lib)*100:.0f}% diversity)

Write a 4-part scientific characterisation:
PART 1 — DRUG-LIKENESS: Physicochemical space vs oral drug standards (Ro5, Veber rules).
PART 2 — CHEMICAL DIVERSITY: Scaffold coverage and chemical space analysis.
PART 3 — ADMET RISK PROFILE: Likely liabilities based on MW/LogP/TPSA trends.
PART 4 — PRIORITISATION STRATEGY: Which compound subsets to advance and why.
Be concise, specific, and scientifically rigorous. No bullet points.
"""
                            st.session_state["lib_gemini"]=call_gemini(prompt,api_key)
                        except Exception as e:
                            st.error(f"Gemini error: {e}")
                if st.session_state.get("lib_gemini"):
                    st.info(st.session_state["lib_gemini"])

            # Full ADMET table
            st.markdown('<p class="section-hdr">🧬 Full ADMET Results — All 66 Properties</p>',unsafe_allow_html=True)
            st.dataframe(adf,use_container_width=True,height=350)
            st.download_button("📥 Download Full ADMET Table (CSV)",
                               data=adf.to_csv(index=False).encode(),
                               file_name="library_admet_66.csv",mime="text/csv",use_container_width=True)

            # Deep dive selector
            st.markdown('<p class="section-hdr">🔬 Launch Deep Dive on a Molecule</p>',unsafe_allow_html=True)
            mol_opts={f"#{i+1}  —  {s[:65]}{'…' if len(s)>65 else ''}": s for i,s in enumerate(smiles_lib)}
            selected_label=st.selectbox("Choose a compound:",list(mol_opts.keys()))
            selected_smi=mol_opts[selected_label]

            if st.button("🚀 Launch Deep Dive on This Molecule →",type="primary",use_container_width=True):
                st.session_state["deep_dive_smi"]=selected_smi; st.session_state["from_library"]=True
                st.session_state["stage"]=0; [st.session_state.pop(k,None) for k in ["v7_gemini","v8_gemini","optimization_run"]]
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ████  MODE B — SINGLE MOLECULE DEEP DIVE  ████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<p class="section-hdr">🔬 Single Molecule Deep Dive</p>',unsafe_allow_html=True)

    prefilled=st.session_state.get("deep_dive_smi","CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4")
    if st.session_state.pop("from_library",False):
        st.info(f"🔗 Loaded from Library Analysis: `{prefilled[:80]}…`")

    raw_smiles=st.text_input("Enter SMILES:",prefilled,help="Default: Terfenadine (hERG liability example)")
    c1,c2=st.columns(2)
    with c1:
        fav=["hERG","DILI","BBB_Martins","Solubility","Pgp_Broccatelli","Clearance_Hepatocyte","Half_Life","Ames"]
        prop1=st.selectbox("🎯 Primary property (XAI target):",fav+[p for p in all_props if p not in fav])
    with c2:
        prop2=st.selectbox("⚖️ Secondary property (trade-off axis):",fav+[p for p in all_props if p not in fav],index=3)

    if prop1==prop2:
        st.warning("⚠️ Primary and Secondary properties are the same — trade-off analysis will not be meaningful. Select two different properties for selectivity-driven optimisation.")

    # ── Step 1: Diagnostics ───────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔬 Run Full Diagnostics",type="primary",use_container_width=True):
        st.session_state.update({"stage":1,"v7_gemini":None,"v8_gemini":None,"optimization_run":False})

        with st.spinner("⚖️ pH 7.4 correction…"):
            ph_smi,changed=correct_ph(raw_smiles); st.session_state.ph_smiles=ph_smi

        if changed: st.info(f"🧪 **pH correction applied:** `{ph_smi}`")
        else:        st.info("🧪 **pH correction:** Molecule unchanged at pH 7.4.")

        mol_2d=Chem.MolFromSmiles(ph_smi)
        if not mol_2d: st.error("❌ Invalid SMILES."); st.session_state.stage=0; st.stop()

        ad=check_ad(ph_smi)
        (st.success if ad>.3 else (st.warning if ad>.15 else st.error))(
            f"{'🟢' if ad>.3 else '🟡' if ad>.15 else '🔴'} **Applicability Domain:** "
            f"{'High' if ad>.3 else 'Moderate' if ad>.15 else 'Low — novel chemotype'} confidence (Sim {ad:.2f})")

        pains=check_pains(mol_2d)
        if pains: st.warning(f"⚠️ **PAINS:** {', '.join(pains)}")
        else:     st.success("✅ **PAINS:** No structural alerts.")

        with st.spinner("🧠 Running 66 ADMET networks…"):
            preds_df=admet_model.predict(smiles=[ph_smi]); st.session_state.preds_df=preds_df
            st.session_state.base_score1=float(preds_df[prop1].values[0])
            if prop2 in preds_df.columns: st.session_state.base_score2=float(preds_df[prop2].values[0])
            else: st.warning(f"⚠️ '{prop2}' not found — trade-off skipped."); st.session_state.base_score2=float("nan")

        with st.spinner(f"🔬 Occlusion XAI — masking {mol_2d.GetNumAtoms()} atoms…"):
            norm_w,abs_w,dir_w=occlusion_xai(ph_smi,prop1,st.session_state.base_score1,admet_model)
            st.session_state.update({"norm_w":norm_w,"abs_w":abs_w,"dir_w":dir_w})
            ranking=sorted([(i,w,mol_2d.GetAtomWithIdx(i).GetSymbol()) for i,w in enumerate(abs_w)],key=lambda x:x[1],reverse=True)
            st.session_state.top_targets=[x[0] for x in ranking if x[1]>.02 and x[2]!="H"][:3]
            st.session_state.top_atoms_str=", ".join([f"{x[2]} (N{x[0]})" for x in ranking if x[1]>.02 and x[2]!="H"][:3])

        with st.spinner("🎨 Generating visualisations…"):
            st.session_state.radar_buf=radar_chart(preds_df.to_dict(orient="records")[0])
            AllChem.Compute2DCoords(mol_2d)

            # Sharp 2D map
            dr=rdMolDraw2D.MolDraw2DSVG(500,500); dr.drawOptions().useBWAtomPalette()
            h_atoms=[i for i,w in enumerate(norm_w) if w>.05]
            h_colors={i:plt.cm.YlOrRd(norm_w[i])[:3] for i in h_atoms}
            rdMolDraw2D.PrepareAndDrawMolecule(dr,mol_2d,highlightAtoms=h_atoms,highlightAtomColors=h_colors)
            dr.FinishDrawing(); st.session_state.svg_2d=dr.GetDrawingText()

            # Contour map
            d2=Draw.MolDraw2DCairo(500,500); d2.drawOptions().useBWAtomPalette()
            SimilarityMaps.GetSimilarityMapFromWeights(mol_2d,[float(w) for w in abs_w],d2,colorMap=CMAP_WR,contourLines=10,alpha=.35)
            d2.FinishDrawing(); st.session_state.contour_buf=io.BytesIO(d2.GetDrawingText())

            # Node bar chart
            fig,ax=plt.subplots(figsize=(8,5)); fig.patch.set_facecolor("white"); ax.set_facecolor("white")
            pairs=[(f"{mol_2d.GetAtomWithIdx(i).GetSymbol()} (N{i})",w) for i,w in enumerate(abs_w) if w>.01]
            if pairs:
                pairs=sorted(pairs,key=lambda x:x[1])[-15:]; xlabs,xvals=zip(*pairs); mx=max(xvals)
                ax.barh(xlabs,xvals,color=[plt.cm.YlOrRd(v/mx) for v in xvals],edgecolor="black")
            ax.set_xlabel("Occlusion Impact |Δ Score|",fontsize=11); ax.set_title(f"Occlusion Sensitivity — {prop1}",fontweight="bold")
            plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=150); plt.close(fig); st.session_state.node_buf=buf

            st.session_state.edge_svg=edge_ablation_svg(ph_smi,prop1,st.session_state.base_score1,admet_model)
            st.session_state.motif_buf=motif_ablation(ph_smi,prop1,st.session_state.base_score1,admet_model)
            st.session_state.tradeoff_buf=(tradeoff_scatter(ph_smi,prop1,prop2,st.session_state.base_score1,
                st.session_state.base_score2,admet_model,dir_w) if not pd.isna(st.session_state.base_score2) else None)
            st.session_state.heatmap_buf=mutation_heatmap(ph_smi,prop1,st.session_state.base_score1,abs_w,admet_model)
            st.session_state.gasteiger_buf=gasteiger_map(mol_2d)
            st.session_state.logp_buf=logp_map(mol_2d)

    # ── Render diagnostics ────────────────────────────────────────────────────
    if st.session_state.get("stage",0)>=1:
        ph_smi=st.session_state.ph_smiles; base_score=st.session_state.base_score1
        preds_df=st.session_state.preds_df; pd_=preds_df.to_dict(orient="records")[0]
        norm_w=st.session_state.norm_w

        col_l,col_r=st.columns([1.6,1])
        with col_r:
            st.markdown("### 📊 ADMET Profile")
            st.session_state.radar_buf.seek(0); st.image(st.session_state.radar_buf,use_container_width=True)
            st.caption(f"{prop1} = **{base_score:.3f}**")
            admet_widget("hERG Cardiotoxicity","hERG",pd_,.7)
            admet_widget("Liver Toxicity (DILI)","DILI",pd_,.7)
            admet_widget("Ames Mutagenicity","Ames",pd_,.5)
            admet_widget("BBB Penetration","BBB_Martins",pd_,.5,low_good=False)
            admet_widget("P-gp Substrate","Pgp_Broccatelli",pd_,.5,low_good=False)
            with st.expander("📊 All 66 predictions"): st.dataframe(preds_df.T,use_container_width=True)
            st.download_button("📥 ADMET CSV",data=preds_df.to_csv(index=False).encode(),file_name="admet_single.csv",mime="text/csv",use_container_width=True)

        with col_l:
            st.markdown("### 🎯 XAI Analysis")
            xtabs=st.tabs(["🧬 3D","🗺️ 2D Sharp","☁️ Contour","📊 Node","🔗 Bond","🧩 Motif","⚖️ Trade-Off","🧪 Mutations","⚛️ Physicochemical"])

            with xtabs[0]:
                mol_3d=Chem.AddHs(Chem.MolFromSmiles(ph_smi))
                AllChem.EmbedMolecule(mol_3d,AllChem.ETKDG()); AllChem.MMFFOptimizeMolecule(mol_3d)
                n_hvy=Chem.MolFromSmiles(ph_smi).GetNumAtoms(); w3d=list(norm_w)+[0.0]*(mol_3d.GetNumAtoms()-n_hvy)
                cmap3=mcolors.LinearSegmentedColormap.from_list("gyr",["green","yellow","red"])
                view=py3Dmol.view(width=600,height=450); view.addModel(Chem.MolToMolBlock(mol_3d),"mol")
                for idx,w in enumerate(w3d):
                    if mol_3d.GetAtomWithIdx(idx).GetAtomicNum()==1: continue
                    view.setStyle({"serial":idx},{"sphere":{"color":mcolors.to_hex(cmap3(w)),"radius":float(.3+w*.45),"opacity":.85},"stick":{"radius":.15,"color":"lightgrey"}})
                view.zoomTo(); showmol(view,height=450,width=600); st.caption("🔴 Red = high impact · 🟢 Green = low impact")

            with xtabs[1]:
                components.html(st.session_state.svg_2d,width=500,height=500)
                st.download_button("🖼️ Download SVG",data=st.session_state.svg_2d,file_name=f"{prop1}_sharp.svg",mime="image/svg+xml")

            with xtabs[2]:
                st.info("Continuous probability cloud — publication-ready contour map.")
                st.session_state.contour_buf.seek(0); st.image(st.session_state.contour_buf,use_container_width=True)
                st.session_state.contour_buf.seek(0)
                st.download_button("☁️ Download PNG",data=st.session_state.contour_buf,file_name=f"{prop1}_contour.png",mime="image/png")

            with xtabs[3]:
                st.session_state.node_buf.seek(0); st.image(st.session_state.node_buf,use_container_width=True)
                st.session_state.node_buf.seek(0)
                st.download_button("📊 Download PNG",data=st.session_state.node_buf,file_name=f"{prop1}_node.png",mime="image/png")
                st.caption("Zeiler & Fergus (2014) adapted for molecular graphs")

            with xtabs[4]:
                st.info("Each bond removed individually — Δ score identifies critical connectivity.")
                components.html(st.session_state.edge_svg,width=500,height=500)
                st.download_button("🖼️ Download SVG",data=st.session_state.edge_svg,file_name=f"{prop1}_bond.svg",mime="image/svg+xml")

            with xtabs[5]:
                if st.session_state.motif_buf:
                    img_dl(st.session_state.motif_buf,"Ring/motif ablation","🧩 Download PNG",f"{prop1}_motif.png")
                else: st.warning("No ring systems found.")

            with xtabs[6]:
                if st.session_state.get("tradeoff_buf"):
                    st.info(f"Bottom-left quadrant = atoms driving BOTH {prop1} AND {prop2}.")
                    img_dl(st.session_state.tradeoff_buf,"Quadrant analysis","⚖️ Download PNG","tradeoff.png")
                else: st.warning(f"'{prop2}' not available for trade-off analysis.")

            with xtabs[7]:
                if st.session_state.heatmap_buf:
                    img_dl(st.session_state.heatmap_buf,"Point mutation array","🧪 Download PNG","mutations.png")
                else: st.warning("No significant mutation targets found.")

            with xtabs[8]:
                pc1,pc2=st.columns(2)
                with pc1:
                    st.markdown("**⚡ Gasteiger Partial Charges**")
                    st.caption("Red = δ+ electrophilic · Blue = δ− nucleophilic")
                    img_dl(st.session_state.get("gasteiger_buf"),"Gasteiger map","⚡ Download","gasteiger.png")
                with pc2:
                    st.markdown("**🧴 Crippen LogP Contributions**")
                    st.caption("Pink = lipophilic hotspots · Green = hydrophilic regions")
                    img_dl(st.session_state.get("logp_buf"),"LogP map","🧴 Download","logp_contrib.png")

        # Gemini diagnostic
        st.markdown("---")
        st.markdown(f"### 🤖 AI Diagnostic Report — {prop1}")
        if not api_key:
            st.warning("Enter your Gemini API Key in the sidebar.")
        elif not st.session_state.get("v7_gemini"):
            with st.spinner("Gemini interpreting XAI results…"):
                try:
                    top_str=st.session_state.top_atoms_str or "diffuse structural features"
                    prompt=f"""
You are a Lead Medicinal Chemist reviewing a computational XAI analysis.
Molecule SMILES (pH 7.4): {ph_smi}
Target ADMET property: {prop1}
Model prediction: {base_score:.3f}
XAI method: Occlusion Sensitivity Analysis (Zeiler & Fergus, 2014)
Top causal atoms: {top_str}

Write a concise 2-part scientific report:
PART 1 — ADMET INTERPRETATION: What does {base_score:.3f} for {prop1} mean for drug developability?
PART 2 — MECHANISTIC EXPLANATION: Why did the network identify {top_str} as primary drivers?
End with ONE single highest-priority medicinal chemistry recommendation.
"""
                    st.session_state.v7_gemini=call_gemini(prompt,api_key)
                except Exception as e: st.error(f"Gemini error: {e}")
        if st.session_state.get("v7_gemini"): st.info(st.session_state.v7_gemini)

    # ── Step 2: Decision gate ─────────────────────────────────────────────────
    if st.session_state.get("stage",0)>=1:
        st.markdown("---")
        dec=st.radio("Proceed to generative optimisation?",
                     ["🔍 No — reviewing data only.","🚀 Yes — generate optimised analogues."],key="decision_radio")
        if dec.startswith("🚀"): st.session_state.stage=max(st.session_state.stage,2)

    # ── Step 3: Engine selector + generation ─────────────────────────────────
    if st.session_state.get("stage",0)>=2:
        st.markdown("---")
        st.markdown("### ⚗️ Step 3: Generation Engine")

        r4_exe=find_r4_exe(); r4_prior=find_r4_prior(); r4_ready=r4_exe and r4_prior
        badge="✅ Detected" if r4_ready else "⚠️ Not installed"
        border="#27AE60" if r4_ready else "#E67E22"

        ec1,ec2=st.columns(2)
        with ec1:
            st.markdown("""<div class='engine-card' style='border:2px solid #4A90D9'>
            <h4>🧩 CReM + Fallback</h4><b>Conservative bioisostere optimisation</b><br><br>
            • ChEMBL fragment DB (radius 1+2)<br>• 16 bioisostere transforms<br>
            • Similarity walk (Tanimoto ≥ 0.35)<br>• ✅ No additional install required</div>""",unsafe_allow_html=True)
        with ec2:
            st.markdown(f"""<div class='engine-card' style='border:2px solid {border}'>
            <h4>🤖 REINVENT4 &nbsp;<small style='color:{border}'>{badge}</small></h4>
            <b>AI-driven scaffold hopping (AstraZeneca)</b><br><br>
            • RL + Transformer — true de novo chemical space<br>
            • Mol2Mol property-conditioned generation<br>
            • Apache 2.0 open-source · GPU recommended</div>""",unsafe_allow_html=True)

        st.markdown("")
        engine=st.radio("Select engine:",["🧩 CReM + Fallback (instant)","🤖 REINVENT4 (scaffold hopping)"],key="engine_radio",horizontal=True)
        use_r4=engine.startswith("🤖")

        if use_r4 and not r4_ready:
            st.error("REINVENT4 not found on this system.")
            with st.expander("📋 Install REINVENT4"):
                st.markdown("""
**Requirements:** Python 3.10–3.12
```bash
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4 && pip install -e .
reinvent --help
```
Find prior: `python -c "import reinvent,os; print(os.path.dirname(reinvent.__file__))"`
After install, refresh this page — the engine will be auto-detected.
""")

        r4_num,r4_temp,r4_sol,r4_clt,r4_ld=200,1.2,"Solubility_no_change","Clint_no_change","LogD_(1.9, 2.1]"
        if use_r4:
            with st.expander("⚙️ REINVENT4 Settings",expanded=True):
                rc1,rc2=st.columns(2)
                with rc1:
                    r4_exe_in=st.text_input("reinvent executable:",value=r4_exe or "",placeholder="/path/to/reinvent")
                    r4_pri_in=st.text_input("Mol2Mol prior (.prior):",value=r4_prior or "",placeholder="/path/to/mol2mol.prior")
                    if r4_exe_in: r4_exe=r4_exe_in
                    if r4_pri_in: r4_prior=r4_pri_in
                with rc2:
                    r4_num=st.slider("Molecules to generate:",50,500,200,50)
                    r4_temp=st.slider("Temperature:",0.5,2.0,1.2,.1,help="Higher = more novel/diverse")
                st.markdown("**🎛️ Property Condition Tokens**")
                cc1,cc2,cc3=st.columns(3)
                with cc1: r4_sol=st.selectbox("Solubility:",["Solubility_no_change","Solubility_low->high","Solubility_high->low"])
                with cc2: r4_clt=st.selectbox("Clearance:",["Clint_no_change","Clint_low->high","Clint_high->low"])
                with cc3: r4_ld=st.selectbox("LogD bin:",sorted([
                    "LogD_(-0.1, 0.1]","LogD_(0.1, 0.3]","LogD_(0.3, 0.5]","LogD_(0.5, 0.7]","LogD_(0.7, 0.9]",
                    "LogD_(0.9, 1.1]","LogD_(1.1, 1.3]","LogD_(1.3, 1.5]","LogD_(1.5, 1.7]","LogD_(1.7, 1.9]",
                    "LogD_(1.9, 2.1]","LogD_(2.1, 2.3]","LogD_(2.3, 2.5]","LogD_(2.5, 2.7]","LogD_(2.7, 2.9]",
                    "LogD_(2.9, 3.1]","LogD_(3.1, 3.3]","LogD_(3.3, 3.5]","LogD_(3.5, 3.7]","LogD_(3.7, 3.9]",
                    "LogD_(3.9, 4.1]","LogD_(4.1, 4.3]","LogD_(4.3, 4.5]","LogD_(4.5, 4.7]","LogD_(4.7, 4.9]",
                    "LogD_(4.9, 5.1]","LogD_(5.1, 5.3]","LogD_(5.3, 5.5]","LogD_(5.5, 5.7]","LogD_(5.7, 5.9]",
                    "LogD_(5.9, 6.1]","LogD_(6.1, 6.3]","LogD_(6.3, 6.5]","LogD_(6.5, 6.7]","LogD_(6.7, 6.9]",
                    "LogD_(6.9, inf]","LogD_(-inf, -6.9]","LogD_(-6.9, -6.7]","LogD_(-6.7, -6.5]",
                    "LogD_(-6.5, -6.3]","LogD_(-6.3, -6.1]","LogD_(-6.1, -5.9]","LogD_(-5.9, -5.7]",
                    "LogD_(-5.7, -5.5]","LogD_(-5.5, -5.3]","LogD_(-5.3, -5.1]","LogD_(-5.1, -4.9]",
                    "LogD_(-4.9, -4.7]","LogD_(-4.7, -4.5]","LogD_(-4.5, -4.3]","LogD_(-4.3, -4.1]",
                    "LogD_(-4.1, -3.9]","LogD_(-3.9, -3.7]","LogD_(-3.7, -3.5]","LogD_(-3.5, -3.3]",
                    "LogD_(-3.3, -3.1]","LogD_(-3.1, -2.9]","LogD_(-2.9, -2.7]","LogD_(-2.7, -2.5]",
                    "LogD_(-2.5, -2.3]","LogD_(-2.3, -2.1]","LogD_(-2.1, -1.9]","LogD_(-1.9, -1.7]",
                    "LogD_(-1.7, -1.5]","LogD_(-1.5, -1.3]","LogD_(-1.3, -1.1]","LogD_(-1.1, -0.9]",
                    "LogD_(-0.9, -0.7]","LogD_(-0.7, -0.5]","LogD_(-0.5, -0.3]","LogD_(-0.3, -0.1]",
                ]), index=20)

        st.markdown("#### 🎯 Target Atoms & Filters")
        mol_2d_ph=Chem.MolFromSmiles(st.session_state.ph_smiles)
        heavy=[i for i in range(mol_2d_ph.GetNumAtoms()) if mol_2d_ph.GetAtomWithIdx(i).GetSymbol()!="H"]
        override=st.multiselect("Atoms to mutate (CReM/SimWalk — REINVENT4 uses full molecule):",
                                options=heavy,default=st.session_state.top_targets,
                                format_func=lambda x:f"{mol_2d_ph.GetAtomWithIdx(x).GetSymbol()} (N{x})")
        oc1,oc2=st.columns(2)
        with oc1:
            direction=st.selectbox("Goal:",["Minimise (reduce toxicity/liability)","Maximise (increase desired property)"])
            high_is_bad="Minimise" in direction
        with oc2:
            use_ro5=st.checkbox("Enforce Lipinski Ro5",value=True)

        # ── Secondary constraint (NEW V18) ────────────────────────────────────
        st.markdown("#### 🔒 Secondary Property Constraint")
        sc1, sc2 = st.columns([2,1])
        with sc1:
            use_sec_constraint = st.checkbox(
                f"Preserve secondary target '{prop2}' — discard analogues that worsen it",
                value=False,
                help="If checked, any generated molecule whose secondary property score worsens "
                     "beyond the tolerance below will be excluded from results. "
                     "Applies to both CReM and REINVENT4."
            )
        with sc2:
            sec_tolerance = st.slider(
                "Max allowed Δ (worsening):", 0.00, 0.20, 0.05, 0.01,
                disabled=not use_sec_constraint,
                help="0.05 means the secondary score may worsen by at most 0.05 from parent."
            )
        if use_sec_constraint and prop1 == prop2:
            st.error("❌ Cannot constrain secondary when Primary == Secondary. Please select different properties.")
            use_sec_constraint = False

        btn="🤖 Launch REINVENT4" if use_r4 else "🧩 Launch CReM + Fallback"
        if st.button(btn,type="primary",use_container_width=True):
            ph_s=st.session_state.ph_smiles; base_s=st.session_state.base_score1

            if use_r4:
                if not r4_exe or not os.path.isfile(r4_exe): st.error("❌ Cannot find reinvent executable."); st.stop()
                if not r4_prior or not os.path.isfile(r4_prior): st.error("❌ Cannot find Mol2Mol prior."); st.stop()
                with st.spinner(f"🤖 REINVENT4 generating {r4_num} molecules…"):
                    muts,tier_msg=run_reinvent4(ph_s,r4_exe,r4_prior,r4_num,r4_temp,f"{r4_sol} {r4_clt} {r4_ld}")
            else:
                if not override: st.error("❌ Select at least one target atom."); st.stop()
                crem_ok=os.path.exists(CREM_DB)
                if not crem_ok: st.warning("⚠️ CReM DB not found — using Bioisostere + SimWalk only.")
                with st.spinner("🧩 Running generation cascade…"):
                    muts,tier_msg=run_cascade(ph_s,override,CREM_DB,crem_ok)

            st.markdown(tier_msg)
            if not muts: st.error("No valid molecules generated. Try the other engine or different atoms."); st.stop()
            st.info(f"⚙️ {len(muts)} raw structures. Filtering…")

            if use_ro5:
                muts=[m for m in muts if passes_lipinski(m["smiles"])]
                st.info(f"After Ro5: {len(muts)} drug-like structures remain.")
                if not muts: st.error("All structures failed Ro5. Uncheck filter to see all."); st.stop()

            with st.spinner(f"🚀 ADMET-AI scoring {len(muts)} molecules…"):
                m_smi=[m["smiles"] for m in muts]; m_pred=admet_model.predict(smiles=m_smi)

            sa=[get_sa_score(s) for s in m_smi]
            res=pd.DataFrame({"SMILES":m_smi,"Method":[m.get("method","Unknown") for m in muts],
                              "Target_Atom_Mutated":[m["target_atom"] for m in muts],
                              "Original_Score":base_s,"New_Score":m_pred[prop1].values,"SA_Score":sa})
            res["Delta"]=res["New_Score"]-res["Original_Score"]
            res["Improved"]=(res["New_Score"]<base_s) if high_is_bad else (res["New_Score"]>base_s)

            # ── Secondary scoring (NEW V18) ───────────────────────────────────
            base_s2 = st.session_state.get("base_score2", float("nan"))
            if prop2 in m_pred.columns and not pd.isna(base_s2):
                res["Secondary_Score"] = m_pred[prop2].values
                res["Delta_Secondary"]  = res["Secondary_Score"] - base_s2
                if use_sec_constraint:
                    before = len(res)
                    res = res[res["Delta_Secondary"] <= sec_tolerance]
                    st.info(f"🔒 Secondary constraint '{prop2}' (tolerance ±{sec_tolerance:.2f}): "
                            f"{before - len(res)} molecules removed · {len(res)} remain.")
            else:
                res["Secondary_Score"] = float("nan")
                res["Delta_Secondary"]  = float("nan")

            res["Ro5_Pass"]=use_ro5; res=res.sort_values("New_Score",ascending=high_is_bad)
            st.session_state.update({"res_df":res,"optimization_run":True,"v8_gemini":None,
                                     "use_sec_constraint":use_sec_constraint,"prop2_saved":prop2})
            st.session_state.stage=max(st.session_state.stage,3)

    if st.session_state.get("optimization_run") and "res_df" in st.session_state:
        res=st.session_state.res_df; imp=res[res["Improved"]==True]
        ph_s=st.session_state.ph_smiles; base_s=st.session_state.base_score1
        _prop2_saved = st.session_state.get("prop2_saved", prop2)
        _has_sec = "Delta_Secondary" in res.columns and not res["Delta_Secondary"].isna().all()
        _constrained = st.session_state.get("use_sec_constraint", False)

        st.markdown("---")
        _sec_note = f" · 🔒 {_prop2_saved} constrained" if _constrained else ""
        st.success(f"✅ **{len(res)}** molecules generated · **{len(imp)}** improved {prop1}{_sec_note}")

        mc=res["Method"].value_counts()
        mc_cols=st.columns(len(mc))
        for col,(method,cnt) in zip(mc_cols,mc.items()): col.metric(method,cnt)

        st.download_button(f"📥 Download All {len(res)} Generated Molecules (CSV)",
                           data=res.to_csv(index=False).encode(),
                           file_name=f"{prop1}_generated.csv",mime="text/csv",use_container_width=True)

        if not imp.empty:
            top5=imp.head(5)
            st.markdown("#### 🏆 Top 5 Optimised Molecules")
            mols_draw=[Chem.MolFromSmiles(s) for s in top5["SMILES"]]
            # Build legend — include Δ_Secondary if available
            legends=[]
            for _,row in top5.iterrows():
                leg = f"[{row['Method']}]\n{prop1}:{row['New_Score']:.3f} Δ{row['Delta']:+.3f}\nSA:{row['SA_Score']:.1f}"
                if _has_sec and not pd.isna(row.get("Delta_Secondary", float("nan"))):
                    _ds = row["Delta_Secondary"]
                    _flag = "✓" if _ds <= 0.05 else "⚠"
                    leg += f"\n{_flag}{_prop2_saved[:8]}:Δ{_ds:+.3f}"
                legends.append(leg)
            img=Draw.MolsToGridImage(mols_draw,molsPerRow=5,subImgSize=(260,260),legends=legends,returnPNG=False)
            buf=io.BytesIO(); img.save(buf,format="PNG"); st.image(buf,use_container_width=True)
            st.caption("SA Score: 1 = trivially synthesisable · 10 = nearly impossible (Ertl & Schuffenhauer, 2009)")

            # Show secondary summary table if available
            if _has_sec:
                base_s2 = st.session_state.get("base_score2", float("nan"))
                st.markdown(f"**Secondary property '{_prop2_saved}' — parent baseline: `{base_s2:.3f}`**")
                _disp = top5[["SMILES","Method","New_Score","Delta","Secondary_Score","Delta_Secondary","SA_Score"]].copy()
                _disp.columns = ["SMILES","Method",f"{prop1}",f"Δ_{prop1}",f"{_prop2_saved}",f"Δ_{_prop2_saved}","SA"]
                def _clr(val):
                    try:
                        v=float(val)
                        if v > 0.05: return "background-color:#3d0a0a;color:#f87171"
                        if v < -0.05: return "background-color:#0a2d0a;color:#4ade80"
                        return "color:#94a3b8"
                    except: return ""
                st.dataframe(_disp.style.applymap(_clr, subset=[f"Δ_{_prop2_saved}",f"Δ_{prop1}"]),
                             use_container_width=True)

            st.markdown("### 🤖 AI Comparative Analysis")
            if not api_key: st.warning("Enter Gemini API Key to unlock.")
            elif not st.session_state.get("v8_gemini"):
                with st.spinner("Gemini writing comparative analysis…"):
                    try:
                        best=top5.iloc[0]
                        _sec_line = ""
                        if _has_sec and not pd.isna(best.get("Delta_Secondary", float("nan"))):
                            _sec_line = f"\nSecondary property {_prop2_saved}: {best['Secondary_Score']:.3f} (Δ = {best['Delta_Secondary']:+.3f} vs parent {st.session_state.get('base_score2',float('nan')):.3f})"
                        prompt=f"""
You are a Lead Medicinal Chemist reviewing an AI-generated lead optimisation.
Original molecule (pH 7.4): {ph_s}
Original {prop1}: {base_s:.3f}
Generation method: {best['Method']}
Best optimised analogue: {best['SMILES']}
New {prop1}: {best['New_Score']:.3f}  (Δ = {best['Delta']:+.3f}){_sec_line}
SA Score: {best['SA_Score']:.1f}/10

Write a rigorous 3-part scientific report:
PART 1 — STRUCTURAL CHANGE: Describe the structural difference between parent and analogue.
PART 2 — PHARMACOLOGICAL RATIONALE: Explain WHY this change improved {prop1}.{"  Also explain whether the secondary property " + _prop2_saved + " was preserved or changed and what that means clinically." if _sec_line else ""}
PART 3 — SYNTHESIS FEASIBILITY: Comment on SA Score {best['SA_Score']:.1f}/10 and synthetic strategy.
"""
                        st.session_state.v8_gemini=call_gemini(prompt,api_key)
                    except Exception as e: st.error(f"Gemini error: {e}")
            if st.session_state.get("v8_gemini"): st.success(st.session_state.v8_gemini)
        else:
            st.warning(f"No molecules improved {prop1}. Try different atoms, relax Ro5, or switch to REINVENT4.")
