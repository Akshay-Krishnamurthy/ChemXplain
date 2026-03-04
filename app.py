"""
ChemXplain V13: State-of-the-Art Lead Optimization
=================================================
XAI Method: Occlusion Sensitivity Analysis (Zeiler & Fergus, 2014)
Pipeline: pH Correction → Applicability Domain → PAINS → ADMET Radar →
          Occlusion XAI (Sharp + Contour) → Physicochemical Maps (Gasteiger + LogP) →
          Interactive Target Override → Exhaustive CReM → SA Scoring

New in V13:
  - Gasteiger-Marsilli partial charge contour map (electron density overlay)
  - Crippen logP atom-contribution contour map (lipophilicity hotspot overlay)
  Both rendered via RDKit SimilarityMaps with the new C++ drawing backend
  (Landrum, G. RDKit Blog, 2020-01-03)
"""

import streamlit as st
import py3Dmol
from stmol import showmol
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, RDConfig, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.Chem import Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
import sys
import pandas as pd
import torch
from admet_ai import ADMETModel
from crem.crem import mutate_mol

# Thread-safe plotting for Streamlit
matplotlib.use('Agg')

# ── Gemini SDK Compatibility ──────────────────────────────────────────────────
try:
    from google import genai
    LEGACY_GEMINI = False
except ImportError:
    import google.generativeai as genai
    LEGACY_GEMINI = True

# ── SA Scorer: try RDKit Contrib path, fallback gracefully ──────────────────
SA_AVAILABLE = False
_sa_paths = [
    os.path.join(RDConfig.RDContribDir, 'SA_Score'),
    os.path.join(os.path.dirname(RDConfig.__file__), '..', 'Contrib', 'SA_Score'),
]
for _p in _sa_paths:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        try:
            import sascorer
            SA_AVAILABLE = True
            break
        except ImportError:
            pass

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="ChemXplain V13")
st.title("🧬 ChemXplain V13: Research-Grade Lead Optimization")
st.markdown(
    "**Pipeline:** pH 7.4 Correction → Applicability Domain → PAINS → ADMET Radar → "
    "Occlusion XAI → **Physicochemical Maps** → Exhaustive Generative AI → SA Scoring"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    st.sidebar.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("⚠️ CPU mode (no GPU detected)")

api_key    = st.sidebar.text_input("Gemini API Key:", type="password")
CREM_DB    = st.sidebar.text_input("CReM DB path:", value="chembl22_sa2_hac12.db")
if not os.path.exists(CREM_DB):
    st.sidebar.error(f"❌ CReM DB not found at: {CREM_DB}")
else:
    st.sidebar.success("✅ CReM DB found")

# ── Load ADMET models once ────────────────────────────────────────────────────
@st.cache_resource
def load_admet():
    return ADMETModel()

with st.spinner("⏳ Loading 66 deep learning models…"):
    admet_model = load_admet()

if "all_properties" not in st.session_state:
    _tmp = admet_model.predict(smiles=["C"])
    st.session_state.all_properties = sorted(list(_tmp.columns))

all_props = st.session_state.all_properties

# ── Stage initialisation ──────────────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = 0

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – USER INPUTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
raw_smiles = st.text_input(
    "Enter SMILES:",
    "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4",
    help="Terfenadine (default) — withdrawn antihistamine with hERG liability"
)

c1, c2 = st.columns(2)
with c1:
    favorites = ["hERG", "DILI", "BBB_Martins", "Solubility",
                 "Clearance_Hepatocyte", "Half_Life"]
    prop_opts  = favorites + [p for p in all_props if p not in favorites]
    prop1      = st.selectbox("🎯 Primary property (XAI target):", prop_opts)
with c2:
    prop2 = st.selectbox("⚖️ Secondary property (trade-off axis):", prop_opts, index=3)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – SCIENTIFIC HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def correct_ph(smi: str) -> tuple[str, bool]:
    try:
        from dimorphite_dl import protonate_smiles
        results = protonate_smiles(smi, ph_min=7.4, ph_max=7.4)
        if results and results[0] != smi:
            return results[0], True
        return smi, False
    except ImportError:
        try:
            from dimorphite_dl import DimorphiteDL
            dl = DimorphiteDL(min_ph=7.4, max_ph=7.4, max_variants=1, quiet=True)
            results = dl.protonate(smi)
            if results and results[0] != smi:
                return results[0], True
            return smi, False
        except Exception:
            return smi, False
    except Exception:
        return smi, False

def check_pains(mol) -> list[str]:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    entries = catalog.GetMatches(mol)
    return [e.GetDescription() for e in entries]

def check_applicability_domain(smi: str) -> float:
    ref_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", 
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "CN1CCC=C1C2=CN=CC=C2"
    ]
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    max_sim = 0.0
    for rs in ref_smiles:
        rmol = Chem.MolFromSmiles(rs)
        if rmol:
            rfp = AllChem.GetMorganFingerprintAsBitVect(rmol, 2, 2048)
            sim = DataStructs.TanimotoSimilarity(fp, rfp)
            if sim > max_sim: max_sim = sim
    return max_sim

def generate_radar_chart(preds_dict):
    labels = ['hERG', 'DILI', 'Ames', 'BBB (Martins)', 'CYP3A4']
    keys = ['hERG', 'DILI', 'Ames', 'BBB_Martins', 'CYP3A4_Inhibitor']
    values = [preds_dict.get(k, 0) for k in keys]
    values += [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1) 
    ax.set_title("ADMET Toxicity Risk Profile", y=1.1, fontweight="bold")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return buf

def passes_lipinski(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if not mol: return False
    violations = 0
    if Descriptors.MolWt(mol) > 500: violations += 1
    if Descriptors.MolLogP(mol) > 5: violations += 1
    if rdMolDescriptors.CalcNumHBD(mol) > 5: violations += 1
    if rdMolDescriptors.CalcNumHBA(mol) > 10: violations += 1
    return violations <= 1

def get_sa_score(smi: str) -> float:
    if not SA_AVAILABLE: return float('nan')
    mol = Chem.MolFromSmiles(smi)
    if not mol: return float('nan')
    return sascorer.calculateScore(mol)

def occlusion_xai(smi: str, prop: str, base_score: float, model) -> tuple[list, list, list]:
    base_mol = Chem.MolFromSmiles(smi)
    if not base_mol: return [], [], []
    try: Chem.Kekulize(base_mol, clearAromaticFlags=True)
    except Exception: pass
    n = base_mol.GetNumAtoms()
    pert_smiles, valid_idx = [], []
    for i in range(n):
        rw = Chem.RWMol(base_mol)
        rw.GetAtomWithIdx(i).SetAtomicNum(0)
        rw.GetAtomWithIdx(i).SetFormalCharge(0)
        try:
            pert_smiles.append(Chem.MolToSmiles(rw))
            valid_idx.append(i)
        except Exception: pass
    abs_w, dir_w = [0.0] * n, [0.0] * n
    if pert_smiles:
        preds = model.predict(smiles=pert_smiles)
        scores = preds[prop].values
        for idx, sc in zip(valid_idx, scores):
            if not pd.isna(sc):
                abs_w[idx] = abs(base_score - sc)
                dir_w[idx] = sc - base_score
    mn, mx = min(abs_w), max(abs_w)
    norm_w = [(w - mn) / (mx - mn) if mx != mn else 0.0 for w in abs_w]
    return norm_w, abs_w, dir_w

def edge_ablation_svg(smi: str, prop: str, base_score: float, model) -> str:
    mol = Chem.MolFromSmiles(smi)
    pert_smiles, valid_bonds = [], []
    for bond in mol.GetBonds():
        rw = Chem.RWMol(mol)
        rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        try:
            pert_smiles.append(Chem.MolToSmiles(rw))
            valid_bonds.append(bond.GetIdx())
        except Exception: pass
    bond_w = {b.GetIdx(): 0.0 for b in mol.GetBonds()}
    if pert_smiles:
        preds = model.predict(smiles=pert_smiles)
        for bidx, sc in zip(valid_bonds, preds[prop].values):
            if not pd.isna(sc): bond_w[bidx] = abs(base_score - sc)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    drawer.drawOptions().useBWAtomPalette()
    mx = max(bond_w.values()) if bond_w else 0
    cmap = plt.cm.YlOrRd
    h_bonds, h_colors = [], {}
    for bidx, w in bond_w.items():
        nw = (w / mx) if mx > 0 else 0
        if nw > 0.05:
            h_bonds.append(bidx)
            h_colors[bidx] = cmap(nw)[:3]
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightBonds=h_bonds, highlightBondColors=h_colors)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def motif_ablation_plot(smi: str, prop: str, base_score: float, model):
    mol = Chem.MolFromSmiles(smi)
    rings = mol.GetRingInfo().AtomRings()
    if not rings: return None
    pert_smiles, labels = [], []
    for i, ring in enumerate(rings):
        rw = Chem.RWMol(mol)
        for ai in ring: rw.GetAtomWithIdx(ai).SetAtomicNum(0)
        try:
            pert_smiles.append(Chem.MolToSmiles(rw))
            labels.append(f"Ring {i+1} ({len(ring)} atoms)")
        except Exception: pass
    if not pert_smiles: return None
    impacts = []
    preds = model.predict(smiles=pert_smiles)
    for sc in preds[prop].values:
        impacts.append(abs(base_score - sc) if not pd.isna(sc) else 0.0)
    fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.6)))
    sns.barplot(x=impacts, y=labels, palette="Reds_d", ax=ax, edgecolor=".2")
    ax.set_xlabel(f"Occlusion Impact on {prop} (|Δ Score|)", fontsize=10)
    ax.set_title("Ring/Motif Ablation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf

def mutation_heatmap(smi: str, prop: str, base_score: float, abs_w: list, model):
    mol = Chem.MolFromSmiles(smi)
    ranking = sorted([(i, w, mol.GetAtomWithIdx(i).GetSymbol()) for i, w in enumerate(abs_w)], key=lambda x: x[1], reverse=True)
    targets = [(i, sym) for i, w, sym in ranking if w > 0.01 and sym != 'H'][:5]
    if not targets: return None
    mutations = {'N': 7, 'O': 8, 'F': 9, 'S': 16}
    y_labels  = [f"{sym} (Node {i})" for i, sym in targets]
    matrix    = np.full((len(targets), len(mutations)), np.nan)
    pert_smiles, tracking = [], []
    for row, (ai, _) in enumerate(targets):
        for col, (sym, anum) in enumerate(mutations.items()):
            rw = Chem.RWMol(mol)
            atom = rw.GetAtomWithIdx(ai)
            if atom.GetAtomicNum() == anum:
                matrix[row, col] = base_score
                continue
            atom.SetAtomicNum(anum)
            atom.SetFormalCharge(0)
            try:
                Chem.SanitizeMol(rw)
                pert_smiles.append(Chem.MolToSmiles(rw))
                tracking.append((row, col))
            except Exception: pass
    if pert_smiles:
        preds = model.predict(smiles=pert_smiles)
        for (r, c), sc in zip(tracking, preds[prop].values): matrix[r, c] = sc
    fig, ax = plt.subplots(figsize=(6, max(3, len(targets) * 0.7)))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm", xticklabels=list(mutations.keys()), yticklabels=y_labels, ax=ax, linewidths=0.5, linecolor='grey')
    ax.set_title(f"Point Mutation Array\n(Base {prop}: {base_score:.3f})", fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf

def tradeoff_scatter(smi, p1, p2, s1, s2, model, dir_w1):
    _, _, dir_w2 = occlusion_xai(smi, p2, s2, model)
    mol    = Chem.MolFromSmiles(smi)
    labels = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(len(dir_w1))]
    abs_w1 = np.abs(dir_w1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.axvline(0, color='gray', lw=1, ls='--')
    sc = ax.scatter(dir_w1, dir_w2, c=abs_w1, cmap='coolwarm', s=100, edgecolor='black', alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label=f"|Impact on {p1}|")
    for i in np.argsort(abs_w1)[-5:]:
        if abs_w1[i] > 0.01:
            ax.annotate(labels[i], (dir_w1[i], dir_w2[i]), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    kw = dict(fontsize=9, alpha=0.55, transform=ax.transAxes)
    ax.text(0.51, 0.97, "Atom suppresses P1, drives P2",    va='top',  **kw)
    ax.text(0.01, 0.97, "Atom suppresses BOTH",             va='top',  **kw)
    ax.text(0.01, 0.02, "Atom DRIVES BOTH → best target",   va='bottom', color='red', fontweight='bold', **kw)
    ax.text(0.51, 0.02, "Atom drives P1, suppresses P2",    va='bottom', **kw)
    ax.set_xlabel(f"Δ {p1} when atom removed  (+→atom suppressed {p1})", fontsize=10)
    ax.set_ylabel(f"Δ {p2} when atom removed  (+→atom suppressed {p2})", fontsize=10)
    ax.set_title("Occlusion Quadrant Analysis\n(Identifying multi-property liabilities)", fontsize=12)
    ax.grid(True, ls=':', alpha=0.4)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return buf

def run_crem_exhaustive(smi: str, target_atoms: list, db: str) -> list:
    """Exhaustive CReM generation (max_replacements=None). Unlocks maximum valid bioisosteres."""
    mol = Chem.MolFromSmiles(smi)
    all_muts = []
    for ai in target_atoms:
        try:
            for m in mutate_mol(mol, db_name=db, radius=1, max_replacements=None, replace_ids=[ai], return_mol=False):
                all_muts.append({"smiles": m, "target_atom": ai})
        except Exception: pass
    seen = {}
    for d in all_muts: seen[d["smiles"]] = d
    return list(seen.values())

# ── NEW V13: Physicochemical Map Generators ───────────────────────────────────

def gasteiger_charge_map(mol) -> io.BytesIO:
    """
    Gasteiger-Marsilli partial charge contour map.
    Renders atom-level electron density as a continuous Gaussian probability
    cloud using RDKit's C++ SimilarityMaps backend.
    Reference: Landrum, G. RDKit Blog (2020-01-03).
               Gasteiger & Marsili, Tetrahedron (1980).
    Positive contours (red) = electron-poor atoms.
    Negative contours (blue) = electron-rich atoms.
    """
    from rdkit.Chem import rdPartialCharges
    try:
        mol_h = Chem.AddHs(mol)
        rdPartialCharges.ComputeGasteigerCharges(mol_h)
        # Collect charges for heavy atoms only (same indexing as mol)
        chgs = [
            mol_h.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")
            for i in range(mol.GetNumAtoms())
        ]
        # Replace any NaN/Inf that Gasteiger occasionally produces
        chgs = [c if np.isfinite(c) else 0.0 for c in chgs]
        d = Draw.MolDraw2DCairo(500, 500)
        d.drawOptions().useBWAtomPalette()
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, chgs, d,
            colorMap='RdBu_r',   # red = δ+, blue = δ−
            contourLines=10,
            alpha=0.30,
        )
        d.FinishDrawing()
        return io.BytesIO(d.GetDrawingText())
    except Exception as e:
        return None

def logp_contrib_map(mol) -> io.BytesIO:
    """
    Crippen logP atom-contribution contour map.
    Each heavy atom's additive contribution to the total Wildman-Crippen
    logP is visualised as a continuous overlay.
    Reference: Landrum, G. RDKit Blog (2020-01-03).
               Wildman & Crippen, J. Chem. Inf. Comput. Sci. (1999).
    Red regions = lipophilic hotspots (high +logP contribution).
    Blue regions = hydrophilic regions (negative logP contribution).
    """
    try:
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        logp_weights = [float(x[0]) for x in contribs]  # x[0]=logP, x[1]=MR
        d = Draw.MolDraw2DCairo(500, 500)
        d.drawOptions().useBWAtomPalette()
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, logp_weights, d,
            colorMap='PiYG_r',   # green = hydrophilic, pink = lipophilic
            contourLines=10,
            alpha=0.30,
        )
        d.FinishDrawing()
        return io.BytesIO(d.GetDrawingText())
    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
if st.button("🔬 Step 1: Run Full Diagnostics", type="primary"):
    st.session_state.stage          = 1
    st.session_state.v7_gemini      = None
    st.session_state.v8_gemini      = None
    st.session_state.optimization_run = False

    with st.spinner("⚖️ Correcting molecule to physiological pH 7.4…"):
        ph_smi, changed = correct_ph(raw_smiles)
        st.session_state.ph_smiles = ph_smi

    if changed: st.info(f"🧪 **pH correction applied.** Dominant species at pH 7.4:\n`{ph_smi}`")
    else: st.info("🧪 **pH correction:** Molecule unchanged at pH 7.4.")

    mol_2d = Chem.MolFromSmiles(ph_smi)
    if not mol_2d:
        st.error("❌ Invalid SMILES — cannot parse molecule.")
        st.session_state.stage = 0
        st.stop()

    # AD BADGE
    ad_score = check_applicability_domain(ph_smi)
    if ad_score > 0.3:
        st.success(f"🟢 **Applicability Domain:** High Confidence. Scaffold matches known pharmaceutical space (Sim: {ad_score:.2f}).")
    elif ad_score > 0.15:
        st.warning(f"🟡 **Applicability Domain:** Moderate Confidence. Scaffold is somewhat novel (Sim: {ad_score:.2f}).")
    else:
        st.error(f"🔴 **Applicability Domain:** Low Confidence. Alien chemotype detected. ADMET predictions may hallucinate (Sim: {ad_score:.2f}).")

    # PAINS
    pains_hits = check_pains(mol_2d)
    if pains_hits:
        st.warning(f"⚠️ **PAINS Alert** (Baell & Holloway, 2010): {', '.join(pains_hits)}. This molecule may be a frequent-hitter assay artifact. Interpret ADMET predictions with caution.")
    else:
        st.success("✅ **PAINS screening:** No structural alerts found.")

    with st.spinner("🧠 Running 66 ADMET neural networks…"):
        preds_df = admet_model.predict(smiles=[ph_smi])
        st.session_state.preds_df    = preds_df
        st.session_state.base_score1 = float(preds_df[prop1].values[0])
        # Guard: prop2 column may not exist if the model doesn't output it
        if prop2 in preds_df.columns:
            st.session_state.base_score2 = float(preds_df[prop2].values[0])
        else:
            st.warning(f"⚠️ Property '{prop2}' not found in model output. Trade-off axis will be skipped.")
            st.session_state.base_score2 = float('nan')

    with st.spinner(f"🔬 Occlusion Sensitivity Analysis — masking {mol_2d.GetNumAtoms()} atoms…"):
        norm_w, abs_w, dir_w = occlusion_xai(ph_smi, prop1, st.session_state.base_score1, admet_model)
        st.session_state.norm_w = norm_w
        st.session_state.abs_w  = abs_w
        st.session_state.dir_w  = dir_w

        ranking = sorted([(i, w, mol_2d.GetAtomWithIdx(i).GetSymbol()) for i, w in enumerate(abs_w)], key=lambda x: x[1], reverse=True)
        st.session_state.top_targets   = [x[0] for x in ranking if x[1] > 0.02 and x[2] != 'H'][:3]
        st.session_state.top_atoms_str = ", ".join([f"{x[2]} (Node {x[0]})" for x in ranking if x[1] > 0.02 and x[2] != 'H'][:3])

    with st.spinner("🎨 Generating visualisations…"):
        st.session_state.radar_buf = generate_radar_chart(preds_df.to_dict(orient='records')[0])
        
        # 1. SHARP 2D MAP
        AllChem.Compute2DCoords(mol_2d)
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
        drawer.drawOptions().useBWAtomPalette()
        h_atoms  = [i for i, w in enumerate(norm_w) if w > 0.05]
        h_colors = {i: plt.cm.YlOrRd(norm_w[i])[:3] for i in h_atoms}
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_2d, highlightAtoms=h_atoms, highlightAtomColors=h_colors)
        drawer.FinishDrawing()
        st.session_state.svg_2d = drawer.GetDrawingText()

        # 2. CONTOUR SIMILARITY MAP (Publication Style)
        # Cast abs_w to Python float list — RDKit ContourAndDrawGaussians rejects numpy.float32
        abs_w_float = [float(w) for w in st.session_state.abs_w]
        d = Draw.MolDraw2DCairo(500, 500)
        d.drawOptions().useBWAtomPalette()
        # draw2d must be the 3rd positional arg in newer RDKit versions
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol_2d,
            abs_w_float,
            d,
            colorMap='Reds',
            contourLines=10,
            alpha=0.25,
        )
        d.FinishDrawing()
        st.session_state.contour_buf = io.BytesIO(d.GetDrawingText())

        # 3. NODE BAR PLOT
        fig, ax = plt.subplots(figsize=(8, 5))
        pairs = [(f"{mol_2d.GetAtomWithIdx(i).GetSymbol()} (Node {i})", w) for i, w in enumerate(abs_w) if w > 0.01]
        if pairs:
            pairs = sorted(pairs, key=lambda x: x[1])[-15:]
            xlabs, xvals = zip(*pairs)
            mx = max(xvals)
            ax.barh(xlabs, xvals, color=[plt.cm.YlOrRd(v / mx) for v in xvals], edgecolor='black')
        else:
            ax.text(0.5, 0.5, "No significant atoms found", ha='center', va='center')
        ax.set_xlabel("Occlusion Impact |Δ Score|", fontsize=11)
        ax.set_title(f"Occlusion Sensitivity Analysis\nMethod: Zeiler & Fergus (2014) — Target: {prop1}", fontweight='bold')
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig)
        st.session_state.node_buf = buf

        st.session_state.edge_svg   = edge_ablation_svg(ph_smi, prop1, st.session_state.base_score1, admet_model)
        st.session_state.motif_buf  = motif_ablation_plot(ph_smi, prop1, st.session_state.base_score1, admet_model)
        if not pd.isna(st.session_state.base_score2):
            st.session_state.tradeoff_buf = tradeoff_scatter(ph_smi, prop1, prop2, st.session_state.base_score1, st.session_state.base_score2, admet_model, dir_w)
        else:
            st.session_state.tradeoff_buf = None
        st.session_state.heatmap_buf = mutation_heatmap(ph_smi, prop1, st.session_state.base_score1, abs_w, admet_model)

        # 4. NEW V13 — Physicochemical Maps
        st.session_state.gasteiger_buf = gasteiger_charge_map(mol_2d)
        st.session_state.logp_buf      = logp_contrib_map(mol_2d)

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get('stage', 0) >= 1:
    ph_smi     = st.session_state.ph_smiles
    base_score = st.session_state.base_score1
    preds_df   = st.session_state.preds_df
    norm_w     = st.session_state.norm_w
    preds_dict = preds_df.to_dict(orient='records')[0]

    col_left, col_right = st.columns([1.6, 1])

    with col_right:
        st.markdown("### 📊 ADMET Predictions")
        st.image(st.session_state.radar_buf, use_container_width=True)
        st.caption(f"Molecule at pH 7.4 · {prop1} = **{base_score:.3f}**")

        def show_metric(label, key, thresh, low_is_good=True):
            val = preds_dict.get(key, float('nan'))
            if pd.isna(val):
                st.info(f"ℹ️ **{label}:** N/A")
                return
            if low_is_good:
                if   val > thresh:       st.error(  f"🔴 **{label}:** {val:.3f} — HIGH RISK")
                elif val > thresh - 0.2: st.warning(f"🟡 **{label}:** {val:.3f} — MODERATE")
                else:                    st.success( f"🟢 **{label}:** {val:.3f} — SAFE")
            else:
                if   val > thresh:       st.success(f"🟢 **{label}:** {val:.3f} — HIGH")
                elif val > thresh - 0.2: st.warning(f"🟡 **{label}:** {val:.3f} — MODERATE")
                else:                    st.error(  f"🔴 **{label}:** {val:.3f} — LOW")

        show_metric("hERG Cardiotoxicity",   "hERG",            0.7)
        show_metric("Liver Toxicity (DILI)", "DILI",            0.7)
        show_metric("Ames Mutagenicity",     "Ames",            0.5)
        show_metric("BBB Penetration",       "BBB_Martins",     0.5, low_is_good=False)
        show_metric("CYP3A4 Inhibition",     "CYP3A4_Inhibitor",0.7)

        with st.expander("📊 All 66 predictions"):
            st.dataframe(preds_df.T, use_container_width=True)
        st.download_button("📥 Download CSV", data=preds_df.to_csv(index=False).encode(), file_name="admet_predictions.csv", mime="text/csv", use_container_width=True)

    with col_left:
        st.markdown("### 🎯 Occlusion Sensitivity XAI")
        # Added the new "Contour Map" Tab here
        tabs = st.tabs(["🧬 3D View", "🗺️ 2D Sharp Map", "☁️ Contour Map", "📊 Node Plot", "🔗 Bond Map", "🧩 Motif Plot", "⚖️ Trade-Off", "🧪 Mutations", "⚛️ Physicochemical Maps"])

        with tabs[0]:
            mol_3d = Chem.AddHs(Chem.MolFromSmiles(ph_smi))
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol_3d)
            n_heavy = Chem.MolFromSmiles(ph_smi).GetNumAtoms()
            w_3d    = list(norm_w) + [0.0] * (mol_3d.GetNumAtoms() - n_heavy)
            cmap_3d = mcolors.LinearSegmentedColormap.from_list("gyr", ["green", "yellow", "red"])
            view    = py3Dmol.view(width=600, height=450)
            view.addModel(Chem.MolToMolBlock(mol_3d), 'mol')
            for idx, w in enumerate(w_3d):
                if mol_3d.GetAtomWithIdx(idx).GetAtomicNum() == 1: continue
                view.setStyle({'serial': idx}, {
                    'sphere': {'color': mcolors.to_hex(cmap_3d(w)), 'radius': float(0.3 + w * 0.45), 'opacity': 0.85},
                    'stick':  {'radius': 0.15, 'color': 'lightgrey'}
                })
            view.zoomTo()
            showmol(view, height=450, width=600)
            st.caption("🔴 Red/Large = high occlusion impact · 🟢 Green/Small = low impact")

        with tabs[1]:
            st.components.v1.html(st.session_state.svg_2d, width=500, height=500)
            st.download_button("🖼️ Download Sharp Map (SVG)", data=st.session_state.svg_2d, file_name=f"{prop1}_2d_map_sharp.svg", mime="image/svg+xml")

        with tabs[2]:
            st.info("**Similarity Contour Map:** A publication-ready visualization displaying occlusion sensitivity as continuous probability clouds.")
            st.session_state.contour_buf.seek(0)
            st.image(st.session_state.contour_buf, use_container_width=True)
            st.session_state.contour_buf.seek(0)
            st.download_button("☁️ Download Contour Map (PNG)", data=st.session_state.contour_buf, file_name=f"{prop1}_contour_map.png", mime="image/png")

        with tabs[3]:
            st.session_state.node_buf.seek(0)
            st.image(st.session_state.node_buf, use_container_width=True)
            st.caption("*Method: Occlusion Sensitivity Analysis (Zeiler & Fergus, 2014) adapted for molecular graphs*")
            st.session_state.node_buf.seek(0)
            st.download_button("📊 Download Plot (PNG)", data=st.session_state.node_buf, file_name=f"{prop1}_node_ablation.png", mime="image/png")

        with tabs[4]:
            st.info("**Bond Ablation:** Removes each bond entirely (not just an atom), re-predicts. Identifies critical connectivity.")
            st.components.v1.html(st.session_state.edge_svg, width=500, height=500)
            st.download_button("🖼️ Download SVG", data=st.session_state.edge_svg, file_name=f"{prop1}_bond_map.svg", mime="image/svg+xml")

        with tabs[5]:
            st.info("**Motif Ablation:** Masks entire ring systems simultaneously. Reveals macro-structural contributions.")
            if st.session_state.motif_buf:
                st.session_state.motif_buf.seek(0)
                st.image(st.session_state.motif_buf, use_container_width=True)
                st.session_state.motif_buf.seek(0)
                st.download_button("🧩 Download PNG", data=st.session_state.motif_buf, file_name=f"{prop1}_motif.png", mime="image/png")
            else:
                st.warning("No ring systems found in this molecule.")

        with tabs[6]:
            st.info(f"**Quadrant Analysis:** Atoms in the **bottom-left** drive BOTH {prop1} and {prop2} — highest-priority targets for modification.")
            if st.session_state.get('tradeoff_buf'):
                st.session_state.tradeoff_buf.seek(0)
                st.image(st.session_state.tradeoff_buf, use_container_width=True)
                st.session_state.tradeoff_buf.seek(0)
                st.download_button("⚖️ Download PNG", data=st.session_state.tradeoff_buf, file_name="tradeoff.png", mime="image/png")
            else:
                st.warning(f"Trade-off plot unavailable — property '{prop2}' not found in model output.")

        with tabs[7]:
            st.info("**Point Mutation Array:** Swaps each high-impact atom to N/O/F/S bioisosteres. Score = predicted ADMET value of mutant. Lower = better for toxicity targets.")
            if st.session_state.heatmap_buf:
                st.session_state.heatmap_buf.seek(0)
                st.image(st.session_state.heatmap_buf, use_container_width=True)
                st.session_state.heatmap_buf.seek(0)
                st.download_button("🧪 Download PNG", data=st.session_state.heatmap_buf, file_name="mutation_heatmap.png", mime="image/png")
            else:
                st.warning("No significant atoms found for point mutation.")

        with tabs[8]:
            st.info(
                "**Physicochemical Maps** — Two complementary atom-level property overlays rendered with "
                "RDKit's C++ SimilarityMaps backend (Landrum, 2020). "
                "These are *independent* of the AI model — they show ground-truth quantum-chemical properties "
                "that help explain *why* the GNN assigned high liability scores to specific atoms."
            )
            pcol1, pcol2 = st.columns(2)

            with pcol1:
                st.markdown("#### ⚡ Gasteiger Partial Charges")
                st.caption(
                    "Atom-level electron density (Gasteiger & Marsili, 1980). "
                    "🔴 **Red/warm** = electron-poor (δ+, electrophilic). "
                    "🔵 **Blue/cool** = electron-rich (δ−, nucleophilic). "
                    "Cross-reference with your XAI map: if a high-occlusion atom is also strongly δ+, "
                    "it likely drives hERG or metabolic liability through electrophilic interactions."
                )
                g_buf = st.session_state.get("gasteiger_buf")
                if g_buf:
                    g_buf.seek(0)
                    st.image(g_buf, use_container_width=True)
                    g_buf.seek(0)
                    st.download_button(
                        "⚡ Download Gasteiger Map (PNG)",
                        data=g_buf,
                        file_name="gasteiger_charge_map.png",
                        mime="image/png",
                    )
                else:
                    st.warning("Gasteiger map could not be generated for this molecule.")

            with pcol2:
                st.markdown("#### 🧴 Crippen LogP Contributions")
                st.caption(
                    "Per-atom additive logP contributions (Wildman & Crippen, 1999). "
                    "🌸 **Pink/warm** = lipophilic hotspots (+logP). "
                    "🟢 **Green/cool** = hydrophilic regions (−logP). "
                    "Lipophilic hotspots that overlap with high-occlusion XAI atoms are the strongest "
                    "candidates for bioisosteric replacement to improve solubility or reduce hERG risk."
                )
                l_buf = st.session_state.get("logp_buf")
                if l_buf:
                    l_buf.seek(0)
                    st.image(l_buf, use_container_width=True)
                    l_buf.seek(0)
                    st.download_button(
                        "🧴 Download LogP Map (PNG)",
                        data=l_buf,
                        file_name="logp_contrib_map.png",
                        mime="image/png",
                    )
                else:
                    st.warning("LogP contribution map could not be generated for this molecule.")

    # ── Gemini Diagnostic Report ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🤖 AI Diagnostic Report — {prop1}")
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar to unlock this report.")
    elif not st.session_state.get("v7_gemini"):
        with st.spinner("Gemini is interpreting the occlusion results…"):
            try:
                top_str = st.session_state.top_atoms_str or "diffuse structural features"
                prompt = f"""
                You are a Lead Medicinal Chemist and AI researcher reviewing a computational analysis.
                Molecule SMILES (pH 7.4 corrected): {ph_smi}
                Target ADMET property: {prop1}
                Model prediction score: {base_score:.3f}
                XAI method: Occlusion Sensitivity Analysis (Zeiler & Fergus, 2014)
                Top causal atoms identified: {top_str}
                
                Write a concise 2-part scientific report:
                PART 1 — ADMET INTERPRETATION:
                What does a score of {base_score:.3f} for {prop1} mean for drug developability?
                PART 2 — XAI MECHANISTIC EXPLANATION:
                Explain WHY the neural network identified {top_str} as the primary drivers, based on established chemical principles.
                End with ONE single highest-priority medicinal chemistry recommendation to improve {prop1}.
                """
                if LEGACY_GEMINI:
                    genai.configure(api_key=api_key)
                    gmodel = genai.GenerativeModel('gemini-2.5-flash')
                    st.session_state.v7_gemini = gmodel.generate_content(prompt).text
                else:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.session_state.v7_gemini = response.text
            except Exception as e:
                st.error(f"Gemini error: {e}")

    if st.session_state.get("v7_gemini"):
        st.info(st.session_state.v7_gemini)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – USER DECISION GATE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get('stage', 0) >= 1:
    st.markdown("---")
    st.markdown("### 🧬 Step 2: Proceed to Generative Optimisation?")
    decision = st.radio(
        "Based on the diagnostics above, do you want to generate optimised bioisosteres?",
        ["🔍 No — reviewing data only.", "🚀 Yes — generate bioisosteres."],
        key="decision_radio"
    )
    if decision == "🚀 Yes — generate bioisosteres.":
        st.session_state.stage = max(st.session_state.stage, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – GENERATIVE OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get('stage', 0) >= 2:
    st.markdown("---")
    st.markdown("### ⚗️ Step 3: CReM Bioisostere Generation")

    # INTERACTIVE TARGET OVERRIDE
    st.markdown("#### 🎯 Interactive Target Override")
    mol_2d_ph = Chem.MolFromSmiles(st.session_state.ph_smiles)
    heavy_atoms = [i for i in range(mol_2d_ph.GetNumAtoms()) if mol_2d_ph.GetAtomWithIdx(i).GetSymbol() != 'H']
    
    override_targets = st.multiselect(
        "Select specific atom indices to mutate (Defaults to GNN Top 3):",
        options=heavy_atoms,
        default=st.session_state.top_targets,
        format_func=lambda x: f"{mol_2d_ph.GetAtomWithIdx(x).GetSymbol()} (Node {x})"
    )

    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        direction   = st.selectbox("Optimisation goal:", ["Minimise (reduce toxicity/liability)", "Maximise (increase desired property)"])
        high_is_bad = "Minimise" in direction
    with c_opt2:
        use_lipinski = st.checkbox("Enforce Lipinski Rule of 5", value=True, help="Uncheck if working with large modern drugs like PROTACs or macrocycles.")

    if st.button("🚀 Step 3: Launch Exhaustive Generative Loop", type="primary"):
        if not os.path.exists(CREM_DB):
            st.error(f"❌ CReM database not found: {CREM_DB}")
            st.stop()
        if not override_targets:
            st.error("❌ You must select at least one atom to mutate.")
            st.stop()

        ph_smi     = st.session_state.ph_smiles
        base_score = st.session_state.base_score1

        with st.spinner(f"🧬 CReM executing exhaustive generation from ChEMBL..."):
            mutant_dicts = run_crem_exhaustive(ph_smi, override_targets, CREM_DB)

        if not mutant_dicts:
            st.error("CReM found no valid replacements. Try a different target atom.")
            st.stop()

        st.info(f"Generated {len(mutant_dicts)} raw structures. Filtering...")
        
        if use_lipinski:
            mutant_dicts = [m for m in mutant_dicts if passes_lipinski(m['smiles'])]
            st.info(f"After Ro5 filter: {len(mutant_dicts)} drug-like structures remain.")
            if not mutant_dicts:
                st.error("All generated structures failed Lipinski Ro5. Try unchecking the Lipinski filter.")
                st.stop()

        with st.spinner(f"🚀 GPU Re-scoring all {len(mutant_dicts)} molecules with ADMET-AI…"):
            m_smiles = [m['smiles'] for m in mutant_dicts]
            m_preds  = admet_model.predict(smiles=m_smiles)

        sa_scores = [get_sa_score(s) for s in m_smiles]
        if not SA_AVAILABLE:
            st.warning("⚠️ SA Score unavailable — sascorer not found. SA column will show NaN.")

        res_df = pd.DataFrame({
            "SMILES":              m_smiles,
            "Target_Atom_Mutated": [m['target_atom'] for m in mutant_dicts],
            "Original_Score":      base_score,
            "New_Score":           m_preds[prop1].values,
            "SA_Score":            sa_scores,   
        })
        res_df["Delta"]    = res_df["New_Score"] - res_df["Original_Score"]
        res_df["Improved"] = (res_df["New_Score"] < base_score) if high_is_bad else (res_df["New_Score"] > base_score)
        res_df["Ro5_Pass"] = use_lipinski   

        sort_asc = high_is_bad
        res_df = res_df.sort_values("New_Score", ascending=sort_asc)

        st.session_state.res_df            = res_df
        st.session_state.optimization_run  = True
        st.session_state.v8_gemini         = None
        st.session_state.stage             = max(st.session_state.stage, 3)

# ── Render optimisation results ───────────────────────────────────────────────
if st.session_state.get("optimization_run") and "res_df" in st.session_state:
    res_df       = st.session_state.res_df
    improved_df  = res_df[res_df["Improved"] == True]
    ph_smi       = st.session_state.ph_smiles
    base_score   = st.session_state.base_score1

    st.markdown("---")
    st.success(f"✅ Exhaustive Generation complete — **{len(res_df)}** valid molecules scored. **{len(improved_df)}** improved {prop1}.")

    st.download_button(
        f"📥 Download all {len(res_df)} Exhaustively Generated Molecules (CSV)",
        data=res_df.to_csv(index=False).encode(),
        file_name=f"{prop1}_exhaustive_generation.csv", mime="text/csv",
        use_container_width=True
    )

    if not improved_df.empty:
        top5 = improved_df.head(5)
        st.markdown("#### 🏆 Top 5 Optimised Molecules")

        mols_draw = [Chem.MolFromSmiles(s) for s in top5["SMILES"]]
        legends   = [f"Score: {row['New_Score']:.3f}  Δ{row['Delta']:+.3f}  SA:{row['SA_Score']:.1f}" for _, row in top5.iterrows()]
        img = Draw.MolsToGridImage(mols_draw, molsPerRow=5, subImgSize=(250, 250), legends=legends, returnPNG=False)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.image(buf, use_container_width=True)
        st.caption("SA Score: 1 = trivially synthesisable · 10 = nearly impossible (Ertl & Schuffenhauer, 2009)")

        # ── Gemini Final Report ───────────────────────────────────────────────
        st.markdown("### 🤖 AI Comparative Analysis")
        if not api_key:
            st.warning("Enter Gemini API key in sidebar to unlock.")
        elif not st.session_state.get("v8_gemini"):
            with st.spinner("Gemini writing comparative analysis…"):
                try:
                    best  = top5.iloc[0]
                    p2    = f"""
                    You are a Lead Medicinal Chemist reviewing an AI-generated lead optimisation.
                    Original molecule (pH 7.4): {ph_smi}
                    Original {prop1} score: {base_score:.3f}
                    Best AI-optimised bioisostere: {best['SMILES']}
                    New {prop1} score: {best['New_Score']:.3f}  (Δ = {best['Delta']:+.3f})
                    SA Score: {best['SA_Score']:.1f}/10 (1=easy to synthesise, 10=nearly impossible)
                    
                    Write a strict scientific comparative report in 3 parts:
                    PART 1 — STRUCTURAL CHANGE: Compare the two SMILES.
                    PART 2 — PHARMACOLOGICAL RATIONALE: Explain WHY this specific structural change improved {prop1}.
                    PART 3 — SYNTHESIS FEASIBILITY: Given SA Score {best['SA_Score']:.1f}/10, comment on synthetic accessibility.
                    """
                    if LEGACY_GEMINI:
                        genai.configure(api_key=api_key)
                        gm = genai.GenerativeModel('gemini-2.5-flash')
                        st.session_state.v8_gemini = gm.generate_content(p2).text
                    else:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=p2)
                        st.session_state.v8_gemini = response.text
                except Exception as e:
                    st.error(f"Gemini error: {e}")

        if st.session_state.get("v8_gemini"):
            st.success(st.session_state.v8_gemini)
    else:
        st.warning(f"No molecules improved {prop1} vs the original score of {base_score:.3f}. Try relaxing filters or mutating different target atoms.")
