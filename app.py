import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Excel export
try:
    import openpyxl  # noqa: F401
    EXCEL_AVAILABLE = True
except Exception:
    EXCEL_AVAILABLE = False

# Optional RDKit support
RDKIT_AVAILABLE = True
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:
    RDKIT_AVAILABLE = False

# Optional PyTorch Geometric support
PYG_AVAILABLE = True
try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GINEConv, global_add_pool
except Exception:
    PYG_AVAILABLE = False

# Optional Altair static export backend
VLCONVERT_AVAILABLE = True
try:
    import vl_convert as vlc  # noqa: F401
except Exception:
    VLCONVERT_AVAILABLE = False

# Optional matplotlib fallback export
MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except Exception:
    MATPLOTLIB_AVAILABLE = False

st.set_page_config(page_title="ChronicAI", layout="wide")

APP_NAME = "ChronicAI"
APP_VERSION = "1.1.0"
MODEL_VERSION = "chronicai-gnn-21t-v1"
PREPROCESSING_VERSION = "std-v2"
CHECKPOINT_FILENAME = "gnn_champion_full_checkpoint.pt"
DEFAULT_THRESHOLD = 6.0
POTENCY_PRESETS = [6.0, 7.0, 8.0, 9.0, 10.0]
CHART_TYPES = [
    "Bar", "Horizontal bar", "Line", "Area", "Scatter", "Bubble",
    "Box", "Strip", "Histogram", "Density", "Lollipop", "Heatmap"
]
HEATMAP_SCHEMES = ["viridis", "plasma", "blues", "tealblues", "goldred", "purplebluegreen"]

TARGET_GROUPS = {
    "Oncology": ["ONC_EGFR", "ONC_ERBB2", "ONC_FGFR1", "ONC_MTOR", "ONC_PDGFRB", "ONC_VEGFR2"],
    "Neurology": ["NEURO_ACHE", "NEURO_DRD2", "NEURO_HTR2A", "NEURO_MAOA", "NEURO_SLC6A4"],
    "Cardiovascular": ["CVD_ACE", "CVD_ADRB2", "CVD_AGTR1"],
    "Metabolic": ["META_PPARA", "META_PPARG"],
    "Diabetes-related": ["T1D_DPP4", "T1D_GLP1R", "T1D_INSR", "T2D_FFAR1", "T2D_SGLT2"],
}
ALL_TARGETS = [t for group in TARGET_GROUPS.values() for t in group]

PALETTE_PRESETS = {
    "Nature-like": {
        "Oncology": "#7B4EA3",
        "Neurology": "#4C6CB3",
        "Cardiovascular": "#2E86AB",
        "Metabolic": "#2A9D8F",
        "Diabetes-related": "#C56A3D",
        "Unknown": "#6B7280",
    },
    "Cool": {
        "Oncology": "#3B82F6",
        "Neurology": "#6366F1",
        "Cardiovascular": "#06B6D4",
        "Metabolic": "#14B8A6",
        "Diabetes-related": "#0EA5E9",
        "Unknown": "#64748B",
    },
    "Warm": {
        "Oncology": "#B91C1C",
        "Neurology": "#EA580C",
        "Cardiovascular": "#F59E0B",
        "Metabolic": "#CA8A04",
        "Diabetes-related": "#9A3412",
        "Unknown": "#78716C",
    },
    "Color-blind-safe": {
        "Oncology": "#0072B2",
        "Neurology": "#56B4E9",
        "Cardiovascular": "#009E73",
        "Metabolic": "#E69F00",
        "Diabetes-related": "#CC79A7",
        "Unknown": "#6B7280",
    },
}

DEFAULT_SESSION = {
    "nav_page": "Overview",
    "input_raw_df": None,
    "input_processed_df": None,
    "input_excluded_df": None,
    "input_audit": None,
    "input_metadata": None,
    "screening_long_df": None,
    "screening_summary_df": None,
    "screening_metadata": None,
    "chart_palette_name": "Nature-like",
    "disease_colors": PALETTE_PRESETS["Nature-like"].copy(),
    "figure_height": 420,
    "figure_width": 980,
    "axis_label_angle": -35,
    "font_scale": 1.0,
    "export_dpi": 300,
    "heatmap_scheme": "viridis",
}
for k, v in DEFAULT_SESSION.items():
    st.session_state.setdefault(k, v)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.0rem; padding-bottom: 2.2rem; max-width: 1520px;}
    .hero-wrap {border: 1px solid #D8E3F0; border-radius: 28px; padding: 26px 28px; background: linear-gradient(135deg, #EEF4FF 0%, #F8FBFF 48%, #FFF7ED 100%); box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);}
    .hero-title {font-size: 2.25rem; font-weight: 850; color: #0F172A; line-height: 1.02; margin-bottom: 0.6rem;}
    .hero-sub {font-size: 1rem; color: #334155; line-height: 1.6; margin-bottom: 0.9rem;}
    .mini-pill {display:inline-block; padding:6px 11px; border-radius:999px; background:#E8EEFF; color:#2F4BB8; border:1px solid #C6D4FF; font-size:0.79rem; font-weight:750; margin: 4px 6px 0 0;}
    .soft-card {border:1px solid #DCE6F3; border-radius:22px; padding:17px 18px; background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFF 100%); box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05); min-height: 128px;}
    .metric-card {border-radius:18px; padding:15px 16px; border:1px solid #DCE6F3; background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%); min-height:112px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);}
    .metric-label {font-size:0.83rem; font-weight:780; color:#475569; margin-bottom:8px;}
    .metric-value {font-size:1.78rem; font-weight:850; line-height:1.0; color:#0F172A; margin-bottom:8px;}
    .metric-sub {font-size:0.79rem; color:#64748B; line-height:1.45;}
    .page-note {border:1px solid #DDE7F5; border-radius:16px; padding:15px 16px; background:#F7FAFF; color:#334155;}
    .workflow-step {border:1px solid #DDE7F5; border-radius:20px; padding:17px; background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFF 100%); min-height:148px; text-align:center; box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);}
    .workflow-badge {display:inline-block; width:38px; height:38px; line-height:38px; border-radius:999px; background:#DDE7FF; color:#2E4DB7; font-weight:850; margin-bottom:10px;}
    .tint-blue {background: linear-gradient(180deg, #F4F8FF 0%, #FFFFFF 100%);}
    .tint-indigo {background: linear-gradient(180deg, #F5F3FF 0%, #FFFFFF 100%);}
    .tint-teal {background: linear-gradient(180deg, #F0FDFA 0%, #FFFFFF 100%);}
    .tint-amber {background: linear-gradient(180deg, #FFF7ED 0%, #FFFFFF 100%);}
    .tint-rose {background: linear-gradient(180deg, #FFF1F2 0%, #FFFFFF 100%);}
    .status-pass {display:inline-block; padding:4px 9px; border-radius:999px; background:#ECFDF5; color:#166534; border:1px solid #A7F3D0; font-weight:800; font-size:0.78rem;}
    .status-fail {display:inline-block; padding:4px 9px; border-radius:999px; background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; font-weight:800; font-size:0.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def get_disease_color_map() -> Dict[str, str]:
    return st.session_state.get("disease_colors", PALETTE_PRESETS["Nature-like"])


def render_metric_card(title: str, value, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_csv_download(df: pd.DataFrame, filename: str, label: str):
    st.download_button(label=label, data=df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


def safe_excel_download(df: pd.DataFrame, filename: str, label: str):
    if not EXCEL_AVAILABLE:
        st.caption("Excel export unavailable because openpyxl is not installed.")
        return
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    st.download_button(label=label, data=buffer.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def safe_json_download(obj, filename: str, label: str):
    st.download_button(label=label, data=json.dumps(obj, indent=2, default=str).encode("utf-8"), file_name=filename, mime="application/json")


def infer_smiles_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if str(c).strip().lower() in {"smiles", "canonical_smiles", "input_smiles"}]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if "smiles" in str(c).lower():
            return c
    return None


def infer_id_column(df: pd.DataFrame) -> Optional[str]:
    preferred = {"compound_id", "compoundid", "compound id", "id", "name", "compound_cid"}
    for c in df.columns:
        if str(c).strip().lower() in preferred:
            return c
    return None


def target_to_group(target: str) -> str:
    for group, targets in TARGET_GROUPS.items():
        if target in targets:
            return group
    return "Unknown"


def to_display_potency(value: float, lower: float = 4.0, upper: float = 10.0) -> float:
    try:
        return round(float(np.clip(float(value), lower, upper)), 3)
    except Exception:
        return np.nan


def add_potency_band(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return "Unknown"
    if v < 6:
        return f"Weak (<6; {v:.2f})"
    if v < 7:
        return f"Moderate (6–7; {v:.2f})"
    if v < 8:
        return f"Good (7–8; {v:.2f})"
    if v < 9:
        return f"Strong (8–9; {v:.2f})"
    return f"Very strong (9–10; {v:.2f})"


def developability_label_from_rules(lipinski, veber, egan) -> str:
    vals = [int(float(x)) if pd.notna(x) else 0 for x in [lipinski, veber, egan]]
    if sum(vals) == 3:
        return "Passed ✅"
    return "Failed ❌"


def developability_color(label: str) -> str:
    return "#15803D" if "Passed" in str(label) else "#B91C1C"


def wrapped_title(title: str, width: int = 34) -> str:
    import textwrap
    if title is None:
        return ""
    title = str(title).replace("_", " ")
    return "\n".join(textwrap.wrap(title, width=width))


def pretty_label(name: str) -> str:
    mapping = {
        "median_pchembl": "Median pChEMBL",
        "mean_pchembl": "Mean pChEMBL",
        "max_pchembl": "Maximum pChEMBL",
        "hit_fraction": "Hit fraction",
        "hit_count": "Hit count",
        "prioritization_score": "Prioritization score",
        "priority_tier": "Priority tier",
        "predicted_pchembl": "Predicted pChEMBL",
        "compound_id": "Compound",
        "disease_group": "Disease group",
        "top_target_pchembl": "Top-target pChEMBL",
    }
    return mapping.get(str(name), str(name).replace("_", " ").title())


def compact_height(n_items: int, mode: str = "bar") -> int:
    n_items = max(1, int(n_items))
    if mode == "hbar":
        return max(220, min(520, 70 + 34 * n_items))
    if mode == "heatmap":
        return max(240, min(620, 80 + 18 * n_items))
    if mode == "scatter":
        return 420
    return max(260, min(440, 90 + 26 * n_items))


def render_summary_band(title: str, value: str, subtitle: str, accent: str = "#3B82F6"):
    st.markdown(
        f"""
        <div style="border:1px solid #D9E4F2;border-left:5px solid {accent};border-radius:18px;padding:16px 18px;background:linear-gradient(180deg,#FFFFFF 0%,#F8FBFF 100%);">
            <div style="font-size:0.82rem;font-weight:800;color:#475569;letter-spacing:0.02em;">{title}</div>
            <div style="font-size:1.85rem;font-weight:850;color:#0F172A;line-height:1.05;margin-top:6px;">{value}</div>
            <div style="font-size:0.84rem;color:#64748B;line-height:1.4;margin-top:8px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def canonicalize_smiles(smiles: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if smiles is None or not str(smiles).strip():
        return None, None, None, "Empty SMILES"
    smiles = str(smiles).strip()
    if not RDKIT_AVAILABLE:
        return smiles, None, None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, "Invalid SMILES"
        clean = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
        canonical = Chem.MolToSmiles(parent, canonical=True)
        mol2 = Chem.MolFromSmiles(canonical)
        inchikey = Chem.MolToInchiKey(mol2)
        murcko = MurckoScaffold.MurckoScaffoldSmiles(mol=mol2)
        return canonical, inchikey, murcko, None
    except Exception as e:
        return None, None, None, str(e)


def compute_descriptors(smiles: str) -> Dict[str, Optional[float]]:
    blank = {"MW": None, "logP": None, "TPSA": None, "HBD": None, "HBA": None, "RotB": None, "QED": None, "Lipinski_pass": None, "Veber_pass": None, "Egan_pass": None}
    if not RDKIT_AVAILABLE:
        return blank
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return blank
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    qed = QED.qed(mol)
    return {
        "MW": round(mw, 3), "logP": round(logp, 3), "TPSA": round(tpsa, 3), "HBD": int(hbd), "HBA": int(hba),
        "RotB": int(rotb), "QED": round(qed, 4), "Lipinski_pass": int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
        "Veber_pass": int(rotb <= 10 and tpsa <= 140), "Egan_pass": int(logp <= 5.88 and tpsa <= 131.6),
    }


def preprocess_dataframe(df: pd.DataFrame, smiles_col: str, id_col: Optional[str], reference_inchikeys: Optional[set]):
    records, seen = [], set()
    for idx, row in df.iterrows():
        raw_smiles = row.get(smiles_col, None)
        compound_id = row.get(id_col, f"cmpd_{idx + 1}") if id_col else f"cmpd_{idx + 1}"
        canonical_smiles, inchikey, murcko, error = canonicalize_smiles(raw_smiles)
        is_valid = error is None
        duplicate = False
        if is_valid:
            key = inchikey if inchikey else canonical_smiles
            if key in seen:
                duplicate = True
            else:
                seen.add(key)
        overlap = int(reference_inchikeys is not None and inchikey in reference_inchikeys) if inchikey else 0
        desc = compute_descriptors(canonical_smiles) if is_valid and canonical_smiles else compute_descriptors("")
        keep = int(is_valid and not duplicate)
        records.append({
            "row_index": idx, "compound_id": compound_id, "input_smiles": raw_smiles, "canonical_smiles": canonical_smiles,
            "inchikey": inchikey, "murcko_scaffold": murcko, "is_valid": int(is_valid), "error": error,
            "is_duplicate_removed": int(duplicate), "overlap_with_reference": overlap, "keep_for_screening": keep, **desc,
        })
    full_df = pd.DataFrame(records)
    retained = full_df[full_df["keep_for_screening"] == 1].copy()
    excluded = full_df[full_df["keep_for_screening"] == 0].copy()
    audit = {
        "uploaded_rows": int(len(full_df)), "valid_count": int(full_df["is_valid"].sum()),
        "invalid_count": int(len(full_df) - full_df["is_valid"].sum()), "duplicate_removed_count": int(full_df["is_duplicate_removed"].sum()),
        "reference_overlap_count": int(full_df["overlap_with_reference"].sum()), "retained_count": int(retained.shape[0]),
        "excluded_count": int(excluded.shape[0]), "rdkit_available": RDKIT_AVAILABLE,
    }
    metadata = {
        "app_name": APP_NAME, "app_version": APP_VERSION, "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION, "prediction_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "input_compound_count": audit["uploaded_rows"], "retained_compound_count": audit["retained_count"],
        "duplicate_removed_count": audit["duplicate_removed_count"], "reference_overlap_count": audit["reference_overlap_count"],
    }
    return retained, excluded, audit, metadata


def make_status_message(audit: Dict) -> Tuple[str, str]:
    if audit["retained_count"] == 0:
        return "error", "No valid unique compounds remain after preprocessing."
    if audit["invalid_count"] > 0 or audit["duplicate_removed_count"] > 0:
        return "warning", "Preprocessing completed with exclusions. Review invalid or duplicate entries."
    return "success", "Preprocessing completed. Retained compounds are ready for screening."


def resolve_checkpoint_path(filename: str) -> str:
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for p in [script_dir / filename, script_dir.parent / filename, Path.cwd() / filename]:
        if p.exists():
            return str(p)
    return str(script_dir / filename)


CHECKPOINT_PATH = resolve_checkpoint_path(CHECKPOINT_FILENAME)


def normalize_args(args_obj) -> Dict:
    if isinstance(args_obj, dict):
        return args_obj
    if hasattr(args_obj, "__dict__"):
        return vars(args_obj)
    return {}


class ConditionalGNN(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int, num_layers: int, dropout: float, n_targets: int, target_emb_dim: int, use_target_embedding: bool = True):
        super().__init__()
        self.use_target_embedding = use_target_embedding
        self.dropout = dropout
        self.node_proj = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_in))
            self.norms.append(nn.BatchNorm1d(hidden))
        if use_target_embedding:
            self.t_emb = nn.Embedding(n_targets, target_emb_dim)
            head_in = hidden + target_emb_dim
        else:
            self.t_emb = None
            head_in = hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def encode(self, batch):
        x = self.node_proj(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, batch.edge_index, batch.edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_add_pool(x, batch.batch)

    def forward(self, batch, target_idx):
        g = self.encode(batch)
        if self.use_target_embedding:
            g = torch.cat([g, self.t_emb(target_idx)], dim=1)
        return self.head(g).squeeze(-1)


@st.cache_resource
def load_checkpoint_bundle(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return {"load_ok": False, "error": f"Checkpoint file not found: {checkpoint_path}"}
    if not PYG_AVAILABLE:
        return {"load_ok": False, "error": "torch_geometric is not available."}
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            return {"load_ok": False, "error": "Checkpoint is not a dictionary object."}
        args = normalize_args(ckpt.get("args", {}))
        state_dict = ckpt.get("state_dict")
        target_map = ckpt.get("target_map")
        n_targets = ckpt.get("n_targets")
        hidden = int(args["hidden"])
        num_layers = int(args["num_layers"])
        dropout = float(args["dropout"])
        target_emb_dim = int(args["target_emb_dim"])
        use_target_embedding = not bool(args.get("no_target_embedding", False))
        node_in = state_dict["node_proj.weight"].shape[1]
        edge_in = [v for k, v in state_dict.items() if "convs.0.lin.weight" in k][0].shape[1]
        model = ConditionalGNN(node_in, edge_in, hidden, num_layers, dropout, int(n_targets), target_emb_dim, use_target_embedding)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return {"load_ok": True, "model": model, "target_map": target_map, "n_targets": n_targets, "args": args}
    except Exception as e:
        return {"load_ok": False, "error": str(e)}


def atom_features(mol) -> List[List[float]]:
    feats = []
    for atom in mol.GetAtoms():
        hyb = atom.GetHybridization()
        chiral = int(atom.GetChiralTag())
        feats.append([
            atom.GetAtomicNum() / 100.0, atom.GetDegree() / 6.0, atom.GetFormalCharge() / 5.0,
            atom.GetTotalNumHs() / 8.0, atom.GetTotalValence() / 8.0, float(atom.GetIsAromatic()),
            float(atom.IsInRing()), atom.GetMass() / 200.0, float(hyb == Chem.rdchem.HybridizationType.SP),
            float(hyb == Chem.rdchem.HybridizationType.SP2), float(hyb == Chem.rdchem.HybridizationType.SP3),
            float(hyb == Chem.rdchem.HybridizationType.SP3D), float(hyb == Chem.rdchem.HybridizationType.SP3D2),
            float(chiral == int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)),
            float(chiral == int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)),
            float(atom.GetNumRadicalElectrons() > 0), atom.GetImplicitValence() / 8.0, atom.GetExplicitValence() / 8.0,
            float(atom.GetNoImplicit()), atom.GetIsotope() / 100.0,
        ])
    return feats


def bond_features(mol):
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        stereo = int(bond.GetStereo())
        feat = [
            float(bt == Chem.rdchem.BondType.SINGLE), float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE), float(bt == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()), float(bond.IsInRing()), stereo / 5.0,
        ]
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])
    if len(edge_index) == 0:
        edge_index, edge_attr = [[0, 0]], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float32)


def smiles_to_pyg_data(smiles: str):
    if not (RDKIT_AVAILABLE and PYG_AVAILABLE):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor(atom_features(mol), dtype=torch.float32)
    edge_index, edge_attr = bond_features(mol)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def run_multitarget_screening(processed_df: pd.DataFrame, selected_targets: List[str], threshold: float):
    if processed_df is None or processed_df.empty:
        return None, None, {"run_ok": False, "error": "No retained compounds available."}
    loader = load_checkpoint_bundle(CHECKPOINT_PATH)
    if not loader.get("load_ok", False):
        return None, None, {"run_ok": False, "error": loader.get("error", "Model loader failed.")}
    model, target_map = loader["model"], loader["target_map"]
    rows = []
    with torch.no_grad():
        for _, row in processed_df.iterrows():
            smi = row.get("canonical_smiles")
            if smi is None or pd.isna(smi):
                continue
            graph = smiles_to_pyg_data(smi)
            if graph is None:
                continue
            batch = Batch.from_data_list([graph])
            for target in selected_targets:
                if target not in target_map:
                    continue
                target_idx = torch.tensor([target_map[target]], dtype=torch.long)
                raw_pred = float(model(batch, target_idx).cpu().numpy()[0])
                disp_pred = to_display_potency(raw_pred)
                drug_label = developability_label_from_rules(row.get("Lipinski_pass"), row.get("Veber_pass"), row.get("Egan_pass"))
                rows.append({
                    "compound_id": row.get("compound_id"), "canonical_smiles": row.get("canonical_smiles"), "inchikey": row.get("inchikey"),
                    "target": target, "disease_group": target_to_group(target), "raw_prediction": raw_pred, "predicted_pchembl": disp_pred,
                    "potency_band": add_potency_band(disp_pred), "active_like_flag": int(disp_pred >= threshold),
                    "QED": row.get("QED"), "Lipinski_pass": row.get("Lipinski_pass"), "Veber_pass": row.get("Veber_pass"),
                    "Egan_pass": row.get("Egan_pass"), "druglikeness_status": drug_label,
                    "overlap_with_reference": row.get("overlap_with_reference"),
                    "reliability_flag": "reference-overlap" if row.get("overlap_with_reference", 0) == 1 else "standard-screening",
                })
    if not rows:
        return None, None, {"run_ok": False, "error": "No predictions were generated."}
    long_df = pd.DataFrame(rows)
    summary_df = (
        long_df.groupby(["compound_id", "canonical_smiles", "inchikey"], dropna=False)
        .agg(
            max_predicted_pchembl=("predicted_pchembl", "max"), mean_predicted_pchembl=("predicted_pchembl", "mean"),
            n_targets_active_like=("active_like_flag", "sum"), n_targets_screened=("target", "count"), QED=("QED", "first"),
            Lipinski_pass=("Lipinski_pass", "first"), Veber_pass=("Veber_pass", "first"), Egan_pass=("Egan_pass", "first"),
            druglikeness_status=("druglikeness_status", "first"), overlap_with_reference=("overlap_with_reference", "first"),
        )
        .reset_index().sort_values(["max_predicted_pchembl", "mean_predicted_pchembl"], ascending=False).reset_index(drop=True)
    )
    meta = {
        "run_ok": True, "screening_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_targets": selected_targets, "selected_target_count": len(selected_targets), "threshold_pchembl": threshold,
        "compound_count": int(processed_df.shape[0]), "compound_target_predictions": int(long_df.shape[0]),
        "model_version": MODEL_VERSION, "preprocessing_version": PREPROCESSING_VERSION, "app_version": APP_VERSION,
    }
    return long_df, summary_df, meta


def get_color_scale_domain_range():
    cmap = get_disease_color_map()
    domain = list(cmap.keys())
    return domain, [cmap[k] for k in domain]


def altair_theme_config(chart):
    return (
        chart
        .configure_view(stroke=None)
        .configure_axis(
            labelFontSize=int(13 * st.session_state["font_scale"]),
            titleFontSize=int(15 * st.session_state["font_scale"]),
            labelColor="#334155",
            titleColor="#0F172A",
            gridColor="#DCE4EE",
            gridOpacity=0.75,
            tickColor="#94A3B8",
        )
        .configure_legend(
            titleFontSize=int(14 * st.session_state["font_scale"]),
            labelFontSize=int(12 * st.session_state["font_scale"]),
            orient="bottom",
            direction="horizontal",
            columns=3,
            symbolSize=180,
        )
        .configure_title(
            fontSize=int(20 * st.session_state["font_scale"]),
            color="#0F172A",
            anchor="start",
            fontWeight=800,
            offset=12,
        )
    )


def make_generic_chart(df: pd.DataFrame, chart_type: str, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None, title: str = ""):
    domain, range_ = get_color_scale_domain_range()
    base_w = int(st.session_state.get("figure_width", 980))
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return None

    plot_df = df.copy()
    n_items = plot_df[x].nunique() if x in plot_df.columns else len(plot_df)
    chart_type = chart_type or "Bar"
    legend_needed = bool(color and color in plot_df.columns and plot_df[color].nunique() > 1)
    legend_obj = alt.Legend(title=pretty_label(color), orient="bottom", direction="horizontal", columns=3) if legend_needed else None
    color_enc = alt.Color(f"{color}:N", scale=alt.Scale(domain=domain, range=range_), legend=legend_obj) if color and color in plot_df.columns else alt.value("#3F8FB5")

    # choose more compact defaults
    if chart_type in ["Horizontal bar", "Lollipop"]:
        height = compact_height(n_items, "hbar")
        width = min(max(base_w, 880), 1280)
    elif chart_type in ["Bar", "Line", "Area", "Box", "Strip"]:
        height = compact_height(n_items, "bar")
        width = min(max(base_w, 860), 1260)
    elif chart_type in ["Scatter", "Bubble"]:
        height = compact_height(n_items, "scatter")
        width = min(max(base_w, 900), 1280)
    elif chart_type in ["Histogram", "Density"]:
        height = 340
        width = min(max(base_w, 820), 1120)
    else:
        height = 340
        width = min(max(base_w, 860), 1220)

    angle = -28 if n_items <= 8 else -35
    base = alt.Chart(plot_df).properties(height=height, width=width, title=wrapped_title(title, 44))
    y_title = pretty_label(y)
    x_title = pretty_label(x)

    if chart_type == "Bar":
        bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.92, size=min(46, max(18, int(340 / max(n_items, 1))))).encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=angle, labelLimit=220), sort=None, title=x_title),
            y=alt.Y(f"{y}:Q", title=y_title),
            color=color_enc, tooltip=list(plot_df.columns),
        )
        if n_items <= 12:
            labels = base.mark_text(dy=-8, fontSize=11, color="#334155").encode(
                x=alt.X(f"{x}:N", sort=None), y=alt.Y(f"{y}:Q"), text=alt.Text(f"{y}:Q", format=".2f")
            )
            chart = bars + labels
        else:
            chart = bars
    elif chart_type == "Horizontal bar":
        ordered = plot_df.sort_values(y, ascending=False)
        bars = alt.Chart(ordered).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.92, size=20).encode(
            y=alt.Y(f"{x}:N", sort='-x', title=x_title),
            x=alt.X(f"{y}:Q", title=y_title),
            color=color_enc, tooltip=list(ordered.columns),
        )
        labels = alt.Chart(ordered).mark_text(align="left", dx=6, fontSize=11, color="#334155").encode(
            y=alt.Y(f"{x}:N", sort='-x'), x=alt.X(f"{y}:Q"), text=alt.Text(f"{y}:Q", format=".2f")
        )
        chart = (bars + labels).properties(height=height, width=width, title=wrapped_title(title, 44))
    elif chart_type == "Line":
        chart = base.mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=angle, labelLimit=220), title=x_title), y=alt.Y(f"{y}:Q", title=y_title), color=color_enc, tooltip=list(plot_df.columns)
        )
    elif chart_type == "Area":
        chart = base.mark_area(opacity=0.65).encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=angle, labelLimit=220), title=x_title), y=alt.Y(f"{y}:Q", title=y_title), color=color_enc, tooltip=list(plot_df.columns)
        )
    elif chart_type == "Scatter":
        chart = base.mark_circle(size=60, opacity=0.52).encode(
            x=alt.X(f"{x}:Q", title=x_title), y=alt.Y(f"{y}:Q", title=y_title), color=color_enc, tooltip=list(plot_df.columns)
        )
    elif chart_type == "Bubble":
        size_col = size if size and size in plot_df.columns else y
        chart = base.mark_circle(opacity=0.48).encode(
            x=alt.X(f"{x}:Q", title=x_title), y=alt.Y(f"{y}:Q", title=y_title),
            size=alt.Size(f"{size_col}:Q", legend=alt.Legend(title=pretty_label(size_col), orient="right"), scale=alt.Scale(range=[30, 350])),
            color=color_enc, tooltip=list(plot_df.columns)
        )
    elif chart_type == "Box":
        chart = base.mark_boxplot(extent="min-max", size=18).encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=angle, labelLimit=220), title=x_title), y=alt.Y(f"{y}:Q", title=y_title), color=color_enc, tooltip=list(plot_df.columns)
        )
    elif chart_type == "Strip":
        chart = base.mark_circle(size=48, opacity=0.52).encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=angle, labelLimit=220), title=x_title), y=alt.Y(f"{y}:Q", title=y_title), color=color_enc, tooltip=list(plot_df.columns)
        )
    elif chart_type == "Histogram":
        chart = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, opacity=0.92).encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=18), title=x_title),
            y=alt.Y("count():Q", title="Count"), color=alt.value("#4F46E5"), tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
    elif chart_type == "Density":
        chart = base.transform_density(x, as_=[x, "density"]).mark_area(opacity=0.72).encode(
            x=alt.X(f"{x}:Q", title=x_title), y=alt.Y("density:Q", title="Density"), color=alt.value("#4F46E5")
        )
    elif chart_type == "Lollipop":
        ordered = plot_df.sort_values(y, ascending=False)
        stems = alt.Chart(ordered).mark_rule(color="#CBD5E1", strokeWidth=2).encode(
            y=alt.Y(f"{x}:N", sort='-x', title=x_title), x=alt.value(0), x2=alt.X2(f"{y}:Q")
        )
        dots = alt.Chart(ordered).mark_circle(size=110, opacity=0.95).encode(
            y=alt.Y(f"{x}:N", sort='-x', title=x_title), x=alt.X(f"{y}:Q", title=y_title), color=color_enc, tooltip=list(ordered.columns)
        )
        labels = alt.Chart(ordered).mark_text(align="left", dx=7, fontSize=11, color="#334155").encode(
            y=alt.Y(f"{x}:N", sort='-x'), x=alt.X(f"{y}:Q"), text=alt.Text(f"{y}:Q", format=".2f")
        )
        chart = (stems + dots + labels).properties(height=height, width=width, title=wrapped_title(title, 44))
    else:
        chart = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=angle, labelLimit=220), title=x_title), y=alt.Y(f"{y}:Q", title=y_title), color=color_enc, tooltip=list(plot_df.columns)
        )
    return altair_theme_config(chart)

def make_heatmap(screening_long_df: pd.DataFrame, ranked_compounds: List[str], selected_targets: List[str], mode: str = "relative"):
    if screening_long_df is None or screening_long_df.empty or not ranked_compounds or not selected_targets:
        return None
    df = screening_long_df[screening_long_df["compound_id"].isin(ranked_compounds) & screening_long_df["target"].isin(selected_targets)].copy()
    if df.empty:
        return None
    df["compound_id"] = pd.Categorical(df["compound_id"], categories=ranked_compounds, ordered=True)
    df["target"] = pd.Categorical(df["target"], categories=selected_targets, ordered=True)
    if mode == "relative":
        df["display_value"] = df.groupby("compound_id")["predicted_pchembl"].transform(lambda s: s - s.mean())
        ctitle = "Relative target preference"
        color_scale = alt.Scale(scheme="blueorange", domainMid=0)
    elif mode == "band":
        band_map = {"Weak": 1, "Moderate": 2, "Good": 3, "Strong": 4, "Very strong": 5}
        df["display_value"] = df["potency_band"].astype(str).map(lambda s: band_map.get(s.split(" ")[0], 0))
        ctitle = "Potency band"
        color_scale = alt.Scale(scheme=st.session_state["heatmap_scheme"], domain=[1, 5])
    else:
        df["display_value"] = df["predicted_pchembl"]
        ctitle = "Predicted potency"
        color_scale = alt.Scale(domain=[4, 10], scheme=st.session_state["heatmap_scheme"])
    height = compact_height(len(ranked_compounds), "heatmap")
    width = min(max(int(st.session_state.get("figure_width", 980)), 760), 1120)
    chart = alt.Chart(df).mark_rect(stroke="#FFFFFF", strokeWidth=0.7).encode(
        x=alt.X("target:N", sort=selected_targets, axis=alt.Axis(labelAngle=-28, labelLimit=180), title="Target"),
        y=alt.Y("compound_id:N", sort=ranked_compounds, title="Compound"),
        color=alt.Color("display_value:Q", title=ctitle, scale=color_scale, legend=alt.Legend(orient="right")),
        tooltip=["compound_id", "target", "disease_group", alt.Tooltip("predicted_pchembl:Q", title="Predicted potency", format=".2f"), "potency_band", "druglikeness_status", "reliability_flag"],
    ).properties(height=height, width=width, title=wrapped_title(f"Top-ranked compound–target map ({mode})", 42))
    return altair_theme_config(chart)

def export_altair_chart_bytes(chart, fmt: str) -> Optional[bytes]:
    if chart is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
            path = tmp.name
        kwargs = {}
        if fmt == "png":
            kwargs["ppi"] = int(st.session_state.get("export_dpi", 300))
            kwargs["scale_factor"] = 2
        chart.save(path, **kwargs)
        with open(path, "rb") as f:
            data = f.read()
        try:
            os.remove(path)
        except Exception:
            pass
        return data
    except Exception:
        return None


def export_matplotlib_bar(df: pd.DataFrame, x: str, y: str, color_col: Optional[str] = None, horizontal: bool = False) -> Optional[Tuple[bytes, bytes]]:
    if not MATPLOTLIB_AVAILABLE or df is None or df.empty or x not in df.columns or y not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    cmap = get_disease_color_map()
    if color_col and color_col in df.columns:
        colors = [cmap.get(str(v), "#4F46E5") for v in df[color_col].tolist()]
    else:
        colors = ["#4F46E5"] * len(df)
    if horizontal:
        ax.barh(df[x].astype(str), df[y].astype(float), color=colors)
    else:
        ax.bar(df[x].astype(str), df[y].astype(float), color=colors)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png_buf, pdf_buf = io.BytesIO(), io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=int(st.session_state.get("export_dpi", 300)), bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_buf.getvalue(), pdf_buf.getvalue()


def render_chart_export_block(chart, prefix: str, fallback_df: Optional[pd.DataFrame] = None, fallback_x: Optional[str] = None, fallback_y: Optional[str] = None, fallback_color: Optional[str] = None, fallback_horizontal: bool = False):
    c1, c2, c3, c4 = st.columns(4)
    png_bytes = export_altair_chart_bytes(chart, "png")
    pdf_bytes = export_altair_chart_bytes(chart, "pdf")
    svg_bytes = export_altair_chart_bytes(chart, "svg")
    if png_bytes is None or pdf_bytes is None:
        fallback = export_matplotlib_bar(fallback_df, fallback_x, fallback_y, fallback_color, fallback_horizontal) if fallback_df is not None and fallback_x and fallback_y else None
        if fallback is not None:
            png_bytes, pdf_bytes = fallback
    with c1:
        if png_bytes is not None:
            st.download_button("Download PNG", png_bytes, f"{prefix}.png", mime="image/png", key=f"{prefix}_png")
        else:
            st.caption("PNG export unavailable")
    with c2:
        if pdf_bytes is not None:
            st.download_button("Download PDF", pdf_bytes, f"{prefix}.pdf", mime="application/pdf", key=f"{prefix}_pdf")
        else:
            st.caption("PDF export unavailable")
    with c3:
        if svg_bytes is not None:
            st.download_button("Download SVG", svg_bytes, f"{prefix}.svg", mime="image/svg+xml", key=f"{prefix}_svg")
        else:
            st.caption("SVG export unavailable")
    with c4:
        if chart is not None:
            st.download_button("Download JSON spec", chart.to_json().encode("utf-8"), f"{prefix}.json", mime="application/json", key=f"{prefix}_json")


def render_figure_controls(prefix: str):
    with st.expander(f"{prefix} figure settings", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.session_state["figure_width"] = st.slider("Figure width", 700, 1500, int(st.session_state.get("figure_width", 980)), 20)
        with c2:
            st.session_state["figure_height"] = st.slider("Figure height", 320, 1000, int(st.session_state["figure_height"]), 20)
        with c3:
            st.session_state["axis_label_angle"] = st.slider("Label angle", -90, 45, int(st.session_state["axis_label_angle"]), 5)
        with c4:
            st.session_state["font_scale"] = st.slider("Font scale", 0.85, 1.8, float(st.session_state["font_scale"]), 0.05)
        with c5:
            st.session_state["export_dpi"] = st.selectbox("Export DPI", [150, 300, 600], index=[150, 300, 600].index(st.session_state["export_dpi"]))
        st.session_state["heatmap_scheme"] = st.selectbox("Heatmap palette", HEATMAP_SCHEMES, index=HEATMAP_SCHEMES.index(st.session_state["heatmap_scheme"]))


def build_rank_table(screening_summary_df: pd.DataFrame, screening_long_df: pd.DataFrame) -> pd.DataFrame:
    top_target_df = (
        screening_long_df.sort_values(["compound_id", "predicted_pchembl"], ascending=[True, False]).drop_duplicates("compound_id")
        [["compound_id", "target", "disease_group", "predicted_pchembl", "potency_band"]]
        .rename(columns={
            "target": "top_target",
            "disease_group": "dominant_disease_group",
            "predicted_pchembl": "top_target_pchembl",
            "potency_band": "top_target_band",
        })
    )
    rank_df = screening_summary_df.merge(top_target_df, on="compound_id", how="left")
    rank_df["potency_band"] = rank_df["max_predicted_pchembl"].apply(add_potency_band)
    if "druglikeness_status" not in rank_df.columns:
        rank_df["druglikeness_status"] = rank_df.apply(
            lambda r: developability_label_from_rules(r.get("Lipinski_pass"), r.get("Veber_pass"), r.get("Egan_pass")),
            axis=1,
        )
    else:
        missing_mask = rank_df["druglikeness_status"].isna()
        if missing_mask.any():
            rank_df.loc[missing_mask, "druglikeness_status"] = rank_df.loc[missing_mask].apply(
                lambda r: developability_label_from_rules(r.get("Lipinski_pass"), r.get("Veber_pass"), r.get("Egan_pass")),
                axis=1,
            )
    rank_df["druglikeness_status"] = rank_df["druglikeness_status"].fillna("Failed ❌")
    rank_df = rank_df.sort_values(["max_predicted_pchembl", "n_targets_active_like", "mean_predicted_pchembl"], ascending=False).reset_index(drop=True)
    rank_df["rank"] = range(1, len(rank_df) + 1)
    return rank_df


def compute_developability_score(row: pd.Series) -> float:
    return 0.4 * float(row.get("Lipinski_pass", 0) or 0) + 0.3 * float(row.get("Veber_pass", 0) or 0) + 0.3 * float(row.get("Egan_pass", 0) or 0)


def assign_developability_status(score: float) -> str:
    if score >= 0.999:
        return "Passed ✅"
    if score >= 0.60:
        return "Failed ❌"
    return "Failed ❌"


def _safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    smin, smax = float(s.min()), float(s.max())
    if smax <= smin:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - smin) / (smax - smin)


def make_prioritization_summary(screening_long_df: pd.DataFrame, screening_summary_df: pd.DataFrame) -> pd.DataFrame:
    top_target_df = (
        screening_long_df.sort_values(["compound_id", "predicted_pchembl"], ascending=[True, False]).drop_duplicates("compound_id")
        [["compound_id", "target", "disease_group", "predicted_pchembl", "reliability_flag"]]
        .rename(columns={"target": "top_target", "disease_group": "dominant_disease_group", "predicted_pchembl": "top_target_pchembl", "reliability_flag": "top_target_reliability_flag"})
    )
    prior = screening_summary_df.merge(top_target_df, on="compound_id", how="left").copy()
    prior["potency_band"] = prior["max_predicted_pchembl"].apply(add_potency_band)
    prior["developability_score"] = prior.apply(compute_developability_score, axis=1)
    prior["developability_status"] = prior["developability_score"].apply(assign_developability_status)
    prior["overlap_penalty"] = pd.to_numeric(prior.get("overlap_with_reference", 0), errors="coerce").fillna(0).astype(float)
    prior["norm_max"] = _safe_minmax(prior["max_predicted_pchembl"])
    prior["norm_mean"] = _safe_minmax(prior["mean_predicted_pchembl"])
    prior["norm_breadth"] = _safe_minmax(prior["n_targets_active_like"])
    prior["norm_qed"] = _safe_minmax(prior["QED"])
    prior["norm_dev"] = _safe_minmax(prior["developability_score"])
    prior["prioritization_score"] = (0.35 * prior["norm_max"] + 0.20 * prior["norm_mean"] + 0.20 * prior["norm_breadth"] + 0.15 * prior["norm_dev"] + 0.10 * prior["norm_qed"] - 0.05 * prior["overlap_penalty"]).clip(0.0, 1.0)
    prior["priority_tier"] = pd.cut(prior["prioritization_score"], bins=[-0.001, 0.35, 0.65, 1.0], labels=["Tier 3", "Tier 2", "Tier 1"]).astype(str)
    prior["ranking_rationale"] = prior.apply(lambda r: "; ".join([x for x in ["broad multi-target signal" if r.get("n_targets_active_like", 0) >= 3 else None, "drug-likeness passed" if r.get("druglikeness_status") == "Passed ✅" else "drug-likeness failed", f"top target: {r['top_target']}" if pd.notna(r.get("top_target")) else None, "reference overlap present" if int(r.get("overlap_with_reference", 0) or 0) == 1 else None] if x]) or "no additional annotation", axis=1)
    prior = prior.sort_values(["prioritization_score", "max_predicted_pchembl", "n_targets_active_like"], ascending=False).reset_index(drop=True)
    prior["rank"] = range(1, len(prior) + 1)
    return prior


st.sidebar.title(APP_NAME)
pages = ["Overview", "Input compounds", "Multitarget screening", "Prioritization dashboard"]
page = st.sidebar.selectbox("Navigation", pages, index=pages.index(st.session_state.get("nav_page", "Overview")))
st.session_state["nav_page"] = page

with st.sidebar.expander("Visual theme", expanded=True):
    preset = st.selectbox("Palette preset", list(PALETTE_PRESETS.keys()), index=list(PALETTE_PRESETS.keys()).index(st.session_state["chart_palette_name"]))
    if preset != st.session_state["chart_palette_name"]:
        st.session_state["chart_palette_name"] = preset
        st.session_state["disease_colors"] = PALETTE_PRESETS[preset].copy()
    current = st.session_state["disease_colors"]
    for group in ["Oncology", "Neurology", "Cardiovascular", "Metabolic", "Diabetes-related", "Unknown"]:
        current[group] = st.color_picker(group, current.get(group, PALETTE_PRESETS[preset].get(group, "#6B7280")), key=f"color_{group}")
    st.session_state["disease_colors"] = current

with st.sidebar.expander("Diagnostics", expanded=False):
    st.caption(f"RDKit available: {RDKIT_AVAILABLE}")
    st.caption(f"PyG available: {PYG_AVAILABLE}")
    st.caption(f"vl-convert available: {VLCONVERT_AVAILABLE}")
    st.caption(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    st.caption(f"Checkpoint exists: {os.path.exists(CHECKPOINT_PATH)}")

if page == "Overview":
    left, right = st.columns([1.25, 0.95])
    with left:
        st.markdown(f"""
        <div class="hero-wrap">
            <div class="hero-title">🧠 {APP_NAME}</div>
            <div class="hero-sub">A multitarget screening and prioritization platform for chronic-disease drug discovery, designed to profile compounds across a Core-21 target panel and convert model outputs into decision-ready pChEMBL-centered ranking views.</div>
            <span class="mini-pill">21 targets</span>
            <span class="mini-pill">5 disease groups</span>
            <span class="mini-pill">pChEMBL-centered scoring</span>
            <span class="mini-pill">Developability-aware triage</span>
            <span class="mini-pill">PNG/PDF/SVG export</span>
        </div>
        """, unsafe_allow_html=True)
    with right:
        c1, c2 = st.columns(2)
        with c1:
            render_metric_card("Model panel", "Core-21", "Multitask screening across oncology, neurology, cardiometabolic and diabetes-related targets.")
        with c2:
            render_metric_card("Output scale", "Predicted potency", "Displayed outputs are clipped to a realistic 4–10 potency range for deployment readability.")
        c3, c4 = st.columns(2)
        with c3:
            render_metric_card("Figure export", "PNG / PDF / SVG", "Major figures support direct export for reports and manuscript preparation.")
        with c4:
            render_metric_card("Intended use", "Triage", "This app supports computational prioritization and does not replace experimental validation.")
    st.write("")
    st.markdown('<div class="page-note" style="background:linear-gradient(90deg,#EEF4FF 0%,#F9FBFF 100%);"><b>Designed for.</b> Computational triage, shortlist construction, multitarget profile review, and figure/table export for reports and manuscript preparation.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("### Why this platform is useful")
    st.caption("Each module is designed to move from interpretable potency prediction to shortlist-ready decision support.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="soft-card tint-blue"><h4>🎯 Multitarget prioritization</h4><p>Profile each compound across a chronic-disease panel instead of relying on single-target ranking logic.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-card tint-indigo"><h4>📈 pChEMBL-based interpretation</h4><p>Predictions are displayed as predicted pChEMBL with threshold bands at 6, 7, 8, 9, and 10.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="soft-card tint-teal"><h4>📊 Flexible figure studio</h4><p>Choose from more than 8 chart types, customize colors, and export figures in multiple formats.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("### Workflow")
    wf1, wf2, wf3, wf4 = st.columns(4)
    cards = [
        ("1", "📥 Upload compounds", "Single SMILES, batch paste, or CSV upload with automatic column detection."),
        ("2", "🧼 Preprocess structures", "Validate, standardize, deduplicate, calculate descriptors, and optionally audit overlap."),
        ("3", "🎯 Screen Core-21", "Run multitarget prediction across all 21 targets, a disease group, or a custom subset."),
        ("4", "📈 Prioritize and export", "Review ranked compounds, compare profiles, build heatmaps, and export results."),
    ]
    for col, (badge, title, desc) in zip([wf1, wf2, wf3, wf4], cards):
        with col:
            st.markdown(f'<div class="workflow-step"><div class="workflow-badge">{badge}</div><div style="font-weight:700; margin-bottom:8px;">{title}</div><div style="font-size:0.9rem; color:#475569; line-height:1.45;">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    disease_cards = [
        ("🎗️", "Oncology", "Receptor and signaling targets for oncology-related prioritization."),
        ("🧠", "Neurology", "Neurobiology-relevant endpoints for central nervous system screening."),
        ("❤️", "Cardiovascular", "Cardiovascular targets for chronic disease prioritization."),
        ("⚙️", "Metabolic", "Metabolic endpoints for pathway-oriented screening."),
        ("🩸", "Diabetes-related", "Diabetes-associated targets relevant to glycemic regulation."),
    ]
    disease_bg = {
        "Oncology": "#FFF1F6",
        "Neurology": "#F4F5FF",
        "Cardiovascular": "#EEF8FF",
        "Metabolic": "#F0FDFA",
        "Diabetes-related": "#FFF7ED",
    }
    for col, (icon, name, desc) in zip([dc1, dc2, dc3, dc4, dc5], disease_cards):
        with col:
            color = get_disease_color_map().get(name, "#64748B")
            bg = disease_bg.get(name, "#F8FAFC")
            st.markdown(f'<div class="soft-card" style="background:{bg}; border-top:4px solid {color};"><div style="font-size:1.8rem;">{icon}</div><div style="font-weight:800; color:{color}; margin-top:6px;">{name}</div><div style="margin-top:8px; color:#475569; font-size:0.88rem; line-height:1.45;">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.info("The public deployment is designed for computational triage. Docking, pharmacology, and clinical conclusions should be treated as downstream validation layers rather than outputs of this interface.")
    if st.button("Start screening", type="primary"):
        st.session_state["nav_page"] = "Input compounds"
        st.rerun()

elif page == "Input compounds":
    st.title("Compound input and preprocessing")
    st.markdown('<div class="page-note"><b>Preprocessing logic.</b> Structures are parsed, standardized, deduplicated, and annotated before screening. Only retained unique valid compounds proceed to the Core-21 model.</div>', unsafe_allow_html=True)
    st.write("")
    input_mode = st.radio("Input mode", ["Single SMILES", "Batch paste", "CSV upload"], horizontal=True)
    raw_df = None
    if input_mode == "Single SMILES":
        c1, c2 = st.columns([1, 2])
        with c1:
            compound_id = st.text_input("Compound ID (optional)")
        with c2:
            smiles = st.text_input("Enter SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O")
        if st.button("Process input", type="primary"):
            if not smiles.strip():
                st.error("Please enter a SMILES string.")
            else:
                raw_df = pd.DataFrame({"compound_id": [compound_id.strip() if compound_id.strip() else "cmpd_1"], "SMILES": [smiles.strip()]})
    elif input_mode == "Batch paste":
        st.code("SMILES\nID,SMILES", language="text")
        batch_text = st.text_area("Paste compounds", height=180, placeholder="cmpd_1,CCO\ncmpd_2,CC(=O)O")
        first_col_has_ids = st.checkbox("First column contains compound IDs", value=True)
        if st.button("Process batch input", type="primary"):
            if not batch_text.strip():
                st.error("Please paste at least one line of input.")
            else:
                rows = []
                for i, line in enumerate(batch_text.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if first_col_has_ids and len(parts) >= 2:
                        rows.append({"compound_id": parts[0], "SMILES": parts[-1]})
                    else:
                        rows.append({"compound_id": f"cmpd_{i}", "SMILES": parts[0]})
                raw_df = pd.DataFrame(rows)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read the CSV file: {e}")
    if raw_df is not None and not raw_df.empty:
        st.session_state["input_raw_df"] = raw_df.copy()
        st.markdown("### Raw upload preview")
        st.dataframe(raw_df.head(10), use_container_width=True, height=240)
        smiles_guess = infer_smiles_column(raw_df)
        id_guess = infer_id_column(raw_df)
        c1, c2 = st.columns(2)
        with c1:
            smiles_col = st.selectbox("SMILES column", options=list(raw_df.columns), index=list(raw_df.columns).index(smiles_guess) if smiles_guess in raw_df.columns else 0)
        with c2:
            id_options = ["<auto-generate>"] + list(raw_df.columns)
            id_default = id_options.index(id_guess) if id_guess in id_options else 0
            id_col_selection = st.selectbox("Compound ID column", options=id_options, index=id_default)
        reference_file = st.file_uploader("Optional reference InChIKey CSV for overlap audit", type=["csv"])
        reference_inchikeys = None
        if reference_file is not None:
            try:
                ref_df = pd.read_csv(reference_file)
                for col in ref_df.columns:
                    vals = ref_df[col].dropna().astype(str)
                    if len(vals) > 0 and vals.str.len().median() >= 20:
                        reference_inchikeys = set(vals.tolist())
                        break
            except Exception as e:
                st.warning(f"Reference overlap file could not be parsed: {e}")
        if st.button("Run preprocessing", type="primary"):
            retained, excluded, audit, metadata = preprocess_dataframe(raw_df, smiles_col, None if id_col_selection == "<auto-generate>" else id_col_selection, reference_inchikeys)
            st.session_state["input_processed_df"] = retained
            st.session_state["input_excluded_df"] = excluded
            st.session_state["input_audit"] = audit
            st.session_state["input_metadata"] = metadata
    processed_df = st.session_state.get("input_processed_df")
    excluded_df = st.session_state.get("input_excluded_df")
    audit = st.session_state.get("input_audit")
    metadata = st.session_state.get("input_metadata")
    if audit is not None:
        msg_type, msg_text = make_status_message(audit)
        getattr(st, msg_type)(msg_text)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Uploaded rows", audit["uploaded_rows"], "Rows detected in the submitted input.")
        with c2:
            render_metric_card("Valid structures", audit["valid_count"], "Passed parsing and validation.")
        with c3:
            render_metric_card("Invalid structures", audit["invalid_count"], "Excluded because parsing failed.")
        with c4:
            render_metric_card("Retained for screening", audit["retained_count"], "Compounds proceeding to multitarget screening.")
        c5, c6, c7 = st.columns(3)
        with c5:
            render_metric_card("Duplicates removed", audit["duplicate_removed_count"], "Removed after canonicalization-based deduplication.")
        with c6:
            render_metric_card("Reference overlap", audit["reference_overlap_count"], "Flagged against the uploaded reference set.")
        with c7:
            render_metric_card("RDKit status", "On" if audit["rdkit_available"] else "Off", "Descriptor and chemistry-aware preprocessing backend.")
        st.markdown("### Descriptor chart")
        render_figure_controls("Preprocessing")
        if processed_df is not None and not processed_df.empty and processed_df["QED"].notna().any():
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                chart_type = st.selectbox("Chart type", ["Histogram", "Density", "Box", "Strip", "Bar"], key="pre_chart_type")
            with c2:
                descriptor = st.selectbox("Descriptor", ["MW", "logP", "TPSA", "QED"], key="pre_descriptor")
            with c3:
                x_col = descriptor if chart_type in ["Histogram", "Density"] else "compound_id"
                st.caption(f"X field: {x_col}")
            with c4:
                st.caption(f"Y field: {descriptor if chart_type not in ['Histogram', 'Density'] else 'count/density'}")
            if chart_type in ["Histogram", "Density"]:
                chart = make_generic_chart(processed_df[[descriptor]].dropna(), chart_type, descriptor, descriptor, title=f"{descriptor} distribution")
                fallback_df, fx, fy = None, None, None
            else:
                plot_df = processed_df[["compound_id", descriptor]].dropna().head(40)
                chart = make_generic_chart(plot_df, chart_type, "compound_id", descriptor, title=f"{descriptor} snapshot")
                fallback_df, fx, fy = plot_df, "compound_id", descriptor
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
                render_chart_export_block(chart, f"preprocessing_{descriptor.lower()}_{chart_type.lower().replace(' ', '_')}", fallback_df, fx, fy)
        st.markdown("### Data views")
        if processed_df is not None:
            st.markdown("#### Retained compounds")
            st.dataframe(processed_df, use_container_width=True, height=280)
        if excluded_df is not None and not excluded_df.empty:
            st.markdown("#### Excluded compounds")
            st.dataframe(excluded_df, use_container_width=True, height=220)
        st.markdown("### Export")
        ex1, ex2, ex3, ex4 = st.columns(4)
        with ex1:
            if processed_df is not None and not processed_df.empty:
                safe_csv_download(processed_df, "chronicai_preprocessed_retained.csv", "Retained CSV")
        with ex2:
            if processed_df is not None and not processed_df.empty:
                safe_excel_download(processed_df, "chronicai_preprocessed_retained.xlsx", "Retained Excel")
        with ex3:
            if excluded_df is not None and not excluded_df.empty:
                safe_csv_download(excluded_df, "chronicai_preprocessed_excluded.csv", "Excluded CSV")
        with ex4:
            if metadata is not None:
                safe_json_download(metadata, "chronicai_preprocessing_metadata.json", "Metadata JSON")
        if audit["retained_count"] > 0 and st.button("Continue to multitarget screening", type="primary"):
            st.session_state["nav_page"] = "Multitarget screening"
            st.rerun()

elif page == "Multitarget screening":
    st.title("Multitarget screening")
    st.markdown('<div class="page-note"><b>Interpretation.</b> The interface reports <b>predicted pChEMBL</b> values and threshold-derived active-like calls. These are computational screening outputs for comparative triage, not experimental confirmation.</div>', unsafe_allow_html=True)
    processed_df = st.session_state.get("input_processed_df")
    if processed_df is None or processed_df.empty:
        st.warning("No retained compounds are available yet. Please complete preprocessing first.")
        st.stop()
    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.2, 1.2])
    with c1:
        scope = st.radio("Target scope", ["All 21 targets", "Disease group", "Custom target subset"], index=0)
    with c2:
        selected_group = st.selectbox("Disease group", list(TARGET_GROUPS.keys()), disabled=scope != "Disease group")
    with c3:
        selected_custom_targets = st.multiselect("Custom targets", ALL_TARGETS, default=TARGET_GROUPS["Oncology"], disabled=scope != "Custom target subset")
    with c4:
        threshold_mode = st.radio("Threshold mode", ["Preset", "Custom"], horizontal=True)
        threshold = st.selectbox("Preset pChEMBL threshold", POTENCY_PRESETS, index=0) if threshold_mode == "Preset" else st.slider("Custom pChEMBL threshold", 4.0, 10.0, float(DEFAULT_THRESHOLD), 0.1)
    if scope == "All 21 targets":
        selected_targets = ALL_TARGETS
    elif scope == "Disease group":
        selected_targets = TARGET_GROUPS[selected_group]
    else:
        selected_targets = selected_custom_targets
    if not selected_targets:
        st.warning("No targets are selected for the current configuration.")
        st.stop()
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        render_metric_card("Retained compounds", int(processed_df.shape[0]), "Unique valid compounds available for screening.")
    with s2:
        render_metric_card("Target scope", scope, "Current screening scope.")
    with s3:
        render_metric_card("Targets in run", len(selected_targets), "Targets selected for the current run.")
    with s4:
        render_metric_card("Threshold", f"pChEMBL ≥ {float(threshold):.1f}", "Operational active-like cutoff.")
    with s5:
        render_metric_card("Evaluations", int(processed_df.shape[0]) * len(selected_targets), "Planned compound–target predictions.")
    if st.button("Run multitarget screening", type="primary"):
        long_df, summary_df, meta = run_multitarget_screening(processed_df, selected_targets, float(threshold))
        st.session_state["screening_long_df"] = long_df
        st.session_state["screening_summary_df"] = summary_df
        st.session_state["screening_metadata"] = meta
    screening_long_df = st.session_state.get("screening_long_df")
    screening_summary_df = st.session_state.get("screening_summary_df")
    screening_metadata = st.session_state.get("screening_metadata")
    if screening_long_df is not None and not screening_long_df.empty:
        st.markdown("## Run summary")
        target_summary = screening_long_df.groupby(["target", "disease_group"], dropna=False).agg(compounds_scored=("compound_id", "nunique"), hit_count=("active_like_flag", "sum"), hit_fraction=("active_like_flag", "mean"), median_pchembl=("predicted_pchembl", "median"), mean_pchembl=("predicted_pchembl", "mean"), max_pchembl=("predicted_pchembl", "max")).reset_index().sort_values(["hit_fraction", "median_pchembl"], ascending=False)
        group_summary = screening_long_df.groupby("disease_group", dropna=False).agg(compounds_scored=("compound_id", "nunique"), targets_in_view=("target", "nunique"), hit_count=("active_like_flag", "sum"), hit_fraction=("active_like_flag", "mean"), median_pchembl=("predicted_pchembl", "median"), mean_pchembl=("predicted_pchembl", "mean")).reset_index().sort_values(["hit_fraction", "median_pchembl"], ascending=False)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            render_metric_card("Compounds screened", screening_metadata["compound_count"], "Unique compounds in the current run.")
        with k2:
            render_metric_card("Targets screened", screening_metadata["selected_target_count"], "Targets scored in the current run.")
        with k3:
            render_metric_card("Predictions", screening_metadata["compound_target_predictions"], "Compound–target scores generated.")
        with k4:
            render_metric_card("Active-like calls", int(screening_long_df["active_like_flag"].sum()), "Calls above the selected threshold.")
        with k5:
            render_metric_card("Median pChEMBL", f"{float(pd.to_numeric(screening_long_df['predicted_pchembl'], errors='coerce').median()):.2f}", "Median predicted pChEMBL across the run.")
        with k6:
            render_metric_card("Compounds with hits", int((screening_summary_df["n_targets_active_like"] > 0).sum()), "Compounds with at least one active-like call.")
        render_figure_controls("Screening")
        st.markdown("## Target-level view")
        c1, c2 = st.columns([1.1, 1.2])
        with c1:
            chart_type = st.selectbox("Chart type", ["Horizontal bar", "Lollipop", "Bar", "Line"], index=0, key="target_chart_type")
        with c2:
            y_metric = st.selectbox("Metric", ["median_pchembl", "mean_pchembl", "hit_fraction", "hit_count", "max_pchembl"], index=0, key="target_metric")
        st.caption(f"Showing {pretty_label(y_metric)} by target. A dashed threshold reference is conceptually {threshold:.1f} pChEMBL.")
        target_chart = make_generic_chart(target_summary, chart_type, "target", y_metric, color="disease_group", title=f"Target-level summary ({pretty_label(y_metric)})")
        if target_chart is not None:
            st.altair_chart(target_chart, use_container_width=True)
            render_chart_export_block(target_chart, f"screening_target_{y_metric}_{chart_type.lower().replace(' ', '_')}", target_summary, "target", y_metric, "disease_group", chart_type == "Horizontal bar")
        st.markdown("## Disease-group summary")
        group_metric = st.selectbox("Disease-group metric", ["hit_fraction", "median_pchembl", "mean_pchembl", "hit_count"], index=0)
        if group_summary.shape[0] <= 1:
            row = group_summary.iloc[0]
            accent = get_disease_color_map().get(str(row["disease_group"]), "#3B82F6")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                render_summary_band("Disease group", str(row["disease_group"]), "Current scoped disease family.", accent)
            with g2:
                render_summary_band(pretty_label(group_metric), f"{float(row[group_metric]):.2f}" if group_metric != "hit_count" else f"{int(row[group_metric])}", "Primary summary metric for the current selection.", accent)
            with g3:
                render_summary_band("Compounds scored", f"{int(row['compounds_scored'])}", "Unique compounds evaluated in this group.", accent)
            with g4:
                render_summary_band("Targets in view", f"{int(row['targets_in_view'])}", "Targets contributing to the current summary.", accent)
        else:
            group_chart_type = st.selectbox("Disease-group chart type", ["Bar", "Horizontal bar", "Lollipop"], index=0)
            group_chart = make_generic_chart(group_summary, group_chart_type, "disease_group", group_metric, color="disease_group", size="compounds_scored", title=f"Disease-group summary ({pretty_label(group_metric)})")
            if group_chart is not None:
                st.altair_chart(group_chart, use_container_width=True)
                render_chart_export_block(group_chart, f"screening_group_{group_metric}_{group_chart_type.lower().replace(' ', '_')}", group_summary, "disease_group", group_metric, "disease_group", group_chart_type == "Horizontal bar")
        st.markdown("## Compound-level ranking")
        rank_df = build_rank_table(screening_summary_df, screening_long_df)
        rank_show = rank_df.copy()
        rank_show["priority_percentile"] = (1 - (rank_show["rank"] - 1) / max(len(rank_show) - 1, 1)).round(3)
        rank_show["druglikeness_status"] = rank_show["druglikeness_status"].fillna("Failed ❌")
        compact_table = rank_show[["rank", "compound_id", "top_target", "top_target_pchembl", "potency_band", "n_targets_active_like", "druglikeness_status", "priority_percentile"]].rename(columns={"top_target_pchembl": "Top-target potency", "potency_band": "Potency category", "druglikeness_status": "Drug-likeness", "n_targets_active_like": "Breadth", "priority_percentile": "Rank percentile"})
        st.dataframe(compact_table, use_container_width=True, height=340)
        with st.expander("Show advanced ranking columns", expanded=False):
            st.dataframe(rank_show[["rank", "compound_id", "dominant_disease_group", "top_target", "top_target_pchembl", "max_predicted_pchembl", "mean_predicted_pchembl", "potency_band", "n_targets_active_like", "druglikeness_status", "QED", "overlap_with_reference"]].rename(columns={"top_target_pchembl": "Top-target potency", "max_predicted_pchembl": "Max predicted potency", "mean_predicted_pchembl": "Mean predicted potency", "potency_band": "Potency category", "druglikeness_status": "Drug-likeness"}), use_container_width=True, height=320)
        st.markdown("## Top-N heatmap")
        h1, h2 = st.columns([1, 1])
        with h1:
            top_n = st.slider("Top-N compounds", 5, min(50, max(5, len(rank_df))), min(12, len(rank_df)))
        with h2:
            heatmap_mode = st.selectbox("Heatmap mode", ["relative", "absolute", "band"], index=0)
        heatmap = make_heatmap(screening_long_df, rank_df.head(top_n)["compound_id"].tolist(), selected_targets, mode=heatmap_mode)
        if heatmap is not None:
            st.altair_chart(heatmap, use_container_width=True)
            render_chart_export_block(heatmap, f"screening_heatmap_top_{top_n}_{heatmap_mode}")
        st.markdown("## Selected compound profile")
        selected_compound = st.selectbox("Select a compound", rank_df["compound_id"].astype(str).tolist(), index=0)
        profile_df = screening_long_df[screening_long_df["compound_id"].astype(str) == str(selected_compound)].sort_values("predicted_pchembl", ascending=False)
        default_profile = "Lollipop" if profile_df["target"].nunique() <= 6 else "Horizontal bar"
        profile_chart_type = st.selectbox("Profile chart type", ["Lollipop", "Horizontal bar", "Bar", "Line", "Scatter"], index=["Lollipop", "Horizontal bar", "Bar", "Line", "Scatter"].index(default_profile))
        profile_chart = make_generic_chart(profile_df, profile_chart_type, "target", "predicted_pchembl", color="disease_group", title=f"Target profile for {selected_compound}")
        if profile_chart is not None:
            st.altair_chart(profile_chart, use_container_width=True)
            render_chart_export_block(profile_chart, f"screening_profile_{selected_compound}", profile_df, "target", "predicted_pchembl", "disease_group", profile_chart_type == "Horizontal bar")
        st.dataframe(profile_df[["target", "disease_group", "predicted_pchembl", "potency_band", "druglikeness_status", "active_like_flag", "reliability_flag"]].rename(columns={"predicted_pchembl": "Predicted potency", "potency_band": "Potency category", "druglikeness_status": "Drug-likeness"}), use_container_width=True, height=240)
        st.markdown("## Export tables")
        ex1, ex2, ex3, ex4 = st.columns(4)
        with ex1:
            safe_csv_download(screening_long_df, "chronicai_screening_long.csv", "Target-level CSV")
        with ex2:
            safe_excel_download(screening_long_df, "chronicai_screening_long.xlsx", "Target-level Excel")
        with ex3:
            safe_csv_download(screening_summary_df, "chronicai_screening_summary.csv", "Compound summary CSV")
        with ex4:
            safe_json_download(screening_metadata, "chronicai_screening_metadata.json", "Metadata JSON")
        if st.button("Continue to prioritization dashboard", type="primary"):
            st.session_state["nav_page"] = "Prioritization dashboard"
            st.rerun()
    elif screening_metadata is not None and not screening_metadata.get("run_ok", False):
        st.error(screening_metadata.get("error", "Screening failed."))

elif page == "Prioritization dashboard":
    st.title("Prioritization dashboard")
    st.markdown('<div class="page-note"><b>Interpretation.</b> Prioritization combines maximum predicted pChEMBL, mean predicted pChEMBL, breadth of active-like calls, developability support, and QED into a transparent triage score.</div>', unsafe_allow_html=True)
    screening_long_df = st.session_state.get("screening_long_df")
    screening_summary_df = st.session_state.get("screening_summary_df")
    screening_metadata = st.session_state.get("screening_metadata")
    if screening_long_df is None or screening_summary_df is None or screening_long_df.empty or screening_summary_df.empty:
        st.warning("No screening results are currently available. Please complete multitarget screening first.")
        st.stop()
    prior_df = make_prioritization_summary(screening_long_df, screening_summary_df)
    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.0, 1.2])
    with f1:
        selected_group_filter = st.selectbox("Disease-group filter", ["All"] + list(TARGET_GROUPS.keys()), index=0)
    with f2:
        developability_filter = st.selectbox("Developability filter", ["All", "Passed ✅", "Failed ❌"], index=0)
    with f3:
        top_n = st.slider("Top-N compounds", 5, min(50, max(5, len(prior_df))), min(20, len(prior_df)))
    with f4:
        min_score = st.slider("Minimum prioritization score", 0.0, 1.0, 0.0, 0.01)
    filtered = prior_df.copy()
    if selected_group_filter != "All":
        filtered = filtered[filtered["dominant_disease_group"] == selected_group_filter].copy()
    if developability_filter != "All":
        filtered = filtered[filtered["developability_status"] == developability_filter].copy()
    filtered = filtered[pd.to_numeric(filtered["prioritization_score"], errors="coerce").fillna(0) >= min_score].copy()
    if filtered.empty:
        st.warning("No compounds match the current dashboard filters.")
        st.stop()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        render_metric_card("Filtered compounds", int(filtered.shape[0]), "Compounds remaining after dashboard filters.")
    with k2:
        render_metric_card("Top score", f"{float(filtered['prioritization_score'].max()):.3f}", "Maximum prioritization score in the filtered set.")
    with k3:
        render_metric_card("Tier 1", int((filtered["priority_tier"] == "Tier 1").sum()), "Highest-priority compounds.")
    with k4:
        render_metric_card("Drug-likeness passed", int((filtered["druglikeness_status"] == "Passed ✅").sum()), "Compounds passing Lipinski, Veber, and Egan rules.")
    with k5:
        render_metric_card("Compounds with hits", int((filtered["n_targets_active_like"] > 0).sum()), "Compounds with at least one active-like target call.")
    with k6:
        render_metric_card("Reference overlap", int(pd.to_numeric(filtered["overlap_with_reference"], errors="coerce").fillna(0).sum()), "Compounds flagged by reference overlap.")
    st.markdown("## Ranked prioritization table")
    dash_show = filtered.copy()
    if "potency_band" not in dash_show.columns:
        dash_show["potency_band"] = dash_show["max_predicted_pchembl"].apply(add_potency_band)
    dash_show["priority_percentile"] = (1 - (dash_show["rank"] - 1) / max(len(dash_show) - 1, 1)).round(3)
    shortlist = dash_show[["rank", "priority_tier", "compound_id", "top_target", "top_target_pchembl", "potency_band", "n_targets_active_like", "druglikeness_status", "priority_percentile", "prioritization_score"]].rename(columns={"top_target_pchembl": "Top-target potency", "potency_band": "Potency category", "druglikeness_status": "Drug-likeness", "n_targets_active_like": "Breadth", "priority_percentile": "Rank percentile", "prioritization_score": "Priority score"})
    st.dataframe(shortlist, use_container_width=True, height=360)
    with st.expander("Show advanced prioritization columns", expanded=False):
        st.dataframe(dash_show[["rank", "compound_id", "priority_tier", "prioritization_score", "dominant_disease_group", "top_target", "top_target_pchembl", "potency_band", "n_targets_active_like", "druglikeness_status", "developability_status", "QED", "max_predicted_pchembl", "mean_predicted_pchembl", "overlap_with_reference", "ranking_rationale"]].rename(columns={"top_target_pchembl": "Top-target potency", "max_predicted_pchembl": "Max predicted potency", "mean_predicted_pchembl": "Mean predicted potency", "potency_band": "Potency category", "druglikeness_status": "Drug-likeness", "developability_status": "Rule-based status"}), use_container_width=True, height=320)
    render_figure_controls("Dashboard")
    st.markdown("## Prioritization scatter / bubble view")
    chart_type = st.selectbox("Chart type", ["Scatter", "Bubble", "Horizontal bar", "Lollipop"], index=0)
    if chart_type in ["Scatter", "Bubble"]:
        scatter = make_generic_chart(filtered, chart_type, "prioritization_score", "QED", color="priority_tier", size="n_targets_active_like", title="Prioritization score versus QED")
        st.altair_chart(scatter, use_container_width=True)
        render_chart_export_block(scatter, f"dashboard_prioritization_{chart_type.lower()}")
    else:
        bar = make_generic_chart(filtered.head(top_n), chart_type, "compound_id", "prioritization_score", color="dominant_disease_group", title="Top-ranked compounds by prioritization score")
        st.altair_chart(bar, use_container_width=True)
        render_chart_export_block(bar, f"dashboard_ranked_{chart_type.lower().replace(' ', '_')}", filtered.head(top_n), "compound_id", "prioritization_score", "dominant_disease_group", chart_type == "Horizontal bar")
    st.markdown("## Priority-tier composition")
    tier_df = filtered.groupby("priority_tier", dropna=False).size().reset_index(name="count")
    tier_chart = make_generic_chart(tier_df, "Bar", "priority_tier", "count", title="Priority-tier composition")
    st.altair_chart(tier_chart, use_container_width=True)
    render_chart_export_block(tier_chart, "dashboard_priority_tiers", tier_df, "priority_tier", "count")
    st.markdown("## Top-N heatmap")
    hm1, hm2 = st.columns([1,1])
    with hm1:
        heatmap_mode = st.selectbox("Heatmap mode", ["relative", "absolute", "band"], index=0, key="dash_heatmap_mode")
    with hm2:
        st.caption("Relative mode is recommended when many shortlisted compounds saturate at the display ceiling.")
    heatmap = make_heatmap(screening_long_df, filtered.head(top_n)["compound_id"].tolist(), screening_long_df["target"].dropna().unique().tolist(), mode=heatmap_mode)
    if heatmap is not None:
        st.altair_chart(heatmap, use_container_width=True)
        render_chart_export_block(heatmap, f"dashboard_heatmap_top_{top_n}_{heatmap_mode}")
    st.markdown("## Selected compound review")
    selected_compound = st.selectbox("Select a prioritized compound", filtered["compound_id"].astype(str).tolist(), index=0)
    selected_row = filtered[filtered["compound_id"].astype(str) == str(selected_compound)].head(1)
    d1, d2 = st.columns([0.95, 1.05])
    with d1:
        if not selected_row.empty:
            r = selected_row.iloc[0]
            render_metric_card("Priority tier", r.get("priority_tier", "NA"), f"Prioritization score: {float(r.get('prioritization_score', 0)):.3f}")
            render_metric_card("Top target", r.get("top_target", "NA"), f"Top-target potency: {float(r.get('top_target_pchembl', 0)):.2f}")
            render_metric_card("Drug-likeness", r.get("druglikeness_status", "NA"), f"Potency category: {r.get('potency_band', 'NA')}")
            render_metric_card("Active-like breadth", int(r.get("n_targets_active_like", 0)), f"Dominant disease group: {r.get('dominant_disease_group', 'NA')}")
    with d2:
        profile_df = screening_long_df[screening_long_df["compound_id"].astype(str) == str(selected_compound)].sort_values("predicted_pchembl", ascending=False)
        default_profile = "Lollipop" if profile_df["target"].nunique() <= 6 else "Horizontal bar"
        profile_chart_type = st.selectbox("Selected-compound chart type", ["Lollipop", "Horizontal bar", "Bar", "Line", "Scatter"], index=["Lollipop", "Horizontal bar", "Bar", "Line", "Scatter"].index(default_profile))
        profile_chart = make_generic_chart(profile_df, profile_chart_type, "target", "predicted_pchembl", color="disease_group", title=f"Target profile for {selected_compound}")
        st.altair_chart(profile_chart, use_container_width=True)
        render_chart_export_block(profile_chart, f"dashboard_profile_{selected_compound}", profile_df, "target", "predicted_pchembl", "disease_group", profile_chart_type == "Horizontal bar")
    st.dataframe(profile_df[["target", "disease_group", "predicted_pchembl", "potency_band", "druglikeness_status", "active_like_flag", "reliability_flag"]].rename(columns={"predicted_pchembl": "Predicted potency", "potency_band": "Potency category", "druglikeness_status": "Drug-likeness"}), use_container_width=True, height=240)
    st.markdown("## Export")
    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        safe_csv_download(filtered, "chronicai_prioritization_dashboard.csv", "Prioritization CSV")
    with ex2:
        safe_excel_download(filtered, "chronicai_prioritization_dashboard.xlsx", "Prioritization Excel")
    with ex3:
        safe_csv_download(filtered.head(top_n), f"chronicai_top_{top_n}.csv", f"Top-{top_n} CSV")
    with ex4:
        safe_json_download({"screening_metadata": screening_metadata, "dashboard_filters": {"selected_group_filter": selected_group_filter, "developability_filter": developability_filter, "top_n": top_n, "min_score": min_score}}, "chronicai_dashboard_metadata.json", "Metadata JSON")
