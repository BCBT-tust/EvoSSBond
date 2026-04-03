import numpy as np
import json
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa

BACKBONE_ATOMS = ['N', 'CA', 'C', 'O', 'CB']
CA_DIST_MIN = 3.0
CA_DIST_MAX = 7.5

AA3 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

def parse_structure(structure_file):
    structure_file = Path(structure_file)
    try:
        if structure_file.suffix == '.cif':
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        return parser.get_structure(structure_file.stem, structure_file)
    except Exception as e:
        print(f"结构解析失败: {e}")
        return None


def estimate_cb(res):
    try:
        n  = res['N'].get_vector()
        ca = res['CA'].get_vector()
        c  = res['C'].get_vector()
        b  = ca - n
        cv = ca - c
        a  = b ** cv
        d  = (b + cv).normalized()
        cb = ca + (-0.58 * d + 0.49 * a.normalized())
        return np.array(cb.get_array())
    except Exception:
        return None


def extract_backbone(res):
    coords = []
    for atom in BACKBONE_ATOMS:
        if atom in res:
            coords.append(res[atom].coord.copy())
        elif atom == 'CB' and res.resname == 'GLY':
            cb = estimate_cb(res)
            if cb is None:
                return None
            coords.append(cb)
        else:
            return None
    return np.array(coords)


def make_features(c1, c2):
    pair = np.vstack([c1, c2])
    n = 10
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dm[i, j] = np.linalg.norm(pair[i] - pair[j])
    tri = np.triu_indices(n, k=1)
    return pair, dm, dm[tri]


def get_flexibility(res, is_alphafold=False):
    vals = [a.bfactor for a in res.get_atoms() if a.bfactor > 0]
    if not vals:
        return None
    mean_val = float(np.mean(vals))
    if is_alphafold:
        return 100.0 - mean_val
    return mean_val

def get_candidate_pairs(structure, mode='full_scan', is_alphafold=False):
    model = next(iter(structure))
    residues = []

    for chain in model:
        seq_idx = 0
        for res in chain:
            if res.id[0] != ' ':
                continue
            if not is_aa(res, standard=True):
                continue
            if 'CA' not in res:
                continue
            residues.append({
                'res':      res,
                'chain':    chain.id,
                'res_id':   res.id[1],
                'res_name': res.resname,
                'seq_idx':  seq_idx,
                'flex':     get_flexibility(res, is_alphafold)
            })
            seq_idx += 1

    print(f"  The number of protein residues: {len(residues)}")

    if mode == 'cys_only':
        candidates = [r for r in residues if r['res_name'] == 'CYS']
        print(f" The number of CYS residues: {len(candidates)}")
    else:
        candidates = residues

    seq_str = ''.join(AA3.get(r['res_name'], 'X') for r in residues)

    pairs = []
    n = len(candidates)
    for i in range(n):
        r1_info = candidates[i]
        r1 = r1_info['res']
        c1 = extract_backbone(r1)
        if c1 is None:
            continue
        for j in range(i + 1, n):
            r2_info = candidates[j]
            r2 = r2_info['res']
            ca_dist = np.linalg.norm(r1['CA'].coord - r2['CA'].coord)
            if not (CA_DIST_MIN <= ca_dist <= CA_DIST_MAX):
                continue
            c2 = extract_backbone(r2)
            if c2 is None:
                continue
            _, _, feat_45 = make_features(c1, c2)
            cb_dist = np.linalg.norm(c1[4] - c2[4])
            pairs.append({
                'chain1': r1_info['chain'], 'res1_id': r1_info['res_id'],
                'res1_name': r1_info['res_name'], 'seq_idx1': r1_info['seq_idx'],
                'flex1': r1_info['flex'],
                'chain2': r2_info['chain'], 'res2_id': r2_info['res_id'],
                'res2_name': r2_info['res_name'], 'seq_idx2': r2_info['seq_idx'],
                'flex2': r2_info['flex'],
                'ca_dist': float(ca_dist), 'cb_dist': float(cb_dist),
                'feat_45': feat_45,
            })

    print(f"  Geometric pre-screening candidate pairs log: {len(pairs)}")
    return pairs, seq_str, residues


def compute_zs_scores(pairs, sequence, model_name='esm2_t33_650M_UR50D'):
    try:
        import esm
        import torch
    except ImportError:
        print("  ESM is not installed, so skipping the ZS calculation.")
        return [{'zs_res1': 0.0, 'zs_res2': 0.0,
                 'zs_joint': 0.0, 'zs_coev': 0.0} for _ in pairs]

    print(f"  Load the ESM-2 model: {model_name}")
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    unique_positions = set()
    for p in pairs:
        unique_positions.add(p['seq_idx1'])
        unique_positions.add(p['seq_idx2'])

    print(f"  Calculate the single-site ZS scores for {len(unique_positions)} ")
    single_zs = {}
    for pos in unique_positions:
        if pos >= len(sequence):
            single_zs[pos] = 0.0
            continue
        masked = sequence[:pos] + '<mask>' + sequence[pos+1:]
        _, _, tokens = batch_converter([("p", masked)])
        tokens = tokens.to(device)
        with torch.no_grad():
            logits = model(tokens, repr_layers=[])['logits'][0, pos+1]
        log_probs = torch.log_softmax(logits, dim=-1)
        cys_idx = alphabet.get_idx('C')
        orig_idx = alphabet.get_idx(sequence[pos])
        single_zs[pos] = float(log_probs[cys_idx] - log_probs[orig_idx])

    print(f"  Calculate the combined ZS score for {len(pairs)} pairs of residues...")
    from tqdm import tqdm
    zs_results = []
    for p in tqdm(pairs, desc="ZSJoint calculation"):
        i, j = p['seq_idx1'], p['seq_idx2']
        if i >= len(sequence) or j >= len(sequence):
            zs_results.append({'zs_res1': 0.0, 'zs_res2': 0.0,
                               'zs_joint': 0.0, 'zs_coev': 0.0})
            continue
        seq_list = list(sequence)
        seq_list[i] = '<mask>'
        seq_list[j] = '<mask>'
        masked_joint = ''.join(seq_list)
        _, _, tokens = batch_converter([("p", masked_joint)])
        tokens = tokens.to(device)
        with torch.no_grad():
            logits = model(tokens, repr_layers=[])['logits'][0]
        cys_idx = alphabet.get_idx('C')
        lp_i = torch.log_softmax(logits[i+1], dim=-1)
        lp_j = torch.log_softmax(logits[j+1], dim=-1)
        orig_i = alphabet.get_idx(sequence[i])
        orig_j = alphabet.get_idx(sequence[j])
        zs_i = float(lp_i[cys_idx] - lp_i[orig_i])
        zs_j = float(lp_j[cys_idx] - lp_j[orig_j])
        zs_joint = zs_i + zs_j
        zs_coev = zs_joint - (single_zs[i] + single_zs[j])
        zs_results.append({
            'zs_res1': single_zs[i], 'zs_res2': single_zs[j],
            'zs_joint': zs_joint, 'zs_coev': zs_coev
        })
    return zs_results

def predict(pairs, zs_scores, model_path):
    import joblib
    model = joblib.load(model_path)
    X = np.array([
        np.concatenate([
            p['feat_45'],
            [z['zs_res1'], z['zs_res2'], z['zs_joint'], z['zs_coev']]
        ]) for p, z in zip(pairs, zs_scores)
    ])
    probs = model.predict_proba(X)[:, 1]
    return probs

def format_results(pairs, zs_scores, probs, top_n=20, output_file=None):
    import pandas as pd
    rows = []
    for p, z, prob in zip(pairs, zs_scores, probs):
        if prob >= 0.9:
            grade = '★★★ Strongly recommend'
        elif prob >= 0.7:
            grade = '★★  Suggestion for verification'
        elif prob >= 0.5:
            grade = '★   Carefully consider'
        else:
            grade = '-   Not recommended'
        need_mut1 = '' if p['res1_name'] == 'CYS' else '→Cys'
        need_mut2 = '' if p['res2_name'] == 'CYS' else '→Cys'
        rows.append({
            'Ranking': 0,
            'Residue1': f"{p['chain1']}{p['res1_id']}({p['res1_name']}{need_mut1})",
            'Residue2': f"{p['chain2']}{p['res2_id']}({p['res2_name']}{need_mut2})",
            'P(SS_bond)': round(float(prob), 4),
            'CA_distance(Å)': round(p['ca_dist'], 2),
            'CB_distance(Å)': round(p['cb_dist'], 2),
            'ZS_joint': round(z['zs_joint'], 3),
            'ZS_coev': round(z['zs_coev'], 3),
            'Flexibility1': round(p['flex1'], 1) if p['flex1'] else 'N/A',
            'Flexibility2': round(p['flex2'], 1) if p['flex2'] else 'N/A',
            'Recommendation_Level': grade,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('P(SS_bond)', ascending=False).reset_index(drop=True)
    df['Ranking'] = df.index + 1

    print(f"\n{'='*70}")
    print(f"Prediction results Top-{top_n}")
    print(f"{'='*70}")
    display_cols = ['Ranking', 'Residue1', 'Residue2', 'P(SS_bond)',
                    'CA_distance(Å)', 'ZS_joint', 'Recommendation_Level']
    print(df[display_cols].head(top_n).to_string(index=False))

    print(f"\nStatistics:")
    print(f"  Total candidate count: {len(df)}")
    print(f"  P>0.9 (Strongly recommend): {(df['P(SS_bond)']>=0.9).sum()}")
    print(f"  P>0.7 (Suggestion for verification): {(df['P(SS_bond)']>=0.7).sum()}")
    print(f"  P>0.5 (Carefully consider): {(df['P(SS_bond)']>=0.5).sum()}")

    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nThe result has been saved: {output_file}")
    return df
    
def main():
    ap = argparse.ArgumentParser(description='EvoSSBond: Prediction of engineered disulfide bond sites in proteins')
    
    ap.add_argument('structure', 
                    help='Input protein structure file (.pdb or .cif)')
    
    ap.add_argument('model_dir', 
                    help='Directory containing the trained model (m2_xgb_49dim.pkl)')
    
    ap.add_argument('--mode', choices=['cys_only', 'full_scan'],
                    default='full_scan', 
                    help='Prediction mode (default: full_scan)')
    
    ap.add_argument('--alphafold', action='store_true',
                    help='Indicates that the input structure is from AlphaFold (B-factor column contains pLDDT)')
    
    ap.add_argument('--top', type=int, default=20, 
                    help='Number of top-ranked predictions to display')
    
    ap.add_argument('--output', default=None, 
                    help='Output file path (default: <structure_name>_predictions.csv)')
    
    ap.add_argument('--esm_model', default='esm2_t33_650M_UR50D',
                    help='ESM-2 model version to use')
    
    args = ap.parse_args()

    structure_file = Path(args.structure)
    model_dir = Path(args.model_dir)
    output_file = args.output or f"{structure_file.stem}_predictions.csv"

    print(f"\nEvoSSBond: Disulfide Bond Site Prediction")
    print(f"{'='*50}")
    print(f"Target protein: {structure_file.name}")
    print(f"Prediction mode: {args.mode}")

    print(f"\n[1/4] Parsing structure...")
    structure = parse_structure(structure_file)
    if structure is None:
        sys.exit(1)

    print(f"\n[2/4] Extracting candidate residue pairs...")
    pairs, sequence, residues = get_candidate_pairs(
        structure, mode=args.mode, is_alphafold=args.alphafold)
    
    if not pairs:
        print("No candidate residue pairs found.")
        sys.exit(1)

    print(f"\n[3/4] Computing zero-shot evolutionary scores...")
    zs_scores = compute_zs_scores(pairs, sequence, args.esm_model)

    print(f"\n[4/4] Running model prediction...")
    model_path = model_dir / 'm2_xgb_49dim.pkl'
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    probs = predict(pairs, zs_scores, model_path)

    format_results(pairs, zs_scores, probs, 
                   top_n=args.top, output_file=output_file)


if __name__ == '__main__':
    main()
