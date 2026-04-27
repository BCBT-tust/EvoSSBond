"""
EvoSSBond Dual-Model Diagnostic (standalone)
"""

import numpy as np
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

AA_MAP = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


def parse_structure(structure_file):
    structure_file = Path(structure_file)
    pdb_id = structure_file.stem
    try:
        if structure_file.suffix == '.cif':
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        return parser.get_structure(pdb_id, structure_file)
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
                'flex':     get_flexibility(res, is_alphafold),
            })
            seq_idx += 1

    print(f"  蛋白质残基数: {len(residues)}")

    if mode == 'cys_only':
        candidates = [r for r in residues if r['res_name'] == 'CYS']
        print(f"  CYS残基数: {len(candidates)}")
    else:
        candidates = residues
        print(f"  候选残基数: {len(candidates)}")

    seq_str = ''.join(AA_MAP.get(r['res_name'], 'X') for r in residues)

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

    print(f"  几何预筛选后候选对数: {len(pairs)}")
    return pairs, seq_str, residues


def compute_zs_scores(pairs, sequence, model_name='esm2_t33_650M_UR50D'):
    try:
        import esm
        import torch
    except ImportError:
        print("  ESM未安装，跳过ZS计算")
        return [{'zs_res1': 0.0, 'zs_res2': 0.0,
                 'zs_joint': 0.0, 'zs_coev': 0.0} for _ in pairs]

    print(f"  加载ESM-2模型: {model_name}")
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    unique_positions = set()
    for p in pairs:
        unique_positions.add(p['seq_idx1'])
        unique_positions.add(p['seq_idx2'])

    print(f"  计算 {len(unique_positions)} 个位点的单点ZS分数...")
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

    print(f"  计算 {len(pairs)} 个残基对的联合ZS分数...")
    from tqdm import tqdm
    zs_results = []
    for p in tqdm(pairs, desc="ZS联合计算"):
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
            'zs_joint': zs_joint, 'zs_coev': zs_coev,
        })
    return zs_results


# ────────────────────────────────────────────────────────
# 双模型预测
# ────────────────────────────────────────────────────────

def predict_dual(pairs, zs_scores, m1_path, m2_path):
    """同一组候选对，分别用 m1 (45D) 和 m2 (49D) 预测。"""
    import joblib

    # 45 维：仅结构特征
    X45 = np.array([p['feat_45'] for p in pairs])
    if X45.shape[1] != 45:
        sys.exit(f"特征维度错误：m1 输入应为 45 维，实际 {X45.shape[1]}")

    # 49 维：结构 + ZS
    X49 = np.array([
        np.concatenate([
            p['feat_45'],
            [z['zs_res1'], z['zs_res2'], z['zs_joint'], z['zs_coev']]
        ]) for p, z in zip(pairs, zs_scores)
    ])
    if X49.shape[1] != 49:
        sys.exit(f"特征维度错误：m2 输入应为 49 维，实际 {X49.shape[1]}")

    print(f"\n[m1] 加载 {m1_path.name} ({m1_path})")
    m1_model = joblib.load(m1_path)
    probs_m1 = m1_model.predict_proba(X45)[:, 1]

    print(f"[m2] 加载 {m2_path.name} ({m2_path})")
    m2_model = joblib.load(m2_path)
    probs_m2 = m2_model.predict_proba(X49)[:, 1]

    return probs_m1, probs_m2


def build_results_table(pairs, zs_scores, probs):
    import pandas as pd
    rows = []
    for p, z, prob in zip(pairs, zs_scores, probs):
        rows.append({
            'res1_key':       f"{p['chain1']}{p['res1_id']}",
            'res2_key':       f"{p['chain2']}{p['res2_id']}",
            'res1_name':      p['res1_name'],
            'res2_name':      p['res2_name'],
            'P':              float(prob),
            'CA_distance':    round(p['ca_dist'], 2),
            'CB_distance':    round(p['cb_dist'], 2),
            'ZS_joint':       round(z['zs_joint'], 3),
            'ZS_coev':        round(z['zs_coev'], 3),
            'is_native_pair': (p['res1_name'] == 'CYS' and p['res2_name'] == 'CYS'),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('P', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df


def find_target_pair(df, res1_id, res2_id, chain='A'):
    keys = {f"{chain}{res1_id}", f"{chain}{res2_id}"}
    for idx, row in df.iterrows():
        if {row['res1_key'], row['res2_key']} == keys:
            return row
    return None


def parse_target_pairs(target_str):
    if not target_str:
        return []
    pairs = []
    for chunk in target_str.split(','):
        a, b = chunk.strip().split(':')
        pairs.append((int(a), int(b)))
    return pairs


def diagnostic_summary(df_m1, df_m2, target_pairs, structure_name):
    print("\n" + "=" * 90)
    print(f"诊断摘要: {structure_name}")
    print("=" * 90)

    print("\n--- m1 (45D, 结构-only) Top 5 ---")
    print(df_m1[['Rank', 'res1_key', 'res2_key', 'res1_name', 'res2_name',
                 'P', 'CB_distance', 'ZS_joint', 'is_native_pair']].head(5).to_string(index=False))

    print("\n--- m2 (49D, 结构+ZS) Top 5 ---")
    print(df_m2[['Rank', 'res1_key', 'res2_key', 'res1_name', 'res2_name',
                 'P', 'CB_distance', 'ZS_joint', 'is_native_pair']].head(5).to_string(index=False))

    n_native_top10_m1 = int(df_m1.head(10)['is_native_pair'].sum())
    n_native_top10_m2 = int(df_m2.head(10)['is_native_pair'].sum())
    n_native_total    = int(df_m1['is_native_pair'].sum())
    print(f"\n--- 天然 Cys-Cys 对（已经是 Cys 的对）---")
    print(f"  候选对中天然对总数: {n_native_total}")
    print(f"  m1 Top 10 中天然对: {n_native_top10_m1}")
    print(f"  m2 Top 10 中天然对: {n_native_top10_m2}")
    if n_native_top10_m2 > n_native_top10_m1 and n_native_total > 0:
        print(f"  ⚠️  m2 在 Top 10 中显著偏好天然对 → 提示 ZS 捷径")

    if target_pairs:
        print(f"\n--- 目标工程位点排名对比 ---")
        header = f"{'目标对':<14} {'m1 排名':<10} {'m1 P':<10} {'m2 排名':<10} {'m2 P':<10} {'排名提升 (m2→m1)'}"
        print(header)
        print("-" * len(header))
        for r1, r2 in target_pairs:
            row_m1 = find_target_pair(df_m1, r1, r2)
            row_m2 = find_target_pair(df_m2, r1, r2)
            if row_m1 is not None and row_m2 is not None:
                drank = int(row_m2['Rank']) - int(row_m1['Rank'])
                arrow = '↑' if drank > 0 else ('↓' if drank < 0 else '=')
                print(f"{r1}-{str(r2):<10} #{int(row_m1['Rank']):<8} {row_m1['P']:<10.4f} "
                      f"#{int(row_m2['Rank']):<8} {row_m2['P']:<10.4f} "
                      f"{arrow} {drank:+d}")
            else:
                print(f"{r1}-{r2}  ← 候选对中未找到（被几何预筛过滤）")

    print("\n--- 解读 ---")
    print("  * 若目标对在 m1 上的排名远高于 m2 → 证实 m2 存在 ZS 捷径")
    print("  * 若 m2 Top 10 多为天然 Cys-Cys 对而 m1 不是 → 证实 ZS 捷径")
    print("  * 若 m1 上目标对仍排名很低 → 几何信号不足，需更深层修复")


# ────────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='EvoSSBond 双模型诊断脚本')
    ap.add_argument('structure', help='输入结构文件 (.pdb 或 .cif)')
    ap.add_argument('model_dir', help='模型目录（含 m1_xgb_45dim.pkl 和 m2_xgb_49dim.pkl）')
    ap.add_argument('--alphafold', action='store_true',
                    help='输入为 AlphaFold 结构（B 因子列为 pLDDT）')
    ap.add_argument('--mode', default='full_scan', choices=['full_scan', 'cys_only'])
    ap.add_argument('--output_prefix', default=None, help='输出文件前缀')
    ap.add_argument('--target_pairs', default=None,
                    help='要追踪的残基对，格式 "666:680,2:28"')
    ap.add_argument('--esm_model', default='esm2_t33_650M_UR50D')
    ap.add_argument('--m1_name', default='m1_xgb_45dim.pkl')
    ap.add_argument('--m2_name', default='m2_xgb_49dim.pkl')
    args = ap.parse_args()

    structure_path = Path(args.structure)
    model_dir = Path(args.model_dir)
    prefix = args.output_prefix or structure_path.stem
    target_pairs = parse_target_pairs(args.target_pairs)

    m1_path = model_dir / args.m1_name
    m2_path = model_dir / args.m2_name

    if not m1_path.exists():
        sys.exit(f"m1 模型未找到: {m1_path}")
    if not m2_path.exists():
        sys.exit(f"m2 模型未找到: {m2_path}")

    print(f"\nEvoSSBond 双模型诊断")
    print("=" * 50)
    print(f"目标蛋白: {structure_path}")
    print(f"m1 (45D): {m1_path}")
    print(f"m2 (49D): {m2_path}")
    print(f"模式: {args.mode}, AlphaFold: {args.alphafold}")
    if target_pairs:
        print(f"追踪目标对: {target_pairs}")

    print(f"\n[1/4] 解析结构...")
    structure = parse_structure(structure_path)
    if structure is None:
        sys.exit(1)

    print(f"\n[2/4] 提取候选残基对...")
    pairs, sequence, residues = get_candidate_pairs(
        structure, mode=args.mode, is_alphafold=args.alphafold)
    if not pairs:
        sys.exit("无候选对")

    print(f"\n[3/4] 计算 ESM-2 ZS 分数...")
    zs_scores = compute_zs_scores(pairs, sequence, args.esm_model)

    print(f"\n[4/4] 运行 m1 与 m2 双模型预测...")
    probs_m1, probs_m2 = predict_dual(pairs, zs_scores, m1_path, m2_path)

    df_m1 = build_results_table(pairs, zs_scores, probs_m1)
    df_m2 = build_results_table(pairs, zs_scores, probs_m2)

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    df_m1.to_csv(f"{prefix}_m1_predictions.csv", index=False, encoding='utf-8-sig')
    df_m2.to_csv(f"{prefix}_m2_predictions.csv", index=False, encoding='utf-8-sig')
    print(f"\n保存: {prefix}_m1_predictions.csv  ({len(df_m1)} 对)")
    print(f"保存: {prefix}_m2_predictions.csv  ({len(df_m2)} 对)")

    import pandas as pd
    df_compare = df_m1[['Rank', 'res1_key', 'res2_key', 'res1_name', 'res2_name',
                        'P', 'CB_distance', 'ZS_joint', 'is_native_pair']].copy()
    df_compare = df_compare.rename(columns={'Rank': 'rank_m1', 'P': 'P_m1'})
    df_m2_min = df_m2[['res1_key', 'res2_key', 'Rank', 'P']].rename(
        columns={'Rank': 'rank_m2', 'P': 'P_m2'})
    df_compare = df_compare.merge(df_m2_min, on=['res1_key', 'res2_key'], how='left')
    df_compare['rank_shift_m2_minus_m1'] = df_compare['rank_m2'] - df_compare['rank_m1']
    df_compare = df_compare.sort_values('rank_m1')
    df_compare.to_csv(f"{prefix}_comparison.csv", index=False, encoding='utf-8-sig')
    print(f"保存: {prefix}_comparison.csv")

    diagnostic_summary(df_m1, df_m2, target_pairs, structure_path.stem)


if __name__ == '__main__':
    main()
