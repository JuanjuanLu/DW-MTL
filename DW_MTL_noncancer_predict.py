#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DW-MTL non-cancer RTV batch prediction script.

This script is a simplified version of the original molecular-graph interpretation
notebook. It keeps only the functions needed for non-cancer RTV prediction on new compounds:

1. Read SMILES from a CSV file.
2. Convert SMILES into DGL molecular graphs.
3. Load one or more trained DW-MTL/MGA models and their hyperparameter pkl files.
4. Predict only main RTV tasks, such as OSF, ISF, IUR, RfD, RfC and BMD.
5. Save RfD, RfC and BMD prediction results to a CSV file.

How to use
----------
Modify the USER CONFIGURATION section below, especially INPUT_CSV, OUTPUT_CSV,
FAILED_CSV and MODEL_JOBS, then run:

python DW_MTL_noncancer_predict.py

Notes
-----
- Classification auxiliary tasks are not exported.
- Main regression task outputs are reported directly in the model output scale.
- Mechanistic interpretation, atom weights and visualization are not included.
"""

import os
import pickle as pkl
import random
from typing import List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.readout import sum_nodes
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def set_random_seed(seed: int = 10) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Map inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
    else:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


# -----------------------------------------------------------------------------
# Molecular graph construction
# -----------------------------------------------------------------------------

def atom_features(atom, explicit_H: bool = False, use_chirality: bool = True) -> np.ndarray:
    results = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As',
             'Se', 'Br', 'Te', 'I', 'At', 'other']
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [Chem.rdchem.HybridizationType.SP,
             Chem.rdchem.HybridizationType.SP2,
             Chem.rdchem.HybridizationType.SP3,
             Chem.rdchem.HybridizationType.SP3D,
             Chem.rdchem.HybridizationType.SP3D2,
             'other']
        )
        + [atom.GetIsAromatic()]
    )

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) \
                      + [atom.HasProp('_ChiralityPossible')]
        except Exception:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results, dtype=np.float32)


def etype_features(bond, use_chirality: bool = True, atompair: bool = True) -> int:
    bt = bond.GetBondType()
    bond_type_flags = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
    ]
    a = int(np.argmax(bond_type_flags))

    b = 1 if bond.GetIsConjugated() else 0
    c = 1 if bond.IsInRing() else 0
    index = a + b * 4 + c * 8

    if use_chirality:
        stereo_flags = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
        d = int(np.argmax(stereo_flags))
        index = index + d * 16

    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        atom_pair_flags = one_of_k_atompair_encoding(
            atom_pair_str,
            [
                ['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                ['CBr', 'BrC'], ['others']
            ]
        )
        e = int(np.argmax(atom_pair_flags))
        index = index + e * 64

    return int(index)


def construct_RGCN_bigraph_from_smiles(smiles: str, expected_feature_dim: int = 40) -> dgl.DGLGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError(f"No atoms found in SMILES: {smiles}")

    g = dgl.graph(([], []), num_nodes=num_atoms)

    atom_feature_list = []
    for atom in mol.GetAtoms():
        feat = torch.from_numpy(atom_features(atom, explicit_H=False, use_chirality=True))
        if feat.shape[0] < expected_feature_dim:
            feat = torch.cat([feat, torch.zeros(expected_feature_dim - feat.shape[0])])
        elif feat.shape[0] > expected_feature_dim:
            feat = feat[:expected_feature_dim]
        atom_feature_list.append(feat)
    g.ndata['atom'] = torch.stack(atom_feature_list).float()

    src_list, dst_list, etype_list = [], [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        etype = etype_features(bond, use_chirality=True, atompair=True)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_list.extend([etype, etype])

    if src_list:
        g.add_edges(src_list, dst_list)
        g.edata['etype'] = torch.tensor(etype_list, dtype=torch.long)
    else:
        g.edata['etype'] = torch.tensor([], dtype=torch.long)

    return g


# -----------------------------------------------------------------------------
# Model definition: MGA / DW-MTL backbone
# -----------------------------------------------------------------------------

class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super().__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight = return_weight
        self.atom_weighting_specific = nn.ModuleList(
            [self.atom_weight(self.in_feats) for _ in range(self.task_num)]
        )
        self.shared_weighting = self.atom_weight(self.in_feats)

    def forward(self, bg, feats):
        feat_list = []
        atom_list = []
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')

        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            return feat_list
        return shared_feats_sum

    @staticmethod
    def atom_weight(in_feats):
        return nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())


class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels=64 * 21, activation=F.relu,
                 loop=False, residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super().__init__()
        self.activation = activation
        self.graph_conv_layer = RelGraphConv(
            in_feats, out_feats, num_rels=num_rels, regularizer='basis',
            num_bases=None, bias=True, activation=activation,
            self_loop=loop, dropout=rgcn_drop_out
        )
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, norm=None):
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        return new_feats


class BaseGNN(nn.Module):
    def __init__(self, gnn_out_feats, n_tasks, rgcn_drop_out=0.5,
                 return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.0):
        super().__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.return_mol_embedding = return_mol_embedding
        self.weighted_sum_readout = WeightAndSum(
            gnn_out_feats, self.task_num, return_weight=self.return_weight
        )
        self.fc_in_feats = gnn_out_feats

        self.fc_layers1 = nn.ModuleList([
            self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])
        self.fc_layers2 = nn.ModuleList([
            self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])
        self.fc_layers3 = nn.ModuleList([
            self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats)
            for _ in range(self.task_num)
        ])
        self.output_layer1 = nn.ModuleList([
            self.output_layer(classifier_hidden_feats, 1)
            for _ in range(self.task_num)
        ])

    def forward(self, bg, node_feats, etype, norm=None):
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype, norm)

        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        prediction_all = None
        for i in range(self.task_num):
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            prediction_all = predict if prediction_all is None else torch.cat([prediction_all, predict], dim=1)

        if self.return_mol_embedding:
            return feats_list[0]
        if self.return_weight:
            return prediction_all, atom_weight_list, node_feats
        return prediction_all

    @staticmethod
    def fc_layer(dropout, in_feats, hidden_feats):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats)
        )

    @staticmethod
    def output_layer(hidden_feats, out_feats):
        return nn.Sequential(nn.Linear(hidden_feats, out_feats))


class MGA(BaseGNN):
    def __init__(self, in_feats, rgcn_hidden_feats, n_tasks, return_weight=False,
                 classifier_hidden_feats=128, loop=False, return_mol_embedding=False,
                 rgcn_drop_out=0.5, dropout=0.0):
        super().__init__(
            gnn_out_feats=rgcn_hidden_feats[-1],
            n_tasks=n_tasks,
            classifier_hidden_feats=classifier_hidden_feats,
            return_mol_embedding=return_mol_embedding,
            return_weight=return_weight,
            rgcn_drop_out=rgcn_drop_out,
            dropout=dropout,
        )

        for out_feats in rgcn_hidden_feats:
            self.gnn_layers.append(
                RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out)
            )
            in_feats = out_feats


# -----------------------------------------------------------------------------
# Data loading and prediction
# -----------------------------------------------------------------------------

def read_smiles_from_csv(csv_path: str, smiles_col: Optional[str] = None) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    if smiles_col is not None:
        if smiles_col not in df.columns:
            raise ValueError(f"smiles_col='{smiles_col}' was not found. Available columns: {list(df.columns)}")
        used_col = smiles_col
    else:
        candidate_cols = [
            'SMILES', 'smiles', 'Smiles', 'canonical_smiles',
            'Canonical_Smiles', 'CANONICAL_SMILES', 'mol_smiles'
        ]
        used_col = next((col for col in candidate_cols if col in df.columns), df.columns[0])

    smiles_list = df[used_col].fillna('').astype(str).str.strip().tolist()
    return df, used_col, smiles_list


def create_molecule_dataset(smiles_list: List[str], expected_feature_dim: int):
    dataset = []
    failed_records = []

    for row_idx, smiles in enumerate(tqdm(smiles_list, desc="Constructing molecular graphs")):
        if not smiles:
            failed_records.append({'row_index': row_idx, 'SMILES': smiles, 'reason': 'empty SMILES'})
            continue
        try:
            graph = construct_RGCN_bigraph_from_smiles(smiles, expected_feature_dim=expected_feature_dim)
            dataset.append((row_idx, smiles, graph))
        except Exception as exc:
            failed_records.append({'row_index': row_idx, 'SMILES': smiles, 'reason': str(exc)})

    return dataset, failed_records


def collate_molgraphs(data):
    row_indices, smiles, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return row_indices, smiles, bg


def load_args(config_path: str) -> dict:
    with open(config_path, 'rb') as f:
        args = pkl.load(f)
    if not isinstance(args, dict):
        raise TypeError("The config pkl file should contain a dictionary of model hyperparameters.")
    return args


def infer_task_names(args: dict, n_tasks_from_cli: Optional[int] = None) -> List[str]:
    if 'select_task_list' in args and args['select_task_list']:
        return list(args['select_task_list'])

    if n_tasks_from_cli is not None:
        return [f'task_{i + 1}' for i in range(n_tasks_from_cli)]

    raise ValueError(
        "Task names were not found in the config file. Please provide --n_tasks or use a config "
        "containing args['select_task_list']."
    )


def load_model(model_path: str, args: dict, n_tasks: int, device: torch.device) -> MGA:
    model = MGA(
        in_feats=args.get('in_feats', 40),
        rgcn_hidden_feats=args['rgcn_hidden_feats'],
        n_tasks=n_tasks,
        return_weight=False,
        classifier_hidden_feats=args['classifier_hidden_feats'],
        loop=args.get('loop', False),
        return_mol_embedding=False,
        rgcn_drop_out=args.get('rgcn_drop_out', 0.5),
        dropout=args.get('drop_out', args.get('dropout', 0.0)),
    )

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, MGA):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model



def select_main_tasks(task_names: List[str], main_task_names: List[str]) -> Tuple[List[str], List[int]]:
    """Select only main-task outputs from the full multitask prediction vector."""
    available = {name: idx for idx, name in enumerate(task_names)}
    selected_names = []
    selected_indices = []

    for task in main_task_names:
        if task in available:
            selected_names.append(task)
            selected_indices.append(available[task])

    if not selected_names:
        raise ValueError(
            "None of MAIN_TASK_NAMES were found in the model task list.\n"
            f"MAIN_TASK_NAMES = {main_task_names}\n"
            f"Model task list = {task_names}"
        )

    return selected_names, selected_indices


def predict_dataset_main_tasks(
    model: MGA,
    dataset,
    task_names: List[str],
    main_task_names: List[str],
    batch_size: int,
    device: torch.device,
    prediction_prefix: str = "",
) -> pd.DataFrame:
    """
    Predict only selected main tasks.

    The model may contain classification auxiliary tasks and regression main tasks.
    This function keeps only MAIN_TASK_NAMES and does not output classification tasks.
    """
    selected_names, selected_indices = select_main_tasks(task_names, main_task_names)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_molgraphs,
    )

    all_records = []
    with torch.no_grad():
        for row_indices, smiles, bg in tqdm(data_loader, desc=f"Predicting {prediction_prefix or 'model'}"):
            bg = bg.to(device)
            atom_feats = bg.ndata['atom'].float().to(device)
            bond_feats = bg.edata['etype'].long().to(device)

            outputs = model(bg, atom_feats, bond_feats, norm=None)
            preds = outputs[:, selected_indices].detach().cpu().numpy()

            for i, row_idx in enumerate(row_indices):
                record = {'row_index': row_idx, 'SMILES': smiles[i]}
                for j, task_name in enumerate(selected_names):
                    col_name = f"{prediction_prefix}_{task_name}" if prediction_prefix else task_name
                    record[col_name] = float(preds[i, j])
                all_records.append(record)

    return pd.DataFrame(all_records)


def save_failed_smiles(failed_records, failed_csv: Optional[str]) -> None:
    if not failed_records:
        return
    failed_df = pd.DataFrame(failed_records)
    if failed_csv is None:
        failed_csv = 'failed_smiles.csv'
    failed_dir = os.path.dirname(os.path.abspath(failed_csv))
    if failed_dir:
        os.makedirs(failed_dir, exist_ok=True)
    failed_df.to_csv(failed_csv, index=False, encoding='utf-8-sig')
    print(f"Failed SMILES saved to: {failed_csv}")


# =============================================================================
# User configuration
# =============================================================================
# 修改下面这些路径后，直接运行本脚本即可。
# python DW_MTL_noncancer_predict.py

INPUT_CSV = r"C:\Users\Administrator\Desktop\DW-MTL-main\data\test chemicals.csv"
SMILES_COL = "SMILES"          # 如果你的列名不是 SMILES，请修改这里；若想自动识别，可设为 None
OUTPUT_CSV = r"C:\Users\Administrator\Desktop\DW-MTL-main\data\test chemicals predictionnoncancer.csv"
FAILED_CSV = r"/path/to/noncancer_failed_smiles.csv"

KEEP_INPUT_COLUMNS = True       # True: 在原始表格后面追加预测结果；False: 只输出 row_index, SMILES 和预测结果
BATCH_SIZE = 32
DEVICE = "cuda"                # 可选 "cuda" 或 "cpu"；没有 GPU 时会自动使用 CPU
SEED = 10

# 只保留非致癌主任务输出，不输出 CYP 分类辅助任务。
MAIN_TASK_NAMES = ["rfd", "rfc", "bmd"]

# 可以同时放入一个或多个模型。
# - name: 输出列名前缀。若只跑一个模型且不想加前缀，可设为 ""。
# - model_path: 训练好的模型权重 .pth
# - config_path: 训练时保存的参数 .pkl，里面应包含 select_task_list、rgcn_hidden_feats 等信息
#
# 示例：如果你只预测致癌 RTV，就只保留 cancer 这一项；
# 如果你同时预测致癌和非致癌 RTV，就保留两个模型。
MODEL_JOBS = [
    {
        # 输出列名将为 RfD, RfC, BMD；如果希望加前缀，可改为 "noncancer"
        "name": "",
        "model_path": r"C:\Users\Administrator\Desktop\DW-MTL-main\DW-MTL models\DW-MTL-noncancer.pth",
        "config_path": r"C:\Users\Administrator\Desktop\DW-MTL-main\DW-MTL models\DW-MTL-noncacer.pkl",
    },
]


# =============================================================================
# Main prediction workflow
# =============================================================================

def main():
    set_random_seed(SEED)

    device = torch.device('cuda' if DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not MODEL_JOBS:
        raise ValueError("MODEL_JOBS is empty. Please add at least one model path and config path.")

    for job in MODEL_JOBS:
        if not os.path.exists(job["model_path"]):
            raise FileNotFoundError(f"Model file was not found: {job['model_path']}")
        if not os.path.exists(job["config_path"]):
            raise FileNotFoundError(f"Config pkl file was not found: {job['config_path']}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV was not found: {INPUT_CSV}")

    # Use the first config to determine atom feature dimension.
    first_args = load_args(MODEL_JOBS[0]["config_path"])
    expected_feature_dim = int(first_args.get('in_feats', 40))

    raw_df, used_col, smiles_list = read_smiles_from_csv(INPUT_CSV, SMILES_COL)
    print(f"Using SMILES column: {used_col}")
    print(f"Input molecules: {len(smiles_list)}")
    print(f"Input atom feature dimension: {expected_feature_dim}")

    dataset, failed_records = create_molecule_dataset(smiles_list, expected_feature_dim)
    print(f"Successfully constructed graphs: {len(dataset)}")
    print(f"Failed SMILES: {len(failed_records)}")

    if not dataset:
        save_failed_smiles(failed_records, FAILED_CSV)
        raise RuntimeError("No valid molecular graphs were constructed. Please check the SMILES input.")

    merged_pred_df = None

    for job in MODEL_JOBS:
        job_name = job.get("name", "")
        args = load_args(job["config_path"])
        task_names = infer_task_names(args, n_tasks_from_cli=None)
        n_tasks = len(task_names)

        selected_names, selected_indices = select_main_tasks(task_names, MAIN_TASK_NAMES)
        print("-" * 80)
        print(f"Model: {job_name or 'model'}")
        print(f"All tasks in model: {task_names}")
        print(f"Main tasks to output: {selected_names}")

        model = load_model(job["model_path"], args, n_tasks, device)
        pred_df = predict_dataset_main_tasks(
            model=model,
            dataset=dataset,
            task_names=task_names,
            main_task_names=MAIN_TASK_NAMES,
            batch_size=min(BATCH_SIZE, len(dataset)),
            device=device,
            prediction_prefix=job_name,
        )

        if merged_pred_df is None:
            merged_pred_df = pred_df
        else:
            # Keep row_index and SMILES only once; append new prediction columns.
            add_cols = [c for c in pred_df.columns if c not in ['row_index', 'SMILES']]
            merged_pred_df = merged_pred_df.merge(
                pred_df[['row_index'] + add_cols],
                on='row_index',
                how='outer'
            )

    if KEEP_INPUT_COLUMNS:
        output_df = raw_df.reset_index().rename(columns={'index': 'row_index'}).merge(
            merged_pred_df.drop(columns=['SMILES']), on='row_index', how='left'
        )
    else:
        output_df = merged_pred_df

    output_dir = os.path.dirname(os.path.abspath(OUTPUT_CSV))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Prediction results saved to: {OUTPUT_CSV}")

    save_failed_smiles(failed_records, FAILED_CSV)
    print("Prediction completed.")


if __name__ == '__main__':
    main()
