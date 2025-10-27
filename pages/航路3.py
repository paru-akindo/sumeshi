# main.py
import streamlit as st
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import re
import requests
from io import StringIO
from copy import deepcopy
from math import prod

st.set_page_config(page_title="航路買い物（貪欲閉路群）", layout="wide")

# ITEMS はスプレッドシートのヘッダ名に合わせる
ITEMS = [
    ("鳳梨",100),("魚肉",100),("酒",100),("水稲",100),("木材",100),("ヤシ",100),
    ("海鮮",200),("絹糸",200),("水晶",200),("茶葉",200),("鉄鉱",200),
    ("香料",300),("玉器",300),("白銀",300),("皮革",300),
    ("真珠",500),("燕の巣",500),("陶器",500),
    ("象牙",1000),("鹿茸",1000)
]

# スプレッドシート設定（あなたの既定値）
SPREADSHEET_ID = "1ft5FlwM5kaZK7B4vLQg2m1WYe5nWNb0udw5isFwDWy0"
GID = "805544474"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}"
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={GID}"

st.markdown(f'<div style="margin-bottom:8px;"><a href="{SPREADSHEET_URL}" target="_blank" rel="noopener noreferrer">スプレッドシートを開く</a> | <a href="{CSV_URL}" target="_blank" rel="noopener noreferrer">CSV を開く</a></div>', unsafe_allow_html=True)

@st.cache_data(ttl=60)
def fetch_price_matrix_from_csv_auto(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    s = r.content.decode("utf-8")
    df = pd.read_csv(StringIO(s))
    if df.shape[1] < 2:
        raise ValueError("スプレッドシートに品目列/港列が見つかりません。")
    items_names = [name for name, _ in ITEMS]
    first_col_name = df.columns[0]
    first_col_values = df[first_col_name].astype(str).tolist()
    any_first_col_is_item = any(v in items_names for v in first_col_values)
    if any_first_col_is_item:
        df_items = df.set_index(df.columns[0])
        df_t = df_items.transpose().reset_index()
        port_col = df_t.columns[0]
        ports = df_t[port_col].astype(str).tolist()
        price_matrix = {name: {} for name, _ in ITEMS}
        for name, _ in ITEMS:
            if name in df_t.columns:
                for idx, p in enumerate(ports):
                    raw = df_t.at[idx, name]
                    try:
                        price_matrix[name][p] = int(raw)
                    except Exception:
                        price_matrix[name][p] = 0
            else:
                for p in ports:
                    price_matrix[name][p] = 0
        return ports, price_matrix
    else:
        port_col = df.columns[0]
        ports = df[port_col].astype(str).tolist()
        price_matrix = {name: {} for name, _ in ITEMS}
        for name, _ in ITEMS:
            if name in df.columns:
                for idx, p in enumerate(ports):
                    raw = df.at[idx, name]
                    try:
                        price_matrix[name][p] = int(raw)
                    except Exception:
                        price_matrix[name][p] = 0
            else:
                for p in ports:
                    price_matrix[name][p] = 0
        return ports, price_matrix

def greedy_plan_for_destination_general(current_port: str, dest_port: str, cash: int, stock: Optional[Dict[str,int]], price_matrix: Dict[str,Dict[str,int]]):
    cash = int(cash) if cash is not None else 0
    candidates = []
    for item, base in ITEMS:
        avail = float('inf') if stock is None else stock.get(item, 0)
        if avail <= 0:
            continue
        buy = price_matrix[item].get(current_port, 0)
        sell = price_matrix[item].get(dest_port, 0)
        unit_profit = sell - buy
        if unit_profit <= 0 or buy <= 0:
            continue
        candidates.append((item, buy, sell, unit_profit, avail))
    candidates.sort(key=lambda x: x[3], reverse=True)
    remaining_cash = cash
    plan = []
    for item, buy, sell, unit_profit, avail in candidates:
        if buy <= 0:
            continue
        max_by_cash = remaining_cash // buy if remaining_cash >= buy else 0
        qty = min(avail if avail != float('inf') else max_by_cash, max_by_cash)
        if qty <= 0:
            continue
        plan.append((item, int(qty), int(buy), int(sell), int(unit_profit)))
        remaining_cash -= qty * buy
        if remaining_cash <= 0:
            break
    total_cost = sum(q * b for _, q, b, _, _ in plan)
    total_profit = sum(q * up for _, q, _, _, up in plan)
    remaining_cash_after_sell = cash + total_profit
    return plan, int(total_cost), int(total_profit), int(remaining_cash_after_sell)

def compute_single_step_multipliers(price_matrix: Dict[str,Dict[str,int]], ports: List[str], cash: int):
    """
    全ての (p -> q) に対して stock=None で greedy を呼び、
    乗数 cash_after / cash_before を返す mapping と candidates list
    """
    mapping = {}
    candidates = []
    for p in ports:
        for q in ports:
            if p == q:
                continue
            plan, cost, profit, cash_after = greedy_plan_for_destination_general(p, q, cash, None, price_matrix)
            if cash <= 0:
                multiplier = 1.0
            else:
                multiplier = float(cash_after) / float(max(1, cash))
            mapping.setdefault(p, {})[q] = {'multiplier': multiplier, 'profit': profit, 'plan': plan, 'cash_after': cash_after}
            candidates.append((p, q, multiplier))
    # sort candidates by multiplier desc
    candidates.sort(key=lambda x: x[2], reverse=True)
    return mapping, candidates

def build_greedy_cycles_from_start(start_port: str, mapping: Dict, cash: int):
    """
    start_port から貪欲に最良遷移（最大乗数）を選んで辿り、閉路になったらそのルートを返す。
    戻り値: route (list), steps (list of dicts), final_cash, avg_multiplier_per_move, total_multiplier
    """
    visited = []
    cur = start_port
    cur_cash = cash
    steps = []
    visited_set = set()
    while True:
        # among possible next ports from cur, choose max multiplier
        next_candidates = mapping.get(cur, {})
        if not next_candidates:
            break
        # pick next q with highest multiplier
        q, info = max(next_candidates.items(), key=lambda kv: kv[1]['multiplier'])
        # record step
        steps.append({'from': cur, 'to': q, 'plan': info['plan'], 'step_profit': info['profit'], 'cash_after': info['cash_after'], 'multiplier': info['multiplier']})
        cur = q
        # detect cycle: if q has been seen in visited -> close cycle from its first occurrence
        if q in visited:
            # find index of first occurrence
            idx = visited.index(q)
            route = [start_port] + visited[idx:] + [q]
            # extract steps corresponding to the cycle segment
            # steps list corresponds 1:1 to transitions taken, so cycle steps are steps[idx+1 :]
            cycle_steps = steps[idx+1:]
            # compute multipliers
            multipliers = [s['multiplier'] for s in cycle_steps]
            total_mul = prod(multipliers) if multipliers else 1.0
            avg_mul = total_mul ** (1.0 / len(multipliers)) if multipliers else 1.0
            return route, cycle_steps, int(cycle_steps[-1]['cash_after']) if cycle_steps else cash, avg_mul, total_mul
        visited.append(q)
        visited_set.add(q)
        # safety: if route too long, break
        if len(visited) > len(mapping):
            break
    return None, None, None, None, None

def generate_routes_greedy_cover(ports: List[str], mapping: Dict, cash: int, top_k_start: int = 3):
    """
    - compute ordered candidates by single-step multiplier
    - for start indices in top_k_start (1..k), produce greedy cycles until all ports covered or candidates exhausted
    - returns list of route-groups per start-rank attempt
    """
    # flatten single-step bests to pick starting ports order
    singles = []
    for p in mapping:
        for q, info in mapping[p].items():
            singles.append((p, q, info['multiplier']))
    singles.sort(key=lambda x: x[2], reverse=True)
    # determine unique starting ports in order of best outgoing multiplier
    start_ports_order = []
    for p, q, m in singles:
        if p not in start_ports_order:
            start_ports_order.append(p)
    results_per_start = []
    for start_choice in start_ports_order[:top_k_start]:
        remaining_ports = set(ports)
        if start_choice in remaining_ports:
            remaining_ports.remove(start_choice)
        routes = []
        # repeat until no remaining ports or cannot find further cycles
        while remaining_ports:
            route, steps, final_cash, avg_mul, total_mul = build_greedy_cycles_from_start(start_choice, mapping, cash)
            if route is None:
                break
            # collect covered ports in this route (exclude start_choice if present at ends)
            covered = set(route)
            routes.append({'route': route, 'steps': steps, 'final_cash': final_cash, 'avg_mul': avg_mul, 'total_mul': total_mul, 'covered': covered})
            # remove covered ports from remaining_ports
            remaining_ports -= covered
            # pick next start as the highest-outgoing-multiplier port in remaining_ports if any, else stop
            next_start = None
            best_m = -1.0
            for p in remaining_ports:
                outs = mapping.get(p, {})
                if not outs:
                    continue
                q, info = max(outs.items(), key=lambda kv: kv[1]['multiplier'])
                if info['multiplier'] > best_m:
                    best_m = info['multiplier']
                    next_start = p
            if next_start is None:
                break
            start_choice = next_start
            if start_choice in remaining_ports:
                remaining_ports.remove(start_choice)
        results_per_start.append({'initial_start': start_choice, 'routes': routes, 'remaining_ports': remaining_ports})
    return results_per_start

# UI
st.title("貪欲閉路群探索（乗数ベース）")
try:
    ports, price_matrix = fetch_price_matrix_from_csv_auto(CSV_URL)
except Exception as e:
    st.error(f"CSV読み込み失敗: {e}")
    st.stop()

col1, col2 = st.columns([1,2])
with col1:
    cash = st.number_input("比較用の仮所持金", min_value=1, value=50000, step=1000)
    top_k_start = st.number_input("開始候補として試す上位k", min_value=1, max_value=10, value=3)
    depth_info = st.markdown("アルゴリズム: 各遷移は在庫無限で greedy を実行し、乗数（到着資産/出発資産）を計算。")

with col2:
    st.write(f"検出対象港数: {len(ports)}")
    if st.checkbox("価格表を表示"):
        rows = []
        for name, _ in ITEMS:
            row = {"品目": name}
            for p in ports:
                row[p] = price_matrix[name].get(p, 0)
            rows.append(row)
        df_all = pd.DataFrame(rows).set_index("品目")
        st.dataframe(df_all, height=480)

if st.button("閉路群を生成（貪欲）"):
    mapping, candidates = compute_single_step_multipliers(price_matrix, ports, cash)
    results = generate_routes_greedy_cover(ports, mapping, cash, top_k_start=top_k_start)
    # display
    for attempt in results:
        st.markdown(f"## 初期スタート候補: {attempt['initial_start']}")
        if not attempt['routes']:
            st.write("ルートが生成できませんでした。")
            continue
        for i, r in enumerate(attempt['routes'], start=1):
            st.markdown(f"### ルート {i}: {' → '.join(r['route'])}")
            st.markdown(f"- カバー港数: {len(r['covered'])}  最終資産乗数: {r['total_mul'] if 'total_mul' in r else r['final_cash']/cash:.4f}  平均乗数/移動: {r['avg_mul']:.6f}")
            with st.expander("ステップ詳細"):
                for sidx, s in enumerate(r['steps'], start=1):
                    st.write(f"{sidx}. {s['from']} → {s['to']} : 想定利益 {s['step_profit']:,} , 到着時資産 {s['cash_after']:,} , 乗数 {s['multiplier']:.6f}")
    st.success("完了")
