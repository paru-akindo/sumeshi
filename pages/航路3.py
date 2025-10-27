# main.py
import streamlit as st
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import requests
from io import StringIO
from copy import deepcopy
from math import prod

st.set_page_config(page_title="航路買い物（貪欲閉路群 再計算版）", layout="wide")

# --------------------
# 設定: ITEMS はスプレッドシートの品目列ヘッダと一致させること
# --------------------
ITEMS = [
    ("鳳梨",100),("魚肉",100),("酒",100),("水稲",100),("木材",100),("ヤシ",100),
    ("海鮮",200),("絹糸",200),("水晶",200),("茶葉",200),("鉄鉱",200),
    ("香料",300),("玉器",300),("白銀",300),("皮革",300),
    ("真珠",500),("燕の巣",500),("陶器",500),
    ("象牙",1000),("鹿茸",1000)
]

# --------------------
# スプレッドシート設定（公開CSV）
# --------------------
SPREADSHEET_ID = "1ft5FlwM5kaZK7B4vLQg2m1WYe5nWNb0udw5isFwDWy0"
GID = "805544474"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}"
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={GID}"

st.markdown(
    f'<div style="margin-bottom:8px;"><a href="{SPREADSHEET_URL}" target="_blank" rel="noopener noreferrer">スプレッドシートを開く（編集・表示）</a> | <a href="{CSV_URL}" target="_blank" rel="noopener noreferrer">CSV を開く</a></div>',
    unsafe_allow_html=True
)

# --------------------
# CSV 取得と price_matrix 生成（行=港 or 列=港 を自動判定）
# --------------------
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

# --------------------
# greedy: current_port で買い dest_port で売る最適貪欲プラン
# stock == None -> 在庫無限（購入は cash 制約のみ）
# returns plan(list of tuples), total_cost, total_profit, remaining_cash_after_sell
# plan entries: (item, qty, buy, sell, unit_profit)
# --------------------
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

# --------------------
# subset版: from_ports -> to_ports のみ計算する効率的関数
# --------------------
def compute_single_step_multipliers_subset(price_matrix: Dict[str,Dict[str,int]], from_ports: List[str], to_ports: List[str], cash: int):
    """
    from_ports から to_ports への単一遷移のみ計算する最適化版。
    戻り値: mapping (dict), candidates (list sorted by multiplier)
    mapping: mapping[p][q] = {'multiplier','profit','plan','cash_after'}
    """
    mapping = {}
    candidates = []
    to_set = set(to_ports)
    for p in from_ports:
        mapping.setdefault(p, {})
        for q in to_ports:
            if p == q:
                continue
            plan, cost, profit, cash_after = greedy_plan_for_destination_general(p, q, cash, None, price_matrix)
            multiplier = float(cash_after) / float(max(1, cash))
            mapping[p][q] = {'multiplier': multiplier, 'profit': profit, 'plan': plan, 'cash_after': cash_after}
            candidates.append((p, q, multiplier))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return mapping, candidates

# --------------------
# build_greedy_cycles_from_start（allowed_ports を追加）
# --------------------
def build_greedy_cycles_from_start(start_port: str, mapping: Dict, cash: int, allowed_ports: Optional[set] = None):
    """
    start_port から貪欲に最良遷移（最大乗数）を選んで辿り、閉路になったらそのルートを返す。
    allowed_ports が指定されていれば、その集合に含まれる港のみを遷移候補にする。
    戻り値: route (list), steps (list of dicts), final_cash, avg_multiplier_per_move, total_multiplier
    """
    visited = []
    cur = start_port
    cur_cash = cash
    steps = []

    while True:
        next_candidates_all = mapping.get(cur, {})
        # 制限集合があればここでフィルタ
        if allowed_ports is not None:
            next_candidates = {q:info for q,info in next_candidates_all.items() if q in allowed_ports or q == start_port}
        else:
            next_candidates = next_candidates_all

        if not next_candidates:
            break

        # pick next q with highest multiplier among allowed
        q, info = max(next_candidates.items(), key=lambda kv: kv[1]['multiplier'])

        # record step
        steps.append({'from': cur, 'to': q, 'plan': info['plan'], 'step_profit': info['profit'], 'cash_after': info['cash_after'], 'multiplier': info['multiplier']})
        cur = q

        # detect cycle: if q has been seen in visited -> close cycle from its first occurrence
        if q in visited:
            idx = visited.index(q)
            route_segment = visited[idx:] + [q]
            route = [start_port] + route_segment
            cycle_steps = steps[-len(route_segment):] if route_segment else []
            multipliers = [s['multiplier'] for s in cycle_steps]
            total_mul = prod(multipliers) if multipliers else 1.0
            avg_mul = total_mul ** (1.0 / len(multipliers)) if multipliers else 1.0
            final_cash = int(cycle_steps[-1]['cash_after']) if cycle_steps else cash
            return route, cycle_steps, final_cash, avg_mul, total_mul

        visited.append(q)

        # safety: break if route too long
        if len(visited) > max(1, len(mapping)):
            break

    return None, None, None, None, None

# --------------------
# generate_routes_greedy_cover（確定後に remaining_ports で再計算）
# --------------------
def generate_routes_greedy_cover_with_recalc(ports: List[str], price_matrix: Dict, cash: int, top_k_start: int = 3):
    """
    - top_k_start 個の開始候補それぞれについて試行
    - 各試行は remaining_ports を管理し、確定ルートの港を削除して次を探す
    - 各ループで remaining_ports に基づいて mapping を再計算する（正確な最適化のため）
    """
    results_per_start = []

    # 初回 mapping (全港 -> 全港)
    mapping_full, singles = compute_single_step_multipliers_subset(price_matrix, ports, ports, cash)

    # flatten singles to get ordering of starting ports by best outgoing
    singles_sorted = sorted(singles, key=lambda x: x[2], reverse=True)
    start_ports_order = []
    for p, q, m in singles_sorted:
        if p not in start_ports_order:
            start_ports_order.append(p)

    for initial_start_choice in start_ports_order[:top_k_start]:
        remaining_ports = set(ports)
        # we'll treat the chosen start as available, but remove it from remaining_ports choice list so we don't pick it again arbitrarily
        if initial_start_choice in remaining_ports:
            remaining_ports.remove(initial_start_choice)

        routes = []
        current_start = initial_start_choice

        # mapping to use in this iteration: recompute from remaining_ports ∪ {current_start} as both from/to
        while True:
            # allowed set for transitions: remaining_ports plus current_start (we allow returning to start)
            allowed_for_calc = set(remaining_ports) | {current_start}
            # build mapping restricted to allowed ports (from_ports = allowed, to_ports = allowed)
            mapping, _ = compute_single_step_multipliers_subset(price_matrix, list(allowed_for_calc), list(allowed_for_calc), cash)

            # attempt to build a cycle starting from current_start with allowed set
            route, steps, final_cash, avg_mul, total_mul = build_greedy_cycles_from_start(current_start, mapping, cash, allowed_ports=allowed_for_calc)
            if route is None:
                break

            covered = set(route)
            routes.append({'route': route, 'steps': steps, 'final_cash': final_cash, 'avg_mul': avg_mul, 'total_mul': total_mul, 'covered': covered})

            # remove covered ports from remaining_ports
            remaining_ports -= covered

            if not remaining_ports:
                break

            # choose next start from remaining_ports based on current mapping (we recompute mapping for remaining_ports)
            mapping_remain, singles_remain = compute_single_step_multipliers_subset(price_matrix, list(remaining_ports), list(remaining_ports), cash)
            # pick port with best outgoing multiplier among remaining_ports
            next_start = None
            best_m = -1.0
            for p in mapping_remain:
                outs = mapping_remain.get(p, {})
                if not outs:
                    continue
                q, info = max(outs.items(), key=lambda kv: kv[1]['multiplier'])
                if info['multiplier'] > best_m:
                    best_m = info['multiplier']
                    next_start = p

            if next_start is None:
                break

            current_start = next_start
            if current_start in remaining_ports:
                remaining_ports.remove(current_start)

        results_per_start.append({'initial_start': initial_start_choice, 'routes': routes, 'remaining_ports': remaining_ports})

    return results_per_start

# --------------------
# UI: 単一遷移ベスト表を先に表示、解析はユーザがボタンで実行
# --------------------
st.title("貪欲閉路群探索（再計算版）")

try:
    ports, price_matrix = fetch_price_matrix_from_csv_auto(CSV_URL)
except Exception as e:
    st.error(f"CSV読み込み失敗: {e}")
    st.stop()

# show single-step bests on load (using default cash 50000)
cash_default = 50000
mapping_preview, candidates_preview = compute_single_step_multipliers_subset(price_matrix, ports, ports, cash_default)

st.subheader("各港から一手で最適な行き先（在庫無限評価）")
rows = []
for p in ports:
    outs = mapping_preview.get(p, {})
    if not outs:
        rows.append({"出発港": p, "最適到着": "-", "乗数": "-", "想定利益": 0, "買う物(一部)": ""})
        continue
    best_q, info = max(outs.items(), key=lambda kv: kv[1]['multiplier'])
    plan = info.get('plan', [])
    items_summary = "; ".join([f"{item}×{qty}" for item, qty, buy, sell, up in plan][:4])
    rows.append({"出発港": p, "最適到着": best_q, "乗数": f"{info['multiplier']:.6f}", "想定利益": int(info['profit']), "買う物(一部)": items_summary})

df_preview = pd.DataFrame(rows)
st.dataframe(df_preview, height=320)

st.markdown("上は仮所持金 50,000 を基準にした各港の単一遷移最適先と想定利益です。必要なら解析用所持金を変更して再計算してください。")

# Controls for analysis
col1, col2 = st.columns([1,1])
with col1:
    cash = st.number_input("解析用の仮所持金", min_value=1, value=50000, step=1000)
    top_k_start = st.number_input("開始候補として試す上位k", min_value=1, max_value=10, value=1)
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

if st.button("閉路群を生成（貪欲・再計算）"):
    with st.spinner("解析中..."):
        results = generate_routes_greedy_cover_with_recalc(ports, price_matrix, cash, top_k_start=int(top_k_start))
    # display
    for attempt in results:
        st.markdown(f"## 初期スタート候補: {attempt['initial_start']}")
        if not attempt['routes']:
            st.write("ルートが生成できませんでした。")
            continue
        for i, r in enumerate(attempt['routes'], start=1):
            st.markdown(f"### ルート {i}: {' → '.join(r['route'])}")
            total_mul = r.get('total_mul', r['final_cash'] / float(max(1, cash)))
            st.markdown(f"- カバー港数: {len(r['covered'])}  最終資産乗数: {total_mul:.6f}  平均乗数/移動: {r['avg_mul']:.6f}")
            with st.expander("ステップ詳細"):
                for sidx, s in enumerate(r['steps'], start=1):
                    st.write(f"{sidx}. {s['from']} → {s['to']} : 想定利益 {s['step_profit']:,} , 到着時資産 {s['cash_after']:,} , 乗数 {s['multiplier']:.6f}")
    st.success("完了")
