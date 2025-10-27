# main.py
import streamlit as st
from typing import Dict, List, Optional
import pandas as pd
import requests
from io import StringIO
from math import prod

st.set_page_config(page_title="航路買い物（単一品目近似・再計算版）", layout="wide")

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
# 単一品目仮定の簡易 greedy（利益率最大の単一品目を選ぶ）
# --------------------
def greedy_one_item_for_destination(current_port: str, dest_port: str, cash: int, price_matrix: Dict[str,Dict[str,int]]):
    """
    current_port で買い、dest_port で売る場合に利益率が最大の単一品目を選ぶ近似。
    戻り値: chosen_item or None, buy_price, sell_price, qty_bought, step_profit, cash_after
    """
    cash = int(max(1, cash))
    best_item = None
    best_rate = -1.0
    best_buy = 0
    best_sell = 0

    for item, _ in ITEMS:
        buy = price_matrix[item].get(current_port, 0)
        sell = price_matrix[item].get(dest_port, 0)
        if buy <= 0:
            continue
        unit_profit = sell - buy
        if unit_profit <= 0:
            continue
        rate = unit_profit / float(buy)
        if rate > best_rate:
            best_rate = rate
            best_item = item
            best_buy = buy
            best_sell = sell

    if best_item is None:
        return None, 0, 0, 0, 0, cash

    qty = cash // best_buy if best_buy > 0 else 0
    if qty <= 0:
        return best_item, best_buy, best_sell, 0, 0, cash

    step_profit = qty * (best_sell - best_buy)
    cash_after = cash + step_profit
    return best_item, best_buy, best_sell, int(qty), int(step_profit), int(cash_after)

# --------------------
# compute_single_step_multipliers_oneitem
# --------------------
def compute_single_step_multipliers_oneitem(price_matrix: Dict[str,Dict[str,int]], from_ports: List[str], to_ports: List[str], cash: int):
    """
    各 (p->q) を greedy_one_item_for_destination で評価し、乗数だけを返す。
    mapping[p][q] = {'multiplier','chosen_item','cash_after'}
    """
    mapping = {}
    candidates = []
    for p in from_ports:
        mapping.setdefault(p, {})
        for q in to_ports:
            if p == q:
                continue
            item, buy, sell, qty, profit, cash_after = greedy_one_item_for_destination(p, q, cash, price_matrix)
            multiplier = float(cash_after) / float(max(1, cash))
            mapping[p][q] = {
                'multiplier': multiplier,
                'chosen_item': item,
                'cash_after': cash_after,
            }
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
    steps = []

    while True:
        next_candidates_all = mapping.get(cur, {})
        if allowed_ports is not None:
            next_candidates = {q: info for q, info in next_candidates_all.items() if q in allowed_ports or q == start_port}
        else:
            next_candidates = next_candidates_all

        if not next_candidates:
            break

        q, info = max(next_candidates.items(), key=lambda kv: kv[1]['multiplier'])

        steps.append({
            'from': cur,
            'to': q,
            'multiplier': info.get('multiplier'),
            'chosen_item': info.get('chosen_item'),
        })
        cur = q

        if q in visited:
            idx = visited.index(q)
            route_segment = visited[idx:] + [q]
            route = [start_port] + route_segment
            cycle_steps = steps[-len(route_segment):] if route_segment else []
            multipliers = [s['multiplier'] for s in cycle_steps]
            total_mul = prod(multipliers) if multipliers else 1.0
            avg_mul = total_mul ** (1.0 / len(multipliers)) if multipliers else 1.0
            final_cash = None
            return route, cycle_steps, final_cash, avg_mul, total_mul

        visited.append(q)

        if len(visited) > max(1, len(mapping)):
            break

    return None, None, None, None, None

# --------------------
# generate_routes_greedy_cover_with_recalc（確定後に remaining_ports で再計算）
# --------------------
def generate_routes_greedy_cover_with_recalc(ports: List[str], price_matrix: Dict, cash: int, top_k_start: int = 3):
    """
    - top_k_start 個の開始候補それぞれについて試行
    - 各試行は remaining_ports を管理し、確定ルートの港を削除して次を探す
    - 各ループで remaining_ports に基づいて mapping を再計算する
    """
    results_per_start = []

    mapping_full, singles = compute_single_step_multipliers_oneitem(price_matrix, ports, ports, cash)

    singles_sorted = sorted(singles, key=lambda x: x[2], reverse=True)
    start_ports_order = []
    for p, q, m in singles_sorted:
        if p not in start_ports_order:
            start_ports_order.append(p)

    for initial_start_choice in start_ports_order[:top_k_start]:
        remaining_ports = set(ports)
        if initial_start_choice in remaining_ports:
            remaining_ports.remove(initial_start_choice)

        routes = []
        current_start = initial_start_choice

        while True:
            allowed_for_calc = set(remaining_ports) | {current_start}
            mapping, _ = compute_single_step_multipliers_oneitem(price_matrix, list(allowed_for_calc), list(allowed_for_calc), cash)

            route, steps, final_cash, avg_mul, total_mul = build_greedy_cycles_from_start(current_start, mapping, cash, allowed_ports=allowed_for_calc)
            if route is None:
                break

            covered = set(route)
            routes.append({'route': route, 'steps': steps, 'avg_mul': avg_mul, 'total_mul': total_mul, 'covered': covered})

            remaining_ports -= covered

            if not remaining_ports:
                break

            mapping_remain, singles_remain = compute_single_step_multipliers_oneitem(price_matrix, list(remaining_ports), list(remaining_ports), cash)
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
# UI
# --------------------
st.title("貪欲閉路群探索（単一品目近似・再計算版）")

try:
    ports, price_matrix = fetch_price_matrix_from_csv_auto(CSV_URL)
except Exception as e:
    st.error(f"CSV読み込み失敗: {e}")
    st.stop()

# 固定の解析用所持金（入力欄は表示しない、内部でこの値を使う）
CASH_DEFAULT = 50000

# show single-step bests on load (using fixed cash)
mapping_preview, candidates_preview = compute_single_step_multipliers_oneitem(price_matrix, ports, ports, CASH_DEFAULT)

st.subheader("各港から一手で最適な行き先（在庫無限・単一品目仮定）")
rows = []
for p in ports:
    outs = mapping_preview.get(p, {})
    if not outs:
        rows.append({"出発港": p, "最適到着": "-", "乗数": "-", "買う物": "-"})
        continue
    best_q, info = max(outs.items(), key=lambda kv: kv[1]['multiplier'])
    items_summary = f"{info.get('chosen_item') or '-'}"
    multiplier = info.get('multiplier', 1.0)
    rows.append({"出発港": p, "最適到着": best_q, "乗数": f"{multiplier:.2f}", "買う物": items_summary})

df_preview = pd.DataFrame(rows)
st.dataframe(df_preview, height=320)

st.markdown("上は所持金を内部固定（50,000）にして算出した各港の単一遷移最適先です。表示は乗数を小数点2桁で丸めています。")

col1, col2 = st.columns([1,1])
with col1:
    top_k_start = st.number_input("開始候補として試す上位k", min_value=1, max_value=10, value=1)
with col2:
    st.write(f"検出対象港数: {len(ports)}")
    if st.checkbox("価格表を表示"):
        rows_price = []
        for name, _ in ITEMS:
            row = {"品目": name}
            for p in ports:
                row[p] = price_matrix[name].get(p, 0)
            rows_price.append(row)
        df_all = pd.DataFrame(rows_price).set_index("品目")
        st.dataframe(df_all, height=480)

if st.button("閉路群を生成（貪欲・再計算）"):
    with st.spinner("解析中..."):
        results = generate_routes_greedy_cover_with_recalc(ports, price_matrix, CASH_DEFAULT, top_k_start=int(top_k_start))
    for attempt in results:
        st.markdown(f"## 初期スタート候補: {attempt['initial_start']}")
        if not attempt['routes']:
            st.write("ルートが生成できませんでした。")
            continue
        for i, r in enumerate(attempt['routes'], start=1):
            st.markdown(f"### ルート {i}: {' → '.join(r['route'])}")
            total_mul = r.get('total_mul', r.get('total_mul', r['avg_mul']))
            st.markdown(f"- カバー港数: {len(r['covered'])}  総乗数: {total_mul:.2f}  平均乗数/移動: {r['avg_mul']:.2f}")
            with st.expander("ステップ詳細"):
                for sidx, s in enumerate(r['steps'], start=1):
                    bought = f"{s.get('chosen_item')}" if s.get('chosen_item') else "-"
                    st.write(f"{sidx}. {s['from']} → {s['to']} : 購入 {bought} , 乗数 {s['multiplier']:.2f}")
    st.success("完了")
