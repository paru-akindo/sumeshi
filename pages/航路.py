# main.py
import streamlit as st
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import re
import requests
from io import StringIO

st.set_page_config(page_title="効率よく買い物しよう！", layout="wide")

# --------------------
# 定数: 品目・基礎値・港はソース（スプレッドシート）と一致させること
# --------------------
# ITEMS の順序・名称はスプレッドシートの列ヘッダーと揃えてください（左端は港名列）
ITEMS = [
    ("鳳梨",100),("魚肉",100),("酒",100),("水稲",100),("木材",100),("ヤシ",100),
    ("海鮮",200),("絹糸",200),("水晶",200),("茶葉",200),("鉄鉱",200),
    ("香料",300),("玉器",300),("白銀",300),("皮革",300),
    ("真珠",500),("燕の巣",500),("陶器",500),
    ("象牙",1000),("鹿茸",1000)
]

# --------------------
# Google スプレッドシート（公開CSV）設定
# 使い方:
#   {SPREADSHEET_ID} は URL の /d/.../ の部分
#   {GID} は該当シートの gid パラメータ（通常最初のシートは 0）
# CSV_URL の形式:
#   https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}
# --------------------
SPREADSHEET_ID = "1ft5FlwM5kaZK7B4vLQg2m1WYe5nWNb0udw5isFwDWy0"
GID = "805544474"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}"

# --------------------
# ヘルパー: 厳格整数入力（空欄許容）
# --------------------
def numeric_input_optional_strict(label: str, key: str, placeholder: str = "", allow_commas: bool = True, min_value: int = None, max_value: int = None):
    invalid_flag = f"{key}_invalid"
    if invalid_flag not in st.session_state:
        st.session_state[invalid_flag] = False

    raw = st.text_input(label, value="", placeholder=placeholder, key=key)

    s = (raw or "").strip()
    if s == "":
        st.session_state[invalid_flag] = False
        return None

    if allow_commas:
        s = s.replace(",", "")
    s = s.translate(str.maketrans("０１２３４５６７８９－＋．，", "0123456789-+.,"))
    if not re.fullmatch(r"\d+", s):
        st.error(f"「{label}」は整数の半角数字のみで入力してください。入力値: {raw}")
        st.session_state[invalid_flag] = True
        return None

    try:
        val = int(s)
    except Exception:
        st.error(f"「{label}」の数値変換に失敗しました。入力: {raw}")
        st.session_state[invalid_flag] = True
        return None

    if min_value is not None and val < min_value:
        st.error(f"「{label}」は {min_value} 以上で入力してください。")
        st.session_state[invalid_flag] = True
        return None
    if max_value is not None and val > max_value:
        st.error(f"「{label}」は {max_value} 以下で入力してください。")
        st.session_state[invalid_flag] = True
        return None

    st.session_state[invalid_flag] = False
    return val

# --------------------
# 既存の最適化ロジック（そのまま）
# --------------------
def greedy_plan_for_destination(current_port: str, dest_port: str, cash: int, stock: Dict[str,int], price_matrix: Dict[str,Dict[str,int]]):
    candidates = []
    for item, base in ITEMS:
        avail = stock.get(item, 0)
        if avail <= 0:
            continue

        buy = price_matrix.get(item, {}).get(current_port, 0)
        sell = price_matrix.get(item, {}).get(dest_port, 0)
        unit_profit = sell - buy
        if unit_profit <= 0:
            continue

        candidates.append((item, buy, sell, unit_profit, avail))

    # ★ 総利益 = unit_profit × min(avail, cash // buy)
    def score(c):
        item, buy, sell, unit_profit, avail = c
        if buy <= 0:
            return -10**18
        max_by_cash = cash // buy
        qty = min(avail, max_by_cash)
        return unit_profit * qty

    candidates.sort(key=score, reverse=True)

    remaining_cash = cash
    plan = []
    for item, buy, sell, unit_profit, avail in candidates:
        max_by_cash = remaining_cash // buy if buy > 0 else 0
        qty = min(avail, max_by_cash)
        if qty <= 0:
            continue
        plan.append((item, qty, buy, sell, unit_profit))
        remaining_cash -= qty * buy
        if remaining_cash <= 0:
            break

    total_cost = sum(q * b for _, q, b, _, _ in plan)
    total_profit = sum(q * up for _, q, _, _, up in plan)
    return plan, total_cost, total_profit

# --------------------
# ★ 完全修正版 CSV 転置ロジック
# --------------------
@st.cache_data(ttl=60)
def fetch_price_matrix_from_csv_auto(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    s = r.content.decode("utf-8")
    df = pd.read_csv(StringIO(s))

    # ★ Unnamed 列を自動除去（ここが重要）
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    
    # 行＝品目、列＝港 → 必ず転置する
    item_col = df.columns[0]
    df_items = df.set_index(item_col)
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

# --------------------
# UI
# --------------------
st.title("効率よく買い物しよう！")

# 先頭に追加
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={GID}"

# 画面上部（タイトルのすぐ下など）に表示
st.markdown(f'<div style="margin-top:6px;"><a href="{SPREADSHEET_URL}" target="_blank" rel="noopener noreferrer">スプレッドシートを開く（編集・表示）</a></div>', unsafe_allow_html=True)

# 価格取得
try:
    ports, price_matrix = fetch_price_matrix_from_csv_auto(CSV_URL)
except Exception as e:
    st.error(f"スプレッドシート（CSV）からの読み込みに失敗しました: {e}")
    st.stop()

# 中央レイアウト
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    current_port = st.selectbox("現在港", ports, index=0)
    cash = numeric_input_optional_strict("所持金", key="cash_input", placeholder="例: 5000", allow_commas=True, min_value=0)

    # お買い得上位Nの選定: 現在港の価格 / 基礎値 が小さい順に上位を取る
    item_scores = []
    for name, base in ITEMS:
        buy = price_matrix.get(name, {}).get(current_port, 0)
        # if buy is zero or missing, treat as very large ratio to de-prioritize
        try:
            ratio = buy / float(base) if base != 0 and buy > 0 else float("inf")
        except Exception:
            ratio = float("inf")
        item_scores.append((name, buy, base, ratio))
    item_scores.sort(key=lambda t: (t[3], t[1]))

    # UI: 在庫入力対象の上位表示数を可変にする（デフォルト5）
    max_n = len(ITEMS)
    top_n = st.slider(
        "在庫入力対象 上位何品目を表示するか",
        min_value=1,
        max_value=max_n,
        value=5,
        step=1,
        key="top_n_items"
    )

    # 可変化した上位リスト（変数名を top_items に変更）
    top_items = item_scores[:top_n]

    st.write("在庫入力対象（お買い得上位）")
    stock_inputs = {}
    # 行ごと2カラムで表示（スマホでも順序崩れない）
    for row_start in range(0, len(top_items), 2):
        c_left, c_right = st.columns(2)
        name, buy, base, ratio = top_items[row_start]
        pct = int(round((buy - base) / base * 100)) if base != 0 and buy > 0 else 0
        label = f"{name}（価格: {buy}, 補正: {pct:+d}%）"
        with c_left:
            stock_inputs[name] = numeric_input_optional_strict(label, key=f"stk_{name}", placeholder="在庫数", allow_commas=True, min_value=0)
        if row_start + 1 < len(top_items):
            name2, buy2, base2, ratio2 = top_items[row_start+1]
            pct2 = int(round((buy2 - base2) / base2 * 100)) if base2 != 0 and buy2 > 0 else 0
            label2 = f"{name2}（価格: {buy2}, 補正: {pct2:+d}%）"
            with c_right:
                stock_inputs[name2] = numeric_input_optional_strict(label2, key=f"stk_{name2}", placeholder="在庫数", allow_commas=True, min_value=0)

    top_k = st.slider("上位何港を出すか（上位k）", min_value=1, max_value=10, value=3)

    if st.button("検索"):
        if cash is None:
            st.error("所持金を入力してください（空欄不可）。")
        else:
            # 在庫の不正入力フラグチェック
            invalid_found = False
            for name in stock_inputs.keys():
                if st.session_state.get(f"stk_{name}_invalid", False):
                    st.error(f"{name} の入力が不正です。半角整数で入力してください。")
                    invalid_found = True
            if invalid_found:
                st.error("不正入力があるため中止します。")
            else:
                # current_stock 初期化（全品目 0）、トップNの入力値だけ反映
                current_stock = {n: 0 for n, _ in ITEMS}
                for name in stock_inputs:
                    val = stock_inputs.get(name)
                    current_stock[name] = int(val) if val is not None else 0

                results = []
                for dest in ports:
                    if dest == current_port:
                        continue
                    plan, cost, profit = greedy_plan_for_destination(current_port, dest, cash, current_stock, price_matrix)
                    results.append((dest, plan, cost, profit))
                results.sort(key=lambda x: x[3], reverse=True)
                top_results = results[:top_k]

                if not top_results or all(r[3] <= 0 for r in top_results):
                    st.info("所持金・在庫の範囲で利益が見込める到着先が見つかりませんでした。")
                else:
                    for rank, (dest, plan, cost, profit) in enumerate(top_results, start=1):
                        # そのまま貼れる置換コード（シンプル）
                        st.markdown(
                            f'''
                            <style>
                              .dest-row {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; padding:6px 8px; border-radius:6px; }}
                              /* ライトモード用 */
                              @media (prefers-color-scheme: light) {{
                                .dest-row {{ background: rgba(255,255,255,0.0); }}
                                .dest-row .label {{ color:#444; }}
                                .dest-row .value {{ color:#111; }}
                                .dest-row .sep {{ color:#ccc; }}
                              }}
                              /* ダークモード用 */
                              @media (prefers-color-scheme: dark) {{
                                .dest-row {{ background: rgba(0,0,0,0.0); }}
                                .dest-row .label {{ color:#cfcfcf; }}
                                .dest-row .value {{ color:#ffffff; }}
                                .dest-row .sep {{ color:#666; }}
                              }}
                              /* 強制対策: ブラウザ側で色を上書きされる場合に備えた!important指定 */
                              .dest-row .label, .dest-row .value, .dest-row .sep {{ -webkit-text-fill-color: initial !important; }}
                            </style>

                            <div class="dest-row">
                              <span class="label" style="font-size:0.85em; margin-right:4px;">到着先</span>
                              <span class="value" style="font-size:1.15em; font-weight:700;">{dest}</span>
                              <span class="sep" style="margin:0 8px;">|</span>
                              <span class="label" style="font-size:0.85em; margin-right:4px;">想定利益</span>
                              <span class="value" style="font-size:1.15em; font-weight:700;">{profit:,}</span>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )

                        if not plan:
                            st.write("購入候補がありません（利益が出ない、もしくは在庫不足）。")
                            continue

                        # 購入候補だけを使って DataFrame を作成（余計な列や NaN を生まない）
                        df_rows = []
                        for item, qty, buy, sell, unit_profit in plan:
                            df_rows.append({
                                "品目": item,
                                "購入数": int(qty),
                                "想定利益": int(qty * unit_profit)
                            })
                        df_out = pd.DataFrame(df_rows)

                        # 合計行（数値列は合計、その他はラベル）
                        totals = {
                            "品目": "合計",
                            "購入数": int(df_out["購入数"].sum()) if not df_out.empty else 0,
                            "想定利益": int(df_out["想定利益"].sum()) if not df_out.empty else 0
                        }
                        df_disp = pd.concat([df_out, pd.DataFrame([totals])], ignore_index=True)

                        # フォーマットして表示（行数に応じた高さを指定）
                        try:
                            num_format = {
                                "購入数": "{:,.0f}",
                                "想定利益": "{:,.0f}"
                            }
                            styled = df_disp.style.format(num_format, na_rep="")
                            st.dataframe(styled, height=max(200, 40 * (len(df_disp) + 1)))
                        except Exception:
                            st.table(df_disp)

                        st.write("---")

with col3:
    if st.checkbox("価格表を表示"):
        # 表示用 DataFrame を作る
        rows = []
        for name, _ in ITEMS:
            row = {"品目": name}
            for p in ports:
                row[p] = price_matrix[name].get(p, 0)
            rows.append(row)
        df_all = pd.DataFrame(rows).set_index("品目")
        st.dataframe(df_all, height=600)
