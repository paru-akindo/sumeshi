# main.py
import streamlit as st
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import re
import requests
from io import StringIO
from copy import deepcopy

st.set_page_config(page_title="航路買い物", layout="wide")

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
# SPREADSHEET_ID と GID を自分のものに置き換えてください
# SPREADSHEET_URL は編集用のURL（リンク表示用）
# --------------------
SPREADSHEET_ID = "1ft5FlwM5kaZK7B4vLQg2m1WYe5nWNb0udw5isFwDWy0"
GID = "805544474"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}"
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit#gid={GID}"

# --------------------
# ヘルパー: 厳格整数テキスト入力（空欄許容）
# --------------------
def numeric_input_optional_strict(label: str, key: str, placeholder: str = "", allow_commas: bool = True, min_value: Optional[int] = None, max_value: Optional[int] = None):
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
# CSV 取得: 行=港 or 行=品目 を自動判定して price_matrix を作る
# price_matrix: { item_name: { port_name: price_int, ... }, ... }
# returns ports(list), price_matrix(dict)
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
        # 行=品目, 列=港 の形式 -> 転置して行=港に合わせる
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
        # 行=港, 列=品目 の形式
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
# 拡張 greedy: stock==None -> 在庫無限(購入は cash 制約のみ)
# returns plan, total_cost, total_profit, remaining_cash_after_sell
# plan: list of (item, qty, buy, sell, unit_profit)
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
    remaining_cash_after_sell = remaining_cash + total_profit
    return plan, int(total_cost), int(total_profit), int(remaining_cash_after_sell)

# --------------------
# lookahead 評価関数
# --------------------
def evaluate_with_lookahead(current_port: str, dest_port: str, cash: int, stock: Dict[str,int], price_matrix: Dict[str,Dict[str,int]], second_k: Optional[int] = None):
    first_plan, first_cost, first_profit, cash_after_sell = greedy_plan_for_destination_general(current_port, dest_port, cash, stock, price_matrix)

    # 二手目候補リスト（dest_port を除く）
    example_item = next(iter(price_matrix))
    ports_list = list(price_matrix[example_item].keys())
    second_candidates = [p for p in ports_list if p != dest_port]

    # 候補絞り（heuristic）: second_k が指定されたら単純スコアで上位Kに絞る
    if second_k is not None and second_k < len(second_candidates):
        scores = []
        for s in second_candidates:
            score = 0
            for item, _ in ITEMS:
                buy = price_matrix[item].get(dest_port, 0)
                sell = price_matrix[item].get(s, 0)
                if buy > 0:
                    unit = sell - buy
                    if unit > 0:
                        score += unit
            scores.append((s, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        second_candidates = [s for s,_ in scores[:second_k]]

    best_second_profit = 0
    best_second_plan = []
    best_second_dest = None
    for s in second_candidates:
        plan2, cost2, profit2, cash_after2 = greedy_plan_for_destination_general(dest_port, s, cash_after_sell, None, price_matrix)
        if profit2 > best_second_profit:
            best_second_profit = profit2
            best_second_plan = plan2
            best_second_dest = s

    total_profit = first_profit + best_second_profit
    return {
        "total_profit": int(total_profit),
        "first_profit": int(first_profit),
        "second_best_profit": int(best_second_profit),
        "first_plan": first_plan,
        "second_plan": best_second_plan,
        "second_dest": best_second_dest,
        "cash_after_first_sell": int(cash_after_sell)
    }

# --------------------
# UI
# --------------------
st.title("何買おうかな")
st.markdown(f'<div style="margin-bottom:8px;"><a href="{SPREADSHEET_URL}" target="_blank" rel="noopener noreferrer">スプレッドシートを開く（編集・表示）</a> | <a href="{CSV_URL}" target="_blank" rel="noopener noreferrer">CSV を開く</a></div>', unsafe_allow_html=True)

# 価格取得
try:
    ports, price_matrix = fetch_price_matrix_from_csv_auto(CSV_URL)
except Exception as e:
    st.error(f"スプレッドシート（CSV）からの読み込みに失敗しました: {e}")
    st.stop()

# 簡易表示オプション
if st.checkbox("取得した港一覧を表示"):
    st.write(ports)

# レイアウト
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    current_port = st.selectbox("現在港", ports, index=0)
    cash = numeric_input_optional_strict("所持金", key="cash_input", placeholder="例: 5000", allow_commas=True, min_value=0)

    # お買い得上位5（現在港基準）
    item_scores = []
    for name, base in ITEMS:
        buy = price_matrix.get(name, {}).get(current_port, 0)
        try:
            ratio = buy / float(base) if base != 0 and buy > 0 else float("inf")
        except Exception:
            ratio = float("inf")
        item_scores.append((name, buy, base, ratio))
    item_scores.sort(key=lambda t: (t[3], t[1]))
    top5 = item_scores[:5]

    st.write("在庫入力（必須：Top5品目）: 現在港の価格/基礎値 小さい順に表示")
    stock_inputs = {}
    for row_start in range(0, len(top5), 2):
        c_left, c_right = st.columns(2)
        name, buy, base, ratio = top5[row_start]
        pct = int(round((buy - base) / base * 100)) if base != 0 and buy>0 else 0
        label = f"{name}（価格: {buy}, 基礎値: {base}, 補正: {pct:+d}%） 在庫数（必須）"
        with c_left:
            stock_inputs[name] = numeric_input_optional_strict(label, key=f"stk_{name}", placeholder="例: 10", allow_commas=True, min_value=0)
        if row_start + 1 < len(top5):
            name2, buy2, base2, ratio2 = top5[row_start+1]
            pct2 = int(round((buy2 - base2) / base2 * 100)) if base2 != 0 and buy2>0 else 0
            label2 = f"{name2}（価格: {buy2}, 基礎値: {base2}, 補正: {pct2:+d}%） 在庫数（必須）"
            with c_right:
                stock_inputs[name2] = numeric_input_optional_strict(label2, key=f"stk_{name2}", placeholder="例: 10", allow_commas=True, min_value=0)

    top_k = st.slider("表示上位何港を出すか（上位k）", min_value=1, max_value=min(10, len(ports)-1), value=3)
    lookahead_mode = st.checkbox("1手先モード（到着先の先を想定）", value=False)
    second_k = None
    if lookahead_mode:
        second_k = st.number_input("二手目候補を上位何港だけ評価するか（1〜 全探索）", min_value=1, max_value=max(1, len(ports)-1), value=min(5, max(1, len(ports)-1)))

    if st.button("検索"):
        # 入力チェック
        if cash is None:
            st.error("所持金を入力してください（空欄不可）。")
        else:
            invalid_found = False
            for name in stock_inputs.keys():
                if st.session_state.get(f"stk_{name}_invalid", False):
                    st.error(f"{name} の入力が不正です。半角整数で入力してください。")
                    invalid_found = True
                if stock_inputs.get(name) is None:
                    st.error(f"{name} の在庫を必ず入力してください（空欄不可）。")
                    invalid_found = True
            if invalid_found:
                st.error("不正入力があるため中止します。")
            else:
                # current_stock を全品目分用意（top5のみ入力値反映）
                current_stock = {n: 0 for n, _ in ITEMS}
                for name in stock_inputs:
                    val = stock_inputs.get(name)
                    current_stock[name] = int(val) if val is not None else 0

                results = []
                with st.spinner("候補評価中..."):
                    for dest in ports:
                        if dest == current_port:
                            continue
                        if lookahead_mode:
                            # 一手目は入力在庫を使う（仕様どおり）
                            eval_res = evaluate_with_lookahead(current_port, dest, cash, current_stock, price_matrix, second_k=second_k)
                            results.append((dest, eval_res["total_profit"], eval_res))
                        else:
                            plan, cost, profit, _ = greedy_plan_for_destination_general(current_port, dest, cash, current_stock, price_matrix)
                            results.append((dest, profit, {"first_plan": plan, "first_profit": profit}))

                results.sort(key=lambda x: x[1], reverse=True)
                top_results = results[:top_k]

                if not top_results or all(r[1] <= 0 for r in top_results):
                    st.info("所持金・在庫の範囲で利益が見込める到着先が見つかりませんでした。")
                else:
                    for rank, (dest, total_profit, meta) in enumerate(top_results, start=1):
                        if lookahead_mode:
                            first_profit = meta["first_profit"]
                            second_profit = meta["second_best_profit"]
                            second_dest = meta["second_dest"]
                            cash_after_first = meta.get("cash_after_first_sell", None)
                            # 見出し：ラベルを小さめ、値を強調（ダークモード対応）
                            st.markdown(
                                f'''
                                <style>
                                  .dest-row {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; padding:6px 8px; border-radius:6px; }}
                                  @media (prefers-color-scheme: light) {{
                                    .dest-row .label {{ color:#444; }}
                                    .dest-row .value {{ color:#111; }}
                                    .dest-row .sep {{ color:#ccc; }}
                                  }}
                                  @media (prefers-color-scheme: dark) {{
                                    .dest-row .label {{ color:#cfcfcf; }}
                                    .dest-row .value {{ color:#fff; }}
                                    .dest-row .sep {{ color:#666; }}
                                  }}
                                  .dest-row .label, .dest-row .value {{ -webkit-text-fill-color: initial !important; }}
                                </style>
                                <div class="dest-row">
                                  <span class="label" style="font-size:0.85em; margin-right:4px;">到着先</span>
                                  <span class="value" style="font-size:1.15em; font-weight:700;">{dest}</span>
                                  <span class="sep" style="margin:0 8px;">|</span>
                                  <span class="label" style="font-size:0.85em; margin-right:4px;">合計想定利益</span>
                                  <span class="value" style="font-size:1.15em; font-weight:700; color:#0b6;">{total_profit:,}</span>
                                  <a href="{SPREADSHEET_URL}" target="_blank" rel="noopener noreferrer" style="margin-left:12px; font-size:0.85em; color:#06c; text-decoration:none;">スプレッドシートを開く</a>
                                </div>
                                ''',
                                unsafe_allow_html=True
                            )
                            st.markdown(f"- 一手目利益: {first_profit:,}  （到着後手元資産: {cash_after_first:,}）")
                            if second_dest:
                                st.markdown(f"- 二手目売却先候補: **{second_dest}**  想定利益: {second_profit:,}")
                            else:
                                st.markdown(f"- 二手目売却先候補なし（想定利益: 0）")

                            # 一手目プラン表
                            if meta.get("first_plan"):
                                df1 = pd.DataFrame([{"品目":i,"購入数":q,"購入単価":b,"売価":s,"単位差益":u,"想定利益":int(q*u)} for i,q,b,s,u in meta["first_plan"]])
                                totals1 = {"品目":"合計","購入数":int(df1["購入数"].sum()) if not df1.empty else 0,"購入単価":"","売価":"","単位差益":"","想定利益":int(df1["想定利益"].sum()) if not df1.empty else 0}
                                df1_disp = pd.concat([df1, pd.DataFrame([totals1])], ignore_index=True)
                                st.write("一手目プラン（現在港→到着先）")
                                try:
                                    st.dataframe(df1_disp.style.format({"購入数":"{:,.0f}","購入単価":"{:,.0f}","売価":"{:,.0f}","単位差益":"{:,.0f}","想定利益":"{:,.0f}"}, na_rep=""), height= max(140, 40*(len(df1_disp)+1)))
                                except Exception:
                                    st.table(df1_disp)

                            # 二手目プラン表
                            if meta.get("second_plan"):
                                df2 = pd.DataFrame([{"品目":i,"購入数":q,"購入単価":b,"売価":s,"単位差益":u,"想定利益":int(q*u)} for i,q,b,s,u in meta["second_plan"]])
                                totals2 = {"品目":"合計","購入数":int(df2["購入数"].sum()) if not df2.empty else 0,"購入単価":"","売価":"","単位差益":"","想定利益":int(df2["想定利益"].sum()) if not df2.empty else 0}
                                df2_disp = pd.concat([df2, pd.DataFrame([totals2])], ignore_index=True)
                                st.write(f"二手目プラン（到着先→売却先: {meta['second_dest']}）")
                                try:
                                    st.dataframe(df2_disp.style.format({"購入数":"{:,.0f}","購入単価":"{:,.0f}","売価":"{:,.0f}","単位差益":"{:,.0f}","想定利益":"{:,.0f}"}, na_rep=""), height= max(140, 40*(len(df2_disp)+1)))
                                except Exception:
                                    st.table(df2_disp)
                            st.write("---")
                        else:
                            plan = meta["first_plan"]
                            profit = meta["first_profit"]
                            # 見出し（ラベル小、値強調）
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
                            df_out = pd.DataFrame([{
                                "品目": item,
                                "購入数": qty,
                                "購入単価": buy,
                                "売価": sell,
                                "単位差益": unit_profit,
                                "想定利益": int(qty * unit_profit)
                            } for item, qty, buy, sell, unit_profit in plan])
                            totals = {"品目":"合計","購入数":int(df_out["購入数"].sum()),"購入単価":"","売価":"","単位差益":"","想定利益":int(df_out["想定利益"].sum())}
                            df_disp = pd.concat([df_out, pd.DataFrame([totals])], ignore_index=True)
                            try:
                                st.dataframe(df_disp.style.format({"購入数":"{:,.0f}","購入単価":"{:,.0f}","売価":"{:,.0f}","単位差益":"{:,.0f}","想定利益":"{:,.0f}"}, na_rep=""), height= max(140, 40*(len(df_disp)+1)))
                            except Exception:
                                st.table(df_disp)

with col3:
    if st.checkbox("価格表を表示"):
        rows = []
        for name, _ in ITEMS:
            row = {"品目": name}
            for p in ports:
                row[p] = price_matrix[name].get(p, 0)
            rows.append(row)
        df_all = pd.DataFrame(rows).set_index("品目")
        st.dataframe(df_all, height=600)
