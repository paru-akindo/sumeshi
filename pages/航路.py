# main.py
import streamlit as st
import requests
import json
import re
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

st.set_page_config(page_title="効率よく買い物しよう！", layout="wide")

# --------------------
# 定数
# --------------------
PORTS = ["博多","開京","明州","泉州","広州","淡水","安南","ボニ","タイ","真臘","スル","三仏斉","ジョホール","大光国","天竺","セイロン","ペルシャ","大食国","ミスル","末羅国"]

ITEMS = [
    ("鳳梨",100),("魚肉",100),("酒",100),("水稲",100),("木材",100),("ヤシ",100),
    ("海鮮",200),("絹糸",200),("水晶",200),("茶葉",200),("鉄鉱",200),
    ("香料",300),("玉器",300),("白銀",300),("皮革",300),
    ("真珠",500),("燕の巣",500),("陶器",500),
    ("象牙",1000),("鹿茸",1000)
]

# --------------------
# jsonbin 設定（必要なら置き換えてください）
# --------------------
JSONBIN_API_KEY = "$2a$10$wkVzPCcsW64wR96r26OsI.HDd3ijLveJn6sxJoSjfzByIRyODPCHq"
JSONBIN_BIN_ID = "68e8d05ed0ea881f409c39c4"
JSONBIN_BASE = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}"
JSONBIN_HEADERS = {"Content-Type": "application/json", "X-Master-Key": JSONBIN_API_KEY}

# --------------------
# 全件リセットのパスワード（ソース直書き）
# --------------------
RESET_PASSWORD = "alan"  # <-- 必要に応じて書き換えてください

# --------------------
# ヘルパー関数
# --------------------
def safe_rerun():
    try:
        getattr(st, "experimental_rerun")()
    except Exception:
        return

def fetch_cfg_from_jsonbin():
    try:
        r = requests.get(f"{JSONBIN_BASE}/latest", headers=JSONBIN_HEADERS, timeout=8)
        r.raise_for_status()
        payload = r.json()
        return payload.get("record", payload)
    except Exception:
        return None

def save_cfg_to_jsonbin(cfg: dict):
    r = requests.put(JSONBIN_BASE, headers=JSONBIN_HEADERS, data=json.dumps(cfg, ensure_ascii=False))
    r.raise_for_status()
    return r

def normalize_items(raw_items) -> List[Tuple[str,int]]:
    out = []
    for it in raw_items:
        if isinstance(it, (list, tuple)):
            out.append((it[0], int(it[1])))
        elif isinstance(it, dict):
            out.append((it["name"], int(it["base"])))
        else:
            raise ValueError("Unknown item format")
    return out

def port_has_actual_prices(port_prices: dict, items: List[Tuple[str,int]]) -> bool:
    for name, _ in items:
        if name not in port_prices or not isinstance(port_prices[name], (int, float)):
            return False
    for name, base in items:
        if int(round(port_prices.get(name))) != int(base):
            return True
    return False

def numeric_input_optional_strict(label: str, key: str, placeholder: str = "", allow_commas: bool = True, min_value: int = None):
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

    val = int(s)
    if min_value is not None and val < min_value:
        st.error(f"「{label}」は {min_value} 以上で入力してください。")
        st.session_state[invalid_flag] = True
        return None

    st.session_state[invalid_flag] = False
    return val

def build_price_matrix_from_prices(prices_cfg: Dict[str, Dict[str,int]], items=ITEMS, ports=PORTS):
    price = {name: {} for name, _ in items}
    for port in ports:
        port_row = prices_cfg.get(port, {})
        for name, _ in items:
            price[name][port] = int(port_row.get(name, 0))
    return price

def greedy_plan_for_destination(current_port: str, dest_port: str, cash: int, stock: Dict[str,int], price_matrix: Dict[str,Dict[str,int]]):
    candidates = []
    for item, base in ITEMS:
        avail = stock.get(item, 0)
        if avail <= 0:
            continue
        buy = price_matrix[item][current_port]
        sell = price_matrix[item][dest_port]
        unit_profit = sell - buy
        if unit_profit <= 0:
            continue
        candidates.append((item, buy, sell, unit_profit, avail))
    candidates.sort(key=lambda x: x[3], reverse=True)
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
# 追加ユーティリティ
# --------------------
def reset_port_to_base(port: str, items_cfg: List[Tuple[str,int]], prices_cfg: Dict[str, Dict[str,int]]):
    new_row = {}
    for name, base in items_cfg:
        new_row[name] = int(base)
    prices_cfg[port] = new_row
    return prices_cfg

def reset_all_ports_to_base(items_cfg: List[Tuple[str,int]], prices_cfg: Dict[str, Dict[str,int]], ports: List[str]):
    for port in ports:
        prices_cfg[port] = {name: int(base) for name, base in items_cfg}
    return prices_cfg

def get_populated_ports(prices_cfg: Dict[str, Dict[str,int]], items_cfg: List[Tuple[str,int]], ports: List[str]):
    populated = []
    for port in ports:
        row = prices_cfg.get(port, {})
        ok = True
        for name, _ in items_cfg:
            if name not in row or not isinstance(row[name], (int, float)):
                ok = False
                break
        if ok:
            populated.append(port)
    return populated

# --------------------
# アプリ本体
# --------------------
st.title("効率よく買い物しよう！")

cfg = fetch_cfg_from_jsonbin()
if cfg is None:
    st.warning("jsonbin から読み込みできませんでした。組み込み定義を使用します。")
    cfg = {"PORTS": PORTS, "ITEMS": [list(i) for i in ITEMS]}

PORTS_CFG = cfg.get("PORTS", PORTS)
ITEMS_CFG = normalize_items(cfg.get("ITEMS", [list(i) for i in ITEMS]))
PRICES_CFG = cfg.get("PRICES", {})

# 全港入力チェック（完全一致基準）
all_populated = True
missing_ports = []
for port in PORTS_CFG:
    port_prices = PRICES_CFG.get(port, {})
    if not port_has_actual_prices(port_prices, ITEMS_CFG):
        all_populated = False
        missing_ports.append(port)

show_price_table = st.checkbox("価格表を表示", value=False, key="chk_price_table")
show_correction_table = st.checkbox("割引率を表示", value=False, key="chk_corr_table")

if "mode" not in st.session_state:
    st.session_state["mode"] = "view"

# --------------------
# シミュレーションは未更新がないときだけ描画する
# --------------------
if all_populated:
    price_matrix = build_price_matrix_from_prices(PRICES_CFG, items=ITEMS_CFG, ports=PORTS_CFG)

    col_left, col_main = st.columns([1, 2])

    with col_left:
        st.header("シミュレーション")
        st.success("すべての港に実価格が入力されています。")

        current_port = st.selectbox("現在港", PORTS_CFG, index=0, key="sel_current_port")
        cash = numeric_input_optional_strict("所持金", key="cash_input", placeholder="例: 5000", allow_commas=True, min_value=0)

        if st.button("管理画面を開く", key="btn_open_admin"):
            st.session_state["mode"] = "admin"
            safe_rerun()

        # 在庫入力対象の選定（price/base 小さい順、同率は価格小さい順）
        item_scores = []
        for item_name, base in ITEMS_CFG:
            this_price = price_matrix[item_name][current_port]
            ratio = this_price / float(base) if base != 0 else float('inf')
            item_scores.append((item_name, ratio, this_price, base))
        item_scores.sort(key=lambda t: (t[1], t[2]))
        top5 = item_scores[:5]

        st.write("在庫入力（上位5）")
        stock_inputs = {}
        cols = st.columns(2)
        for i, (name, ratio, price_val, base_val) in enumerate(top5):
            pct = int(round((price_val - base_val) / float(base_val) * 100)) if base_val != 0 else 0
            sign_pct = f"{pct:+d}%"
            label = f"{name}({sign_pct}) 在庫数"
            help_text = f"現在価格: {price_val} / 基礎値: {base_val}"
            c = cols[i % 2]
            with c:
                stock_inputs[name] = numeric_input_optional_strict(label, key=f"stk_{name}_sim", placeholder="例: 10", allow_commas=True, min_value=0)
                st.caption(help_text)

        top_k = st.slider("表示上位何港を出すか（上位k）", min_value=1, max_value=10, value=3, key="slider_topk")

        if st.button("検索", key="btn_search_sim"):
            if cash is None:
                st.error("所持金を入力してください（空欄不可）。")
            else:
                invalid_found = False
                for name in stock_inputs.keys():
                    if st.session_state.get(f"stk_{name}_sim_invalid", False):
                        st.error(f"{name} の入力が不正です。")
                        invalid_found = True
                if invalid_found:
                    st.error("不正入力があるため中止します。")
                else:
                    current_stock = {n: 0 for n, _ in ITEMS_CFG}
                    for name in stock_inputs:
                        val = stock_inputs.get(name)
                        current_stock[name] = int(val) if val is not None else 0

                    results = []
                    for dest in PORTS_CFG:
                        if dest == current_port:
                            continue
                        plan, cost, profit = greedy_plan_for_destination(current_port, dest, cash, current_stock, price_matrix)
                        results.append((dest, plan, cost, profit))
                    results.sort(key=lambda x: x[3], reverse=True)
                    top_results = results[:top_k]

                    if not top_results or all(r[3] <= 0 for r in top_results):
                        st.info("利益の見込める到着先は見つかりませんでした。")
                    else:
                        for rank, (dest, plan, cost, profit) in enumerate(top_results, start=1):
                            st.markdown(f"### {rank}. 到着先: {dest}　想定合計利益: {profit}　合計購入金額: {cost}")
                            if not plan:
                                st.write("購入候補がありません。")
                                continue
                            df = pd.DataFrame([{
                                "品目": item,
                                "購入数": qty,
                                "購入単価": buy,
                                "売価": sell,
                                "単位差益": unit_profit,
                                "想定利益": qty * unit_profit
                            } for item, qty, buy, sell, unit_profit in plan])
                            totals = {"品目":"合計", "購入数": int(df["購入数"].sum()) if not df.empty else 0, "購入単価": np.nan, "売価": np.nan, "単位差益": np.nan, "想定利益": int(df["想定利益"].sum()) if not df.empty else 0}
                            df_disp = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
                            num_format = {"購入単価":"{:,.0f}", "売価":"{:,.0f}", "単位差益":"{:,.0f}", "購入数":"{:,.0f}", "想定利益":"{:,.0f}"}
                            styled = df_disp.style.format(num_format, na_rep="")
                            st.dataframe(styled, height=240)

    # 右側テーブル表示
    with col_main:
        st.header("テーブル表示")

        if show_price_table:
            rows = []
            for item, base in ITEMS_CFG:
                row = {"品目": item}
                for p in PORTS_CFG:
                    row[p] = price_matrix[item][p]
                rows.append(row)
            df_price = pd.DataFrame(rows).set_index("品目")

            def price_cell_css_simple(price_val, base):
                try:
                    price_int = int(price_val)
                except Exception:
                    return ""
                diff = price_int - base
                if diff < 0:
                    return "background-color: #ffc0cb; color: #000"
                elif diff > 0:
                    return "background-color: #c6f6b6; color: #000"
                else:
                    return ""

            def price_styler_simple(df):
                sty = pd.DataFrame("", index=df.index, columns=df.columns)
                base_map = dict(ITEMS_CFG)
                for item in df.index:
                    base = base_map[item]
                    for col in df.columns:
                        sty.at[item, col] = price_cell_css_simple(df.at[item, col], base)
                return sty

            st.subheader("実価格表")
            styled_price = df_price.style.apply(lambda _: price_styler_simple(df_price), axis=None)
            st.dataframe(styled_price, height=380)

        if show_correction_table:
            rows = []
            for item, base in ITEMS_CFG:
                row = {"品目": item}
                for p in PORTS_CFG:
                    price = price_matrix[item][p]
                    pct = int(round((price - base) / float(base) * 100)) if base != 0 else 0
                    row[p] = pct
                rows.append(row)
            df_corr = pd.DataFrame(rows).set_index("品目")

            def corr_cell_css_simple(pct_val):
                try:
                    v = int(pct_val)
                except Exception:
                    return ""
                if v < 0:
                    return "background-color: #ffc0cb; color: #000"
                elif v > 0:
                    return "background-color: #c6f6b6; color: #000"
                else:
                    return ""

            def corr_styler_simple(df):
                sty = pd.DataFrame("", index=df.index, columns=df.columns)
                for item in df.index:
                    for col in df.columns:
                        sty.at[item, col] = corr_cell_css_simple(df.at[item, col])
                return sty

            st.subheader("割引率表")
            styled_corr = df_corr.style.apply(lambda _: corr_styler_simple(df_corr), axis=None)
            styled_corr = styled_corr.format("{:+d}", na_rep="")
            st.dataframe(styled_corr, height=380)

else:
    st.warning("一部の港が未更新です。管理画面で入力してください。")
    st.write("未更新港:", missing_ports)
    if st.button("管理画面を開く（未更新港を編集）", key="btn_open_admin_from_missing"):
        st.session_state["mode"] = "admin"
        safe_rerun()

# --------------------
# 管理画面（タブ式: 未更新 / 全ポート）
# --------------------
if st.session_state.get("mode") == "admin":
    st.header("管理画面")
    tab_missing, tab_all = st.tabs(["未更新港の編集", "全ポート一覧"])

    with tab_missing:
        st.subheader("未更新港の編集")
        if missing_ports:
            sel_missing = st.selectbox("編集する未更新港", options=missing_ports, key="sel_missing_admin")
            st.markdown(f"## {sel_missing} の入力（未更新）")
            current = PRICES_CFG.get(sel_missing, {})
            cols = st.columns(2)
            inputs_miss = {}
            for i, (name, base) in enumerate(ITEMS_CFG):
                c = cols[i % 2]
                default = "" if name not in current else str(current[name])
                with c:
                    inputs_miss[name] = st.text_input(f"{name} (base: {base})", value=default, key=f"{sel_missing}_{name}_miss_admin")

            col_ok_miss, col_reset_miss, col_refresh_miss = st.columns([1,1,1])
            with col_ok_miss:
                if st.button("保存（未更新港）", key=f"save_miss_{sel_missing}"):
                    new_row = {}
                    invalids = []
                    for name, base in ITEMS_CFG:
                        raw = inputs_miss.get(name, "")
                        s = (raw or "").strip().replace(",", "")
                        s = s.translate(str.maketrans("０１２３４５６７８９－＋．，", "0123456789-+.,"))
                        if s == "" or not re.fullmatch(r"\d+", s):
                            invalids.append(name)
                        else:
                            v = int(s)
                            if v < 0:
                                invalids.append(name)
                            else:
                                new_row[name] = v
                    if invalids:
                        st.error("不正な入力があります: " + ", ".join(invalids))
                    else:
                        PRICES_CFG[sel_missing] = new_row
                        cfg["PRICES"] = PRICES_CFG
                        try:
                            resp = save_cfg_to_jsonbin(cfg)
                            st.success(f"{sel_missing} を保存しました。HTTP {resp.status_code}")
                            new_cfg = fetch_cfg_from_jsonbin()
                            if new_cfg:
                                cfg = new_cfg
                                PRICES_CFG = cfg.get("PRICES", {})
                                safe_rerun()
                            else:
                                st.info("保存成功。ページを手動で再読み込みしてください。")
                        except Exception as e:
                            st.error(f"保存に失敗しました: {e}")

            with col_reset_miss:
                if st.button("この未更新港を base にリセット", key=f"reset_miss_{sel_missing}"):
                    PRICES_CFG = reset_port_to_base(sel_missing, ITEMS_CFG, PRICES_CFG)
                    cfg["PRICES"] = PRICES_CFG
                    try:
                        resp = save_cfg_to_jsonbin(cfg)
                        st.success(f"{sel_missing} を base にリセットしました。HTTP {resp.status_code}")
                        new_cfg = fetch_cfg_from_jsonbin()
                        if new_cfg:
                            cfg = new_cfg
                            PRICES_CFG = cfg.get("PRICES", {})
                            safe_rerun()
                    except Exception as e:
                        st.error(f"リセットに失敗しました: {e}")

            with col_refresh_miss:
                if st.button("この港の最新データを再取得", key=f"refresh_miss_{sel_missing}"):
                    new_cfg = fetch_cfg_from_jsonbin()
                    if new_cfg:
                        cfg = new_cfg
                        PRICES_CFG = cfg.get("PRICES", {})
                        st.success("最新データを取得しました。")
                        safe_rerun()
                    else:
                        st.error("再取得に失敗しました。")
        else:
            st.info("未更新の港はありません。")

    with tab_all:
        st.subheader("全港一覧")
        sel_port_all = st.selectbox("編集する港を選択", options=PORTS_CFG, key="sel_port_all_admin")
        st.markdown(f"## {sel_port_all} の価格（編集可）")
        current_row = PRICES_CFG.get(sel_port_all, {})
        cols2 = st.columns(2)
        inputs_all = {}
        for i, (name, base) in enumerate(ITEMS_CFG):
            c = cols2[i % 2]
            default = "" if name not in current_row else str(current_row[name])
            with c:
                inputs_all[name] = st.text_input(f"{name} (base: {base})", value=default, key=f"{sel_port_all}_{name}_all_admin")

        col_ok_all, col_refresh_all = st.columns([1,1])
        with col_ok_all:
            if st.button("保存（この港）", key=f"save_all_{sel_port_all}"):
                new_row = {}
                invalids = []
                for name, base in ITEMS_CFG:
                    raw = inputs_all.get(name, "")
                    s = (raw or "").strip().replace(",", "")
                    s = s.translate(str.maketrans("０１２３４５６７８９－＋．，", "0123456789-+.,"))
                    if s == "" or not re.fullmatch(r"\d+", s):
                        invalids.append(name)
                    else:
                        v = int(s)
                        if v < 0:
                            invalids.append(name)
                        else:
                            new_row[name] = v
                if invalids:
                    st.error("不正な入力があります: " + ", ".join(invalids))
                else:
                    PRICES_CFG[sel_port_all] = new_row
                    cfg["PRICES"] = PRICES_CFG
                    try:
                        resp = save_cfg_to_jsonbin(cfg)
                        st.success(f"{sel_port_all} を保存しました。HTTP {resp.status_code}")
                        new_cfg = fetch_cfg_from_jsonbin()
                        if new_cfg:
                            cfg = new_cfg
                            PRICES_CFG = cfg.get("PRICES", {})
                            safe_rerun()
                        else:
                            st.info("保存成功。ページを手動で再読み込みしてください。")
                    except Exception as e:
                        st.error(f"保存に失敗しました: {e}")

        with col_refresh_all:
            if st.button("この港の最新データを再取得", key=f"refresh_all_{sel_port_all}"):
                new_cfg = fetch_cfg_from_jsonbin()
                if new_cfg:
                    cfg = new_cfg
                    PRICES_CFG = cfg.get("PRICES", {})
                    st.success("最新データを取得しました。")
                    safe_rerun()
                else:
                    st.error("再取得に失敗しました。")

        # ---------- 全ポート最下部: 全件リセット（パスワード保護） ----------
        st.markdown("---")
        st.write("全港のデータを base 値にリセットします。実行すると現在の全データが上書きされます。")
        pwd = st.text_input("操作パスワードを入力してください", type="password", key="reset_all_pwd")
        if st.button("全港を base 値にリセット（パスワード必須）", key="reset_all_confirm"):
            if pwd == RESET_PASSWORD:
                PRICES_CFG = reset_all_ports_to_base(ITEMS_CFG, PRICES_CFG, PORTS_CFG)
                cfg["PRICES"] = PRICES_CFG
                try:
                    resp = save_cfg_to_jsonbin(cfg)
                    st.success(f"全港を base にリセットしました。HTTP {resp.status_code}")
                    new_cfg = fetch_cfg_from_jsonbin()
                    if new_cfg:
                        cfg = new_cfg
                        PRICES_CFG = cfg.get("PRICES", {})
                        safe_rerun()
                except Exception as e:
                    st.error(f"全件リセットに失敗しました: {e}")
            else:
                st.error("パスワードが違います。操作は中止されました。")

    st.markdown("---")
    if st.button("管理モードを終了して戻る", key="btn_close_admin"):
        st.session_state["mode"] = "view"
        safe_rerun()
