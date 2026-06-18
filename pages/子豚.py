import streamlit as st
from collections import defaultdict
import heapq

# -----------------------------
# 日本語化辞書
# -----------------------------
EVENT_JP = {
    "geisho": "霓裳",
    "enkai": "宴会",
    "tonko": "敦煌",
}

# -----------------------------
# cost → 段階番号（修正版）
# -----------------------------
STAGE_MAP = {
    # 霓裳
    900: 1,
    7600: 2,
    15200: 3,
    60800: 4,

    # 宴会（79125 が正解）
    7920: 1,
    79125: 2,   # ← 修正済み
    158250: 3,
    633000: 4,

    # 敦煌
    30: 1,
    100: 2,
    200: 3,
    800: 4,
}

# -----------------------------
# 選択肢（12種類）
# -----------------------------
OPTIONS = [
    ("geisho", 900, 20, 1),
    ("geisho", 7600, 200, 1),
    ("geisho", 15200, 400, 2),
    ("geisho", 60800, 1600, 2),

    ("enkai", 7920, 20, 1),
    ("enkai", 79125, 200, 1),   # ← 修正済み
    ("enkai", 158250, 400, 2),
    ("enkai", 633000, 1600, 2),

    ("tonko", 30, 20, 1),
    ("tonko", 100, 200, 1),
    ("tonko", 200, 400, 2),
    ("tonko", 800, 1600, 2),
]


# -----------------------------
# 履歴の日本語化
# -----------------------------
def convert_history(history):
    jp = []
    for ev, cost, feed, item in history:
        jp.append({
            "event": EVENT_JP[ev],
            "stage": STAGE_MAP[cost],
            "cost": cost,
            "feed": feed,
            "item": item
        })
    return jp


# -----------------------------
# DP 本体
# -----------------------------
def optimize_training(geisho_pt, enkai_pt, tonko_pt, N):
    dp = [defaultdict(lambda: {"feed": -1, "item": -1, "history": []}) for _ in range(N + 1)]
    dp[0][(geisho_pt, enkai_pt, tonko_pt)] = {"feed": 0, "item": 0, "history": []}

    for i in range(N):
        for (g_left, h_left, s_left), state in dp[i].items():
            feed_now = state["feed"]
            item_now = state["item"]
            hist_now = state["history"]

            for ev, cost, feed_gain, item_gain in OPTIONS:
                if ev == "geisho" and cost > g_left:
                    continue
                if ev == "enkai" and cost > h_left:
                    continue
                if ev == "tonko" and cost > s_left:
                    continue

                ng, nh, ns = g_left, h_left, s_left
                if ev == "geisho":
                    ng -= cost
                elif ev == "enkai":
                    nh -= cost
                else:
                    ns -= cost

                new_feed = feed_now + feed_gain
                new_item = item_now + item_gain
                new_hist = hist_now + [(ev, cost, feed_gain, item_gain)]

                key = (ng, nh, ns)
                old = dp[i + 1][key]

                if new_feed > old["feed"] or new_item > old["item"]:
                    dp[i + 1][key] = {
                        "feed": new_feed,
                        "item": new_item,
                        "history": new_hist
                    }

    results = []
    for (g, h, s), st in dp[N].items():
        results.append((st["feed"], st["item"], st["history"]))

    top_feed = heapq.nlargest(3, results, key=lambda x: x[0])
    top_item = heapq.nlargest(3, results, key=lambda x: x[1])
    top_mix = heapq.nlargest(3, results, key=lambda x: x[0] + x[1] * 100)

    # 日本語化
    top_feed = [(f, i, convert_history(h)) for (f, i, h) in top_feed]
    top_item = [(f, i, convert_history(h)) for (f, i, h) in top_item]
    top_mix = [(f, i, convert_history(h)) for (f, i, h) in top_mix]

    return top_mix, top_feed, top_item


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐷 豚育成 最適化ツール")

geisho = st.number_input("霓裳ポイント", min_value=0, value=0, step=100)
enkai = st.number_input("宴会ポイント", min_value=0, value=0, step=100)
tonko = st.number_input("敦煌ポイント", min_value=0, value=0, step=10)
count = st.number_input("残り育成回数（最大10）", min_value=1, max_value=10, value=5)

if st.button("最適化する"):
    top_mix, top_feed, top_item = optimize_training(geisho, enkai, tonko, count)

    # -----------------------------
    # 🔥 複合スコア（先頭）
    # -----------------------------
    st.subheader("🔥 複合スコア 最大パターン（餌 + アイテム×100）")
    for feed, item, hist in top_mix:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")

    # -----------------------------
    # 🍚 餌 最大
    # -----------------------------
    st.subheader("🍚 餌 最大パターン")
    for feed, item, hist in top_feed:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")

    # -----------------------------
    # 🎁 アイテム 最大
    # -----------------------------
    st.subheader("🎁 アイテム 最大パターン")
    for feed, item, hist in top_item:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")
