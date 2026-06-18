import streamlit as st
from collections import defaultdict
import heapq

# -----------------------------
# 日本語化辞書
# -----------------------------
EVENT_JP = {
    "gosou": "護送",
    "kouzan": "黄山の美",
    "shika": "鹿追",
}

STAGE_MAP = {
    22500: 1,
    225250: 2,
    450500: 3,
    1802000: 4,

    1: 1,
    7: 2,
    13: 3,
    52: 4,

    101000: 1,
    454500: 2,
    909000: 3,
    3636000: 4,
}

# -----------------------------
# 選択肢（12種類）
# -----------------------------
OPTIONS = [
    ("gosou", 22500, 20, 1),
    ("gosou", 225250, 200, 1),
    ("gosou", 450500, 400, 2),
    ("gosou", 1802000, 1600, 2),

    ("kouzan", 1, 20, 1),
    ("kouzan", 7, 200, 1),
    ("kouzan", 13, 400, 2),
    ("kouzan", 52, 1600, 2),

    ("shika", 101000, 20, 1),
    ("shika", 454500, 200, 1),
    ("shika", 909000, 400, 2),
    ("shika", 3636000, 1600, 2),
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
def optimize_training(gosou_pt, kouzan_pt, shika_pt, N):
    dp = [defaultdict(lambda: {"feed": -1, "item": -1, "history": []}) for _ in range(N + 1)]
    dp[0][(gosou_pt, kouzan_pt, shika_pt)] = {"feed": 0, "item": 0, "history": []}

    for i in range(N):
        for (g_left, h_left, s_left), state in dp[i].items():
            feed_now = state["feed"]
            item_now = state["item"]
            hist_now = state["history"]

            for ev, cost, feed_gain, item_gain in OPTIONS:
                if ev == "gosou" and cost > g_left:
                    continue
                if ev == "kouzan" and cost > h_left:
                    continue
                if ev == "shika" and cost > s_left:
                    continue

                ng, nh, ns = g_left, h_left, s_left
                if ev == "gosou":
                    ng -= cost
                elif ev == "kouzan":
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

    return top_feed, top_item, top_mix


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐷 豚育成 最適化ツール（Streamlit版）")

gosou = st.number_input("護送ポイント", min_value=0, value=0, step=1000)
kouzan = st.number_input("黄山の美ポイント", min_value=0, value=0, step=1)
shika = st.number_input("鹿追ポイント", min_value=0, value=0, step=1000)
count = st.number_input("残り育成回数（最大10）", min_value=1, max_value=10, value=5)

if st.button("最適化する"):
    top_feed, top_item, top_mix = optimize_training(gosou, kouzan, shika, count)

    st.subheader("🍚 餌 最大パターン")
    for feed, item, hist in top_feed:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")

    st.subheader("🎁 アイテム 最大パターン")
    for feed, item, hist in top_item:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")

    st.subheader("🔥 複合スコア 最大パターン（餌 + アイテム×100）")
    for feed, item, hist in top_mix:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")
