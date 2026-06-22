import streamlit as st
from collections import defaultdict
import heapq

# ============================================================
# 🔼 画面上部に表示されるイベント定義（ここを編集するだけでOK）
# ============================================================
EVENT_DATA = {
    "yakuyasan": {
        "name": "薬屋",
        "costs": [14300, 143000, 286000, 1144000],
        "feed":  [20, 200, 400, 1600],
        "item":  [1, 1, 2, 2]
    },
    "houshin": {
        "name": "封神演義",
        "costs": [3200, 49000, 98000, 392000],
        "feed":  [20, 200, 400, 1600],
        "item":  [1, 1, 2, 2]
    },
    "dotou": {
        "name": "怒涛斬魁",
        "costs": [7500, 23000, 46000, 184000],
        "feed":  [20, 200, 400, 1600],
        "item":  [1, 1, 2, 2]
    }
}

# ============================================================
# DP 用データ生成（自動）
# ============================================================
EVENT_JP = {}
OPTIONS = []
STAGE_MAP = {}

for key, ev in EVENT_DATA.items():
    EVENT_JP[key] = ev["name"]

    for stage, cost in enumerate(ev["costs"], start=1):
        feed = ev["feed"][stage - 1]
        item = ev["item"][stage - 1]

        OPTIONS.append((key, cost, feed, item))
        STAGE_MAP[cost] = stage


# ============================================================
# 履歴の日本語化
# ============================================================
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


# ============================================================
# DP 本体（育成回数以内）
# ============================================================
def optimize_training(points, N):
    event_keys = list(EVENT_DATA.keys())

    dp = [defaultdict(lambda: {"feed": -1, "item": -1, "history": []})
          for _ in range(N + 1)]

    start_key = tuple(points[k] for k in event_keys)
    dp[0][start_key] = {"feed": 0, "item": 0, "history": []}

    for i in range(N):
        for state_key, state in dp[i].items():
            feed_now = state["feed"]
            item_now = state["item"]
            hist_now = state["history"]

            current_points = dict(zip(event_keys, state_key))

            for ev, cost, feed_gain, item_gain in OPTIONS:
                if cost > current_points[ev]:
                    continue

                new_points = current_points.copy()
                new_points[ev] -= cost

                new_key = tuple(new_points[k] for k in event_keys)

                new_feed = feed_now + feed_gain
                new_item = item_now + item_gain
                new_hist = hist_now + [(ev, cost, feed_gain, item_gain)]

                old = dp[i + 1][new_key]

                if new_feed > old["feed"] or new_item > old["item"]:
                    dp[i + 1][new_key] = {
                        "feed": new_feed,
                        "item": new_item,
                        "history": new_hist
                    }

    # 0〜N の全状態から最適解を取る
    results = []
    for i in range(N + 1):
        for st in dp[i].values():
            results.append((st["feed"], st["item"], st["history"]))

    top_feed = heapq.nlargest(3, results, key=lambda x: x[0])
    top_item = heapq.nlargest(3, results, key=lambda x: x[1])
    top_mix = heapq.nlargest(3, results, key=lambda x: x[0] + x[1] * 100)

    top_feed = [(f, i, convert_history(h)) for (f, i, h) in top_feed]
    top_item = [(f, i, convert_history(h)) for (f, i, h) in top_item]
    top_mix = [(f, i, convert_history(h)) for (f, i, h) in top_mix]

    return top_mix, top_feed, top_item


# ============================================================
# Streamlit UI
# ============================================================
st.title("🐷 豚育成 最適化ツール（イベント定義つき）")

# 🔼 イベント定義を画面上部に表示
st.subheader("📘 イベント定義（ここを編集して使う）")

for key, ev in EVENT_DATA.items():
    st.markdown(f"### {ev['name']}")
    st.write("コスト:", ev["costs"])
    st.write("餌:", ev["feed"])
    st.write("アイテム:", ev["item"])
    st.markdown("---")

# 入力フォーム
points = {}
for key, ev in EVENT_DATA.items():
    points[key] = st.number_input(
        f"{ev['name']}ポイント",
        min_value=0,
        value=0,
        step=100
    )

count = st.number_input("残り育成回数（最大10）", min_value=1, max_value=10, value=5)

if st.button("最適化する"):
    top_mix, top_feed, top_item = optimize_training(points, count)

    st.subheader("🔥 複合スコア 最大パターン")
    for feed, item, hist in top_mix:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")

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
