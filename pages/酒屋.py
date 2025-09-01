# pages/酒屋.py
import streamlit as st

def format_japanese_number(n: int) -> str:
    """
    整数 n を「○億○万○」の形式で返す関数。
    例：
      123456789 → "1億2345万6789"
      5000      → "5000"
      10000     → "1万"
    """
    oku, rest = divmod(n, 10**8)
    man, rest = divmod(rest, 10**4)
    parts = []
    if oku:
        parts.append(f"{oku}億")
    if man:
        parts.append(f"{man}万")
    # 残りがあれば、または億・万が無くても「0」は表示
    if rest or not parts:
        parts.append(str(rest))
    return "".join(parts)
    
st.title("酒屋計算")

# ① GitHub の Raw URL
image_url = "https://raw.githubusercontent.com/paru-akindo/calc/main/image/sake.jpg"

# ② 画像表示
st.image(
    image_url,
    caption="銀塊のアイコン",
    width=300
)

# 入力をすべて整数のみ許可
base_silver = st.number_input(
    "基礎銀塊", 
    min_value=0, step=1, value=0
)
collection_bonus = st.number_input(
    "収蔵品効果", 
    min_value=0, step=1, value=0
)
elixir = st.number_input(
    "美酒", 
    min_value=0, step=1, value=0
)

if st.button("計算する"):
    total = (base_silver + collection_bonus) * elixir * 2  # 整数計算
    formatted = format_japanese_number(total)
    st.success(f"計算結果：{formatted}")
