import streamlit as st
import copy
import pandas as pd

import streamlit as st

st.markdown(
    """
    <style>
    /* ボタンの余白・パディングをさらに狭く */
    div[data-baseweb="button"] {
         padding: 0px 2px !important;
         margin: 0px !important;
         font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 定数
BOARD_SIZE = 5
DEFAULT_MAX_VALUE = 20

def format_board(board, action=None):
    """
    盤面 (list-of-lists) を pandas の DataFrame に変換する。
    ・None（欠損値）は0に置換し、すべて整数で表示する。
    ・行・列ラベルは1〜BOARD_SIZEに設定する。
    ・action が指定される場合（("add", r, c) または ("remove", r, c)）は、
      対象セルを "add" は赤、"remove" は青でハイライトする。
    ・ヘッダーのラベルは灰色で表示。
    """
    df = pd.DataFrame(board)
    df = df.fillna(0).astype(int)
    df.index = [i + 1 for i in range(len(df))]
    df.columns = [i + 1 for i in range(len(df.columns))]
    
    def highlight_action(df):
        styled = pd.DataFrame("", index=df.index, columns=df.columns)
        if action is not None:
            act_type, act_r, act_c = action
            if act_type == "add":
                styled.at[act_r+1, act_c+1] = "background-color: red"
            elif act_type == "remove":
                styled.at[act_r+1, act_c+1] = "background-color: blue"
        return styled

    styler = df.style.apply(highlight_action, axis=None)
    header_styles = [
        {'selector': 'th.col_heading.level0', 'props': 'background-color: gray;'},
        {'selector': 'th.row_heading.level0', 'props': 'background-color: gray;'}
    ]
    styler = styler.set_table_styles(header_styles)
    return styler

# ----------------------------
# MergeGameSimulator クラス
# ----------------------------
class MergeGameSimulator:
    def __init__(self, board):
        self.board = board  # 初期盤面

    def display_board(self, board, action=None):
        """盤面をテーブル形式で表示する。必要に応じて対象セルに色付けする。"""
        st.dataframe(format_board(board, action))
        st.markdown("---")

    def find_clusters(self, board):
        """隣接する同じ数字のクラスタを検出する"""
        visited = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        clusters = []
        def dfs(r, c, value):
            if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                return []
            if visited[r][c] or board[r][c] != value:
                return []
            visited[r][c] = True
            cluster = [(r, c)]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                cluster.extend(dfs(r+dr, c+dc, value))
            return cluster
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] is not None and not visited[r][c]:
                    cluster = dfs(r, c, board[r][c])
                    if len(cluster) >= 3:
                        clusters.append(cluster)
        return clusters

    def merge_clusters(self, board, clusters, fall, user_action=None, max_value=20):
        """
        検出したクラスタを合成し、合成されたセル数を返す。
        user_action が指定されている場合は、1手目ではその対象セルを優先的に更新する。
        """
        total_merged_numbers = 0
        for cluster in clusters:
            values = [board[r][c] for r, c in cluster]
            base_value = values[0]
            new_value = base_value + (len(cluster) - 2)
            total_merged_numbers += len(cluster)
            if user_action and user_action[0] == "add":
                if fall == 0:
                    target_r, target_c = user_action[1], user_action[2]
                else:
                    target_r, target_c = min(cluster, key=lambda x: (-x[0], x[1]))
            else:
                target_r, target_c = min(cluster, key=lambda x: (-x[0], x[1]))
            for r, c in cluster:
                board[r][c] = None
            if new_value < max_value:
                board[target_r][target_c] = new_value
        return total_merged_numbers

    def apply_gravity(self, board):
        """各列の数字を下に落下させる"""
        for c in range(BOARD_SIZE):
            column = [board[r][c] for r in range(BOARD_SIZE) if board[r][c] is not None]
            for r in range(BOARD_SIZE-1, -1, -1):
                board[r][c] = column.pop() if column else None

    def simulate(self, action, max_value=20, suppress_output=False):
        """
        指定したアクション（("add", r, c) または ("remove", r, c)）を適用したときの連鎖シミュレーションを行う。
        初期盤面は対象セルをハイライトして表示する（suppress_output=Falseの場合）。
        すでに空（None）のセルには "add" は適用されません。
        戻り値: (fall_count, total_merged_numbers, 最終盤面)
        """
        board = copy.deepcopy(self.board)
        if not suppress_output:
            st.write("Initial board:")
            self.display_board(board, action=action)
        if action[0] == "add":
            r, c = action[1], action[2]
            if board[r][c] is not None:
                board[r][c] += 1
        elif action[0] == "remove":
            r, c = action[1], action[2]
            board[r][c] = None
        fall_count = 0
        total_merged_numbers = 0
        self.apply_gravity(board)
        while True:
            clusters = self.find_clusters(board)
            if not clusters:
                break
            total_merged_numbers += self.merge_clusters(board, clusters, fall_count, user_action=action, max_value=max_value)
            self.apply_gravity(board)
            fall_count += 1
            if not suppress_output:
                st.write(f"After fall {fall_count}:")
                self.display_board(board)
        return fall_count, total_merged_numbers, board

    def find_best_action(self, max_value=20):
        """
        盤面全体に対して "add" と "remove" を試行し、
        1手のみのシミュレーションで最適な操作（合成セル数優先）を求める。
        戻り値は辞書 {'action': (op, r, c), 'merged': 合成セル数, 'fall': 落下回数, 'board': 最終盤面}。
        """
        candidates = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] is not None:
                    for op in ["add", "remove"]:
                        action = (op, r, c)
                        fall, merged, board_after = self.simulate(action, max_value=max_value, suppress_output=True)
                        candidates.append({
                            'action': action,
                            'merged': merged,
                            'fall': fall,
                            'board': board_after
                        })
        best = max(candidates, key=lambda x: x['merged'])
        return best

    def find_best_action_by_fall(self, max_value=20):
        """
        盤面全体に対して "add" と "remove" を試行し、
        1手のみのシミュレーションで最適な操作（落下回数優先）を求める。
        戻り値は find_best_action と同じ形式の辞書。
        """
        candidates = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] is not None:
                    for op in ["add", "remove"]:
                        action = (op, r, c)
                        fall, merged, board_after = self.simulate(action, max_value=max_value, suppress_output=True)
                        candidates.append({
                            'action': action,
                            'merged': merged,
                            'fall': fall,
                            'board': board_after
                        })
        best = max(candidates, key=lambda x: x['fall'])
        return best

    def find_best_action_multistep(self, max_value=20, threshold=6):
        """
        全パターンの2手候補を最初から網羅的に検証する方式。
        盤面全体に対して、全ての1手候補と、その後のすべての2手候補を試行し、
        1手目＋2手目の合計効果（合成セル数）が最大となる操作シーケンスを求める。
        戻り値は辞書 {'one_move': 1手目候補, 'two_moves': 2手シーケンス候補（あれば）}。
        """
        candidates_1 = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] is not None:
                    for op in ["add", "remove"]:
                        action = (op, r, c)
                        fall, merged, board_after = self.simulate(action, max_value=max_value, suppress_output=True)
                        candidates_1.append({
                            'action': action,
                            'merged': merged,
                            'fall': fall,
                            'board': board_after
                        })
        one_move = max(candidates_1, key=lambda x: x['merged'])
        result = {'one_move': one_move, 'two_moves': None}
        
        best_total = one_move['merged']
        best_sequence = (one_move['action'], None)
        # 各1手候補について、2手目を網羅的に評価
        for cand in candidates_1:
            temp_board = cand['board']
            simul2 = MergeGameSimulator(temp_board)
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if temp_board[r][c] is not None:
                        for op in ["add", "remove"]:
                            action2 = (op, r, c)
                            _, merged2, _ = simul2.simulate(action2, max_value=max_value, suppress_output=True)
                            total = cand['merged'] + merged2
                            if total > best_total:
                                best_total = total
                                best_sequence = (cand['action'], action2)
        if best_sequence[1] is not None:
            result['two_moves'] = {'actions': best_sequence, 'merged': best_total}
        return result

# ----------------------------
# Streamlit アプリ本体
# ----------------------------
st.title("百鬼夜行")

# 盤面の入力方法選択
input_method = st.radio("盤面の入力方法を選択", ("カンマ区切りテキスト入力", "グリッド入力"))

# セッション状態の初期化
if "grid_board_values" not in st.session_state:
    st.session_state.grid_board_values = [[8] * BOARD_SIZE for _ in range(BOARD_SIZE)]
if "csv_board_values" not in st.session_state:
    default_csv = "8,7,6,5,6\n6,9,8,6,5\n9,5,11,5,9\n7,8,9,11,8\n7,11,8,6,7"
    st.session_state.csv_board_values = default_csv
if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = None
if "max_value" not in st.session_state:
    st.session_state.max_value = DEFAULT_MAX_VALUE

st.subheader("最大合成値の設定")
st.session_state.max_value = st.number_input("最大合成値 (max_value)", min_value=1,
                                               value=st.session_state.max_value, key="max_value_input")

board = None
if input_method == "グリッド入力":
    st.subheader("グリッド入力（セルをタップして編集）")
    for r in range(BOARD_SIZE):
        cols = st.columns(BOARD_SIZE)
        for c in range(BOARD_SIZE):
            # ボタンラベルは座標情報を削除し、現状の数字だけを表示する
            if cols[c].button(f"{st.session_state.grid_board_values[r][c]}", key=f"grid_btn_{r}_{c}"):
                st.session_state.selected_cell = (r, c)
    if st.session_state.selected_cell is not None:
        r, c = st.session_state.selected_cell
        st.subheader(f"セルの値を変更")
        new_value = st.slider("新しい値を選択", min_value=0,
                              max_value=st.session_state.max_value,
                              value=st.session_state.grid_board_values[r][c],
                              key=f"grid_slider_{r}_{c}")
        if st.button("確定", key=f"grid_confirm_{r}_{c}"):
            st.session_state.grid_board_values[r][c] = new_value
            st.session_state.selected_cell = None
        if st.button("キャンセル", key=f"grid_cancel_{r}_{c}"):
            st.session_state.selected_cell = None
    board = st.session_state.grid_board_values

else:
    st.subheader("カンマ区切りテキスト入力")
    csv_input = st.text_area("5行のカンマ区切りで盤面を入力",
                             value=st.session_state.csv_board_values,
                             height=150)
    st.session_state.csv_board_values = csv_input
    try:
        lines = csv_input.strip().splitlines()
        parsed_board = []
        for line in lines:
            values = [int(v.strip()) for v in line.split(",")]
            if len(values) != BOARD_SIZE:
                st.error("各行に5つの数値が必要です。")
                parsed_board = None
                break
            parsed_board.append(values)
        if parsed_board is not None and len(parsed_board) != BOARD_SIZE:
            st.error("5行入力してください。")
            parsed_board = None
    except Exception as e:
        st.error(f"入力解析エラー: {e}")
        parsed_board = None
    board = parsed_board

if board is not None:
    st.subheader("入力された盤面")
    st.dataframe(format_board(board))

simulate_button = st.button("実行")

if simulate_button:
    if board is None:
        st.error("盤面が正しく入力されていません。")
    else:
        simulator = MergeGameSimulator(board)
        max_value = st.session_state.max_value

        # 1手目および全パターンの2手候補を網羅的に検証
        multi_result = simulator.find_best_action_multistep(max_value=max_value, threshold=6)
        one_move = multi_result['one_move']
        two_moves = multi_result['two_moves']

        # 常に1手目のみの結果を表示（左上：1手の連鎖数、右上：1手の合成セル数）
        col_top1, col_top2 = st.columns(2)
        with col_top1:
            best_by_fall = simulator.find_best_action_by_fall(max_value=max_value)
            st.subheader("最大連鎖(1手)")
            st.write(f"【{best_by_fall['action'][0]}】 ({best_by_fall['action'][1]+1},{best_by_fall['action'][2]+1})")
            st.write(f"落下回数: {best_by_fall['fall']}")
            st.dataframe(format_board(best_by_fall['board']))
            st.write("手順:")
            simulator.simulate(best_by_fall['action'], max_value=max_value, suppress_output=False)
        with col_top2:
            best_by_merged = simulator.find_best_action(max_value=max_value)
            st.subheader("最大合成(1手)")
            st.write(f"【{best_by_merged['action'][0]}】 ({best_by_merged['action'][1]+1},{best_by_merged['action'][2]+1})")
            st.write(f"合成セル数: {best_by_merged['merged']}")
            st.dataframe(format_board(best_by_merged['board']))
            st.write("手順:")
            simulator.simulate(best_by_merged['action'], max_value=max_value, suppress_output=False)

        # 2手候補がある場合、下部に2手の結果（合成数）を表示
        if two_moves is not None:
            actions = two_moves['actions']
            st.subheader("最大合成(2手)")
            st.write(f"1手目: 【{actions[0][0]}】 ({actions[0][1]+1},{actions[0][2]+1})")
            st.write(f"2手目: 【{actions[1][0]}】 ({actions[1][1]+1},{actions[1][2]+1})")
            st.write(f"合計合成セル数: {two_moves['merged']}")
            st.subheader("手順")
            st.write("【1手目の操作】")
            sim1 = simulator.simulate(actions[0], max_value=max_value, suppress_output=False)
            board_after1 = sim1[2]
            st.write("【2手目の操作】")
            sim2 = MergeGameSimulator(board_after1)
            sim2.simulate(actions[1], max_value=max_value, suppress_output=False)
