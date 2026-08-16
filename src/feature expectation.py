import pandas as pd
import numpy as np
from itertools import product
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


id_col = "id"
time_col = "day"

df = pd.read_csv("./data/merged_219.csv").sort_values([id_col, time_col]).reset_index(drop=True)
df["startSleepTime"] = pd.to_datetime(df["startSleepTime"], format="%Y-%m-%d %I:%M %p", errors="coerce")
df["startSleepTime_decimal"] = df["startSleepTime"].dt.hour + df["startSleepTime"].dt.minute / 60
df["sleep_onset_shifted"] = ((df["startSleepTime_decimal"] + 12) % 24) - 12

df["step_cat"] = pd.cut(
    df["step"],
    bins = [-np.inf, 5000, np.inf],
    labels = [0, 1]
)
df["norm_ent_cat"] = pd.cut(
    df["normalized_location_entropy"],
    bins = [-np.inf, 0.5, 1],
    labels = [0, 1]
)
df["totalSleep_cat"] = pd.cut(
    df["totalSleep"],
    bins = [-np.inf, 420, 540, np.inf],
    labels = [0, 1, 2]
)
df["sleep_onset_cat"] = pd.cut(
    df["sleep_onset_shifted"],
    bins = [-np.inf, 0.66, 3.16, np.inf],
    labels = [0, 1, 2]
)

raw_state_cols = ['feeling', 'appetite', 'feelingOfSleep', 'generally', '스트레스', '우울', '불안', 'intervene']
action_cols = ['step_cat', 'norm_ent_cat', 'totalSleep_cat', 'sleep_onset_cat']
reward_vars = ['feeling', 'appetite', 'feelingOfSleep', 'generally', '스트레스', '우울', '불안']
reward_cols = []

df[[f"{c}_next" for c in reward_vars]] = (
    df.groupby(id_col)[reward_vars].shift(-1)
)
for c in reward_vars:
    df[f"{c}_reward"] = df[f"{c}_next"] - df[c]
    reward_cols.append(f"{c}_reward")

GAMMA = 1.0
N_CLUSTERS = 20


# ============================================================
# 1. State 압축 (KMeans)
# ============================================================
scaler = StandardScaler()
df = df.dropna(subset=raw_state_cols)
scaled_features = scaler.fit_transform(df[raw_state_cols])

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
df["state_cluster"] = kmeans.fit_predict(scaled_features)
N_STATES = N_CLUSTERS

# 각 클러스터의 "대표 feature 값" = 그 클러스터에 속한 데이터의 평균 지표
# (이게 reward를 구성하는 phi(s) 역할을 함)
cluster_features = df.groupby("state_cluster")[raw_state_cols].mean().values  # (N_STATES, n_features)
N_FEATURES = cluster_features.shape[1]


# ============================================================
# 2. Action 인코딩 (4개 카테고리 변수 -> 하나의 정수 인덱스)
# ============================================================
action_values = [
    [0, 1],       # step_cat
    [0, 1],       # norm_ent_cat
    [0, 1, 2],    # totalSleep_cat
    [0, 1, 2],    # sleep_onset_cat
]
all_actions = list(product(*action_values))
N_ACTIONS = len(all_actions)

action_to_idx = {a: i for i, a in enumerate(all_actions)}
df["action_idx"] = df[action_cols].apply(lambda row: action_to_idx[tuple(row)], axis=1)


# ============================================================
# 3. Transition 모델 추정 (경험적 카운트 + smoothing)
# ============================================================
# P[s, a, s'] 형태의 3차원 배열
transition_counts = np.ones((N_STATES, N_ACTIONS, N_STATES)) * 0.1  # Laplace smoothing (완전 0 방지)

for pid, sub_df in df.groupby(id_col):
    sub_df = sub_df.sort_values(time_col)
    states = sub_df["state_cluster"].values
    actions = sub_df["action_idx"].values
    for t in range(len(sub_df) - 1):
        s, a, s_next = states[t], actions[t], states[t + 1]
        transition_counts[s, a, s_next] += 1

# 정규화해서 확률로 변환
transition_probs = transition_counts / transition_counts.sum(axis=2, keepdims=True)


# ============================================================
# 4. Expert feature expectation 계산
#    (실제 관측된 사람들의 궤적에서, 할인된 평균 feature)
# ============================================================
def compute_feature_expectation(trajectories_states):
    """trajectories_states: list of state_cluster 시퀀스(사람별)"""
    total = np.zeros(N_FEATURES)
    for traj in trajectories_states:
        discount = 1.0
        for s in traj:
            total += discount * cluster_features[s]
            discount *= GAMMA
    return total / len(trajectories_states)

expert_trajectories = [
    sub_df.sort_values(time_col)["state_cluster"].values
    for _, sub_df in df.groupby(id_col)
]
mu_expert = compute_feature_expectation(expert_trajectories)
print("Expert feature expectation:", mu_expert)

# ============================================================
# 5. Value Iteration (주어진 w로 최적 정책 계산)
# ============================================================
def value_iteration(w, n_iter=200, tol=1e-4):
    R = cluster_features @ w  # (N_STATES,) - 각 state의 reward
    V = np.zeros(N_STATES)
    for _ in range(n_iter):
        Q = R[:, None] + GAMMA * (transition_probs @ V)  # (N_STATES, N_ACTIONS)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = Q.argmax(axis=1)  # 각 state에서 최적 action
    return policy

# ============================================================
# 6. 특정 정책의 feature expectation을 시뮬레이션으로 추정
# ============================================================
def simulate_feature_expectation(policy, n_episodes=200, episode_len=30, start_states=None):
    if start_states is None:
        start_states = np.random.choice(N_STATES, n_episodes)
    total = np.zeros(N_FEATURES)
    for ep in range(n_episodes):
        s = start_states[ep % len(start_states)]
        discount = 1.0
        for t in range(episode_len):
            total += discount * cluster_features[s]
            a = policy[s]
            s = np.random.choice(N_STATES, p=transition_probs[s, a])
            discount *= GAMMA
    return total / n_episodes

# ============================================================
# 7. Projection 알고리즘 (Abbeel & Ng, 2004)
# ============================================================
def feature_expectation_matching(mu_expert, n_iterations=15):
    # 초기 랜덤 정책의 feature expectation
    w = np.random.randn(N_FEATURES)
    w /= np.linalg.norm(w)
    policy = value_iteration(w)
    mu_policies = [simulate_feature_expectation(policy)]
    mu_bar = mu_policies[0]

    for i in range(n_iterations):
        # projection step
        diff = mu_expert - mu_bar
        w = diff / np.linalg.norm(diff)

        policy = value_iteration(w)
        mu_pi = simulate_feature_expectation(policy)
        mu_policies.append(mu_pi)

        # mu_bar 업데이트 (projection onto line)
        num = (mu_pi - mu_bar) @ (mu_expert - mu_bar)
        denom = (mu_pi - mu_bar) @ (mu_pi - mu_bar)
        mu_bar = mu_bar + (num / denom) * (mu_pi - mu_bar)

        margin = np.linalg.norm(mu_expert - mu_bar)
        print(f"iter {i}: margin = {margin:.4f}, w = {w}")

    return w

w_final = feature_expectation_matching(mu_expert)
print("\n최종 추정된 가중치 (w):", dict(zip(raw_state_cols, w_final)))