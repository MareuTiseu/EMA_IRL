import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


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
sleep_action_cols = ["totalSleep_cat", "sleep_onset_cat"]
df[[f"{c}_aligned" for c in sleep_action_cols]] = (
    df.groupby(id_col)[sleep_action_cols].shift(-1)
)
action_cols = ["step_cat", "norm_ent_cat", "totalSleep_cat_aligned", "sleep_onset_cat_aligned"]


# ============================================================
# 1. Reward 계산 (t+1 - t)
# ============================================================
reward_vars = ['feeling', 'appetite', 'feelingOfSleep', 'generally', '스트레스', '우울', '불안']
reward_direction = {
    'feeling': 1,
    'appetite': 1,
    'feelingOfSleep': 1,
    'generally': 1,
    '스트레스': -1,
    '우울': -1,
    '불안': -1,
}

df[[f"{c}_next" for c in reward_vars]] = (
    df.groupby(id_col)[reward_vars].shift(-1)
)
for c in reward_vars:
    diff = df[f"{c}_next"] - df[c]
    df[f"{c}_reward"] = diff * reward_direction[c]

reward_cols = [f"{c}_reward" for c in reward_vars]
df["total_reward"] = df[reward_cols].sum(axis=1)


# ============================================================
# 2. covariate(intervene 등) + action dummy 준비
# ============================================================
covariate_cols = ["intervene"]

# action은 카테고리별로 각각 dummy화 (36개 조합 대신, 우선 개별 변수로 시작)
df = df.dropna(subset=action_cols)
action_dummies = pd.get_dummies(df[action_cols].astype("Int64").astype(str),
                                  prefix=action_cols, drop_first=True)

model_df = pd.concat([df[[id_col, "total_reward"] + covariate_cols], action_dummies], axis=1)
model_df = model_df.dropna(subset=["total_reward"])  # 마지막 시점(NaN reward) 제외


# ============================================================
# 3. 전체 인구 모델 (fixed effect)
# ============================================================
predictor_cols = covariate_cols + list(action_dummies.columns)
formula = "total_reward ~ " + " + ".join(predictor_cols)

pop_model = smf.ols(formula, data=model_df).fit()
print("=== 전체 인구 회귀 결과 ===")
print(pop_model.summary())


# ============================================================
# 4. Mixed effects 모델 (개인별 + 전체 동시에)
#    action_cols 중 하나만 우선 random slope로 (여러 개 넣으면 수렴 어려움)
# ============================================================
valid_ids = model_df.groupby(id_col).size()
valid_ids = valid_ids[valid_ids >= 10].index
model_df_filtered = model_df[model_df[id_col].isin(valid_ids)]

mixed_model = smf.mixedlm(
    formula,
    data=model_df_filtered,
    groups=model_df_filtered[id_col],
    re_formula="~" + action_dummies.columns[0]  # 예시로 첫 dummy만 random slope
)
mixed_result = mixed_model.fit()
print("\n=== Mixed Effects 모델 (전체 + 개인별) ===")
print(mixed_result.summary())

print("\n전체 집단 가중치 (fixed effects):")
print(mixed_result.fe_params)

print("\n개인별 편차 (random effects), 상위 5명:")
random_effects_df = pd.DataFrame(mixed_result.random_effects).T
print(random_effects_df.head())