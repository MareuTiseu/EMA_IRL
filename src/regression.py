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


phq_df = pd.read_csv("./data/phq_subgroup.csv")
df = pd.merge(df, phq_df, on="id", how="left")



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
# 4. Subgroup analysis
# ============================================================
somatic_flag = (phq_df["somantic"] > phq_df["cognitive"]).astype(int)
subgroup_flags = pd.DataFrame({
    id_col: phq_df["id"],
    "somatic_higher": somatic_flag,
    "depression_diagnosis": phq_df["depression_diagnosis"].astype(bool),
    "on_medication": phq_df["medication_current"].notna(),
})

model_df = pd.merge(model_df, subgroup_flags, on=id_col, how="left")


def run_subgroup_regression(data, group_col, group_value, predictor_cols):
    """group_col == group_value 인 subset에 대해 회귀를 돌리고 결과를 반환"""
    sub = data[data[group_col] == group_value]
    formula = "total_reward ~ " + " + ".join(predictor_cols)
    model = smf.ols(formula, data=sub).fit()
    return model


def compare_subgroups(data, group_col, predictor_cols, labels=("Group 0", "Group 1")):
    """group_col 기준 True/False(또는 0/1) 두 그룹의 회귀 결과를 비교"""
    model_0 = run_subgroup_regression(data, group_col, False, predictor_cols)
    model_1 = run_subgroup_regression(data, group_col, True, predictor_cols)

    print(f"\n=== {group_col} 기준 비교 ===")
    print(f"\n--- {labels[0]} (n={int(model_0.nobs)}) ---")
    print(model_0.summary())
    print(f"\n--- {labels[1]} (n={int(model_1.nobs)}) ---")
    print(model_1.summary())

    # 계수만 뽑아서 나란히 비교하는 테이블
    coef_compare = pd.DataFrame({
        labels[0]: model_0.params,
        f"{labels[0]}_pvalue": model_0.pvalues,
        labels[1]: model_1.params,
        f"{labels[1]}_pvalue": model_1.pvalues,
    })
    print(f"\n--- {group_col} 계수 비교 테이블 ---")
    print(coef_compare)

    return model_0, model_1, coef_compare

# ------------------------------------------------------------
# 4-1. Somatic vs Cognitive (somatic이 더 높은 사람 vs 아닌 사람)
# ------------------------------------------------------------
somatic_model_low, somatic_model_high, somatic_coef = compare_subgroups(
    model_df, "somatic_higher", predictor_cols,
    labels=("Cognitive higher/equal", "Somatic higher")
)

# ------------------------------------------------------------
# 4-2. 우울 진단 여부
# ------------------------------------------------------------
dep_model_no, dep_model_yes, dep_coef = compare_subgroups(
    model_df, "depression_diagnosis", predictor_cols,
    labels=("No diagnosis", "Depression diagnosis")
)

# ------------------------------------------------------------
# 4-3. 약물 복용 여부
# ------------------------------------------------------------
med_model_no, med_model_yes, med_coef = compare_subgroups(
    model_df, "on_medication", predictor_cols,
    labels=("No medication", "On medication")
)



# ============================================================
# 5. Mixed effects 모델 (개인별 + 전체 동시에)
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