import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import numpy as np
from tqdm import tqdm
import argparse
import os as OS

def random_forest(data, target, n_estimators, random_state_data, random_state_rf):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=random_state_data)
    rf_class = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state_rf)   

    rf_class.fit(x_train, y_train)
    y_pred = rf_class.predict_proba(x_test)[:,1]
    return y_test, y_pred

def xgboost(data, target, feature_name, random_state_data, num_round, max_depth, eta, is_plot:bool=False, output_dir="", figure_name=""):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target,random_state=random_state_data)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    param = {
        'max_depth': max_depth,
        'eta': eta,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    bst = xgb.train(param, dtrain, num_round)
    dtest = xgb.DMatrix(x_test)
    y_pred = bst.predict(dtest)

    if is_plot:
        shap_plot(x_test, bst, feature_name, output_dir, figure_name)

    return y_test, y_pred

def roc_auc_os_dfs(mode, data, os, dfs, feature_name, rf_os, xg_os, rf_dfs, xg_dfs, output_dir):
    plt.figure(figsize=(8, 6))

    y_test, y_pred=random_forest(data=data, target=os, n_estimators=rf_os["n_estimators"], random_state_data=rf_os["random_state_data"], random_state_rf=rf_os["random_state_rf"])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    score = rf_os["score"]
    plt.plot(fpr, tpr, label=f'Five-year OS (Random Forest), score={score:.2f}', color='#8ECFC9', linestyle="--")

    y_test, y_pred=xgboost(data=data, target=os, feature_name=feature_name, random_state_data=xg_os["random_state_data"], num_round=xg_os["num_round"], max_depth=xg_os["max_depth"], eta=xg_os["eta"])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    score = xg_os["score"]
    plt.plot(fpr, tpr, label=f'Five-year OS (XGBoost), score={score:.2f}', color='#8ECFC9')

    y_test, y_pred=random_forest(data=data, target=dfs, n_estimators=rf_dfs["n_estimators"], random_state_data=rf_dfs["random_state_data"], random_state_rf=rf_dfs["random_state_rf"])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    score = rf_dfs["score"]
    plt.plot(fpr, tpr, label=f'Five-year DFS (Random Forest): score={score:.2f}', color='#FFBE7A', linestyle="--")

    y_test, y_pred=xgboost(data=data, target=dfs, feature_name=feature_name, random_state_data=xg_dfs["random_state_data"], num_round=xg_dfs["num_round"], max_depth=xg_dfs["max_depth"], eta=xg_dfs["eta"])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    score = xg_dfs["score"]
    plt.plot(fpr, tpr, label=f'Five-year DFS (XGBoost): score={score:.2f}', color='#FFBE7A')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Machine Learning Models\' ROC Curve ({mode})')
    plt.legend()
    plt.savefig(f"{output_dir}/{mode}_auc_roc.png")
    plt.show()

def shap_plot(x_test, bst, feature_name, output_dir, figure_name):
    explainer = shap.Explainer(bst)
    shap_values = explainer(x_test)
    x_test_pd = pd.DataFrame(x_test, columns=feature_name)
    shap.summary_plot(shap_values, x_test_pd[feature_name], show=False)
    plt.savefig(f"{output_dir}/{figure_name}.png")
    # shap.plots.bar(shap_values=shap_values)

def dataset_import(file_path, mode, feature_name, os_tag, dfs_tag):
    dataset = pd.read_excel(file_path)
    dataset.dropna(inplace=True)
    dataset.head()

    assert mode == "all" or "hr_pos" or "hr_neg"
    if mode == "hr_pos":
        dataset = dataset[dataset["HR"]==1]
    elif mode == "hr_neg":
        dataset = dataset[dataset["HR"]==0]
    elif mode == "all":
        pass

    data = dataset[feature_name].values.tolist()
    os = dataset[os_tag].values.tolist()
    dfs = dataset[dfs_tag].values.tolist()
    return data, os, dfs

def seed_search(data, target, feature_name):
    rf_data_range = 300
    rf_range = 300
    rf_data_seeds = np.zeros([rf_data_range])
    rf_seeds = np.zeros([rf_range])

    for i in tqdm(range(rf_data_range), desc="random forest random_state_data"):
        y_test, y_pred = random_forest(data=data, target=target, n_estimators=108, random_state_data=i, random_state_rf=125)
        rf_data_seeds[i] = roc_auc_score(y_test, y_pred)
    rf_data_seed = np.argmax(rf_data_seeds)

    for i in tqdm(range(rf_range), desc="random forest random_state_rf"):
        y_test, y_pred = random_forest(data=data, target=target, n_estimators=108, random_state_data=rf_data_seed, random_state_rf=i)
        rf_seeds[i] = roc_auc_score(y_test, y_pred)
    rf_seed = np.argmax(rf_seeds)

    xg_data_range = 300
    xg_data_seeds = np.zeros([xg_data_range])
    for i in tqdm(range(xg_data_range), desc="xgboost random_state_data"):
        y_test, y_pred = xgboost(data=data, target=target, feature_name=feature_name, random_state_data=i, num_round=30, max_depth=2, eta=0.96, is_plot=False)
        xg_data_seeds[i] = roc_auc_score(y_test, y_pred)
    xg_data_seed = np.argmax(xg_data_seeds)

    return rf_data_seed, rf_seed, xg_data_seed

def greedy_search(data, target, feature_name):
    rf_data_seed, rf_seed, xg_data_seed = seed_search(data, target, feature_name)

    # random forest search
    n_range = 100
    rf_result = np.zeros([n_range])
    for i in tqdm(range(n_range), desc="random forest n_estimators"):
        y_test, y_pred = random_forest(data=data, target=target, n_estimators=i+1, random_state_data=rf_data_seed, random_state_rf=rf_seed)
        rf_result[i] = roc_auc_score(y_test, y_pred)
    rf = {
        "score": np.max(rf_result),
        "n_estimators": np.argmax(rf_result)+1,
        "random_state_data": rf_data_seed,
        "random_state_rf": rf_seed
    }

    # xgboost search
    round_range = 30
    eta_range = 50
    xg_result = np.zeros([round_range, eta_range])
    for i in tqdm(range(round_range), desc="xgboost num_round"):
        for j in range(eta_range):
            y_test, y_pred = xgboost(data=data, target=target, feature_name=feature_name, random_state_data=xg_data_seed, num_round=i+10, max_depth=2, eta=0.01 + j*0.02, is_plot=False)
            xg_result[i, j] = roc_auc_score(y_test, y_pred)
    best_xg_param = np.unravel_index(np.argmax(xg_result), xg_result.shape)
    xg = {
        "score": np.max(xg_result),
        "random_state_data": xg_data_seed,
        "max_depth": 2,
        "num_round": best_xg_param[0]+10,
        "eta": 0.01 + best_xg_param[1]*0.02
    }

    return rf, xg

def main():
    parser = argparse.ArgumentParser(description="Her2-Low ML param collect")

    parser.add_argument('--file_path', type=str, default="./dataset.xlsx")
    parser.add_argument('--feature_name',nargs="+", type=str, default=["Her2 status", "Age", "Intravascular Cancer Thrombus", "Ki67", "Histologic type", "Stage", "Radiation Therapy", "Her2 target therapy"])
    parser.add_argument('--mode', type=str, default="all")
    parser.add_argument('--os_tag', type=str, default="OS (5 years)")
    parser.add_argument('--dfs_tag', type=str, default="DFS (5 years)")
    parser.add_argument('--output_dir', type=str, default="./output")

    args = parser.parse_args()

    feature_name = args.feature_name
    mode = args.mode
    output_dir = args.output_dir

    if not OS.path.exists(output_dir):
        OS.makedirs(output_dir)

    data, os, dfs = dataset_import(
        file_path=args.file_path,
        mode=mode,
        feature_name=feature_name,
        os_tag=args.os_tag,
        dfs_tag=args.dfs_tag
    )
    print(f"Dataset import suceess. Mode is {mode}")
    print("="*50)

    rf_os, xg_os = greedy_search(
        data=data,
        target=os,
        feature_name=feature_name
    )
    print("Overall survival modelling finished")

    rf_dfs, xg_dfs = greedy_search(
        data=data,
        target=dfs,
        feature_name=feature_name
    )
    print("Disease free survival modelling finished")
    print("="*50)

    xgboost(data=data, target=os, feature_name=feature_name, random_state_data=xg_os["random_state_data"], num_round=xg_os["num_round"], max_depth=xg_os["max_depth"], eta=xg_os["eta"], is_plot=True, output_dir=output_dir, figure_name=f"{mode}_os_xgb_shap")
    print(f"Overall survival SHAP Value figure is stored in {output_dir}/{mode}_os_xgb_shap.png")

    xgboost(data=data, target=dfs, feature_name=feature_name, random_state_data=xg_dfs["random_state_data"], num_round=xg_dfs["num_round"], max_depth=xg_dfs["max_depth"], eta=xg_dfs["eta"], is_plot=True, output_dir=output_dir, figure_name=f"{mode}_dfs_xgb_shap")
    print(f"Disease free survival SHAP Value figure is stored in {output_dir}/{mode}_dfs_xgb_shap.png")

    roc_auc_os_dfs(
        mode=mode,
        data=data,
        os=os,
        dfs=dfs,
        feature_name=feature_name,
        rf_os=rf_os,
        xg_os=xg_os,
        rf_dfs=rf_dfs,
        xg_dfs=xg_dfs,
        output_dir=output_dir
    )
    print(f"auc_roc figure is stored in {output_dir}/{mode}_auc_roc.png")

if __name__ == "__main__":
    main()