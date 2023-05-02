import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 学習用データ(train.csv)を読み込み
train_data = pd.read_csv('train.csv')

# 特徴量と目的変数に分ける
X = train_data.drop(['id', 'target'], axis=1)
y = train_data['target']

# 訓練用データと検証用データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

# 学習用データおよび検証用データの特徴量をスケーリングする
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# モデルを初期化する
models = {
    "Logistic Regression": LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
    "Linear SVM": LinearSVC(penalty='l1', dual=False, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# 各モデルのハイパーパラメータグリッドを定義する
param_grids = {
    "Logistic Regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    },
    "Linear SVM": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    },
    "Gradient Boosting": {
        "n_estimators": [10, 50, 100, 150, 200, 250, 300],
        "learning_rate": [0.01, 0.1, 1],
        "max_depth": [3, 5, 7, 9, 11, 13],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
}

# 各モデルをグリッドサーチで学習する
model_accuracies = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    model_accuracies[model_name] = accuracy
    print(f"Accuracy for {model_name}: {accuracy}\n")

# 最も精度の高いモデルを選択
best_model_name = max(model_accuracies, key=model_accuracies.get)

# ベストモデルを選択して再学習
best_model = models[best_model_name]
grid_search = GridSearchCV(best_model, param_grids[best_model_name], cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# テストデータを前処理
test_data = pd.read_csv("test.csv")

# 特徴量を取得
X_test = test_data.drop(["id"], axis=1)

# テストデータの特徴量をスケーリング
X_test_scaled = scaler.transform(X_test)

# 学習済みモデルで予測を実行
y_test = best_model.predict(X_test_scaled)

# 予測結果をファイルに書き出す
output = pd.DataFrame({"id": test_data["id"], "target": y_test})
output.to_csv("submission17.csv", index=False)

print(f"Best Model: {best_model_name}")