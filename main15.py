# 必要なライブラリをインポート
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 学習用データ(train.csv)を読み込み
train_data = pd.read_csv('train.csv')

# 特徴量と目的変数に分ける
X = train_data.drop(['id', 'target'], axis=1)
y = train_data['target']

# 訓練用データと検証用データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 学習用データおよび検証用データの特徴量をスケーリングする
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ハイパーパラメータの探索範囲を定義
param_grid = {
    'rf__n_estimators': [150, 200],
    'rf__max_depth': [15, 20],
    'rf__max_features': ['sqrt'],
    'rf__min_samples_split': [5, 10],
    'rf__min_samples_leaf': [4, 6],
    'rf__bootstrap': [True],
    'rf__criterion': ['gini'],
    'rf__min_impurity_decrease': [0.0],
    'gb__learning_rate': [0.1],
    'gb__max_iter': [200, 300],
    'gb__max_depth': [10, 15, 20],
    'gb__max_leaf_nodes': [None],
    'gb__min_samples_leaf': [6, 8, 10],
    'gb__l2_regularization': [0.0]
}

# ランダムフォレストモデルを作成
rf_model = RandomForestClassifier(random_state=42)

# GBRFモデルを作成
gbrf_model = HistGradientBoostingClassifier(random_state=42)

# GridSearchCV を使用してハイパーパラメータチューニングを実行
estimators = [('rf', rf_model), ('gb',gbrf_model)]
voting_model = VotingClassifier(estimators=estimators, voting='soft')
grid_search = GridSearchCV(
voting_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# ベストのハイパーパラメータを表示
print("Best Parameters: ", grid_search.best_params_)

# 最適なハイパーパラメータでモデルを構築し、訓練データで学習
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# 検証用データを使って予測
y_pred = grid_search.predict(X_val_scaled)

# 精度を計算して表示
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)

# 学習済みモデルを保存する
joblib.dump(grid_search.best_estimator_, 'best_model11.pkl')

# テスト用データ(test.csv)を読み込み
test_data = pd.read_csv('test.csv')

# 特徴量を取得する
X_test = test_data.drop(['id'], axis=1)

# テストデータの特徴量をスケーリング
X_test_scaled = scaler.transform(X_test)

# 学習済みモデルを読み込む
model = joblib.load('best_model11.pkl')

# 予測を行う
y_test = model.predict(X_test_scaled)

# 予測結果をファイルに書き出す
output = pd.DataFrame({'id': test_data['id'], 'target': y_test})
output.to_csv('submission24.csv', index=False)