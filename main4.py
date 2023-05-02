# 必要なライブラリをインポート
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 学習用データ(train.csv)を読み込み
train_data = pd.read_csv('train.csv')

# 外れ値を制限する関数を定義
def clip_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.10)
    Q3 = df[column_name].quantile(0.90)
    IQR = Q3 - Q1
    threshold_min = Q1 - 1.5 * IQR
    threshold_max = Q3 + 1.5 * IQR
    df[column_name] = df[column_name].apply(lambda x: threshold_min if x < threshold_min else (threshold_max if x > threshold_max else x))
    return df

# 外れ値を制限する列名を指定して、clip_outliers()関数を適用
column_names = ['creatinine_phosphokinase', 'ejection_fraction','platelets','serum_creatinine', 'serum_sodium']
for column_name in column_names:
    train_data = clip_outliers(train_data, column_name)

# 特徴量と目的変数に分ける
X = train_data.drop(['id', 'target'], axis=1)
y = train_data['target']

# 訓練用データと検証用データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

# ハイパーパラメータの探索範囲を定義
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [5, 10, 15, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# ランダムフォレストモデルを作成
model = RandomForestClassifier(random_state=42)

# GridSearchCV を使用してハイパーパラメータチューニングを実行
grid_search = GridSearchCV(
    model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# ベストのハイパーパラメータを表示
print("Best Parameters: ", grid_search.best_params_)

# 最適なハイパーパラメータでモデルを構築し、訓練データで学習
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 検証用データを使って予測
y_pred = grid_search.predict(X_val)

# 精度を計算して表示
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)

# 学習済みモデルを保存する
joblib.dump(grid_search.best_estimator_, 'best_model2.pkl')

# テスト用データ(test.csv)を読み込み
test_data = pd.read_csv('test.csv')

# 外れ値を制限する列名を指定して、clip_outliers()関数を適用
for column_name in column_names:
    test_data = clip_outliers(test_data, column_name)

# 特徴量を取得する
X_test = test_data.drop(['id'], axis=1)

# 学習済みモデルを読み込む
model = joblib.load('best_model2.pkl')

# 予測を行う
y_test = model.predict(X_test)

# 予測結果をファイルに書き出す
output = pd.DataFrame({'id': test_data['id'], 'target': y_test})
output.to_csv('submission9.csv', index=False)