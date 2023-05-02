# 必要なライブラリをインポート
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 学習用データ(train.csv)を読み込み
train_data = pd.read_csv('train.csv')

# 外れ値を制限する関数を定義
def clip_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    threshold_min = Q1 - 1.5 * IQR
    threshold_max = Q3 - 1.5 * IQR
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレストモデルを作成し、訓練用データで学習
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 検証用データを使って予測
y_pred = model.predict(X_val)

# 学習済みモデルを保存する
joblib.dump(model, 'model.pkl')

# 精度を計算して表示
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)

# テスト用データ(test.csv)を読み込み
test_data = pd.read_csv('test.csv')

# 外れ値を制限する列名を指定して、clip_outliers()関数を適用
for column_name in column_names:
    test_data = clip_outliers(test_data, column_name)

# 特徴量を取得する
X_test = test_data.drop(['id'], axis=1)

# 学習済みモデルを読み込む
model = joblib.load('model.pkl')

# 予測を行う
y_test = model.predict(X_test)

# 予測結果をファイルに書き出す
output = pd.DataFrame({'id': test_data['id'], 'target': y_test})
output.to_csv('submission3.csv', index=False)