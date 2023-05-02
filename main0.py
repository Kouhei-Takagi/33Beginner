# 必要なライブラリをインポート
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 学習用データ(train.csv)を読み込み
train_data = pd.read_csv('train.csv')
#print(train_data.head())
#print(train_data.isnull().sum())

# 外れ値を除去する関数を定義
def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    threshold_min = Q1 - 1.5 * IQR
    threshold_max = Q3 + 1.5 * IQR
    df = df[(df[column_name] >= threshold_min) & (df[column_name] <= threshold_max)]
    return df

# 外れ値を除去する列名を指定して、remove_outliers()関数を適用
column_name = 'creatinine_phosphokinase'
train_data = remove_outliers(train_data, column_name)

# 外れ値を除去する列名を指定して、remove_outliers()関数を適用
column_name = 'serum_creatinine'
train_data = remove_outliers(train_data, column_name)

# 外れ値を除去する列名を指定して、remove_outliers()関数を適用
column_name = 'serum_sodium'
train_data = remove_outliers(train_data, column_name)
'''
# 目的変数と相関が高い上位N個の特徴量を抽出する
N = 6
corr = train_data.corr()
corr_abs = abs(corr)
corr_abs_sorted = corr_abs.nlargest(N+1, 'target')['target']
selected_features = corr_abs_sorted.index.tolist()[1:]

# 選択した特徴量を抽出する
train_data = train_data[selected_features]
'''
# 特徴量と目的変数に分ける
X = train_data.drop(['id', 'target'], axis=1)
y = train_data['target']

# 訓練用データと検証用データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ロジスティック回帰モデルを作成し、訓練用データで学習
model = LogisticRegression()
model.fit(X_train, y_train)

# 検証用データを使って予測
y_pred = model.predict(X_val)

# 精度を計算して表示
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)
