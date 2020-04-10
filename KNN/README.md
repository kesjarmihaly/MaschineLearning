# k-Nearest Neighbor
KNNの実装をまとめました。Python3の環境が整ったLinuxサーバで動かす際は基本的に以下のコマンドですべてのPythonファイルの実行がなされます。

> bash exe.sh

(注) 基本的にこのコマンドが最後まで実行されると、輪講の課題である実装と実験は終了します。

### knn.py
irisデータセットの4つの特徴量の内、"sepal length"と"sepal width"の2を用い、knnによってテストデータを推定します。kは1~120までを自動的に計算してくれます。結果は"csv"ディレクトリに".csv"として出力します。また、それぞれのkの値における"決定領域"を示す図を"images/desicion_region"ディレクトリに出力します。図における色は各クラス"setosa", "versicolor", "virginica"の決定領域を表しており、点はデータ点で黒丸がついているのはテストデータ点です。さらに、それぞれのkの値における"ヒートマップ"(識別器の予測に対して、どのくらい正解しているのかを示す図)を"images/heatmap"ディレクトリに出力します。ヒートマップにおける『0, 1, 2』はそれぞれ『"setosa, "versicolor", "virginica』クラスを表しています。

### make_gif.py
knn.pyによって出力された各kにおける決定領域の可視化画像を統合し、"gif"ディレクトリに".gif"ファイルとして出力します。

## 参考URL
"k-nn（k近傍法)クラス分類をPythonで実装してみよう！"  
 https://www.sejuku.net/blog/64355
