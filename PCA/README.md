# Principal Component Analysis
PCAの実装をまとめました。Python3の環境が整ったLinuxサーバで動かす際は以下のコマンドで実行できますが、医用画像はデータの次元が非常に高いため、全ての実行には5時間ほど要すると思います。画像データは"./data/"ディレクトリ配下においてください。

> bash exe.sh

（注）これらのファイルはLinuxサーバで動かすことを想定して作りましたが、Linuxサーバに接続できない場合は、requirement.txtを参照し、必要なモジュールを取り入れ、"exe.sh"を参照し、その処理内容の順にプログラムを実行すればできると思います。

### pca_sample.py
irisデータセットを用いて、PCAの原理を簡単に可視化します。

### model.py
PCAの処理プログラムが書かれています。debug関数で簡易な配列を特異値分解を用いてPCAにかけます。

### utils.py
画像データの前処理・後処理を行う関数をおいています。

### pca.py
PCAのプログラムを実行します。入力データは肝臓のレベルセット（ラベルデータを符号付き距離変換したもの）画像であるため、"./origin/"ディレクトリ、ラベル画像を保存するようにしています。また、訓練、テスト、任意形状のPCA再構成画像は"./output/"ディレクトリ内に".mhd"ファイルで出力されるので "3D-Slicer[https://www.slicer.org/]" 等のアプリを用いて見てみてください。**私のスライドにある肝臓ラベル画像は"Pluto"というアプリで作成したものですが、正直使い方を忘れてしまったので、そのままコピペして大丈夫です（去年もそうした）。**

訓練時は20枚のデータをすべて用い、潜在空間に投影します。投影点を表す画像ファイルを"./images/latent.png"に出力します。また、寄与率・累積寄与率についてもデータ及び図のそれぞれを".csv", ".png"として"./npy/"ディレクトリ内に、平均、固有値、固有ベクトル".npy"ファイルとともに出力されます。

テスト時はデータ分割は2fold-cross-validationで20枚のデータを10枚の訓練データと10枚のテストデータに分けて行われます。結果は主成分軸1~9まで変化させたときのそれぞれを出力します。

任意形状もテスト時と同様、主成分軸1~9まで変化させたときのそれぞれの結果を100枚出力します。任意形状数を増加させたい場合は変数"n_shape"を変えてください。

**（注）形状変化はあまりうまく行きません。プログラムの問題かと思いますが、去年の結果画像（パワポ参照）を使うのが賢明でしょう。崩壊した任意形状が発生します。ただ、うまく行っているものもあるので、"3D-Slicer"等で「潜在変数の変化に伴う任意形状の変化」を確認してみましょう。**

### eval.py
PCAの結果をjaccard-indexを用いた"Generalization"と"Specificity"で評価します。計算速度の向上のため、並列計算に特化した"pytorch"を用いてjaccard-indexを計算させますが、"Specificity"の任意形状の個数が多くなればなるほど、計算に時間がかかります。計算結果は"./csv/"に出力されます。

### make_graph.py
"Generalization"と"Specificity"の結果を用いてそれぞれの関係性を示すグラフを作成してくれます。結果は
"./images/"フォルダに出力します。


## 参考URL
https://orizuru.io/blog/machine-learning/pca_kaisetsu/    
"主成分分析とは何なのか、とにかく全力でわかりやすく解説する"  
