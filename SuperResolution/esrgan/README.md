# ESRGAN

pytorchによるESRGANの実装プログラムです（論文とは少し実装方法が異なっているかもしれませんが…）。データセットはSTL10を用いています。最初にl1ノルムを用いてプレトレインします。l1ノルムにしたのは、Generatorの学習が安定したからです。次にESRGANによって超解像を行います。損失や評価値(psnr, ssim)はすべてtensorboardを用いて可視化されます。また、tensorboardから".csv"ファイルが取得できるので、結果として利用してください。

## Requirement
```
torch
torchvision
scikit-image
Pillow
tensorboard
tqdm
```

## Usage
以下のコマンドで実行できます。メモリが足りない場合、コマンドラインからバッチサイズを変更してください。 
>python3 main.py

tensorboardの起動は以下のコマンドで可能です。
>tensorboard --logdir=runs

## Reference
[1] Xintao Wang et.al. "ESRGAN: Enhanced Super-Resolution
Generative Adversarial Networks"  
arXiv:1809.00219v2 17 Sep 2018  

[2]【Intern CV Report】超解像の歴史探訪 -SRGAN編-  
https://buildersbox.corp-sansan.com/entry/2019/04/29/110000
  
[3] SRGANを用いた超解像
https://medium.com/@crosssceneofwindff/srgan%E3%82%92%E7%94%A8%E3%81%84%E3%81%9F%E8%B6%85%E8%A7%A3%E5%83%8F-cf7fac787729  
