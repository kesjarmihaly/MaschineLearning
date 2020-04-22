# 白黒画像のカラー化
画像の自動着色はGANのもたらした最先端の画像処理と言っても過言ではありません。有名なところだと"Paint Chainer"でしょうか。今回はGANの中でもとりわけ有名なpix2pixというモデルを用いて白黒画像をカラー化します。学習が進むに連れ、画像に色がついてきている様子が観察できると思います。

# pix2pix
pix2pixはCondition GANと呼ばれるモデルの一種であり、Discriminatorに本物か偽物かを識別させる際、本物あるいは偽物画像とともに条件(condition)となる画像（今回の場合白黒画像）を入力する。Discriminatorには本来PatchGANが用いられますが、今回の実装には用いていません(参考文献にあるサイトをとにかく参考にしまくったため…)。本格的なpix2pixのプログラムを実装したい場合は、参考文献にある元論文サイトのプログラムを使ってください。GeneratorにはU-netが使われています。また、U-netだけだと、今ひとつGeneratorの損失が下がらなかったので、U-netのエンコーダのあとに、Residual Blockを追加したネットワークを組んだプログラムpix2pix_with_res.pyも用意しておいたので、モージュールのimportを変更して使ってみてください。やや着色が鮮やかになります。損失はtensorboardを用いて確認できます。

## Requirement
```
torch
torchvision
tensorboard
tqdm
```

## Usage
以下のコマンドで実行できます。メモリが足りない場合、utils.pyからバッチサイズを変更してください。 
>python3 main.py

tensorboardの起動は以下のコマンドで可能です。
>tensorboard --logdir=runs

## Reference
[1] P. Isola, J. Zhu, T. Zhou, A. A. Efros. Image-to-Image Translation with Conditional Adversarial Networks. CVPR, 2017.  
https://phillipi.github.io/pix2pix/

[2] pix2pixを1から実装して白黒画像をカラー化してみた（PyTorch）  
https://blog.shikoan.com/pytorch_pix2pix_colorization/