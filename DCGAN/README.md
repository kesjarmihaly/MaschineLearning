# DCGAN

DCGANのソースコードです．基本的にはPytorchのチュートリアルにあるコードを参照しているので，詳細は参照にあるサイトを見てくれるといいかもです．ただ，チュートリアルだとjupyter-noteでの実装を想定しているので，こちらでローカル環境での実装を想定したコードに書き換えています．

## Requirment

> torch  
> torchvision  
> tqdm  
> numpy  
> pandas  
> matplotlib  

## Usage
データセットはselebaデータセットを使います．[こちら](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)から"Align&Cropped Images.zip"をダウンロードしていただいて，展開．main.pyがあるディレクトリに"celeba"フォルダを作成し, 以下のように置いてください．
```
DCGAN   
       -> main.py  
       -> model.py   
       -> README.md  
       -> celeba   
             -> img_align_celeba  
                   -> 000001.jpg  
                   -> 000002.jpg  
                   -> ...
```
実行は以下のコードで可能です．

> python3 main.py

また，環境に応じてコマンドライン引数から各種パラメータも設定できます．

損失の推移はmatplotlibで可視化したデータも保存されますが，CSVファイルとしても出力されるようにしたので，いろいろ実験してみて，学習の安定性を議論してみてください．

## Reference
[1] Alec Radford, Luke Metz, Soumith Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", arXiv:1511.06434 (2015).  

[2] "DCGAN TUTORIAL"
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html



