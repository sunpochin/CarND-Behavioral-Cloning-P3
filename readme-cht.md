# 自駕車課程作業之「我終於也做了點 "Google 工程師會做的事情" 啦」
---

(是說，不管哪種工程師，都一定會的同一件事情是：寫出 bug. 所以寫程式前，要先想好怎麼 test & debug)


說明
----
這個影片是 udacity 自駕車課程其中一個作業，我目前做到的錄影：

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/O4reOzBoT5M/0.jpg)](https://www.youtube.com/watch?v=O4reOzBoT5M)

背景音樂： [Bendito Kejío: Blas Córdoba canta y Chano Domínguez](https://www.youtube.com/watch?v=8P2TBhCObsQ)

它是這篇文章的實做： 
["想學做無人車的工程師注意！Google 工程師教你從零開始學「無人駕駛技術」"](https://buzzorange.com/techorange/2017/06/19/self-drive-simulator-n-test/ ) 

這作業是用軟體模擬器，感覺有點像是在玩賽車遊戲，操作方法、原理在上面那篇文章都有講了。
我的 ConvNet 架構是用 nvidia 2016 出的一個無人車架構，我在心中叫它 [DiaNet.](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 西班牙文中 Dia == Day ，所以 DiaNet 可以翻譯為「天網」。 

如果讀者妳想要試試看用我的 training model 來自動駕駛模擬車，這是我這專案的 [github 連結](https://github.com/sunpochin/CarND-Behavioral-Cloning-P3)，
這是模擬器的 [下載連結](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)，
這裡是udacity 的預設 training set [連結](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)。

實做
---
1. (「Google 工程師」文章提到的) 觀察一： Angle 0 我用 ```np.random.random() ``` 過濾掉 99% , 現在想起來會不會太多了點，也許多留一些 Angle 0 會讓車子輪胎不會總是在轉。 
2. 觀察二：我的參數是給 0.25 .
3. 觀察三：「利用反轉」增加 training set: 我並沒有增加 training set, 而是 on the fly 的用 0.5 的機率來決定這張圖要不要反轉。我沒有增加 training set 的原因是，我想到的實做方法是要另外寫一段 code 在最前面，我就是目前還沒有寫到那段。
4. 觀察四：crop image & resize image, 也都做了，也都是在 keras model 裡面 on the fly 做。
5. 觀察五：Normalization 有做。
6. 架構用 nvidia net.
7. image preprocessing, 加上 ```fit_generator``` 技巧之後，我的筆電 GTX 950M 2.0 GiB 才可以裝得下 nvidia net.



Bug
---
1. 我犯了一個很大的錯誤是沒有做好足夠的 visualization, 也就少了一個 debug 的重要武器。 
這個 bug 的實做細節是我沒有注意到 opencv 讀圖檔時要注意 channel 的順序是否正確，少了一行 
```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) .
```

 	在寫了一支程式比對「實際餵給 keras 的圖檔」跟「自己想像中餵給 keras 的圖檔」後，才發現顏色錯了。這個問題導致車子在走過橋樑之後總是會偏到左邊。
2. 我的個人電腦是 windows 筆電，GPU 似乎不夠快，於是開了 AWS 來 train 的 model, 但 AWS train model 完之後 WinSCP 抓回來執行，得到一個錯誤訊息說我兩邊 keras 版本不同，一個是 2.0.6 另個是 2.0.4. 
	
	最後解決方法是：我在 AWS ubuntu 新開一個 anaconda env, 讓 AWS & windows 10 兩邊通通是 anaconda3, python 3.5.3, keras 2.0.4, tensorflow-gpu 這樣的環境。

後續
---
* 把 conv2d layer 盡量減少，不要用到五層來做同樣的事。
* 試試看 simulator 中第二個 track, 這要重新收集車子訓練時的影像。
  第一個 track 有課程提供的 default 資料，我因為想要減少變因，就使用了 default 資料。不過它好像在錄影時速度保持慢速，所以用這個 dataset train 出來的車子也是慢慢跑。


