# 我終於也做了點 "Google 工程師會做的事情" 啦～
---
說明
----
這個影片是 udacity 自駕車課程其中一個作業，我目前做到的錄影：

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/O4reOzBoT5M/0.jpg)](https://www.youtube.com/watch?v=O4reOzBoT5M)

背景音樂： [Bendito Kejío: Blas Córdoba canta y Chano Domínguez](https://www.youtube.com/watch?v=8P2TBhCObsQ)

它是這篇文章的實做： 
["想學做無人車的工程師注意！Google 工程師教你從零開始學「無人駕駛技術」"](https://buzzorange.com/techorange/2017/06/19/self-drive-simulator-n-test/ ) 

這作業是用軟體模擬器，感覺有點像是在玩賽車遊戲。
原理在上面那篇文章都有講了，我是用 nvidia 出的這個架構，我在心中叫它 [DiaNet](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ ) 

如果讀者妳想要試試看用我的 training model 來自動駕駛模擬車，這是我這專案的 [github 連結](https://github.com/sunpochin/CarND-Behavioral-Cloning-P3)，
這是模擬器的 [下載連結](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)，
這裡是udacity 的預設 training set [連結](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)。





Bug
---
我犯了一個很大的錯誤是沒有做好足夠的 visualization, 也就少了一個 debug 的重要武器。 
這個 bug 的實做細節是我沒有注意到 opencv 讀圖檔時要注意 channel 的順序是否正確，少了一行 
```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) .
```

在寫了一支程式比對「實際餵給 keras 的圖檔」跟「自己想像中餵給 keras 的圖檔」後，才發現顏色錯了。這個問題導致車子在走過橋樑之後總是會偏到左邊。

後續
---
* 把 conv2d layer 盡量減少，不要用到五層來做同樣的事。
* 試試看 simulator 中第二個 track, 這要重新收集車子訓練時的影像。
  第一個 track 有課程提供的 default 資料，我因為想要減少變因，就使用了 default 資料。不過它好像在錄影時速度保持慢速，所以用這個 dataset train 出來的車子也是慢慢跑。


