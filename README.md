# Word2Vector-RNN-SongCi    

### 1简介    
基于RNN和word2vector，以《全宋词》为训练数据，训练一个人工智能写词机   

### 2数据集     
全宋词《QuanSongCi.txt》  

### 3实现过程    
- 使用embedding.py实现word2vector，把高维空间的词嵌入到低维向量空间。  
1）预处理数据，建立词汇表字典记录频数最高的5000个词，生成dictionary.json和reverse_dictionary.json  
2）建立skip-gram模型，训练得到词向量（权重矩阵）  
3）可视化词向量效果（词频最高的500个词分布图，距离较近表示词义接近）  
![](tsne.png 'tsne image')     

- 使用长短时记忆网络（LSTM）构建RNN网络结构，训练并测试结果  

![](val_1.png 'val_1.png')      
![](val_2.png 'val_2.png')      
![](val_3.png 'val_3.png')      
  