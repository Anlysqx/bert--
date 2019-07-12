# bert
### CCKS 2019 面向金融领域的事件主体抽取
### tensorflow 版本代码
### 方法是 在bert 的输出后面 接两个 dense 成 [batch_size,sentence_len,1] 然后 tf.squeeze(,axis=-1) 
### 这两个dense 分别用来预测 实体在原句中的起始下标和终止下标
### bert-finetune 直接 python run little3.py 即可
### 我的运行环境是 titan V 
