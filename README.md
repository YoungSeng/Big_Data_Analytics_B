# Big_Data_Analytics_B

## 软件环境

1. Miniconda3-latest-Linux-x86_64
2. Pycharm 2021.2.1

## Model_1_vi_en.py

主要参考：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

主要贡献：数据清洗：将 Unicode 字符转换为 ASCII，
英文转化为小写只保留’a-z,.!?‘字符，
越南语删除’$#@%^&*()/_\-+|\\'\"0-9 ‘，

`.replace('&amp; amp ; quot ;', '"')
.replace('&apos;', "'").replace('&quot;', '"')
.replace('&amp;', '&').replace('&#91;', '[')
.replace('&#93;', ']').replace('& amp ;', '&')`

以及越南语token配置

模型配置：Attention，teacher_forcing_ratio=0.5，dropout=0.5，
MAX_LENGTH=300，n_iters=1330000

最终结果：在NVIDIA RTX 3080运行21h 3m 11s后，
Loss:5.3968，PPL:220.70095144082947，PLEU:0.0，结果很差

优点：
1. 代码易懂，入门简单
2. 无需依赖其他库

缺点：
1. **效果不好**
2. 没有考虑’unk‘的情况
3. 手动构造字典，没有对低词频的词语进行过滤
4. 没有batch

增加’unk‘的情况，修改一些参数的结果如下。

训练损失曲线如下：

![image](C:\Users\HCSI\PycharmProjects\pythonProject1\NMT\Model_1_vi_en_loss.png)

## Model_2_vi_en.py

主要参考：https://pytorch.org/tutorials/beginner/torchtext_translation.html

考虑换一种模型

主要贡献：数据集的处理，词汇表Vocab的生成，根据经验选择batch

模型配置：torchtext库，增加batch，Global attention，batch=3，epoch=10

最终结果：在NVIDIA RTX 3080运行11h 40m后（A100上运行45m），
Test Loss: 4.778，Test PPL: 118.853，结果很差

[comment]: <> (>'Ông qua đời , bị lịch sử quật ngã .')

[comment]: <> (truth:He died broken by history .)

[comment]: <> (Microsoft Translator:He died, destroyed by history.)

[comment]: <> (Model_1_vi_en:he was through history , history . <EOS>)

[comment]: <> (Model_2_vi_en:and he was the . .)

## Model_2_en_de.py

同上，采用europarl-v7.de-en(1920201对)，min_freq=30（英语：20950，德语：36141），BATCH_SIZE = 96，
80%作为训练集（1782890对），15%作为验证集，5%作为测试集，在A100上26h40min结果
Val. Loss: 4.841 |  Val. PPL: 126.571| Test Loss: 4.834 | Test PPL: 125.668 |，与越南语的类似，
PPL在120左右，但是对单个句子翻译效果感觉没有之前好，可能是词汇量变多了。

## Model_2.1_en_de_Multi30k.py

主要参考：https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb

主要贡献：发现尽管在Test PPL比较好，
但对于一般句子（以上的五句）的词语原训练集中并没有出现，都是<unk> ，
模型效果确实与数据集有关系。

模型配置：Packed Padded Sequences, Masking

运行结果：在A100上训练5min，Test Loss: 3.187，Test PPL:  24.207，BLEU score = 29.20

```
src = ['ein', 'schwarzer', 'hund', 'und', 'ein', 'gefleckter', 'hund', 'kämpfen', '.']
trg = ['a', 'black', 'dog', 'and', 'a', 'spotted', 'dog', 'are', 'fighting']
predicted trg = ['a', 'black', 'dog', 'and', 'a', 'spotted', 'dog', 'fighting', '.', '<eos>']
```

注意力矩阵：

![image](C:\Users\HCSI\PycharmProjects\pythonProject1\NMT\Attention Matrix.png)

## Model_3_vi_en

参考：https://github.com/tensorflow/nmt/tree/tf-1.4

模型配置：Global attention(luong)，num_layers=2，num_units=128，dropout=0.2

运行结果：在RTX 2080 Ti上训练1h 34min，训练集 ppl 14.51，dev ppl 15.10, dev bleu 14.7, test ppl 13.94, test bleu 16.5

```
>'Câu chuyện này chưa kết thúc .' 
truth:This is not a finished story .
Microsoft Translator:This story is not over.
Model_1_vi_en:this is the not . . <EOS> 
Model_2_vi_en:this story is .
Model_3_vi_en:This story isn &apos;t going to end .

>'Ông là ông của tôi .'
truth:He is my grandfather .
Microsoft Translator:You're my grandfather.
Model_1_vi_en:he was my . <EOS>
Model_2_vi_en:he my grandfather .
Model_3_vi_en:He was my grandfather .

>'Tôi chưa bao giờ gặp ông ngoài đời .'
truth:I never knew him in real life .
Microsoft Translator:I've never met you in real life.
Model_1_vi_en:i never never met him meet him . <EOS>
Model_2_vi_en:i never have the outside .
Model_3_vi_en:I never met him outside of life .
```
可以发现，用tensorflow的这个模型翻译效果最好
