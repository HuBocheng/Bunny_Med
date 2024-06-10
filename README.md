# Bunny_Med
Migrate the latest Small VLM Bunny to the medical domain of Surgical VQA and fine-tune it.



## 探索一：数据迁移

利用原先Bunny多轮训练的模板，使用Med-VQA19、Endovis18和Cholec80数据集对Bunny进行训练，再在个数据集的测试集上测试。结果如下

（对原始surgical vqa的lables做了一些修改，去重统计了每个子类的所有）

### Med-VQA19

baseline（为经过训练的bunny）：Acc：45/500

全量训练，小类测试（模型泛化性能好）

| 数据集            | ACC    | Recall（Macro，Micro） | Fscore（Macro，Micro） |
| ----------------- | ------ | ---------------------- | -------------------------------- |
| **Med-VQA (C1)**  | 0.42   |                        |                                  |
| **Med-VQA (C2)**  | 0.52 | 0.1130,0.52 | 0.0805,0.5200 |
| **Med-VQA (C3)**  | 0.464 | 0.1667,0.4828 | 0.1609,0.4828 |
| **Med-VQA (ALL)** | 0.3720 |                        |                                  |

小类训练，小类测试	

| 数据集            | ACC    | Recall（Macro，Micro） | Fscore（Macro，Micro） |
| ----------------- | ------ | ---------------------- | ---------------------- |
| **Med-VQA (C1)**  | 0.42   |                        |                        |
| **Med-VQA (C2)**  |        |                        |                        |
| **Med-VQA (C3)**  |        |                        |                        |
| **Med-VQA (ALL)** | 0.4560 | 0.1710,0.4617          | 0.1250,0.4617          |



### Endovis18
baseline（为经过训练的bunny）：Acc：0.1419
- 不带提示词Acc:27%
- 带提示词Acc:33%





**其他结论：在训练中不加提示词无法使模型效果得到很高的提升**


# 开始

使用Med VQA19进行训练和评估

```bash
cd Bunny
# 进行训练
./script/train/finetune_lora_surgical2.sh 
# 进行评估
./script/eval/full/medVQA.sh
```

使用EndoVis18进行训练和评估

```bash
cd Bunny
# 进行训练
./script/train/finetune_lora_surgical.sh 
# 进行评估
./script/eval/full/Endovis.sh


