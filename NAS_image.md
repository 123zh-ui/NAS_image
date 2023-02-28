# NAS_image
## 文件的结构
1. data文件是用于存储训练所使用的数据集，目前mini——Imagenet的数据集存放在data文件下的images中；具体下载地址在\ista_nas_single\data\images\README.md中有介绍。
2. onnx用于存储NAS搜索完成后的目标网络架构，这里我们将epoch设置为1，为了快速获得搜索后的结构。
3. recovery文件下存放着一个求解器的代码，目的是用于将原始结构系数矩阵进行转换。
4. search文件下存放的是基本的网络结构生成的细节。
     -  models文件中涉及的代码是结构系数矩阵如何搜索结构的
         - model.py评估网络架构的时候使用
         - model_search.py是搜索阶段fullnet如何初始化的，如何搜索的。
     - inner_trainer.py是搜索过程中训练和验证的具体代码
     - operations.py指的是fullnet中所有涉及的操作结构，以及forward
     - optimizer.py指的是选取的优化器，这里用的是Adam
5. train_search_single.py搜索过程中的参数加载，主函数
6. trainer.py指的是整个训练，验证，投影转换，恢复，评估是否结构固定的代码

## 所需环境

- Python >= 3.7(我使用的是3.9)
- PyTorch >= 1.1 and torchvision（ 1.12.1+cu116和0.13.1+cu116）
- CVXPY（我使用1.2.1）
- Mosek（我使用10.0.26）
- Please have a licence file `mosek.lic` （可能不需要） Please have a licence file `mosek.lic` following [this page](https://docs.mosek.com/9.2/licensing/quickstart.html#i-don-t-have-a-license-file-yet), and place this file in the directory `$HOME/mosek/mosek.lic`.

## 使用用法

1. 将整个仓库拉取到本地；
2. 在ista_nas_single\data\images\中下载并解压数据集。
3. 配置环境
4. 直接执行train_search_single.py，参数已经设置好了。

