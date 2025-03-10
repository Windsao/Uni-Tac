Hi!

运行代码方式：
首先cd 到/scratch/mzh1800/t3目录下
source activate /projects/p32294/conda_env/newfort3
运行之前的代码：
python3 scripts/train_nn.py

新改的代码：
python3 scripts/train_nn_student.py

版本复原：

copy t3/configs/config_teacher copy.yaml  to t3/configs/config.yaml
copy t3/configs/network/pretrain1_mae copy 2.yaml to t3/configs/network/pretrain1_mae.yaml

此时运行的代码是 t3/scripts/train_nn.py 这里调用的是test.py，如果需要运行t3官方代码：解除注释line 3,8
test.py 无需gpu，可以在任何地方运行


基本全部设置都在config.yaml中修改，概率出现magic=False 报错，直接删除这行就好了（仅在原本T3官方代码中出现，是版本问题）
T3teacher代码的区别为，load 2个config，并且设置2个模型，冻结teacher模型，并且student模型按照原样training
t3/t3/models/t3.py 中line 15，25-32 可注释，出现为debug所用




