## future-consumption-prediction
# 一、数据 
1.训练集数据：
  训练集的数据为空调的每小时的电耗，一共需要持续一个月的训练数据
  
2.测试集数据：
  测试集的数据每个单独为空调每小时的电耗，持续时间为训练集之后几天的训练数据，或者后几个月中随机抽取几天的电耗
  
3.输入的数据：
  输入的数据为连续的前m个小时的电耗
  输出的数据为后连续的n个小时分别的电耗，先假设n为1

# 二、算法模型
模型：
  RNN(LSTM)(GRU备选）

