import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from rnn_model import RNNModel
from data_loader import load_data, preprocess_data

def get_batch(data, batch_size, episode):
    """对数据进行分批取样操作"""
    # 随机对data里面的样本进行抽取，生成随机抽取样本的索引
    # x_index = np.random.randint(low=0, high=data.shape[0]-1,size=batch_size)
    last_batch_end = episode*batch_size
    
    batch_start  = last_batch_end % (data.shape[0] - batch_size -1)  # 对取样的初始位置进行处理
    x_index = np.arange(batch_size) + batch_start # 不随机对样本的进行抽取
    y_index = x_index + 1
    if batch_start==0:
        print("new epoch")
    # 在所有样本中取出x和y
    x = data[x_index]
    y = data[y_index]
    return x, y

def plot_result(pred, y, cost):
    # plt.plot(xs[0,:], res[0].flatten(), 'r', xs[0,:], pred.flatten()[:TIME_STEPS], 'b--')
    # plt.ylim((-1.2, 1.2))
    plt.plot(cost)
    plt.draw()
    plt.pause(0.3)  # 每 0.3 s 刷新一次

def train(data_train, model, sess):
    """训练模型"""
    # iteraiton = data_train.shape[0] # data.shape[0] 表示的是样本数量
    for i in range(20000):
        x_train, y_train = get_batch(data_train, model.batch_size, episode=i)
        if i == 0:
            feed_dict = {
                model.x: x_train,
                model.y: y_train
            }
        else:
            feed_dict = {
                model.x: x_train,
                model.y: y_train,
                model.cell_init_state: state # 使用上一次的模型的最终状态作为本次模型的初始状态
                # 想办法把这个状态的变化在模型的内部进行处理
            }
        _, cost, state, pred = sess.run(
            [model.optimizer, model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)
        
        # 画出训练的结果
        # plt.ion()
        
        
        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            # plt.plot(np.array(cost))
            # plt.show()
            
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)


def test(data_test, model, sess):
    """测试模型的效果"""
    pass

if __name__=="__main__":
    # RNN模型参数设置
    rnn_config = {
        'batch_size': 64,
        'input_size': 1,
        'output_size': 1,
        'cell_size': 10,
        'seq_length': 10,
        'learning_rate': 0.05
    }
    
    # 导入数据
    if os.path.exists("data/air-condition-consumption.xlsx"):
        data_file = "data/air-condition-consumption.xlsx"
        print(data_file)
    elif os.path.exists("./air-condition-consumption.xlsx"):
        data_file = "./air-condition-consumption.xlsx"
        print(data_file)
    else:
        print("data_file not found! ") 
    sheet_number=2
    
    data = load_data(data_file,sheet_number=sheet_number)
    data_train, data_test = preprocess_data(data, input_size=rnn_config['input_size'], seq_length=rnn_config['seq_length'], month=12)


    # 初始化rnn模型
    model = RNNModel(**rnn_config)

    # 声明对话及数据保存位置
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    
    # 初始化会话
    init = tf.global_variables_initializer()
    sess.run(init)

    if sys.argv[1] is not None and sys.argv[1] == "train": 
        train(data_train, model, sess)        
    else:
        test(data_test, model, sess)