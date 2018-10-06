import tensorflow as tf

class RNNModel():
    def __init__(self, seq_length=10, input_size=1, output_size=1, cell_size=10, batch_size=64, learning_rate=0.005):
        self.seq_length = seq_length
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, input_size, seq_length], name="x")
            self.y = tf.placeholder(tf.float32, [None, input_size, seq_length], name='y')
        
        with tf.variable_scope("input_layer") as scope:
            self.add_input_layer()
        
        with tf.variable_scope("LSTM_cell"):
            self.add_cell()

        with tf.variable_scope("output_layer"):
            self.add_output_layer()

        with tf.variable_scope("cost"):
            self.compust_cost()
        
        with tf.variable_scope("update"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
    def add_input_layer(self):
        """给模型增加输入层，即对输入进行处理,使得输入lstm的输入样本的维度为(seq_length, cell_size)"""
        # x_input 处理成维度分别为(input_size, batch_size*seq_length,)
        X_in = tf.reshape(self.x, [-1, self.input_size], name="x_input_2d")

        # weights 维度为(input_size, cell_size)
        Weights_in = self._weight_variable([self.input_size, self.cell_size])

        # bias 维度为(cell_size, )
        Bias_in = self._bias_variable([self.cell_size,])

        with tf.name_scope('out_of_input_layer'): # 线性模型
            # X_out 维度为 (batch_size * seq_length, cell_size)
            X_out = tf.matmul(X_in, Weights_in) + Bias_in
        # 把 X_out 变为 (batch, seq_length, cell_size)维度的矩阵
        self.X_out = tf.reshape(X_out, [-1, self.seq_length, self.cell_size], name="X_out_3d")

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)

        with tf.name_scope("inital_state"):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.X_out, initial_state=self.cell_init_state, time_major=False) # time_major =False 意味着输入的batch_size是第一个维度
    
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        output = tf.reshape(self.cell_outputs, [-1, self.cell_size], name="X_out_2d")
        
        Weights_out = self._weight_variable([self.cell_size, self.output_size])
        
        Bias_out = self._bias_variable([self.output_size,])
        
        with tf.name_scope("output_of_output_layer"):
            # pred 的维度为(batch * steps, output_size)
            self.pred = tf.matmul(output, Weights_out) + Bias_out

    def compust_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[tf.reshape(self.pred, [-1], name='reshape_pred')],
            targets=[tf.reshape(self.y, [-1], name='reshape_target')],
            weights=[tf.ones([self.batch_size * self.seq_length], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )

        with tf.name_scope("average_cost"):
            self.cost = tf.div(
                tf.reduce_sum(losses, name="losses"),
                self.batch_size,
                name='average_cost'
            )
            tf.summary.scalar('cost', self.cost)


    def _weight_variable(self, shape, name='weights'): 
        """为模型所有的weight初始化"""
        # 由于使用name_scope,不同变量域的weight可以使用相同的变量名，但是分别处于不同的名字域种所以不会冲突
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        """为模型所有的bias初始化"""
        # 由于使用name_scope,不同的bias可以使用相同的变量名，但是分别处于不同的名字域种所以不会冲突
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
    
    @staticmethod
    def ms_error(labels, logits):
        """计算模型的损失值"""
        return tf.square(tf.subtract(labels, logits))
