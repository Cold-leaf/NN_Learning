#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


# 导入 MINST 数据集
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('X_train.shape:',X_train.shape)
print('X_test.shape:',X_test.shape)
print('y_train.shape:',y_train.shape)
print('y_test.shape:',y_test.shape)
 
#参数设置
learning_rate = 0.001   #学习率
training_epochs = 5    #迭代次数
batch_size = 100    #每批数据量的大小
display_step = 1    #展示次数
 
# Network Parameters
n_hidden_1 = 256 # 第一层节点数
n_hidden_2 = 256 # 第二层节点数
n_input = 784 # MNIST data 输入维度 (img shape: 28*28)
n_classes = 10  # MNIST 类别 (0-9 ，一共10类)
 
def onehot(y,start,end,categories='auto'):
    ohot = OneHotEncoder()
    a = np.linspace(start,end-1,end-start)
    b = np.reshape(a,[-1,1]).astype(np.int32)
    ohot.fit(b)
    c = ohot.transform(y).toarray()
    return c

def MNISTLable_TO_ONEHOT(X_Train,Y_Train,X_Test,Y_Test,shuff=True):
    Y_Train = np.reshape(Y_Train,[-1,1])
    Y_Test = np.reshape(Y_Test,[-1,1])
    Y_Train = onehot(Y_Train.astype(np.int32),0,n_classes)
    Y_Test = onehot(Y_Test.astype(np.int32),0,n_classes)
    if shuff ==True:
        X_Train,Y_Train = shuffle(X_Train,Y_Train)
        X_Test,Y_Test = shuffle(X_Test,Y_Test)
        return X_Train,Y_Train,X_Test,Y_Test

X_train,y_train,X_test,y_test = MNISTLable_TO_ONEHOT(X_train,y_train,X_test,y_test)
 
# tf Graph input
tf.compat.v1.disable_eager_execution()
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
 
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
    
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), #（784,256）
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),  #（256,256）
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))   #（256,10）
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
 
# 构建模型
pred = multilayer_perceptron(x, weights, biases)    #预测值（batch_size,n_classes）
 
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))   #交叉熵损失（向量）的平均值
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  #Adam优化器
 
# 初始化变量
init = tf.global_variables_initializer()    #含有tf.Variable的环境下，需要初始化这些变量
 

#tesorflow有一个特有的特点，就是可以提前定义很多变量和函数，但是这些操作并不直接执行，要通过session run的形式才可以执行
# 启动session
with tf.Session() as sess:
    sess.run(init)
 
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        wr_count = 0
        total_batch = int(X_train.shape[0]/batch_size)  #一批100算出需要多少批

        # 遍历全部数据集
        for i in range(total_batch):
        
            batch_x = X_train[i*batch_size:(i+1)*batch_size,:]
            batch_x = np.reshape(batch_x,[-1,28*28])
            batch_y = y_train[i*batch_size:(i+1)*batch_size,:]
            correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1)) #每行最大值所在索引代表其预测类别，比较得出正误矩阵（true、false）
            Accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))  #得出本批次正确率
            _, c , Acc= sess.run([optimizer, cost,Accuracy], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            #tf.cast(wr_count,"int")
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost),"Accuracy:",Acc,wr_count,"/",X_train.shape[0])
    print (" Finished!")
 
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    X_test = np.reshape(X_test,[-1,28*28])
    print ("Test Accuracy:", accuracy.eval({x: X_test, y: y_test}))
    print(sess.run(tf.argmax(y_test[:30],1)),"Real Number")
    print(sess.run(tf.argmax(pred[:30],1),feed_dict={x:X_test,y:y_test}),"Prediction Number")
