基于LSTM-CNN神经网络模型的沪深300ETF的行情预测，利用Tushare提供的沪深300行情数据，利用
Talib库对行情数据进行加工得到行情指标，对行情进行特征工程，构建深度学习所需要的数据集， 采用的框架为
Tensorflow2，用平均绝对误差、均方根差、均方差来确定系数评价模型，最后利用模型预测沪深300ETF的行情，在部分行情预
测上还结合了国内的疫情数据来作为特征。预测准确率在0.75左右。
