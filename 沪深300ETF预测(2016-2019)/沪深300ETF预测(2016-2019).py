import pandas as pd
import tushare as ts
import talib as ta
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, AveragePooling1D
from math import sqrt

pro = ts.pro_api('514302302e066c20b2375f1c4bbb312a43feea7f6b5d78fd56ca84b0')

#从tushare获取所需数据
def getDailyData(list_code, stockcode='510300.SH',startdate='20220228',enddate='20220228'):

    # ts.set_token('514302302e066c20b2375f1c4bbb312a43feea7f6b5d78fd56ca84b0')
    # pro=ts.pro_api()

    tsdata=pro.fund_daily(ts_code=stockcode,start_date=startdate,end_date=enddate,adj=None)
    #沪深300ETF的每日行情
    # 翻转数据,因为tsdata获取到的数据是时间倒序的，翻转后得到的数据是时间正序的
    data={
        # 'ts_code':np.array(tsdata['ts_code'])[::-1],
        'trade_date':np.array(tsdata['trade_date'])[::-1],
        'close':np.array(tsdata['close'])[::-1],
        'open':np.array(tsdata['open'])[::-1],
        'high':np.array(tsdata['high'])[::-1],
        'low':np.array(tsdata['low'])[::-1],
        # 'pre_close':np.array(tsdata['pre_close'])[::-1],
        # 'change':np.array(tsdata['change'])[::-1],
        # 'pct_chg':np.array(tsdata['pct_chg'])[::-1],
        'vol':np.array(tsdata['vol'])[::-1],
        # 'amount':np.array(tsdata['amount'])[::-1]
    }

    print('沪深300ETF的每日行情数据长度',len(data['low']))  # 这里是1200

    shdata=pro.index_daily(ts_code='399300.SZ',start_date=startdate,end_date=enddate)
    #大盘指数的每日行情
    indexdata={
        # 'ts_code':np.array(shdata['ts_code'])[::-1],
        # 'trade_date':np.array(shdata['trade_date'])[::-1],
        'close':np.array(shdata['close'])[::-1],
        'open':np.array(shdata['open'])[::-1],
        'high':np.array(shdata['high'])[::-1],
        'low':np.array(shdata['low'])[::-1],
        #'pre_close':np.array(shdata['pre_close'])[::-1],
        # 'change':np.array(shdata['change'])[::-1],
        # 'pct_chg':np.array(shdata['pct_chg'])[::-1],
        'vol':np.array(shdata['vol'])[::-1],
        # 'amount':np.array(shdata['amount'])[::-1]
    }

    # 构建沪深300ETF的样本股资金流向
    # 个股代码，个股名称，权重，现价
    etf300=list_code
    length=len(data['close'])

    '''由于300支样本股的停盘时间可能与300ETF的停盘时间不一致，
    甚至会出现“中国北车”这样合并停盘的情况，所以300支样本股的
    资金流向并非全部可用,
    len(mf['buy_sm_vol'])<len(flow['buy_sm_vol'])判断沪深300
    的ETF行情数据长度与样本股的资金流向数据 长度是否一致的'''

    #沪深300ETF的资金流向

    flow={
        'buy_sm_vol':np.array([0 for i in range(length)]),
        'sell_sm_vol':np.array([0 for i in range(length)]),
        'buy_md_vol':np.array([0 for i in range(length)]),
        'sell_md_vol':np.array([0 for i in range(length)]),
        'buy_lg_vol':np.array([0 for i in range(length)]),
        'sell_lg_vol':np.array([0 for i in range(length)]),
        'buy_elg_vol':np.array([0 for i in range(length)]),
        'sell_elg_vol':np.array([0 for i in range(length)])
    }

    short=[]
    valid=0.0
    for tup in etf300:
        mf=pro.moneyflow(ts_code=tup[0],start_date=startdate,end_date=enddate)
        daily_data = pro.daily(ts_code=tup[0],start_date=startdate,end_date=enddate)


        if len(mf['buy_sm_vol'])<len(flow['buy_sm_vol']):
            short.append(tup[1])
            continue


        # 个股行情数据长度不等于个股的资金流动数据长度则跳过
        if len(daily_data['close'])!=len(mf['buy_sm_vol']):
            short.append(tup[0])
            continue

        for label in flow.keys():
            # print(len(flow[label]))
            # print('----------')
            # print(len(daily_data['close'][::-1]))
            # (np.array(mf[label])[::-1] 个股buy_sm_vol等资金流向数据 / tup[3]个股现价)*tup[2]个股权重
            flow[label]=np.add(flow[label],np.multiply(np.divide(np.array(mf[label])[::-1],daily_data['close'][::-1]),tup[1]))
        print(tup[0],'添加完毕')
        valid+=tup[1]   # 权重累加
    print('有效因子',valid)
    print('无法添加',short)
    return data,indexdata,flow

# 行列显示
def pandas_pretty_printing():
    pd.set_option('display.max_rows', None)     # 解决行显示不全
    pd.set_option('display.max_columns', None)  # 解决列显示不全
    pd.set_option('max_colwidth', 1000)         # 解决列宽不够
    pd.set_option('display.width', 1000)         # 解决列过早换行
    # pd.set_option('display.float_format', 'float_format')  # 解决浮点数总是科学计数法

#使用talib库函数构造行情指标
def taAnalysis(input_arrays):
    #length=len(input_arrays['close'])
    #Overlap Studies
    upper,middle,lower=ta.BBANDS(input_arrays['close'],matype=ta.MA_Type.T3)
    ma=ta.MA(input_arrays['close'],matype=0)
    ht=ta.HT_TRENDLINE(input_arrays['close'])
    midpoint=ta.MIDPOINT(input_arrays['close'],timeperiod=14)
    midpirce=ta.MIDPRICE(input_arrays['high'],input_arrays['low'],timeperiod=14)
    sar=ta.SAR(input_arrays['high'],input_arrays['low'],acceleration=0.1,maximum=0.1)
    OSlist={
        'upper':upper,
        'middle':middle,
        'lower':lower,
        'ma':ma,
        'ht':ht,
        'midpoint':midpoint,
        'midpirce':midpirce,
        'sar':sar
    }
    #Momentum Indicator
    macd,macdsignal,macdhist=ta.MACD(input_arrays['close'],fastperiod=12,slowperiod=26,signalperiod=9)
    slowk,slowd=ta.STOCH(input_arrays['high'],input_arrays['low'],input_arrays['close'],fastk_period=5,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
    MIlist={
        'macd':macd,
        'macdsignal':macdsignal,
        'macdhist':macdhist,
        'slowk':slowk,
        'slowd':slowd
    }
    #Pattern Recognition
    cdl2crows=ta.CDL2CROWS(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdl3crows=ta.CDL3BLACKCROWS(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdl3inside=ta.CDL3INSIDE(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdl3linestrike=ta.CDL3LINESTRIKE(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdl3outside=ta.CDL3OUTSIDE(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdl3starsinsouth=ta.CDL3STARSINSOUTH(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdl3whitesoldiers=ta.CDL3WHITESOLDIERS(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdlabandonedbaby=ta.CDLABANDONEDBABY(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdladvanceblock=ta.CDLADVANCEBLOCK(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdlbelthold=ta.CDLBELTHOLD(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    cdlbreakaway=ta.CDLBREAKAWAY(input_arrays['open'],input_arrays['high'],input_arrays['low'],input_arrays['close'])
    PRlist={
        'cdl2crows':cdl2crows,
        'cdl3crows':cdl3crows,
        'cdl3inside':cdl3inside,
        'cdl3linestrike':cdl3linestrike,
        'cdl3outside':cdl3outside,
        'cdl3starsinsouth':cdl3starsinsouth,
        'cdl3whitesoldiers':cdl3whitesoldiers,
        'cdlabandonedbaby':cdlabandonedbaby,
        'cdladvanceblock':cdladvanceblock,
        'cdlbelthold':cdlbelthold,
        'cdlbreakaway':cdlbreakaway
    }
    return OSlist,MIlist,PRlist

# 数据清洗
def featureEngineering(data,OSlist,MIlist,PRlist,indexdata,flow,timesteps=3):
    length=len(data['close'])
    original=[]
    #将上证50ETF的各项特征序列拼接为一整块二维向量
    for i in range(length):
        original.append([])
        for key,value in data.items():
            # 将上证50ETF的每日开盘价和大盘指数的每日开盘价从特征集中移除。
            if key!='trade_date' and key!='open':
                original[-1].append(value[i])
        for key,value in OSlist.items():
            original[-1].append(value[i])
        for key,value in MIlist.items():
            original[-1].append(value[i])
        for key,value in PRlist.items():
            original[-1].append(value[i])
        for key,value in indexdata.items():
            if key!='open':
                original[-1].append(value[i])
        original[-1].append(flow['buy_sm_vol'][i]-flow['sell_sm_vol'][i])
        original[-1].append(flow['buy_md_vol'][i]-flow['sell_md_vol'][i])
        original[-1].append(flow['buy_lg_vol'][i]-flow['sell_lg_vol'][i])
        original[-1].append(flow['buy_elg_vol'][i]-flow['sell_elg_vol'][i])
    #数据规范化，构造数据集
    dataX,dataY,date,rate=[],[],[],[]
    for i in range(length-timesteps+1):
        adata=original[i:(i+timesteps)]
        if np.any(np.isnan(np.array(adata))):
            continue
        bdata=MinMaxScaler(feature_range=(0, 1)).fit_transform(adata)
        label=0
        trate=[0,0]
        if i+timesteps<length:
            maxclose=np.amax(adata, axis=0)[0]
            minclose=np.amin(adata, axis=0)[0]
            trate=[maxclose,minclose]
            label=((original[i+timesteps][0])-minclose)/(maxclose-minclose)
            if bdata[-1][0]-(adata[-1][0]-minclose)/(maxclose-minclose)>0.01:
                print('wrong')
        dataX.append(bdata)
        dataY.append(label)
        date.append(data['trade_date'][i+timesteps-1])
        rate.append(trate)
    return dataX,dataY,date,rate

#模型训练
def train(dataX,dataY):
    train_X=np.array(dataX)
    train_Y=np.array(dataY)

    model=Sequential()
    '''
    首先构建三层LSTM神经网络，主要的可调参数为神经元个数，
    神经元个数偏少将无法提取完整的数据特征，个数过多则容易
    出现过拟合。经过反复测试我选择200作为隐层的神经元个数。
    为了进一步防止过拟合，在每一层LSTM之间加入了Dropout
    层，随机断开神经元的连接，抛弃阈值设定为0.2。
    '''
    model.add(LSTM(units=200,return_sequences=True,input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200,return_sequences=True))
    model.add(Dropout(0.2))
    '''
    之后构建两层一维卷积层，卷积核大小各为256个，卷积核大
    小为3，因为训练过程中涉及到较多的参数，所以选取Relu激
    活函数，对过拟合进行控制，在每一次卷积后还加入了池化层，
    对卷积结果下采样，对特征进行压缩，去除冗余信息。
    '''
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same'))
    model.add(AveragePooling1D(pool_size=2, strides=1))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same'))
    model.add(AveragePooling1D(pool_size=2, strides=1))
    model.add(Dropout(0.2))

    '''
    最后加入了4层全连接层，综合各组特征，输出预测结果。
    '''
    model.add(Flatten())
    model.add(Dense(units=256,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=256,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=128,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=1,activation='relu'))
    '''
    损失函数为mae均方误差，优化器采用Adam。Adam算法是
    一种自适应学习率的方法，它利用梯度的一阶矩阵估计和
    二阶矩阵估计动态调整每个参数的学习率。学习率参数设
    定为0.001。
    '''
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(0.001))

    checkpoint_save_path = "./checkpoint/lstmconv_stock.ckpt"
    if os.path.exists(checkpoint_save_path):
        model.load_weights(checkpoint_save_path)
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor='val_loss')
    '''
    模型的epochs为100次，batch_size设定为32，shuffle
    置为True在训练过程中随机打乱输入样本的顺序，使用回调
    函数ModelCheckpoint，将在每个epoch后保存性能最好的
    模型到指定文件中。模型训练时的评价指标为mae均方误差
    '''
    history = model.fit(train_X,
                        train_Y,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1,
                        shuffle=True,
                        callbacks=[cp_callback])
    model.summary()

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return model

#模型使用与预测
def apply(model,testX,testY,date,rate):
    #mode为0时比较明日预测值与当日真实值给出涨跌预测正确性判断
    #mode为1时比较明日预测值与前一日预测的当日预测值给出涨跌预测正确性判断
    mode=0
    #将股票收盘价序列反归一化
    testPredict = model.predict(np.array(testX))
    for i in range(len(testPredict)):
        testPredict[i]=testPredict[i]*(rate[i][0]-rate[i][1])+rate[i][1]
        testY[i]=testY[i]*(rate[i][0]-rate[i][1])+rate[i][1]
        # testX[i][-1][0]=testX[i][-1][0]*(testR[i][0]-testR[i][1])+testR[i][1]
    if len(testY)>1:
        plt.plot(testY[:-1],label='actual')
        plt.plot(testPredict[:-1],label='predict')
        plt.legend()
        plt.show()

        right=0
        count=0
        for i in range(len(testPredict)):
            if testY[i]==0:
                continue
            count+=1
            if mode:
                ref_ans=testY[i-1]
                ref_pre=testPredict[i-1]
            else:
                ref_ans=testY[i-1]
                ref_pre=testY[i-1]
            if (testY[i]>ref_ans and testPredict[i]>ref_pre) or (testY[i]<=ref_ans and testPredict[i]<=ref_pre):
                right+=1
        # calculate MSE 均方误差
        mse=mean_squared_error(testY,testPredict)
        # calculate RMSE 均方根误差
        rmse = sqrt(mean_squared_error(testY,testPredict))
        #calculate MAE 平均绝对误差
        mae=mean_absolute_error(testY,testPredict)
        #calculate R square
        r_square=r2_score(testY,testPredict)
        print('均方误差: %.6f' % mse)
        print('均方根误差: %.6f' % rmse)
        print('平均绝对误差: %.6f' % mae)
        print('R_square: %.6f' % r_square)

        for i in range(len(testPredict)):
            print(date[i],end='\t')
            if mode:
                ref_ans=testY[i-1]
                ref_pre=testPredict[i-1]
            else:
                ref_ans=testY[i-1]
                ref_pre=testY[i-1]
            if testPredict[i]>ref_pre:
                print('↑',end='\t')
                #数据集最后一个数据无法判断正确与否
                if testY[i]==0:
                    print('?')
                elif testY[i]>ref_ans:
                    print('\033[1;32m √ \033[0m')
                else:
                    print('\033[1;31m × \033[0m')
            elif testPredict[i]<=ref_pre:
                print('↓',end='\t')
                #数据集最后一个数据无法判断正确与否
                if testY[i]==0:
                    print('?')
                elif testY[i]<=ref_ans:
                    print('\033[1;32m √ \033[0m')
                else:
                    print('\033[1;31m × \033[0m')

        print(right,'-',count)
        print(float(right)/float(count))
    return testPredict[0],testY[0]

# 获取指数权重个股
def get_tuple(start_date, end_date):
    # 获取指数权重
    df2 = pro.index_weight(index_code='399300.SZ', start_date=start_date, end_date=end_date)

    # 升序排列，方便拼接数据
    df2 =  df2.sort_values(by='con_code',ascending=True)

    # 方便拼接
    df2.rename(columns={'con_code': 'ts_code'})

    # 创建个股str,放入接口
    str2 = ''
    for i in df2['con_code']:
        str2 = str2 + i+','


    # 成分股行情
    df22 = pro.daily(ts_code= str2, trade_date='20220228')

    # 个股名称
    # name_list = []
    # ts_code_list = []
    # for i in df22['ts_code']:
    #     time.sleep(0.5)
    #     df = pro.namechange(ts_code=i, fields='ts_code,name,start_date,end_date,change_reason')
    #     print(df['name'][0])
    #     name_list.append(df['name'][0])
    #     ts_code_list.append(df['ts_code'][0])
    # name = {
    #     'ts_code':ts_code_list,
    #     'name':name_list
    # }
    # name_data = pd.DataFrame(name)

    df22_new = df22[['ts_code','close']]
    df2 = df2.reset_index(drop=True)
    df22_new = df22_new.reset_index(drop=True)
    df2 =  df2.rename(columns = {"con_code": "ts_code"})

    # 拼接数据
    new_data = pd.merge(df2,df22_new)
    # new_data = pd.merge(new_data,name_data)
    new_data = new_data.drop_duplicates()
    new_data = new_data.reset_index(drop=True)

    # 创建tuple列表
    #########################################     这里有个name↓
    list2 = [tuple(x) for x in new_data[['ts_code','weight','close']].values]
    return list2

# 选择沪深300ETF行情年份
def get_data_new():
    # 按照惯例，沪深300指数会在每年的一月初或七月初更新成分股
    time_list = [
                 ['20160101','20160105'],['20180701','20160703'],
                 ['20170101','20170105'],['20170701','20170703'],
                 ['20180101','20180102'],['20180701','20180702'],
                 ['20190101','20190102'],['20190701','20190701'],
                 # ['20200115','20200131'],['20200701','20200701'],
                 # ['20210101','20210105'],['20210701','20210731'],
                 # ['20220101','20220105']
                 ]
    date = 0

    data_all = {
        # 'ts_code':np.array(tsdata['ts_code'])[::-1],
        'trade_date':[],
        'close':[],
        'open':[],
        'high':[],
        'low':[],
        # 'pre_close':np.array(tsdata['pre_close'])[::-1],
        # 'change':np.array(tsdata['change'])[::-1],
        # 'pct_chg':np.array(tsdata['pct_chg'])[::-1],
        'vol':[],
        # 'amount':np.array(tsdata['amount'])[::-1]
    }
    indexdata_all = {
        # 'ts_code':np.array(shdata['ts_code'])[::-1],
        # 'trade_date':np.array(shdata['trade_date'])[::-1],
        'close':[],
        'open':[],
        'high':[],
        'low':[],
        #'pre_close':np.array(shdata['pre_close'])[::-1],
        # 'change':np.array(shdata['change'])[::-1],
        # 'pct_chg':np.array(shdata['pct_chg'])[::-1],
        'vol':[],
        # 'amount':np.array(shdata['amount'])[::-1]
    }
    flow_all = {
        'buy_sm_vol':[],
        'sell_sm_vol':[],
        'buy_md_vol':[],
        'sell_md_vol':[],
        'buy_lg_vol':[],
        'sell_lg_vol':[],
        'buy_elg_vol':[],
        'sell_elg_vol':[]
    }
    for i in time_list:
        date +=1

        list = get_tuple(start_date=i[0], end_date=i[1])
        print(list)

        if date%2 !=0:
            print(i[0][:4]+'0101',i[0][:4]+'0630')
            start = i[0][:4]+'0101'
            end = i[0][:4]+'0630'

        elif date%2 ==0:

            print(i[0][:4]+'0701',i[0][:4]+'1231')
            start = i[0][:4]+'0701'
            end = i[0][:4]+'1231'

        data,indexdata,flow = getDailyData(list, stockcode='510300.SH',startdate=start,enddate=end)

        print(len(data['close']))
        print(len(indexdata['close']))
        print(len(flow['buy_sm_vol']))

        print(data['close'])


        data_all = {key: np.append(data_all[key],data[key]) for key in data}
        indexdata_all = {key: np.append(indexdata_all[key],indexdata[key]) for key in indexdata}
        flow_all = {key: np.append(flow_all[key],flow[key]) for key in flow}

    return data_all,indexdata_all,flow_all

if __name__ == '__main__':

    pandas_pretty_printing()

    #可调参数：
    ###################
    #基金代码
    scode='510300.SH'
    #时间序列长度
    timesteps=35
    #训练集占所有数据集的比例
    train_rate=0.80
    #是否制作新的数据集，如果选择否则使用保存的数据集，当更新数据集时间后要制作新的数据集
    newdataset=False
    #是否使用现有模型进行预测，如果要重新训练模型，该参数应该置为False
    usemodel=True
    ###################

    # 返回 上证50ETF的每日行情 大盘指数的每日行情
    # data,indexdata,flow = getDailyData(stockcode=scode,startdate='20180501',enddate='20220309')


    if newdataset:
        data,indexdata,flow = get_data_new()
        # data,indexdata,flow=getDailyData(stockcode=scode,startdate='20210630',enddate='20211231')
        try:
            f=open('dataset.dst','wb')
            tup=(data,indexdata,flow)
            pickle.dump(tup,f)
            f.close()
        except IOError as e:
            print("error:caculate.write(data,indexdata,flow)",e)
    else:
        try:
            f=open('dataset.dst','rb')
            tup=pickle.load(f)
            (data,indexdata,flow)=tup
        except IOError as e:
            print("error:caculate.read(dataset.dst)",e)

    OSlist,MIlist,PRlist=taAnalysis(data)

    # print('OSlist:\n',pd.DataFrame([OSlist]).shape)
    # print('OSlist:\n',pd.DataFrame([MIlist]).shape)
    # print('OSlist:\n',pd.DataFrame([PRlist]).shape)


    dataX,dataY,date,rate=featureEngineering(data,OSlist,MIlist,PRlist,indexdata,flow,timesteps)

    train_size=int(len(dataY)*train_rate)
    if train_size==len(dataY) and dataY[-1]==0:
        train_size-=1
    trainX=dataX[:train_size]
    trainY=dataY[:train_size]
    testX=dataX[train_size:]
    testY=dataY[train_size:]
    testD=date[train_size:]
    testR=rate[train_size:]

    if not usemodel:
        model=train(trainX,trainY)
        model.save('LSTM模型')
    else:
        model=load_model('LSTM模型')
    apply(model,testX,testY,testD,testR)
