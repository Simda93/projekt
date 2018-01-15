import sklearn
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge,LinearRegression,Lasso,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from matplotlib import pyplot as plt
from keras import metrics
from sklearn import metrics as mtrcs
import tensorflow as tf
import utils
from keras import losses


def load_data(data):
    data = np.loadtxt("unshuffled.csv")
    return data

def split_data(data,test):
    data_train,data_test = train_test_split(data,test_size=test,random_state=188)
    return data_train,data_test

def select_Y(data,Y_column):
    vektor = [i for i in range(data.shape[1]) if i != Y_column]
    data_X = data[:,vektor]
    data_Y = data[:,Y_column].reshape(-1,1)
    return data_X,data_Y

def select_atributes(data,vektor):
    new = data[:,vektor]
    return new
vektors =[[],
[0,1,2,3,4, 5, 7, 9, 11, 13, 16, 18]  #743158   111765  ^2 Ridge
,[2,3,4, 5, 7, 9, 11, 13, 16, 18]  #741256 ^2 Ridge 121412
,[7,9,11] # 0.740950 ^3 Ridge 116682
,[4, 5, 7, 9, 11, 13, 16, 18] #lat,long,pocet_opravnenych,%volicov,%osobne,%postou,%smer,%LSNS  748456  106002
,[ 7, 9, 11, 13, 16, 18] #bez long a lat   745701  111413
, [4, 5, 7, 9, 11, 13] #bez % Smeru a LSNS  747846  107774
,[ 7, 9, 11, 13] # ^3 741230   112431
,[2,3,7, 9, 11, 13, 16, 18]  #  0.124972   0.742004   1.093444
,[2,3,5,7, 9, 11, 13, 16, 18]  #  0.125715   0.740924   1.092516
,[2,3,5,7, 9, 11, 13, 16]
          ] #  0.127874   0.741038   1.089817

#select atributes
vektor = vektors[3]  #najlepsia mnozina atributov na zaklade validacie
file =open("header.csv",'r')
header = file.read().split()
header =np.array(header)[vektor]
print(header)


#load data
data=np.empty([0,0])
data=load_data(data)

#split data
final,test = split_data(data,0.2)
train,valid = split_data(final,0.2501)

#select Y and antributes
X_train,Y_train = select_Y(train,19)
X_valid,Y_valid = select_Y(valid,19)
X_test,Y_test = select_Y(test,19)
X_train=select_atributes(X_train,vektor)
X_valid=select_atributes(X_valid,vektor)
X_test=select_atributes(X_test,vektor)

#normalize
if(True):
    scaler=StandardScaler()
    scaler=scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_valid=scaler.transform(X_valid)



#odhad priemerom

mean_Y = np.mean(Y_train)
asume_mean=np.mean(np.abs(mean_Y - Y_valid))

print("Mean Y_train"+str(mean_Y))
print("Assume mean on train "+str(np.mean(np.abs(mean_Y - Y_train))))
print("Assume mean on valid "+str(asume_mean))


#podomacky grid search
it = 100000
models = []
res=[]
models.append(('LR', LinearRegression()))
models.append(('Ridge1', Ridge()))
models.append(('Ridge0,9', Ridge(alpha=0.9)))
models.append(('Ridge0,7', Ridge(alpha=0.7)))
models.append(('Ridge0.5', Ridge(alpha=0.5)))
models.append(('Lasso1', Lasso()))
models.append(('Lasso0.7', Lasso(alpha=0.7,max_iter=it)))
models.append(('Lasso0.5', Lasso(alpha=0.5,max_iter=it)))
models.append(('Lasso0.3', Lasso(alpha=0.3,max_iter=it)))
for alpha in [1,0.5,0.3,0.2]:
    for l1_ratio in [0.2,0.25,0.4]:
        models.append(('ElasticNet.'+str(alpha)+"."+str(l1_ratio),ElasticNet(max_iter=it,alpha=alpha,l1_ratio=l1_ratio)))
        pass

#vyber modelu
if(True):
    for i in [1, 2,3]: #exponent
        poly = preprocessing.PolynomialFeatures(i)
        X_train_p = poly.fit_transform(X_train)
        X_valid_p = poly.fit_transform(X_valid)
        res.append("squared: "+str(i))
        print("squared: " + str(i))
        for name,model in models:
            print(" ")
            print("~~~~~~~~~~~Model: "+name+"~~~~~~~~~~~~")
            model.fit(X_train_p,Y_train)
            print("Vakid score "+str(model.score(X_valid_p, Y_valid)))
            print("Train score "+str(model.score(X_train_p, Y_train)))
            y_pred = model.predict(X_valid_p)
            print("Predict mae "+str(mtrcs.mean_absolute_error(Y_valid,y_pred)))
            print("assumed - predicted mae "+str(asume_mean - mtrcs.mean_absolute_error(Y_valid,y_pred)))
            print(asume_mean)
            print(mtrcs.mean_absolute_error(Y_valid,y_pred))
            print("R2 score"+str(mtrcs.r2_score(Y_valid,y_pred)))
            msg = "%20s: %10f %10f %10f" % (name, mtrcs.r2_score(Y_valid,y_pred),mtrcs.mean_absolute_error(Y_valid,y_pred),mtrcs.mean_squared_error(Y_valid,y_pred))
            res.append(msg)


for msg in res:
    print(msg)
#NN
if(False):
    nodes,lr,decay =64,0.015,0.001
    model = Sequential()
    model.add(Dense(X_train.shape[1],activation='sigmoid',input_dim=X_train.shape[1]))
    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(nodes,activation='sigmoid'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer=SGD(lr=lr,decay=decay),loss='mean_squared_error',metrics=[metrics.mae])

    history = model.fit(x=X_train,y=Y_train,epochs=2000,validation_data=(X_valid,Y_valid),batch_size=32,verbose=2)

    f,s = 2,1

    plt.figure("%d %f %f"(nodes,lr,decay))
    plt.plot(history.history['loss'][f::s], label='training loss')
    plt.plot(history.history['val_loss'][f::s], label='validation loss')
    plt.legend(loc='best')
    plt.figure("%d %f %f"(nodes, lr, decay))
    plt.plot(history.history['mean_absolute_error'][f::s], label='train mae')
    plt.plot(history.history['val_mean_absolute_error'][f::s], label='validation mae')
    plt.legend(loc='best')
    plt.show()

#Time for testing
if(True):
    RDG=Ridge()
    X_final,Y_final = select_Y(final,19)
    X_final = select_atributes(X_final, vektors[10])
    X_test,Y_test = select_Y(test,19)
    X_test = select_atributes(X_test,vektors[10])
    poly = preprocessing.PolynomialFeatures(2)
    X_final=poly.fit_transform(X_final)
    X_test=poly.fit_transform(X_test)

    scaler = StandardScaler()
    scaler = scaler.fit(X_final)
    X_final = scaler.transform(X_final)
    X_test = scaler.transform(X_test)


    trained = RDG.fit(X_final,Y_final)
    Y_predict = RDG.predict(X_test)
    print(RDG.score(X_final,Y_final))
    print(RDG.score(X_test,Y_test))
    Y_mean = np.mean(Y_final)
    r2 = mtrcs.r2_score(Y_test, Y_predict)
    mae =  mtrcs.mean_absolute_error(Y_test, Y_predict)
    mse = mtrcs.mean_squared_error(Y_test, Y_predict)
    mae_predict = np.mean(np.abs(Y_test-Y_mean))
    mse_predict = np.mean(np.power(np.abs(Y_test-Y_mean),2))
    msg = "%20s: %10f %10f %10f %10f %10f %10f %10f" % ("Testing results r2score,MAE,MSE,MAE diff", r2,mae,mse,mae_predict,mse_predict,mae_predict-mae,mse_predict-mse )
    print(msg)
    X_plot = select_atributes(final,vektor)
    for i in range(len(header)):
        print(header[i])
        print(RDG.coef_[0,i+1])
        plt.figure("Train "+header[i]+" coef: "+str(RDG.coef_[0,i+1]))
        plt.plot(X_test[:, i + 1], Y_test, 'g*')
        plt.plot(X_test[:, i + 1], Y_predict, 'b+')
        plt.figure("Test "+header[i]+" coef: "+str(RDG.coef_[0,i+1]))
        plt.plot(X_test[:,i+1],Y_test,'g*')
        plt.plot(X_test[:, i + 1], Y_predict, 'b+')
        plt.show()





