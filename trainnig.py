import pandas as pd
import numpy as np
import pickle


def data_spilt(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__=='__main__':

    df=pd.read_csv('covid_data1.csv')
    train,test=data_spilt(df,0.2)

    X_train1=train[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()
    X_test1=test[['fever','bodypain','age','runnyNose','diffBreath']].to_numpy()
    y_train1=train[['infectionProb']].to_numpy().reshape(80,)
    y_test1=test[['infectionProb']].to_numpy().reshape(19,)

    from sklearn.linear_model import LogisticRegression
    Lr=LogisticRegression()
    Lr.fit(X_train1,y_train1)

    y_pred=Lr.predict(X_test1)
    print(y_pred)
    inp_features=[[101.65,0,69,0,0]]
    inf_prob=Lr.predict_proba(inp_features)[0][1]
    print(inf_prob)

    file=open('model1.pkl','wb')

    pickle.dump(Lr,file)

    file.close()
