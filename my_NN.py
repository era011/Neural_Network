import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def sigm(x):
    return 1/(1+np.exp(-x))

# def softmax():

def ReLu(x):
    return np.max(0,x)

def ReLu_derivative(x):
    if x<0: 
        return 0
    else:
        return 1

class NeuralNetworkClassifier:
    def __init__(self,input_layer_size=15, hidden_layer_size=(10,),epoch=1000,h=.1):
        self.fitted=False
        self.hidden_layer_size=hidden_layer_size
        self.weights=[]
        for i in self.hidden_layer_size:
            if len(self.weights)==0:
                self.weights.append(np.random.randn(*(input_layer_size,i)))
            else:
                self.weights.append(np.random.randn(*(self.weights[-1].shape[1],i)))  
        self.epoch=epoch
        self.y=None
        self.pred=None
        self.activations=[]
        self.activation=np.vectorize(sigm)  
        self.h=h  
        self.unique_classes=hidden_layer_size[-1]

    def __ones(self,y_train):
        self.y=np.zeros((self.unique_classes,))
        self.y[int(y_train)]=1                                                   # Создание вектора ответа

    
    def fit(self,x_train:np.ndarray,y_train:np.ndarray):
        def apply(arr):
            self.train_on_one_x(arr[:-1],arr[-1])
        arr=np.concatenate((x_train,y_train[:,np.newaxis]),axis=1)    
        for i in range(self.epoch):
            np.apply_along_axis(apply,arr=arr,axis=1)
        
    
    def grad(self,k):
        if k==0:
            gr=self.pred*(1-self.pred)*(self.pred-self.y)
            delta_weights=np.tile(gr,(self.weights[-1].shape[0],1))
            delta_weights=(delta_weights.T*self.activations[-2]*self.h).T
            self.weights[-1]=self.weights[-1]-delta_weights
            return gr   
        else:
            # gradient=((self.grad(k-1)*self.weights[-k])*self.activations[-k]*(1-self.activations[-k])).sum(axis=1)
            gradient=((self.grad(k-1)*self.weights[-k])*self.activations[-k]*(1-self.activations[-k])).sum(axis=1)
            delta_weights=np.tile(gradient,(self.weights[len(self.weights)-k-1].shape[0],1))
            delta_weights=(delta_weights.T*self.activations[-(k+2)]*self.h).T
            self.weights[len(self.weights)-k-1]=self.weights[len(self.weights)-k-1]-delta_weights
            return gradient
        
    def train_on_one_x(self,x,y):
        self.activations.append(x)
        self.pred=self.__pred(x)
        self.__ones(y)
        depth=len(self.weights)-1
        self.grad(depth)
        self.activations.clear()

    def __pred(self,x,k=None):
        if k is None:
            mult=None
            for i in range(len(self.weights)):
                if mult is None:
                    mult=np.dot(x,self.weights[i])
                    mult=self.activation(mult)
                    self.activations.append(mult)
                else:
                    mult=np.dot(mult,self.weights[i])
                    mult=self.activation(mult)
                    self.activations.append(mult)
            return mult
        else:
            mult=None
            for i in range(k):
                if mult is None:
                    mult=np.dot(x,self.weights[i])
                    mult=self.activation(mult)
                else:
                    mult=np.dot(mult,self.weights[i])
                    mult=self.activation(mult)
            return mult 

    def predict(self,X,k=None): 
        y_preds=np.apply_along_axis(self.__pred,axis=1,arr=X,k=k)        
        return np.argmax(y_preds, axis=1)
    

    def print_w(self):
        for i in self.weights:
            print(i)

data_df=load_iris()
data=data_df.data
target=data_df.target
nn=NeuralNetworkClassifier(4,(4,3,3),epoch=500)
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
X_standardized_train = scaler.fit_transform(X_train)
preds_before_fir=nn.predict(X_standardized_train)
print(f'train before fit : {accuracy_score(y_train,preds_before_fir)}')
print(f'test_before fit : {accuracy_score(y_test,nn.predict(scaler.transform(X_test)))}')
nn.fit(X_standardized_train,y_train)
preds_after_fit=nn.predict(scaler.transform(X_test))
print(f'train_after fit : {accuracy_score(y_train,nn.predict(scaler.transform(X_train)))}')
print(f'test_after fit : {accuracy_score(y_test,preds_after_fit)}')


        
