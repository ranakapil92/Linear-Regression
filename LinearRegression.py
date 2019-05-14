import pandas as pd                                         #for arranging data into matricex
from csv import reader                                      #for reading the fiile    
from pandas import DataFrame                                #for data arrangement
import matplotlib.pyplot as plt                           #for ploting the graph
import numpy as np                                          #for matrix operations
from numpy import dot                                         
from numpy.linalg import inv
from sklearn.model_selection import train_test_split        #for partitioning the data randomly



def mean(attr):
    return sum(attr) / float(len(attr))        

def variance(attr, mean):
	return sum([(x-mean)**2 for x in attr])


inputdata = list()
with open("linregdata", 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        inputdata.append(row)

rows= len(inputdata)

#Encoding Gender (Female, Infact, Male) into Numbers (0,1,2) respectively 

for i in range(0,rows):
    if(inputdata[i][0]=="M"):
        inputdata[i][0]=2
    elif(inputdata[i][0]=="I"):
        inputdata[i][0]=1
    elif(inputdata[i][0]=="F"):
        inputdata[i][0]=0   
 
        
#Enconding (Female, Infact, Male ,0,1,2) with three as ((1,0,0 - Female), (0,1,0-Infat),(0,0,1 - Male))  
# I am also adding intercept(First Column) = 1.0
        
for i in range(0,rows):
    if(inputdata[i][0]==0):
        inputdata[i]=[1.0,1.0,0.0]+inputdata[i]
    elif(inputdata[i][0]==1):
        inputdata[i][0]=0.0
        inputdata[i]=[1.0,0.0,1.0]+inputdata[i]
    elif(inputdata[i][0]==2):
        inputdata[i][0]=1
        inputdata[i]=[1.0,0.0,0.0]+inputdata[i]    
        
        
dataset = DataFrame(inputdata)
dataset.columns = [1,2,3,4,5,6,7,8,9,10,11,12]
        
for i in range(1,13):
    dataset[i]= dataset[i].astype(float) 
      
        
dataset.dtypes        

# Till Here whole dataset metrix is ready with whole dataset (1- intercept,3 for gender, 7 Input features, 1 output label)

#Here we are creating a matrix results which will store all errors after training over fraction and lambdas values 

fracs=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
lambdas=np.array([0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,500.0])
results=np.zeros((130,5))
j=0
while(j<122):
    results[j:j+13,1]=lambdas   
    j=j+13
j=0
p=0
while(j<120):
    results[j:j+13,0]=fracs[p]
    j=j+13    
    p=p+1
    
del p
     
     
         

#getting input column in X_main (Feature Matrix) and output value in y (Label Output)
        
X_main = dataset.iloc[:,:11].values
y_main = dataset.iloc[:, 11].values        
        

del rows
del row 
del dataset

# this part use for randomly deviding the wholedataset in training/validation/test dataset
# I am Fixing 20% of whole dataset as test set fix_frac=0.2
#Here 20% test set and remaining will be devide into training/test set

fix_frac=0.2
X_train1, X_test, y_train1, y_test = train_test_split(X_main, y_main, test_size=fix_frac)


w =np.array([0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])



def mylinridgeregeval(X,weights):
    n=len(X)
    
    y_pred=np.empty(n)
    y_pred.fill(0.0)
    for j in range(0,n):
        y_pred[j]=np.dot(weights,X[j])   
        
    return y_pred  
    


def mylinridgereg(X,Y,Lambda):
    
    Xt=np.transpose(X)
    n=len(Xt)
    Lambda_I= Lambda*(np.identity(n))
    
    a = np.zeros(shape=(n,n))
    a=np.dot(Xt,X)
    a=np.add(a,Lambda_I)
    if(np.linalg.det(a)!=0):
        
        b=np.zeros(shape=(n,1))
        c=np.zeros(shape=(n,n))
        
        b=np.dot(Xt,Y)
        c=np.linalg.inv(a)
        c=np.dot(c,b)
        
        return c
    else:
        b=np.zeros(shape=(n,1))
        c=np.zeros(shape=(n,n))
        
        b=np.dot(Xt,Y)
        c=np.linalg.pinv(a)
        c=np.dot(c,b)
        
        return c
        
   
def meansquarederr(T,Tdash):
    error =0.0
    total_error=0.0
    
    error=np.subtract(T,Tdash)
    sq_error=np.square(error)
    total_error=np.sum(sq_error) 
    
    return total_error/float(len(T))  


# This is main execution loop which iterate over all values of lambda and fraction and  fit linear function

j=0
for f in range(0,10):               #loop for Fractions
    frac=results[j,0]
    
    for l in range(0,13):           #loop for lambdas 
        
        L=results[l,1]
        
        print("frac",frac,"Lambda",L)
        error1=0.0
        error2=0.0
        error3=0.0  
        for rep in range(0,100): 
            
            temp_X=np.copy(X_test)
            temp_y=np.copy(y_test)
            
            X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=1-frac)                 #random partitioning
        
            w =np.array([0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
           
            means=np.zeros((11,))
            variances=np.zeros((11,))
            
            #Caculating mean,variance of training set except gender and intercept column as you said in class
            #Standarized the training set
            
            for i in range(4,11):
                m=mean(X_train[:,i])
                means[i]=m
                v=variance(X_train[:,i],m)
                variances[i]=v
                X_train[:,i]=(X_train[:,i]-m)/v 
            
        
            t_initial = mylinridgeregeval(X_train,w)
            error=meansquarederr(t_initial,y_train)
            error1 = error1+error
            
                    
            w = mylinridgereg(X_train,y_train,L)
            
            
            t_train = mylinridgeregeval(X_train,w)
            error=meansquarederr(t_train,y_train)
            error2 = error2+error
        
            #standardized the a copy of fixed  test set
            
            for i in range(4,11):
                temp_X[:,i]=(temp_X[:,i]-means[i])/variances[i] 
            
            
            
            t_test = mylinridgeregeval(temp_X,w)
            error=meansquarederr(t_test,temp_y)
            error3 = error3+error
           
        results[(l+j),2]=error1/100.0
        results[(l+j),3]=error2/100.0
        results[(l+j),4]=error3/100.0
        
        
          
        print(results[(l+j),2],results[(l+j),3],results[(l+j),4])
        print(" ") 
        
    j=j+13
    
    
 
#Graphs Section
  
get_ipython().run_line_magic('matplotlib', 'qt')
    
xg1 = lambdas
xg2=[0,1,2,3,4,5,6,7,8,9,10,11,12]
j=0
plt.figure("Overall Average Training and Testing Error")
for i in range(1,10):
    plt.subplot(3, 3, i)
    plt.xticks(range(len(xg1)), xg1)
    yg1=results[j:j+13,3]
    yg2=results[j:j+13,4]
    plt.plot(xg2, yg1, 'o-')
    plt.plot(xg2, yg2, 'o-')
    plt.legend(['AMS Training Error', 'AMS Testing Error'], loc='upper left')
    plt.ylabel("AMSE for Frac="+str(results[j,0]))
    plt.xlabel("Lambda")
    plt.rc('axes', labelsize=15)

    j=j+13


j=0
min_mste=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]  
min_i=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 
m_i=0
for i in range(0,10):
    min_mste[i]=min(results[j:j+13,4])
    m_i=np.argmin(results[j:j+13,4])
    min_i[i]=results[(j+m_i),1]
    j=j+13
    
    
    
plt.figure("Minimum Average Mean Square Testing Error Vs Fraction")  
xg1=fracs
xg2=[0,1,2,3,4,5,6,7,8,9]  
plt.xticks(range(len(xg1)), xg1)
plt.plot(xg2,min_mste,'o-')
plt.ylabel("Minimum Average Mean Square Testing Error")
plt.xlabel("Fraction")
plt.rc('axes', labelsize=20)



plt.figure("Lambda For Minimum AMS Testing Error vs Fraction")


xg1=fracs
xg2=[0,1,2,3,4,5,6,7,8,9]  
plt.xticks(range(len(xg1)), xg1)
plt.plot(xg2,min_i,'o-')
plt.rc('axes', labelsize=20)
plt.ylabel("Lambda For Minimum Average Mean Square Testing Error")
plt.xlabel("Fraction")

plt.figure("Log Lambda For Minimum AMS Testing Error vs Fraction")


xg1=fracs
xg2=[0,1,2,3,4,5,6,7,8,9]  
plt.xticks(range(len(xg1)), xg1)
min_j=np.log10(min_i)
plt.plot(xg2,min_j,'o-')
plt.rc('axes', labelsize=20)
plt.ylabel("Log base 10 Lambda For Minimum Average Mean Square Testing Error")
plt.xlabel("Fraction")
plt.show()

#here we are predicting output label values for frac=0.9 

X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=1-0.9)        
w =np.array([0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
means=np.zeros((11,))
variances=np.zeros((11,))
for i in range(4,11):
    m=mean(X_train[:,i])
    means[i]=m
    v=variance(X_train[:,i],m)
    variances[i]=v
    X_train[:,i]=(X_train[:,i]-m)/v         
        
w = mylinridgereg(X_train,y_train,0.00000001)        
t_train = mylinridgeregeval(X_train,w)
error=meansquarederr(t_train,y_train)
temp_X=np.copy(X_test)
temp_y=np.copy(y_test)
for i in range(4,11):
    temp_X[:,i]=(temp_X[:,i]-means[i])/variances[i] 

t_test = mylinridgeregeval(temp_X,w)
error=meansquarederr(t_test,temp_y)        

        
        
plt.figure("Predicted Values vs Actual Values")        
plt.ylim(0, 25)
plt.xlim(0,25)        
plt.plot(temp_y,t_test,'ro')
plt.rc('axes', labelsize=20)
plt.ylabel("Predicted Value of Age, Y' ")
plt.xlabel("Actual Value of Age, Y")        
        
plt.show()       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        