import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import keras
from  keras import Sequential
from keras.layers import Dense, Flatten,Conv1D,Dropout
from keras.models import Model
from keras.models import load_model
from keras import regularizers
global ill
global normal

#load signal data 
with open('normal_features', 'rb') as fp:
    normal_features=pickle.load(fp)


with open('ill_features', 'rb') as fp:
    ill_features=pickle.load(fp)

"""
this function for drew history of model
#   * title must be str
#   * for useing function you must complit parametr of function
#   * for drew any parameter[loss , acc, val_loss, val_acc ] set 1 for that element 
"""


def plot_history(history,title = 'none',loss = 0 , acc = 1 ,val_loss = 0 , val_acc = 0,save = 0):
    drew =[acc,val_acc,loss,val_loss]
    if acc == 1:
        a = history.history['acc']
        plt.plot(a)
    if val_acc == 1:
        v_a = history.history['val_acc']
        plt.plot(v_a)
    if loss == 1:
        l = history.history['loss']
        plt.plot(l)
    if val_loss == 1:
        v_l = history.history['val_loss']
        plt.plot(v_l)
    plt.legend(['acc','val_acc','loss','val_loss'])
    plt.title(title)
    if save == 1 :
        plt.savefig(str(title)+".jpg")
    plt.show()



# nromalization data and append data together 

def normal_data(normal_features):
    normal =[]
    for l in normal_features :
        l = l.flatten()
        l =l.reshape((240,480))
        l = (l - l.min())/(l.max()-l.min())
        normal.append(l)
    normal = np.array(normal)
    normal = normal.astype('float32')
    return normal

def ill_data(ill_features):
    ill = []
    for l in ill_features :
        l = l.flatten()
        l =l.reshape((240,480))
        l = (l - l.min())/(l.max()-l.min())
        ill.append(l)
    ill = np.array(ill)
    ill = ill.astype('float32')
    return ill

def data():
    ill = ill_data(ill_features)
    normal = normal_data(normal_features)
    x = np.concatenate([normal,ill])
    x = x.astype('float32')
    return x

# function for build labels

def label(ill_n=ill_features.shape[0],normal_n=normal_features.shape[0]):
    y = []
    for l in range(normal_features.shape[0]):
        y.append(1)
    for l in range(ill_features.shape[0]):
        y.append(0)
    y=np.array(y)
    return y
###load gan model

# gan ill model
ill_gen = load_model('ill.h5')

# gan normal model 
normal_gen = load_model('normal.h5')


#generator gan (make_fake_signal )

def generator(ill_signal,normal_signal,batch_size):
	random_array = np.random.rand(int(batch_size/4),240)
	fake_ill_signal = ill_gen.predict(random_array)
	print(fake_ill_signal.shape)
	random_array = np.random.rand(int(batch_size/4),240)
	fake_normal_signal = normal_gen.predict(random_array)
	print(fake_normal_signal.shape)
	ill =ill_signal
	normal = normal_signal
	ill_signal = ill[np.random.randint(0,ill.shape[0],size=int(batch_size/4))]
	normal_signal = normal[np.random.randint(0,normal.shape[0],size=int(batch_size/4))]

	X_batch = np.concatenate([normal_signal,fake_normal_signal,fake_ill_signal,ill_signal]).astype('float32')
	y_batch = np.array([1]*int(batch_size/2)+[0]*int(batch_size/2))
	return X_batch,y_batch



#load label
y_label = label(ill_n=ill_features.shape[0],normal_n=normal_features.shape[0])

#load data
x_data = data()

#split data and label to test and train data and lebal

x_train, x_val, y_train, y_val = train_test_split(x_data, y_label, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size=0.5)

# function for make model

def model():
    conv1d = Sequential()
    conv1d.add(Conv1D(filters=32,kernel_size=3,input_shape=(240,480),activation='relu'))
    conv1d.add(Conv1D(filters=32,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=64,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=64,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=128,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=128,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=256,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=256,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=256,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=512,kernel_size=3,activation='relu'))
    conv1d.add(Conv1D(filters=512,kernel_size=3,activation='relu'))
    #conv1d.add(Dropout(0.5))
    conv1d.add(Flatten())
    conv1d.add(Dense(512,activation='relu',kernel_regularizer=regularizers.l1_l2(l1= 0.00001,l2 = 0.001)))
    conv1d.add(Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1= 0.00001,l2 = 0.001)))
    return conv1d

    
# load model
m1 = model() 
#summary of model

m1.summary()


adam=keras.optimizers.Adam(lr=0.00001)
m1.compile(optimizer=adam ,loss='binary_crossentropy',metrics=['acc'])
# *train model
"""
#train model with out gen




history = m1.fit(x_train,y_train, batch_size=50, epochs=20,
           shuffle=True,validation_data=(x_test,y_test))
"""

#train model with  gen
ill = ill_data(ill_features)
normal = normal_data(normal_features)

x_train_ill, x_val_ill= train_test_split(ill, test_size=0.3)
x_test_ill, x_val_ill= train_test_split(x_val_ill, test_size=0.5)

x_train_normal, x_val_normal= train_test_split(normal, test_size=0.3)
x_test_normal, x_val_nromal= train_test_split(x_val_normal, test_size=0.5)

x_train ,y_train = generator(x_train_ill,x_train_normal,batch_size=2000) 
x_val ,y_val = generator(x_val_ill,x_test_normal,batch_size=200) 

history = m1.fit(x_train,y_train, batch_size=20, epochs=35,
           shuffle=True,validation_data=(x_val,y_val))
#plot and save out put model
plot_history(history,title = 'conv1d',loss = 0 , acc = 1 ,val_loss = 0 , val_acc = 1,save = 1)
#save model
m1.save('cnn.h5')

#test model 
x_test = np.concatenate([x_test_normal,x_test_ill]).astype('float32')
y_test = np.array([1]*x_test_normal.shape[0]+[0]*x_test_ill.shape[0])
test = m1.evaluate(x_test, y_test)
print('test acc : '+str(test))


input('exit')



#this line for give graphcal model (layers , input and out put layers)


"""
keras.utils.plot_model(
    m1,
    to_file="CONV1D.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=1000)
"""
