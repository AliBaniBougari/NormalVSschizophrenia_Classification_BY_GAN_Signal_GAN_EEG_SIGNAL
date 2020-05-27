from keras.models import *
from keras.layers import *
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Flatten,Activation, Reshape,BatchNormalization,Conv1D, MaxPooling1D #LSTM,SimpleRNN
from keras.layers.advanced_activations import LeakyReLU
import keras
import numpy as np
from tqdm import tqdm
import numpy as np
import os
import os.path as path
import matplotlib.pyplot as plt

normalfiles=os.listdir('normal/')
features=[]

#######################normal################
with tqdm(total=len(normalfiles)) as pbar:
    for file in normalfiles:
        feats=[]
        filename="normal/"+file
        f = open(filename, 'r')     #opening file
        x = f.readline()
        sublist=[]
        i=0
        while x:
            if i<=7679:                     #getting features per channels
                sublist.append(float(x))
            else:
                i=0
                sublist=np.array(sublist)
                sublist = (sublist-sublist.min())/(sublist.max() - sublist.min())
                feats.append(sublist)
                sublist=[]
                sublist.append(float(x))
            i+=1
            x=f.readline()
        feats=np.array(feats)
        features.append(feats)
        pbar.update(1)


features=np.array(features)
print(features.shape)
features = features.astype('float32')

#generator

generator = Sequential()

generator.add(Dense(240*7,input_dim = 240))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
generator.add(Reshape((240,7)))
generator.add(Conv1D(30,kernel_size=7,padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
generator.add(Conv1D(60,kernel_size=7,padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
generator.add(Conv1D(120,kernel_size=7,padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
generator.add(Conv1D(240,kernel_size=7,padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
generator.add(Conv1D(480,kernel_size=3,activation ='tanh',padding='same'))
generator.add(BatchNormalization())

generator.summary()


#discriminator
discriminator = Sequential()

discriminator.add(Conv1D(30,kernel_size=7,input_shape=(240,480),padding='same'))

discriminator.add(LeakyReLU(0.2))
discriminator.add(MaxPooling1D())

discriminator.add(Conv1D(60,kernel_size=7,padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(MaxPooling1D())
discriminator.add(Dropout(0.3))

discriminator.add(Conv1D(120,kernel_size=7,padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(MaxPooling1D())
discriminator.add(Dropout(0.3))

discriminator.add(Conv1D(240,kernel_size=7,padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(MaxPooling1D())

discriminator.add(Conv1D(480,kernel_size=7,padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(MaxPooling1D())

discriminator.add(Dropout(0.4))
discriminator.add(Flatten())
discriminator.add(Dense(512,activation = 'sigmoid'))
discriminator.add(Dense(1,activation = 'sigmoid'))
discriminator.summary()

generator.compile(loss="binary_crossentropy",optimizer=Adam(lr = 0.00000099) ,metrics = ['accuracy'])
discriminator.compile(loss="binary_crossentropy",optimizer=Adam(lr = 0.00000099) ,metrics = ['accuracy'])

input = Input(shape=(240,))
gen = generator(input)
discriminator.trainable = False
output = discriminator(gen)
gen = Model(inputs = input , outputs = output)
gen.compile(loss="binary_crossentropy",optimizer=Adam(lr = 0.00000099),metrics = ['accuracy'])
gen.summary()

def plot_output(n):
    fake = np.random.rand(2,240)
    predictions = generator.predict(fake)
    predictions = (predictions-predictions[1].min())/(predictions[1].max() - predictions[1].min())
    signal_batch = features[np.random.randint(0,features.shape[0],size=2)].reshape((2, 240,480))
    signal_batch = (signal_batch-signal_batch[1].min())/(signal_batch[1].max() - signal_batch[1].min())
    plt.plot(predictions[1].flatten())
    plt.plot(signal_batch[1].flatten())
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
lost =[]

def train(epoch=10, batch_size= 5):
    avg_accuracy= 0
    gen_avg_accuracy = 0
    batch_count = features.shape[0]//batch_size
    for i in range(epoch):
        print('epoch '+str(i+1)+r"/"+str(epoch))
        if avg_accuracy !=0:
            print("avg_ lost : "+str(avg_accuracy/batch_count))
            print("gen_avg_lost : "+str(gen_avg_accuracy/batch_count))
            lost.append(gen_avg_accuracy)
            gen_avg_accuracy = 0
            avg_accuracy = 0 
        pbar = tqdm(total=batch_count)
        for j in range(batch_count):
            fake = np.random.rand(batch_size,240)
            
            signal_batch = features[np.random.randint(0,features.shape[0],size=batch_size)]

            predictions = generator.predict(fake,batch_size=batch_size)
            signal_batch = (signal_batch-signal_batch.min())/(signal_batch.max() - signal_batch.min())
            x = np.concatenate([signal_batch.reshape((batch_size,240,480)),predictions])
            y_discriminator = [1]*(int(batch_size)) + [0]*(int(batch_size))
            discriminator.trainable = True

            x = x.reshape((batch_size*2,240,480))
            histor = discriminator.train_on_batch(x,y_discriminator)
            avg_accuracy += histor[1]

            fake = np.random.rand(batch_size,240)
            y_generator = [1]*batch_size

            discriminator.trainable = False
            histor =gen.train_on_batch(fake,y_generator)
            gen_avg_accuracy +=histor[1]
            pbar.update(1)
            print('\n')
            #plot_output(i)

train(800,features.shape[0])
fake = np.random.rand(10,240)
predictions = generator.predict(fake)
predictions = (predictions-predictions[1].min())/(predictions[1].max() - predictions[1].min())
signal_batch = features[np.random.randint(0,features.shape[0],size=10)].reshape((10, 240,480))
signal_batch = (signal_batch-signal_batch[1].min())/(signal_batch[1].max() - signal_batch[1].min())
plt.plot(predictions[1].flatten())
plt.plot(signal_batch[1].flatten())

plt.show()
plt.plot(lost)
plt.title('gan_loss')
plt.savefig('gan_loss'+".jpg")
plt.show()
generator.save('normal.h5')
"""
keras.utils.plot_model(
    generator,
    to_file="gan.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=1000)

"""
