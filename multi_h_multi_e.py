import time
import numpy as np
import chainer

from hid_ae import ae_Trainer, Autoencoder
from chainer import optimizers, Variable, cuda

from tqdm import tqdm

import h5py
import sys

from sklearn.preprocessing import StandardScaler as SS
from mh_trainer import multi_head, Trainer_one, c_gpu, g_cpu, reshapei
from mhme_trainer import mh_me, Trainer_all

###################################### Data input #########################################

hf = h5py.File('train_bp_420.h5', 'r') 
x_train = hf['x_train'][0:20000]
x_test = hf['x_train'][20000:36300]

y_train = hf['y_train'][0:20000]
y_test = hf['y_train'][20000:36300]
hf.close() 

hf = h5py.File('train_mf_420.h5', 'r') 
x_train = hf['x_train'][0:20000]
x_test = hf['x_train'][20000:25200]

y_train = hf['y_train'][0:20000]
y_test = hf['y_train'][20000:25200]
hf.close() 

###################################### Model building ###############################################

Scalar = SS()
Scalar.fit(x_train)

x_train_transformed = Scalar.transform(x_train)

x_test_transformed = Scalar.transform(x_test)

x_train_transformed = x_train_transformed.astype(np.float32)
x_test_transformed = x_test_transformed.astype(np.float32)

d = 0
batch = 100
epoch = 30
lr1 = 0.0075
lr2 = 0.005

hid = 512
inp = x_train_transformed.shape[1]

#bp932
n1_classes=30
n2_classes=120

n3_classes=260

n4_classes=470

n_classes = y_train.shape[1]

#mf589
n1_classes=30
n2_classes=120

n3_classes=260

n4_classes=339

n_classes = y_train.shape[1]


def out2(ae, x):
    hidn = ae.l1(c_gpu(x))
    hidn_data = g_cpu(hidn.data)
    return hidn_data


def hid_out1(x, model, col_size):
	hid_data = [] 
	for i in tqdm(range(0, x.shape[0], 100)): 
		hid = model(model1.FE(c_gpu(x[i: i+100])))
		hid_data.append(hid.data) 
	hid_data = np.array(hid_data).reshape(x.shape[0], col_size).astype(np.float32)

	return hid_data

############################################ Deep CNN MH Training ####################################
model1 = multi_head(hid, n1_classes, n2_classes, n3_classes, n4_classes, n_classes)
optimizer = optimizers.MomentumSGD(lr1)
optimizer.setup(model1)
cuda.get_device(d).use()
model1.to_gpu(d)

trainer = Trainer_one(epoch, x_train_transformed, y_train, model = model1, optimizer=optimizer, batchsize=batch)
trainer.run(test_data=x_test_transformed, test_label=y_test)

#chainer.serializers.save_npz('mh_I_bp_fe.npz', model1.FE)

#chainer.serializers.save_npz('mh_I_mf_fe.npz', model1.FE)

hid_train1 = hid_out1(reshapei(x_train_transformed), model1.o1, n1_classes)
hid_train2 = hid_out1(reshapei(x_train_transformed), model1.o2, n2_classes)
hid_train3 = hid_out1(reshapei(x_train_transformed), model1.o3, n3_classes)
hid_train4 = hid_out1(reshapei(x_train_transformed), model1.o4, n4_classes)

hid_train5 = hid_out1(reshapei(x_train_transformed), model1.o5, n_classes)
hid_train_g = hid_out1(reshapei(x_train_transformed), model1.o, n_classes)

test1 = hid_out1(reshapei(x_test_transformed), model1.o1, n1_classes)
test2 = hid_out1(reshapei(x_test_transformed), model1.o2, n2_classes)
test3 = hid_out1(reshapei(x_test_transformed), model1.o3, n3_classes)
test4 = hid_out1(reshapei(x_test_transformed), model1.o4, n4_classes)

test5 = hid_out1(reshapei(x_test_transformed), model1.o5, n_classes)
test_g = hid_out1(reshapei(x_test_transformed), model1.o, n_classes)

vaild_g = hid_out1(reshapei(x_vaild), model.o, n_classes)
vaild5 = hid_out1(reshapei(x_vaild), model.o5, n_classes)
vaild_data5 = np.c_[x_vaild, vaild5]

train_data1 = np.c_[x_train_transformed, hid_train1]
test_data1 = np.c_[x_test_transformed, test1]

train_data2 = np.c_[x_train_transformed, hid_train2]
test_data2 = np.c_[x_test_transformed, test2]

train_data3 = np.c_[x_train_transformed, hid_train3]
test_data3 = np.c_[x_test_transformed, test3]

train_data4 = np.c_[x_train_transformed, hid_train4]
test_data4 = np.c_[x_test_transformed, test4]

train_data5 = np.c_[x_train_transformed, hid_train5]
test_data5 = np.c_[x_test_transformed, test5]



############################ AE ################################

ae1 = Autoencoder(train_data1.shape[1], x_train_transformed.shape[1])
optimizer2 = optimizers.MomentumSGD(lr2)
optimizer2.setup(ae1)
cuda.get_device(d).use()
ae1.to_gpu(d)

trainer2 = ae_Trainer(30, train_data1, batchsize=batch, ae=ae1, optimizer=optimizer2)
trainer2.run()

ae2 = Autoencoder(train_data2.shape[1], x_train.shape[1])
optimizer2 = optimizers.MomentumSGD(lr2)
optimizer2.setup(ae2)
cuda.get_device(d).use()
ae2.to_gpu(d)

trainer2 = ae_Trainer(30, train_data2, batchsize=batch, ae=ae2, optimizer=optimizer2)
trainer2.run()

ae3 = Autoencoder(train_data3.shape[1], x_train.shape[1])
optimizer2 = optimizers.MomentumSGD(lr2)
optimizer2.setup(ae3)
cuda.get_device(d).use()
ae3.to_gpu(d)

trainer2 = ae_Trainer(30, train_data3, batchsize=batch, ae=ae3, optimizer=optimizer2)
trainer2.run()

ae4 = Autoencoder(train_data4.shape[1], x_train.shape[1])
optimizer2 = optimizers.MomentumSGD(lr2)
optimizer2.setup(ae4)
cuda.get_device(d).use()
ae4.to_gpu(d)

trainer2 = ae_Trainer(30, train_data4, batchsize=batch, ae=ae4, optimizer=optimizer2)
trainer2.run()

ae5 = Autoencoder(train_data5.shape[1], x_train.shape[1])
optimizer2 = optimizers.MomentumSGD(lr2)
optimizer2.setup(ae5)
cuda.get_device(d).use()
ae5.to_gpu(d)

trainer2 = ae_Trainer(30, train_data5, batchsize=batch, ae=ae5, optimizer=optimizer2)
trainer2.run()

hidn_train1 = out2(ae1, train_data1)
hidn_test1 = out2(ae1, test_data1)

hidn_train2 = out2(ae2, train_data2)
hidn_test2 = out2(ae2, test_data2)

hidn_train3 = out2(ae3, train_data3)
hidn_test3 = out2(ae3, test_data3)

hidn_train4 = out2(ae4, train_data4)
hidn_test4 = out2(ae4, test_data4)

hidn_train5 = out2(ae5, train_data5)
hidn_test5 = out2(ae5, test_data5)

hidn_vaild5 = out2(ae5, vaild_data5)

###########################################Deep CNN MHME Training #################################
model = mh_me(hid, n1_classes, n2_classes, n3_classes, n4_classes, n_classes)
optimizer = optimizers.MomentumSGD(lr1)
optimizer.setup(model)
cuda.get_device(d).use()
model.to_gpu(d)

#chainer.serializers.save_npz("mh_I_bp_fe.npz", model1.FE) #parameters save
#chainer.serializers.load_npz("mh_I_bp_fe.npz", model.FE)  #parameters upload

#chainer.serializers.save_npz("mh_I_mf_fe.npz", model1.FE)
#chainer.serializers.load_npz("mh_I_mf_fe.npz", model.FE)

trainer = Trainer_all(epoch, x_train_transformed, hidn_train1, hidn_train2, hidn_train3, hidn_train4, hidn_train5, y_train, model=model, optimizer=optimizer, batchsize=batch)
trainer.run(test_data=x_test_transformed, test_data1=hidn_test1, test_data2=hidn_test2, test_data3=hidn_test3, test_data4=hidn_test4, test_data5=hidn_test5, test_label=y_test)

##################################################################################################

def hid_out(x, mod, col_size):
	hid_data = [] 
	for i in tqdm(range(0, x.shape[0], 100)): 
		hid = mod(model.FE(c_gpu(x[i: i+100])))
		hid_data.append(hid.data) 
	hid_data = np.array(hid_data).reshape(x.shape[0], col_size).astype(np.float32)

	return hid_data

hid_train1 = hid_out(reshapei(x_train_transformed), model.o1, n1_classes)
hid_train2 = hid_out(reshapei(x_train_transformed), model.o2, n2_classes)
hid_train3 = hid_out(reshapei(x_train_transformed), model.o3, n3_classes)
hid_train4 = hid_out(reshapei(x_train_transformed), model.o4, n4_classes)

hid_train5 = hid_out(reshapei(x_train_transformed), model.o5, n_classes)
hid_train_g = hid_out(reshapei(x_train_transformed), model.o, n_classes)

vaild5 = hid_out1(reshapei(x_vaild_transformed), model.o5, n_classes)
vaild_g = hid_out1(reshapei(x_vaild_transformed), model.o, n_classes)

test1 = hid_out(reshapei(x_test_transformed), model.o1, n1_classes)
test2 = hid_out(reshapei(x_test_transformed), model.o2, n2_classes)
test3 = hid_out(reshapei(x_test_transformed), model.o3, n3_classes)
test4 = hid_out(reshapei(x_test_transformed), model.o4, n4_classes)

test5 = hid_out(reshapei(x_test_transformed), model.o5, n_classes)
test_g = hid_out(reshapei(x_test_transformed), model.o, n_classes)

#D.C:Repeat MHME training until an appropriate stop condition is satisfied.
"""
################################# two ############################################
optimizer = optimizers.MomentumSGD(lr1)
optimizer.setup(model)
cuda.get_device(d).use()
model.to_gpu(d)

trainer = Trainer_all(epoch, x_train_transformed, hidn_train1, hidn_train2, hidn_train3, hidn_train4, hidn_train5, y_train, n_classes, model = model, optimizer=optimizer, batchsize=batch)
trainer.run(test_data=x_test_transformed, test_data1=hidn_test1, test_data2=hidn_test2, test_data3=hidn_test3, test_data4=hidn_test4, test_data5=hidn_test5, test_label=y_test)


####################### three ################################
optimizer = optimizers.MomentumSGD(lr1*0.25)
optimizer.setup(model)
cuda.get_device(d).use()
model.to_gpu(d)

trainer = Trainer_all(epoch, x_train_transformed, hidn_train1, hidn_train2, hidn_train3, hidn_train4, hidn_train5, y_train, n_classes, model = model, optimizer=optimizer, batchsize=batch)
trainer.run(test_data=x_test_transformed, test_data1=hidn_test1, test_data2=hidn_test2, test_data3=hidn_test3, test_data4=hidn_test4, test_data5=hidn_test5, test_label=y_test)


####################### four ################################

optimizer = optimizers.MomentumSGD(lr1*0.125)
optimizer.setup(model)
cuda.get_device(d).use()
model.to_gpu(d)

trainer = Trainer_all(epoch, x_train_transformed, hidn_train1, hidn_train2, hidn_train3, hidn_train4, hidn_train5, y_train, n_classes, model = model, optimizer=optimizer, batchsize=batch)
trainer.run(test_data=x_test_transformed, test_data1=hidn_test1, test_data2=hidn_test2, test_data3=hidn_test3, test_data4=hidn_test4, test_data5=hidn_test5, test_label=y_test)

