"""
	this is just practice python code
	and i not executed and review this code
	i only executed and checked in ipython notebook.

"""

from keras.models import Sequential, model
from keras.layers import *

def sequential_model():
	model = Sequential()

	## Convolution 2d
	model.add(Conv2D(filters = 32, kernel_size =  (5,5),
		padding = "same", activation = "relu", input_shape = (28,28,1)))
	## Max pooling 2d
	model.add(MaxPooling2D(pool_size = (2,2), stride = 2))
	## Drop out
	model.add(Dropout(rate = 0.5))
	## Flatten
	model.add(Flatten())
	## Dense
	model.add(Dense(units = 10, activation = "softmax"))
	## compile
	return model

def functional_model():
	## return to tensor
	X = Input(shape = (28, 28, 1))

	## Convolution 2d
	conv = Conv2D(filters = 64, kernel_size = (5,5),
		padding = "same", activation = "relu")(X)
	## Max Pollling 2d
	pool = MaxPooling2D(pool_size = (2,2))(conv)
	## Drop out
	dropout = Dropout(rate = 0.4)(pool)
	## Dense
	dense = Dense(units = 10, activation = "softmax")(dropout)

	model = Model(inputs = X, outputs = dense)
	return model

def run(model):
	## complie
	model.compile(optimizer = "Adam", loss = "categorical_crossentropy",
		metrics = ['accuracy'])
	## fit
	model.fit(data['train_img'], data['train_lb'], epoch = 10, batch_size = 10)
	## evalute
	score = model.evalute(data['test_img'], data['test_lb'], batch_size = 10)




