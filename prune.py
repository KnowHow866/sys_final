# 3rd module
import tensorflow as tf
from tensorflow import keras as keras
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense

# native module
import os 

# local module
import setting

def inspect_layer_conv(layer):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        weight = layer.weights[0]
        weight = sess.run(weight)
        print(dir(layer))
        print(layer.output)
        print(weight[0 , : , : ,0])

def inspect_layer_dense(layer):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        weight = layer.weights[0]
        weight = sess.run(weight)
        print(weight[ :,0])

def main():
    model = keras.models.load_model(setting.pruning_load_model)

    model.summary()
    print(dir(model))

    for idx, layer in enumerate(model.layers):
        print('Layer insight, No %s '.ljust(45, '.') % idx)
        print(layer)
        print(type(layer))

        if type(layer) == type(Conv2D(4, (3, 3))):
            inspect_layer_conv(layer)
        
        # if type(layer) == type(Dense(512, activation = 'relu')):
        #     inspect_layer_dense(layer)

        print(layer.weights)

        print()

    # 乾脆算變減在來觀察看看 QQ
    pruned_model = delete_channels(model, model.layers[0], channels=[0, 3])
    pruned_model.summary()
    
if __name__ == "__main__":
    main()
