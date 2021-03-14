from tensorflow.keras.layers import *
import numpy as np
import tensorflow as tf
import warnings

class ResidualConvBlock(tf.keras.Sequential):
    def __init__(self, kernel_shapes, filters, strides=(1,1), padding="same", activation=tf.keras.layers.Activation('linear'), norm=tf.keras.layers.BatchNormalization):
        super(ResidualConvBlock, self).__init__(name='')

        self.kernel_shapes = kernel_shapes
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.norm = norm
        
        self.norm_adapter = 1 

        if type(self.filters) is int or len(self.filters) == 1:
            self.filters = int(self.filters)
            self.filters = (self.filters, self.filters, self.filters)
            warnings.warn("'filters' argument should be tuple with >1 variable but you entered variable or 1 length tuple it\n converted to tuple with length 3",
                        RuntimeWarning)
        if (not self.filters[0] == self.filters[-1] == self.filters[-2] or not self.kernel_shapes[0] == self.kernel_shapes[-1] == self.kernel_shapes[-2]) and self.padding == "valid":
            self.padding = "same"
            warnings.warn("if your first and last 2 convolution have different number of filters or diferent filter shapes you\n must use 'same' for padding but you used 'valid' because of this it converted to same",
                        RuntimeWarning)

        self._layers = []
        for i in range(len(self.kernel_shapes)):
            self._layers.append(tf.keras.layers.Conv2D(self.filters[i], self.kernel_shapes[i], strides=self.strides, padding=self.padding, activation=self.activation))
            if self.norm != None:
                self._layers.append(self.norm())
                self.norm_adapter = 2
        self._layers.append(self.activation)

    def call(self, input_tensor, training=False):
        outs = []
        
        out = InputLayer(input_shape=input_tensor.shape)(input_tensor)
        for layer_ind in range(len(self._layers)):
            layer = self._layers[layer_ind]
            out = layer(out)
            outs.append(out)
            """
            if (layer_ind+1)%int(len(self.filters)*self.norm_adapter) == 0 and layer_ind != 0:
                print("h"*50)
                print(layer_ind+1-int(len(self.filters)*self.norm_adapter))
                old_out = outs[layer_ind+1-int(len(self.filters)*self.norm_adapter)]
                out = Concatenate()([out, old_out])
            """
            if (layer_ind+1)==len(self.filters)*self.norm_adapter:
                old_out = outs[layer_ind+1-int(len(self.filters)*self.norm_adapter)]
                out = tf.keras.layers.Concatenate()([out, old_out])
        return out
l = ResidualConvBlock((3, 3, 3), (128, 256, 512), padding="valid")
_=l(tf.zeros([2,5,5,3]))
l.summary()