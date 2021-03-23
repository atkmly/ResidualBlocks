import tensorflow
import warnings

class ResidualConvBlock(tensorflow.keras.Sequential):
    def __init__(self,layers, activation=tensorflow.keras.layers.Activation('linear'), norm=tensorflow.keras.layers.BatchNormalization):
        super(ResidualConvBlock, self).__init__(name='')

        self.layers = layers
        self.activation = activation
        self.norm = norm
        
        self.norm_adapter = 1 

        self._layers = []
        for i in range(len(self.layers)):
            self._layers.append(tensorflow.keras.layers.Dense(self.layers[i]))
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

            if (layer_ind+1)==len(self.filters)*self.norm_adapter:
                old_out = outs[layer_ind+1-int(len(self.filters)*self.norm_adapter)]
                out = tensorflow.keras.layers.Concatenate()([out, old_out])
        return out
