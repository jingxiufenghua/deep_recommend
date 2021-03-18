# wide&deep  根据网络结构自己重新编写


"""

class WideLayer(layers.Layer):
    def __init__(self):
        super(WideLayer, self).__init__()

    def build(self, input_shape):
        super(WideLayer, self).build(input_shape)

        self.weight = self.add_weight(shape=(int((input_shape[1] * (input_shape[1] - 1)) / 2), 1),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)

    def call(self, inputs):
        t=[]
        for ii in tqdm(range(batch_size)):
            X=inputs[ii]
            i=0
            outputs = []
            for j1 in range(X.shape[0]):
                for j2 in range(j1+1,X.shape[0]):
                    out=(self.weight[i]*X[j1]*X[j2])[0]
                    outputs.append(out)
                    i+=1
            t.append(tf.cast(outputs,tf.float32))
        return tf.cast(t,tf.float32)
