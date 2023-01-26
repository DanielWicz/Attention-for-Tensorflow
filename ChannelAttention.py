import tensorflow as tf

class Channel_Attention(tf.keras.Model):
    def __init__(self, kernel_size=1, dropout=0.2):
        super(Channel_Attention, self).__init__(name="")
        # we don't use filter size if we use LC, 1 filter and permute
        self.conv1 = tf.keras.layers.LocallyConnected1D(
            1, kernel_size, activation="relu"
        )
        self.conv2 = tf.keras.layers.LocallyConnected1D(
            1, kernel_size, activation="sigmoid"
        )
        self.permute_layer = tf.keras.layers.Permute((2, 1))
        self.gap = tf.keras.layers.GlobalAvgPool1D(keepdims=True)
        self.dropout = tf.keras.layers.GaussianDropout(dropout)

    def call(self, x, training=False, return_attention_scores=False):
        # collapse timesteps to 1
        x_a = self.gap(x)
        # reconstruct timesteps based on channels
        # (Batch, dim, 1) - over dims
        x_a = self.permute_layer(x_a)
        x_a = self.conv1(x_a, training=training)
        x_a = self.conv2(x_a, training=training)
        # (Batch, 1, dim)
        x_a = self.permute_layer(x_a)
        x_a = self.dropout(x_a, training=training)
        x = x * x_a
        if return_attention_scores:
            return x, x_a
        else:
            return x
