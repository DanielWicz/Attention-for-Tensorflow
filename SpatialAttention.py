import tensorflow as tf

class Spatial_Attention(tf.keras.Model):
    def __init__(self, kernel_size=1, dropout=0.2):
        super(Spatial_Attention, self).__init__(name="")
        self.conv1 = tf.keras.layers.LocallyConnected1D(
            1, kernel_size, activation="sigmoid"
        )
        self.gap = tf.keras.layers.GlobalAvgPool1D(keepdims=True)
        self.gmp = tf.keras.layers.GlobalMaxPool1D(keepdims=True)
        self.permute_layer = tf.keras.layers.Permute((2, 1))
        self.dropout = tf.keras.layers.GaussianDropout(dropout)

    def call(self, x, training=False, return_attention_scores=False):
        # so that we average of channels, not timesteps
        # (Batch, dim, Tq)
        x_a = self.permute_layer(x)
        # (Batch, 2, Tq), collapse channels
        x_a = tf.concat([self.gap(x_a) + self.gmp(x_a)], axis=1)
        x_a = self.permute_layer(x_a)
        # (Batch, Tq, 1)
        # reconstruct channels based on timesteps
        x_a = self.conv1(x_a, training=training)
        x_a = self.dropout(x_a, training=training)
        x = x * x_a
        if return_attention_scores:
            return x, x_a
        else:
            return x
