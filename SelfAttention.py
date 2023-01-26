import tensorflow as tf

class Attention_Layer(tf.keras.Model):
    def __init__(
        self,
        kernel_size=1,
        filters=32,
        channelwise=False,
        dropout=0.2,
        skip_con=False,
    ):
        super(Attention_Layer, self).__init__(name="Attention_Layer")

        self.skip_con = skip_con
        self.skip_con_layer = tf.keras.layers.LocallyConnected1D(filters, kernel_size)

        self.conv_query = tf.keras.layers.LocallyConnected1D(filters, kernel_size)

        self.conv_value = tf.keras.layers.LocallyConnected1D(filters, kernel_size)
        self.conv_key = tf.keras.layers.LocallyConnected1D(filters, kernel_size)
        if channelwise:
            self.permute_layer = tf.keras.layers.Permute((2, 1))
        else:
            self.permute_layer = tf.keras.layers.Activation("linear")

        self.attention_op = tf.keras.layers.Attention(use_scale=True, dropout=dropout)

    def call(self, x, training=False, return_attention_scores=False):
        x_q = self.conv_query(x, training=training)
        x_q = self.permute_layer(x_q)
        x_v = self.conv_value(x, training=training)
        x_v = self.permute_layer(x_v)
        x_k = self.conv_key(x, training=training)
        x_k = self.permute_layer(x_k)
        if self.skip_con:
            x_skip = self.skip_con_layer(x)
        if return_attention_scores:
            x_v, att = self.attention_op(
                [x_q, x_v, x_k],
                return_attention_scores=return_attention_scores,
                training=training,
            )
            x_v = self.permute_layer(x_v)
            if self.skip_con:
                x_v = x_v + x_skip
            return x_v, att
        else:
            x_v = self.attention_op([x_q, x_v, x_k], training=training)
            x_v = self.permute_layer(x_v)
            if self.skip_con:
                x_v = x_v + x_skip
            return x_v
