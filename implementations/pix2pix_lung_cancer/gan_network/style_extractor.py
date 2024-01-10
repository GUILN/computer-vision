from typing import List
import tensorflow as tf

upsample_layer_names: List[str] = []
for i in range(9, 15):
    upsample_layer_names.append("sequential_" + str(i))


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(
        self, style_layers: List[str], content_layers: List[str], u_net: tf.keras.Model
    ):
        super(StyleContentModel, self).__init__()
        self.u_net = self.unet_upsample_layers(u_net, style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def unet_upsample_layers(
        u_net: tf.keras.Model, layer_names: List[str]
    ) -> tf.keras.Model:
        outputs = [u_net.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([u_net.input], outputs)
        return model

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = self.u_net.preprocess_input(inputs)
        outputs = self.u_net(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }
        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }
        return {"content": content_dict, "style": style_dict}
