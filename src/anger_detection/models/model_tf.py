"""DS-CNN model definition (depthwise separable convolutions)."""

from tensorflow.keras import layers, models


def create_dscnn_model(input_shape=(98, 40, 1), num_classes=5, width=24, dropout=0.15):
    """DS-CNN with depthwise separable blocks + BatchNorm for stable training."""

    def conv_bn(filters, kernel=(3, 3)):
        return [
            layers.Conv2D(filters, kernel, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]

    def ds_block(dw_channels, pw_channels):
        return [
            layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(pw_channels, (1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    for layer in conv_bn(width):
        model.add(layer)
    for block in (
        ds_block(width, width),
        ds_block(width, width * 2),
        ds_block(width * 2, width * 2),
        ds_block(width * 2, width * 4),
    ):
        for layer in block:
            model.add(layer)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


if __name__ == "__main__":
    m = create_dscnn_model()
    m.summary()
