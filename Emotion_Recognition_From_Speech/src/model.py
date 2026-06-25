from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    TimeDistributed,
    Flatten,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)


def build_cnn_model(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    x = Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling2D((2,2))(x)

    x = Conv2D(
        64,
        (3,3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D((2,2))(x)

    # Convert CNN feature maps into sequences
    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=False
        )
    )(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model