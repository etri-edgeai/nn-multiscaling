import tensorflow as tf

MODELS = [
    tf.keras.applications.ResNet50V2,
    tf.keras.applications.InceptionResNetV2,
    tf.keras.applications.MobileNetV2,
    tf.keras.applications.MobileNet,
    tf.keras.applications.DenseNet121,
    tf.keras.applications.NASNetMobile,
    tf.keras.applications.EfficientNetB0,
    tf.keras.applications.EfficientNetB1,
    tf.keras.applications.EfficientNetB2,
    tf.keras.applications.EfficientNetB3,
    tf.keras.applications.EfficientNetB6,
    tf.keras.applications.EfficientNetV2B0,
    tf.keras.applications.EfficientNetV2B1,
    tf.keras.applications.EfficientNetV2S
]

endpoints = {
    "EfficientNetB1":[
        "block1b_add",
        "block2c_add",
        "block3c_add",
        "block5d_add",
        "block7b_add"
    ],
    "EfficientNetB2":[
        "block1b_add",
        "block2c_add",
        "block3c_add",
        "block4d_add",
        "block5d_add",
        "block6e_add",
        "block7b_add"
    ],
    "EfficientNetB3":[
        "block1b_add",
        "block2c_add",
        "block3c_add",
        "block5e_add",
        "block7b_add"
    ]
}

def get_model(model_name, config):
    model_class = None
    for model_ in MODELS:
        if model_name in model_.__name__:
            model_class = model_
            break
    model = model_class(weights="imagenet", include_top=False)
    tf.keras.utils.plot_model(model, "backbone.png", show_shapes=True)
    outputs = [
        layer.output for layer in model.layers if layer.name in endpoints[model_name]
    ]
    model = tf.keras.Model(inputs=model.inputs, outputs=[model.outputs]+outputs)

    x_ = tf.keras.Input(shape=model.input.shape[1:])
    xs = model(x_)
    x = tf.keras.layers.GlobalAveragePooling2D()(xs[0][0])
    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Dense(config["task"]["num_classes"], activation='softmax')(x)

    return tf.keras.Model(inputs=x_, outputs=[y] + xs[1:])
