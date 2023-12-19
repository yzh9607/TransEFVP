import os
import numpy as np
import tensorflow as tf
from Encoder import Encoder


def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        p_t = y_true*y_pred + (tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true)-p_t),gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed


def get_model():
    inputESM = tf.keras.layers.Input(shape=(6, 1280))
    inputProtT5 = tf.keras.layers.Input(shape=(4, 1024))
    featureESM = tf.keras.layers.Dense(512)(inputESM)
    featureESM = Encoder(4, 512, 8, 1024, rate=0.3)(featureESM)
    featureESM = tf.keras.layers.Flatten()(featureESM)
    featureProtT5 = tf.keras.layers.Dense(512)(inputProtT5)
    featureProtT5 = Encoder(4, 512, 8, 1024, rate=0.3)(featureProtT5)
    featureProtT5 = tf.keras.layers.Flatten()(featureProtT5)
    featureConcat = tf.keras.layers.Concatenate()([featureESM, featureProtT5])
    feature = tf.keras.layers.Dense(512, activation='relu')(featureConcat)
    feature = tf.keras.layers.Dense(256, activation='relu')(feature)
    feature = tf.keras.layers.Dense(128, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.1)(feature)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(feature)

    model = tf.keras.models.Model(inputs=[inputESM, inputProtT5], outputs=y)
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model
