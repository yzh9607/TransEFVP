import os
import numpy as np
import joblib
import tensorflow as tf
from Encoder import Encoder
from model import get_model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, auc, roc_curve


def train():
    X_train_esm = np.load('../data/train_esm.npy')
    X_train_protT5 = np.load('../data/train_protT5.npy')
    y_train = np.load('../data/train_label.npy')
    
    batch_size = 1024
    earlyStopPatience = 10
    monitor = 'val_loss'
    input_train = [X_train_esm, X_train_protT5]

    encoder_model = get_model()
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=earlyStopPatience, verbose=1, mode='auto')
    log_dir = "./model/logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    history_callback = encoder_model.fit(x=input_train, y=y_train, batch_size=batch_size, epochs=10000, verbose=1,
                                         callbacks=[earlystopping], validation_split=0.1, shuffle=True, )
    encoder_model.save_weights("./model/encoder_model.h5")

    feature_layer = tf.keras.models.Model(inputs=encoder_model.input,
                                          outputs=encoder_model.get_layer('concatenate').output)
    train_feature_output = feature_layer.predict(input_train)
    pca_model = PCA(n_components=0.9)
    train_feature_output_pca = pca_model.fit_transform(train_feature_output)
    svc_model = SVC()
    svc_model.fit(train_feature_output_pca, y_train)
    joblib.dump(pca_model, "./model/pca_model.pkl")
    joblib.dump(svc_model, "./model/svc_model.pkl")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    train()
