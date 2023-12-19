import os
import numpy as np
import joblib
import tensorflow as tf
from Encoder import Encoder
from model import get_model
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, auc, roc_curve


def test():
    X_test_esm = np.load('../data/test_esm.npy')
    X_test_protT5 = np.load('../data/test_protT5.npy')
    y_test = np.load('../data/test_label.npy')
    input_test = [X_test_esm, X_test_protT5]

    encoder_model = get_model()
    encoder_model.load_weights("./model/encoder_model.h5")
    feature_layer = tf.keras.models.Model(inputs=encoder_model.input,
                                          outputs=encoder_model.get_layer('concatenate').output)
    test_feature_output = feature_layer.predict(input_test)
    pca_model = joblib.load("./model/pca_model.pkl")
    test_feature_output_pca = pca_model.transform(test_feature_output)
    svc_model = joblib.load("./model/svc_model.pkl")
    Y_Pred = svc_model.predict(test_feature_output_pca).reshape(-1,)
    
    Y_Pred_new = []
    for value in Y_Pred:
        if value < 0.5:
            Y_Pred_new.append(0)
        else:
            Y_Pred_new.append(1)
    Y_Pred_new = np.array(Y_Pred_new)
    tn, fp, fn, tp = confusion_matrix(y_test, Y_Pred_new).ravel()

    print("ACC: " + str((tp+tn)/(tp+tn+fp+fn)))
    print("Matthews相关系数: " + str(matthews_corrcoef(y_test, Y_Pred_new)))
    print('precision:', tp / (tp + fp))
    print('sensitivity/recall:', tp / (tp + fn))
    print('specificity:', tn / (tn + fp))
    print("F1值: " + str(f1_score(y_test, Y_Pred_new)))
    print('false positive rate:', fp / (tn + fp))
    print('false discovery rate:', fp / (tp + fp))
    print('TN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)
    fpr, tpr, thresholds = roc_curve(y_test, Y_Pred_new)
    print('roc_auc:', auc(fpr, tpr))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    test()
