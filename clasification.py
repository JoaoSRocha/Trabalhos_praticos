import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score


def single_class_accuracy(y_true, y_pred, INTERESTING_CLASS_ID):
    class_id_true = keras.backend.argmax(y_true, axis=-1)
    class_id_preds = keras.backend.argmax(y_pred, axis=-1)
    accuracy_mask = keras.backend.cast(keras.backend.equal(class_id_preds, INTERESTING_CLASS_ID), 'int32')
    class_acc_tensor = keras.backend.cast(keras.backend.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = keras.backend.sum(class_acc_tensor) / keras.backend.maximum(keras.backend.sum(accuracy_mask), 1)
    return class_acc


def sensitivity(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + keras.backend.epsilon())


def specificity(y_true, y_pred):
    true_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = keras.backend.sum(keras.backend.round(keras.backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + keras.backend.epsilon())


def ANN():
    model = keras.Sequential()
    model.add(keras.layers.Dense(15, input_dim=9, init='uniform', activation='relu'))
    model.add(keras.layers.Dense(4, init='uniform', activation='softmax'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.RMSprop(0.001),
        metrics=['accuracy'])
    return model


if __name__ == '__main__':
    with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/labels.pkl', 'rb') as l_id:
        labels = pkl.load(l_id)
    with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/features.pkl', 'rb') as feat_id:
        features = pkl.load(feat_id)
    seed = 7
    np.random.seed(seed)
    # X and Y values
    X = features
    Y = keras.utils.to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    estimator = KerasClassifier(build_fn=ANN, epochs=1000, batch_size=10, verbose=100)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    # model.fit(X_train, y_train, epochs=500, batch_size=10, verbose=100)
    # predictions = model.predict(X_test)
    #
    # print(predictions)
