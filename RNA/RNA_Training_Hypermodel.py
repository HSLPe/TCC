import os
from matplotlib import units
import tensorflow as tf
import keras_tuner as kt
import time
from tensorflow import keras
import datetime as dt
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


from sklearn import datasets
from sklearn.model_selection import train_test_split


os.system('cls' if os.name == 'nt' else 'clear')    # Limpa terminal
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PATH=os.path.dirname(os.path.realpath(__file__))
cancer = datasets.load_breast_cancer()
start_time=time.time()
X_treino, X_valid, y_treino, y_valid = train_test_split(cancer.data,cancer.target,test_size=0.3,train_size=0.7)
SavePath=r'TrainHyper'
def RNA(hp):
    inputs = keras.Input(shape=30,name="Entrada")                             
    x=keras.layers.Dense(
        units=hp.Int("Camada 1",min_value=2,max_value=150,step=1),
        activation="relu",
        name=r'Escondida1')(inputs)          
    x2=keras.layers.Dense(
        units=hp.Int("Camada 2",min_value=2,max_value=150,step=1),
        activation="relu",
        name=r'Escondida2')(x)
    x3=keras.layers.Dense(
        units=hp.Int("Camada 2",min_value=2,max_value=150,step=1),
        activation="relu",
        name=r'Escondida3')(x2)
    outputs = keras.layers.Dense(2,activation="softmax",name="Saida")(x3)    
    RNA=keras.Model(inputs,outputs)                                                                          
    Opt=keras.optimizers.Adam(learning_rate=0.01)
    lossFcn=keras.losses.SparseCategoricalCrossentropy()
    RNA.compile(optimizer=Opt,                              
                loss=lossFcn,
                metrics="accuracy")
    return RNA

tune=kt.Hyperband(RNA,
                objective='val_accuracy',
                max_epochs=15,
                factor=3,
                directory=SavePath,
                project_name='Tuner_Hyperband')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tune.search(X_treino, y_treino, epochs=14, validation_split=0.2, callbacks=[stop_early])
best_hps=tune.get_best_hyperparameters(num_trials=1)[0]

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

modelo=tune.hypermodel.build(best_hps)
history = modelo.fit(X_treino, y_treino, epochs=50, validation_split=0.2)
eval_result = modelo.evaluate(X_valid, y_valid)
print("[test loss, test accuracy]:", eval_result)
plot_model(modelo, to_file=SavePath+r'/model_plot.png', show_shapes=True, show_layer_names=True)


modelo.summary()
modelo.save(SavePath+r'/ModeloHyper.h5')
plt.plot(history.history['accuracy'],'x')
plt.plot(history.history['val_accuracy'],'*')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()

