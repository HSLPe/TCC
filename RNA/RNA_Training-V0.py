import os
from matplotlib import units
import tensorflow as tf
import keras_tuner as kt
import time
from tensorflow import keras
import datetime as dt



from sklearn import datasets
from sklearn.model_selection import train_test_split


os.system('cls' if os.name == 'nt' else 'clear')    # Limpa terminal
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PATH=os.path.dirname(os.path.realpath(__file__))

cancer = datasets.load_breast_cancer()
start_time=time.time()
X_treino, X_valid, y_treino, y_valid = train_test_split(cancer.data,cancer.target,test_size=0.3,train_size=0.7)
# Criando a RNA
def montar_modelo(hp):
    inputs = keras.Input(shape=30)                              # Define que a entrada tem dimensão 30 (Qunaitdade de 'features' do datasset)
    x=keras.layers.Dense(
        units=hp.Int("Camada 1",min_value=5,max_value=50,step=1),
        activation="relu")(inputs)          # Monta primeira camada da RNA com 10 neurônios, ativação relu
    x2=keras.layers.Dense(
        units=hp.Int("Camada 2",min_value=5,max_value=50,step=1),
        activation="relu")(x)              # Monta segunda camada
    outputs = keras.layers.Dense(2,activation="softmax")(x2)     # Camada de saida
    RNA1=keras.Model(inputs,outputs)                            # Monta o modelo                                              # Imprimi no terminal o modelo criado
    #Consolidando o modelo
    #ADAM: Gradiente Descendente Estocástico com estimador adaptativo
    #SGD: Gradiente Descendente Estocastico com momento
    Opt=keras.optimizers.Adam(learning_rate=0.01)
    lossFcn=keras.losses.SparseCategoricalCrossentropy()
    RNA1.compile(optimizer=Opt,                              
                loss=lossFcn,
                metrics="accuracy")
    return RNA1
def montar_modelo2(hp):
    inputs = keras.Input(shape=30)                              # Define que a entrada tem dimensão 30 (Qunaitdade de 'features' do datasset)
    x=keras.layers.Dense(
        units=hp.Int("units3",min_value=5,max_value=50,step=1),
        activation="relu")(inputs)          # Monta primeira camada da RNA com 10 neurônios, ativação relu
    x2=keras.layers.Dense(
        units=hp.Int("units4",min_value=5,max_value=50,step=1),
        activation="relu")(x)              # Monta segunda camada
    outputs = keras.layers.Dense(2,activation="softmax")(x2)     # Camada de saida
    RNA2=keras.Model(inputs,outputs)                            # Monta o modelo
    #Consolidando o modelo
    #ADAM: Gradiente Descendente Estocástico com estimador adaptativo
    #SGD: Gradiente Descendente Estocastico com momento
    Opt=keras.optimizers.Adam(learning_rate=0.01)
    lossFcn=keras.losses.SparseCategoricalCrossentropy()
    RNA2.compile(optimizer=Opt,                              
                loss=lossFcn,
                metrics="accuracy")
    return RNA2

log_dir = PATH+r'/TrainigData/' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

tuner=kt.RandomSearch(
    hypermodel=montar_modelo,
    objective='accuracy',
    max_trials=5,
    overwrite=True,
    directory=log_dir,
    project_name="CheckPoints",
)

# tuner=kt.RandomSearch(
#     hypermodel=montar_modelo2,
#     objective='accuracy',
#     max_trials=5,
#     overwrite=True,
#     directory=log_dir,
#     project_name="RNATeste-acc",
# )
batch_size=5                                               # Tomando uma amostra de treinamento sendo 30% dos dados disponíveis
Epocas=50                                                  # Epocas de treinamento
TrainDataset=tf.data.Dataset.from_tensor_slices((X_treino,y_treino)).batch(batch_size)
ValDataset = tf.data.Dataset.from_tensor_slices((X_valid,y_valid)).batch(batch_size)

tuner.search(X_treino,y_treino,epochs=Epocas,validation_data=(X_valid,y_valid),callbacks=[tensorboard_callback])

Modelos=tuner.get_best_models()[0]
Modelos.save(log_dir+r'/Modelo.h5')
# for i in range(0,len(Modelos)):
#     Modelos[i].save(log_dir+r'/Modelo_'+str(i)+r'.h5')
Modelos.evaluate(ValDataset)
print('Processo concluido em %f s '%(time.time()-start_time))