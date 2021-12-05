import argparse
import os
import numpy as np
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
from tensorflow import keras
import io

STAGE = "Transfer Learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_even_odd_labels(list_of_labels):
    for idx,label in enumerate(list_of_labels):
        list_of_labels[idx] = np.where(label%2 == 0,1,0)
    return list_of_labels

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    pass

    ## get the data
    (X_train_full,y_train_full),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full/255.0
    X_test = X_test/255.0
    X_valid,X_train = X_train_full[:5000],X_train_full[5000:]
    y_valid,y_train = y_train_full[:5000],y_train_full[5000:]

    y_train_binary,y_valid_binary,y_test_binary = update_even_odd_labels([y_train,y_valid,y_test])

    ## set the seeds
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # model.summary()
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn = lambda x: stream.write(f"{(x)}\n"))
            summary_str = stream.getvalue()
        return summary_str    

    ## load the base model
    base_model_path = os.path.join("artifacts","models","base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    logging.info(f"Loaded base model summary: \n{_log_model_summary(base_model)}")

    ## Freeze the weights
    for layer in base_model.layers[:-1]:
    # last layer i.e. output layer is to be trained since the output is even odd instead of 10 outputs i.e. 0 - 9
    # we are building a model using last trained model so we are not training the weights of first and hidden layers
    # since these layers have already trained to recognize 0 - 9
    # training the outlayer only since output is changed now from 0 - 9 to even,odd 
    # Freezing the weights of all layers leaving the outlayer unchanged
        print(f'trainable status before for the {layer.name} : {layer.trainable}')
        layer.trainable = False
        print(f'trainable status after for the {layer.name} : {layer.trainable}')

    base_layer =base_model.layers[:-1]# not taking earlier output layer since we are looking for odd,even output
    ## define the model and train it
    new_model = tf.keras.models.Sequential(base_layer)
    # adding new output layer
    new_model.add(
        tf.keras.layers.Dense(2,activation="softmax",name='Outputlayer')
        )
    logging.info(f"{STAGE} model summary: \n{_log_model_summary(new_model)}")
    
    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    OPTIMIZER = keras.optimizers.SGD(learning_rate=1e-3)
    METRICS =['accuracy']

    new_model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=METRICS)

    # ## Train the model
    history = new_model.fit(X_train, y_train_binary,
        epochs=10,
        validation_data = (X_valid,y_valid_binary),
        verbose=True
    )

    # ## Save the model
    model_dir_path = os.path.join("artifacts","models")    

    model_file_path = os.path.join(model_dir_path,"even_odd_model.h5")
    new_model.save(model_file_path)

    logging.info(f'base model is saved at {model_file_path}')
    logging.info(f'evaluation metrics {new_model.evaluate(X_test, y_test_binary)}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e