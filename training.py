
import os
import glob
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(1)
np.random.seed(1)

import warnings
warnings.filterwarnings("ignore")

class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_path, batch_size=32, shuffle=True):
        from s3fs.core import S3FileSystem
        self.s3 = S3FileSystem()
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_types = ["train_frames_x", "train_frames_y", "train_frames_embed", "train_frames_decoder_input"]
        self.indices = []
        for i in range(1, 2):
            for j in range(1, 2):
                self.indices += [f"_{i}_{j}"]
        self.on_epoch_end()
        self.total_samples = 0
        self.num_samples_per_index = {}
        for index in self.indices:
            data = np.load(self.s3.open(os.path.join(self.file_path, self.file_types[3] + index + ".pkl")), allow_pickle=True)
            self.total_samples += data.shape[0]
            self.num_samples_per_index[index] = data.shape[0]

    def __len__(self):
        return int(np.floor(self.total_samples/self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        lower_idx = index * self.batch_size
        upper_idx = (index+1) * self.batch_size - 1
        
        start = 0
        end = 0
        begin = 0
        finish = 0
        starting_idx = ""
        ending_idx = ""
        
        for idx in self.indices:
            end = start + self.num_samples_per_index[idx]
            if lower_idx < end and lower_idx >= start:
                starting_idx = idx
                begin = lower_idx-start
            if upper_idx < end and upper_idx >= start:
                ending_idx = idx
                finish = upper_idx-start
                break
            start = end
        
        train_data_x, train_data_embed, train_data_decoder_input, train_data_y = [], [], [], []
        started = False
        for idx in self.indices:
            if not started and idx != starting_idx:
                continue
            if idx == starting_idx:
                if idx == ending_idx:
                    train_data_x += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_x"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
                    train_data_y += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_y"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
                    train_data_embed += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_embed"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
                    train_data_decoder_input += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_decoder_input"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
                    break
                else:
                    train_data_x += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_x"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
                    train_data_y += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_y"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
                    train_data_embed += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_embed"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
                    train_data_decoder_input += [self.s3.open(np.load(os.path.join(self.file_path, "train_frames_decoder_input"+idx+".pkl")), allow_pickle=True)[begin:finish+1]]
            elif idx != ending_idx:
                train_data_x += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_x"+idx+".pkl")), allow_pickle=True)]
                train_data_y += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_y"+idx+".pkl")), allow_pickle=True)]
                train_data_embed += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_embed"+idx+".pkl")), allow_pickle=True)]
                train_data_decoder_input += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_decoder_input"+idx+".pkl")), allow_pickle=True)]
            elif idx == ending_idx:
                train_data_x += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_x"+idx+".pkl")), allow_pickle=True)[:finish+1]]
                train_data_y += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_y"+idx+".pkl")), allow_pickle=True)[:finish+1]]
                train_data_embed += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_embed"+idx+".pkl")), allow_pickle=True)[:finish+1]]
                train_data_decoder_input += [np.load(self.s3.open(os.path.join(self.file_path, "train_frames_decoder_input"+idx+".pkl")), allow_pickle=True)[:finish+1]]
        
        train_data_x, train_data_y, train_data_embed, train_data_decoder_input = np.concatenate(train_data_x), np.concatenate(train_data_y), np.concatenate(train_data_embed), np.concatenate(train_data_decoder_input)
        
        return [train_data_x[:, :, 1:], train_data_embed[:, 1, 1:], train_data_decoder_input[:, :, 2:]], train_data_y[:, :, 1:]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def load_data(train_data_path):
    
    train_data = {}
    test_data = {}
    
    file_types = ["train_frames_x_1", "train_frames_y_1", "train_frames_embed_1", "train_frames_decoder_input_1"]
    for file_type in file_types:
        
        data = []
        for file in sorted(glob.glob(os.path.join(train_data_path, file_type+"*"))):
            data.append(np.load(str(file), allow_pickle=True))
        data = np.array(data)
        train_data[file_type] = data
        
        data = []
        for file in sorted(glob.glob(os.path.join("test" + train_data_path[5:], file_type+"*"))):
            data.append(np.load(str(file), allow_pickle=True))
        data = np.array(data)
        test_data[file_type] = data
        
    return train_data, test_data

def define_model(num_embedding_features, num_historical_features, historical_data_window, future_prediction_window):
    
    encoder_input = keras.Input(shape=(historical_data_window, num_historical_features))
    encoder_lstm1 = layers.LSTM(32, return_sequences=True)(encoder_input)
    encoder_lstm1 = layers.ReLU()(encoder_lstm1)
    batch_norm1 = layers.BatchNormalization()(encoder_lstm1)
    encoder_lstm2 = layers.LSTM(64, return_sequences=True)(batch_norm1)
    encoder_lstm2 = layers.ReLU()(encoder_lstm2)
    batch_norm2 = layers.BatchNormalization()(encoder_lstm2)
    encoder_output = layers.LSTM(32)(batch_norm2)
    encoder_output = layers.ReLU()(encoder_output)

    embedding_input = keras.Input(shape=(num_embedding_features, ))
    embedding_layer1 = layers.Dense(32)(embedding_input)
    embedding_layer1 = layers.ReLU()(embedding_layer1)
    batch_norm3 = layers.BatchNormalization()(embedding_layer1)
    embedding_layer2 = layers.Dense(64)(batch_norm3)
    embedding_layer2 = layers.ReLU()(embedding_layer2)
    batch_norm4 = layers.BatchNormalization()(embedding_layer2)
    embedding_output = layers.Dense(32)(batch_norm4)
    embedding_output = layers.ReLU()(embedding_output)

    embedding_encoder_concatenate = layers.Concatenate()([encoder_output, embedding_output])
    embedding_encoder_concatenate = layers.Dense(32)(embedding_encoder_concatenate)
    embedding_encoder_concatenate = layers.ReLU()(embedding_encoder_concatenate)
    future_cpc = keras.Input(shape=(FUTURE_PREDICTION_WINDOW, ))

    decoder_input = layers.RepeatVector(future_prediction_window)(embedding_encoder_concatenate)
    future_cpc = layers.Reshape((-1, 1))(future_cpc)
    decoder_input = layers.Concatenate()([decoder_input, future_cpc])
    decoder_lstm1 = layers.LSTM(32, return_sequences=True)(decoder_input)
    decoder_lstm1 = layers.ReLU()(decoder_lstm1)
    batch_norm5 = layers.BatchNormalization()(decoder_lstm1)
    decoder_lstm2 = layers.LSTM(16, return_sequences=True)(batch_norm5)
    decoder_lstm2 = layers.ReLU()(decoder_lstm2)
    batch_norm6 = layers.BatchNormalization()(decoder_lstm2)
    decoder_output = layers.LSTM(2, return_sequences=True)(batch_norm6)
    decoder_output_fin = layers.ReLU()(decoder_output)

    # Create the model
    ED_lstm_model = tf.keras.Model(inputs=[encoder_input, embedding_input, future_cpc], outputs=decoder_output_fin)
    ED_lstm_model.compile(optimizer="adam", loss='mean_squared_error')
    
    return ED_lstm_model

def prepare_data(train_data, test_data):
    
    for key, value in train_data.items():
        train_data[key] = np.nan_to_num(np.array(value).astype(np.float32)).astype(np.float32)
        
    for key, value in test_data.items():
        test_data[key] = np.nan_to_num(np.array(value).astype(np.float32)).astype(np.float32)
        
    return train_data, test_data

def train_model(epochs, batchsize, model_path, model, train_generator, valid_generator):
    
    history = model.fit_generator(
                         generator = train_generator,
                         validation_data = valid_generator,
                         use_multiprocessing = True
                         workers=2,
                         verbose=1,
                         callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=1,
                                mode='min'
                            ),
                            tf.keras.callbacks.ModelCheckpoint(
                                model_path,
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min',
                                verbose=1
                            )
                         ],
                     )
    return history
                         
if __name__=="__main__":
    
    path = "/opt/ml/processing/input/data/"
    model_path = "/opt/ml/processing/output/trained_model.ckpt"
    epochs = 10
    batch_size = 32

    train_data_gen = DataGenerator("s3://training-data-lstm/processed_data_1/", 32, True, "train")
    val_data_gen = DataGenerator("s3://training-data-lstm/processed_data_1/", 32, True, "test")
    
    num_embedding_features = 14
    num_historical_features = 14
    historical_data_window = 14
    future_prediction_window = 3
                         
    model = define_model(num_embedding_features, num_historical_features, historical_data_window, future_prediction_window)
    history = train_model(epochs, batch_size, model_path, model, train_data_gen, val_data_gen)
    
                        
