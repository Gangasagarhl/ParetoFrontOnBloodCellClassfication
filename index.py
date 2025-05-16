"""
STEPS INVOLVED
1. Load the data set for training and validating, which return the train and valid
2. download the hyper parameters using the grid search
3. For each and every set of hyper parameters , make the model/build the model
4. Then Build the model, using epochs
5. Then save the results into csv
6. Then lad the data in the csv, and make the pareto front
"""


#from packages.build_and_return_model import build_model as BM #no parameters at init 
from packages.pareto_corner import pareto_generation  #csv file path, output directot
from packages.return_model_size import model_size as MS
from packages.save_data_into_csv_for_pareto import save 
from packages.save_summary import save_summary_file  #output directory to save the summary 
#from packages.load_dataset import load_data
import pandas as pd
import ast
from tensorflow.keras.utils import plot_model
import  matplotlib.pyplot as plt
from pareto_generate import pareto
#from pareto_generate import pareto




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense, MaxPooling2D, GlobalAveragePooling2D
)


class load_data: 

    def __init__(self, train_data_path, validation_data_path, batch_size=32, image_height=224, image_width=224, seed=123):
        self.TRAIN_ROOT = train_data_path
        self.VALIDATION_ROOT = validation_data_path
        self.batch_size = batch_size
        self.img_h = image_height
        self.img_w = image_width
        self.seed = seed

    def to_grayscale(self, image, label):
        image = tf.image.rgb_to_grayscale(image)
        return image, label

    def load_data(self):
        train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
            self.TRAIN_ROOT,
            seed=self.seed,
            image_size=(self.img_h, self.img_w),
            batch_size=self.batch_size,
        )
        val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
            self.VALIDATION_ROOT,
            seed=self.seed,
            image_size=(self.img_h, self.img_w),
            batch_size=self.batch_size,
        )

        # Get class names BEFORE mapping to grayscale
        class_names = train_ds_raw.class_names

        # Convert datasets to grayscale
        train_ds = train_ds_raw.map(self.to_grayscale)
        val_ds = val_ds_raw.map(self.to_grayscale)

        return train_ds, val_ds, class_names








class BM:
    def __init__(self):
        pass

    def build_model(self,
        dense_layers: int,
        list_number_of_nuerons_in_each_layer: list[int],
        Number_of_convoluational_filters: list[int],
        kernel_size: list[int]
    ) -> Sequential:
        """
        Builds a Sequential CNN + Dense model.
        
        Args:
        dense_layers: number of Dense layers (also equals number of classes/output units).
        list_number_of_nuerons_in_each_layer: neurons in each Dense layer, length == dense_layers.
        Number_of_convoluational_filters: list of filters for each Conv2D block.
        kernel_size: list of kernel sizes (ints) for each Conv2D block, same length as filters list.
        
        Returns:
        A compiled tf.keras.Sequential model.
        """
        model = Sequential()

        # — Convolutional blocks —
        for idx, (filters, k) in enumerate(zip(Number_of_convoluational_filters, kernel_size)):
            if idx == 0:
                # first conv needs input_shape
                model.add(Conv2D(filters, (k, k),
                                activation='relu',
                                input_shape=(224,224,1)))
                model.add(MaxPooling2D(pool_size=(2,2)))
            else:
                model.add(Conv2D(filters, (k, k), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))

        # flatten before Dense
        
        model.add(GlobalAveragePooling2D())

        # — Dense blocks —
        for neurons in list_number_of_nuerons_in_each_layer:
            model.add(Dense(neurons, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

    
        model.add(Dense(4, activation='softmax'))

        # compile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'loss']
        )
        
        return model




class run_to_build_pareto_front:
    def __init__(self,path_hyper_parameter):
        data =  pd.read_excel(path_hyper_parameter)



        print("Loading the everything is done\nNow it is the time for making the model and pareto front")
        self.number_of_dense_layers = data['Dense layers'].apply(ast.literal_eval)
        self.number_of_nuerons_in_each_dense_layers = data['number of nuerons in dense layers'].apply(ast.literal_eval)

        self.kernel_size_varied =  data['number of convolution layers '].apply(ast.literal_eval)
        self.kernel_size = data['kernel size'].apply(ast.literal_eval)

    
    
    def build_model_and_fit(self,epochs):

        

        train_data, val_data, class_names = load_data(
            train_data_path="Blood_cell_dataset/TEST_SIMPLE",
            validation_data_path="Blood_cell_dataset/TEST_SIMPLE",
            batch_size = 8
        ).load_data()

        count =  0
        buildModel = BM()
        num_classes =  len(class_names)
        ssf = save_summary_file("model_summaries")
        model_size= MS()
        sv = save()
        #loading the hyper parameters to make the things 
        for i,j in zip(self.number_of_dense_layers[0],self.number_of_nuerons_in_each_dense_layers[0]):
            for k in self.kernel_size_varied[0]:


                model =  buildModel.build_model(i,j,k,self.kernel_size[0])
                model.compile( optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                              )
                size_of_the_model = model_size.get_model_size_mb(model)
                print(i,j,k,self.kernel_size[0], size_of_the_model,"Done")
               
            
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=epochs
                )

                #ssf.save_summary(model=model,count=count)
                
                

                final_train_loss = history.history['loss'][-1]
                final_train_acc  = history.history['accuracy'][-1]
                final_val_loss   = history.history['val_loss'][-1]
                final_val_acc    = history.history['val_accuracy'][-1]
                

                sv.save_the_row_to_csv_file(final_train_loss, final_train_acc, final_val_loss, final_val_acc, model_size=size_of_the_model)

                """

                plot_model(
                model,
                to_file=f'model_picture/model_architecture_{count}.png',
                show_shapes=True,
                show_layer_names=True,
                dpi=200,               # Higher resolution
                expand_nested=True,    # If you're using nested models
                rankdir='TB'           # Layout: TB=top-bottom, LR=left-right
                )
                
                # 8. (Optional) Plot curves
                plt.figure()
                plt.plot(history.history['loss'],  label='train loss')
                plt.plot(history.history['val_loss'],label='val   loss')
                plt.plot(history.history['accuracy'],  label='train acc')
                plt.plot(history.history['val_accuracy'],label='val   acc')
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.legend()
                plt.title('Training & Validation Metrics')
                plt.show()
                """

                count += 1
            
                

    def generte_pareto_after_model_built(self,path_to_read_metric,directory_path):
        pareto_generation(path_to_read_metric,dierctory_path=directory_path).genearte()

    def generate_pareto(self,path):
        pareto(path=path).gen()



if __name__ == "__main__":

    run = run_to_build_pareto_front("hyper_paremetes.xlsx")
    run.build_model_and_fit(1)
    
    pareto("metrics_log.csv").gen()

