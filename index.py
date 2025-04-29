"""
STEPS INVOLVED
1. Load the data set for training and validating, which return the train and valid
2. download the hyper parameters using the grid search
3. For each and every set of hyper parameters , make the model/build the model
4. Then Build the model, using epochs
5. Then save the results into csv
6. Then lad the data in the csv, and make the pareto front
"""
from packages.build_and_return_model import build_model as BM #no parameters at init 
from packages.pareto_corner import pareto_generation  #csv file path, output directot
from packages.return_model_size import model_size as MS
from packages.save_data_into_csv_for_pareto import save 
from packages.save_summary import save_summary_file  #output directory to save the summary 
from packages.load_dataset import load_data
import pandas as pd
import ast
from tensorflow.keras.utils import plot_model
import  matplotlib.pyplot as plt





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
            train_data_path="Blood_cell_dataset/TRAIN/",
            validation_data_path="Blood_cell_dataset/TEST/"
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
                
               
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=epochs
                )

                ssf.save_summary(model=model,count=count)
                
                

                final_train_loss = history.history['loss'][-1]
                final_train_acc  = history.history['accuracy'][-1]
                final_val_loss   = history.history['val_loss'][-1]
                final_val_acc    = history.history['val_accuracy'][-1]
                size_of_the_model = model_size.get_model_size_mb(model)

                sv.save_the_row_to_csv_file(final_train_loss, final_train_acc, final_val_loss, final_val_acc, model_size=size_of_the_model)



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


                count += 1
            


                print(i,j,k,self.kernel_size[0], "Done")
                


    def generte_pareto_after_model_built(self,path_to_read_metric,directory_path):
        pareto_generation(path_to_read_metric,dierctory_path=directory_path).genearte()



if __name__ == "__main__":

    run = run_to_build_pareto_front("hyper_paremetes.xlsx")
    run.build_model_and_fit(10)
    run.generte_pareto_after_model_built("metrics_log.csv", directory_path="pareto_outputs")
