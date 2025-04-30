import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os
pio.renderers.default = "browser"

# Load dataset




class pareto_generation:

    def __init__(self,file_path,dierctory_path):
        self.df =  pd.read_csv(file_path)  #read the csv file
        self.output_directory =  dierctory_path #wher to save the parh
        
        

    # Pareto front function
    def pareto_front(self, x_col, y_col, minimize_x=True, minimize_y=True):

        pareto = []
        for i, row in self.df.iterrows():
            is_dominated = False
            for j, other in self.df.iterrows():
                if j == i:
                    continue
                cond_x = other[x_col] <= row[x_col] if minimize_x else other[x_col] >= row[x_col]
                cond_y = other[y_col] <= row[y_col] if minimize_y else other[y_col] >= row[y_col]
                strictly = (other[x_col] < row[x_col] if minimize_x else other[x_col] > row[x_col]) or \
                        (other[y_col] < row[y_col] if minimize_y else other[y_col] > row[y_col])
                if cond_x and cond_y and strictly:
                    is_dominated = True
                    break
            pareto.append(not is_dominated)
        return self.df[pareto]
    

    def genearte(self):

        # Define plots
        plots = [
            ("train_loss", True, "Train Loss vs Model Size", "Train Loss"),
            ("train_accuracy", False, "Train Accuracy vs Model Size", "Train Accuracy"),
            ("val_loss", True, "Validation Loss vs Model Size", "Validation Loss"),
            ("val_accuracy", False, "Validation Accuracy vs Model Size", "Validation Accuracy")
        ]

        # Generate each plot separately
        for metric, minimize_y, title, y_label in plots:
            pf = self.pareto_front(self.df, "model_size", metric, minimize_x=True, minimize_y=minimize_y)

            # Separate Pareto and non-Pareto points
            pareto_ids = set(pf.index)
            non_pf = self.data[~self.df.index.isin(pareto_ids)]

            fig = go.Figure()

            # Non-Pareto points
            fig.add_trace(go.Scatter(
                x=non_pf["model_size(MB)"], y=non_pf[metric], mode="markers",
                marker=dict(color="black", size=8),
                name="Non-Pareto Points"
            ))

            # Pareto front (sorted)
            pf_sorted = pf.sort_values(by="model_size")
            fig.add_trace(go.Scatter(
                x=pf_sorted["model_size(MB)"], y=pf_sorted[metric],
                mode="lines+markers",
                name="Pareto Front",
                line=dict(color="blue", width=3),
                marker=dict(size=10, color="blue")
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Model Size (MB)",
                yaxis_title=y_label,
                title_font_size=24,
                width=900,
                height=600,
                font=dict(size=16),
                xaxis=dict(title_font=dict(size=18, family="Arial Black", color="black")),
                yaxis=dict(title_font=dict(size=18, family="Arial Black", color="black"))

            )

            #fig.show()

            # Save plots
            

            #output_directory = "pareto_outputs"
            if not os.path.exists(self):
                os.makedirs(self.output_directory)


            fig.write_html(f"{self.output_directory}/{metric}_pareto_plot.html")
            fig.write_image(f"{self.output_directory}/{metric}_pareto_plot.png")
