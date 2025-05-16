import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os

# Use browser renderer
pio.renderers.default = "browser"

class pareto:
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def pareto_front(self, df, x_col, y_col, minimize_x=True, minimize_y=True):
        pareto = []
        for i, row in df.iterrows():
            is_dominated = False
            for j, other in df.iterrows():
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
        return df[pareto]

    def gen(self):
        plots = [
            #("train_loss", True, "Train Loss vs Model Size", "Train Loss"),
            ("train_accuracy", False, "Accuracy vs Model Size", "Accuracy"),
            #("val_loss", True, "Validation Loss vs Model Size", "Validation Loss"),
            #("val_accuracy", False, "Validation Accuracy vs Model Size", "Validation Accuracy")
        ]

        output_directory = "pareto_outputs"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for metric, minimize_y, title, y_label in plots:
            pf = self.pareto_front(self.data, "model_size(MB)", metric, minimize_x=True, minimize_y=minimize_y)

            pareto_ids = set(pf.index)
            non_pf = self.data[~self.data.index.isin(pareto_ids)]

            fig = go.Figure()

            # Plot non-Pareto points
            fig.add_trace(go.Scatter(
                x=non_pf["model_size(MB)"], y=non_pf[metric], mode="markers",
                marker=dict(color="black", size=8),
                name="Non-Pareto Points"
            ))

            # Plot Pareto front
            pf_sorted = pf.sort_values(by="model_size(MB)")
            fig.add_trace(go.Scatter(
                x=pf_sorted["model_size(MB)"], y=pf_sorted[metric],
                mode="lines+markers",
                name="Pareto Front",
                line=dict(color="blue", width=3),
                marker=dict(size=10, color="blue")
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Model Size",
                yaxis_title=y_label,
                title_font_size=24,
                width=900,
                height=600,
                font=dict(size=16),
                xaxis=dict(title_font=dict(size=18, family="Arial", color="black")),
                yaxis=dict(title_font=dict(size=18, family="Arial", color="black"))
            )

            fig.show()

            # Save plot as HTML
            fig.write_html(f"{output_directory}/{metric}_pareto_plot.html")

            # Save as PNG (with Kaleido) – try/except in case Kaleido is broken
            try:
                fig.write_image(f"{output_directory}/{metric}_pareto_plot.png")
            except Exception as e:
                print(f"[Warning] Could not save image for {metric}: {e}")
                print("Try reinstalling kaleido with: pip install -U kaleido")


if __name__ == "__main__": 
    pareto(path="metrics_log.csv").gen()
