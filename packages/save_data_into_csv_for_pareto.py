import os 
import csv

class save:
    def __init__(self):
        pass

    def save_the_row_to_csv_file(train_loss, train_acc, val_loss, val_acc, model_size, filename="metrics_log.csv"):
        """
        Appends a row of metrics to a CSV file. If the file doesn't exist,
        creates it and writes a header first.

        Args:
            train_loss (float): Training loss.
            train_acc (float):  Training accuracy.
            val_loss (float):   Validation loss.
            val_acc (float):    Validation accuracy.
            model_size (int):   Model size in parameters (or any unit you prefer).
            filename (str):     CSV filename (default "metrics_log.csv").
        """
        file_exists = os.path.exists(filename)

        # Define the header and the row to write
        header = ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "model_size(MB)"]
        row    = [train_loss, train_acc, val_loss, val_acc, model_size]

        # Open in append mode
        with open(filename, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # If the file didn't exist before, write the header
            if not file_exists:
                writer.writerow(header)

            # Write the new row
            writer.writerow(row)

        print(f"Metrics saved to '{filename}'")
