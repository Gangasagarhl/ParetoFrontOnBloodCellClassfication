import os 
import io

class save_summary_file:
    def __init__(self, out_dir):
        self.out_dir = out_dir



    def save_summary(self,model, count):
        """
        Saves the textual summary of a Keras-style model to:
            /model_summaries/model_<count>.txt

        Args:
            model:            A model instance with a .summary(...) method.
            count (int|str):  Identifier for this summary file.
        """
        # Directory where summaries will be stored
        #self.out_dir = "model_summaries"
        os.makedirs(self.out_dir, exist_ok=True)

        # File path
        file_path = os.path.join(self.out_dir, f"model_{count}.txt")

        # Capture the summary into a string buffer
        stream = io.StringIO()
        model.summary(print_fn=lambda line: stream.write(line + "\n"))
        summary_str = stream.getvalue()
        stream.close()

        # Write to file
        with open(file_path, "w") as f:
            f.write(summary_str)

        print(f"Model summary saved to {file_path}")

