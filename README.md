````markdown
# Pareto Front Visualization: Model Size vs Accuracy

This repository helps you visualize the **Pareto Front** generated from training different models, comparing **Model Size (before training)** and **Accuracy** across a configurable number of epochs.

##  Dataset

The dataset required for training and evaluation is already included in the repository.

##  Required Libraries

All the necessary libraries are listed in the `requirements.txt` file.

##  Getting Started

Follow the steps below to set up and run the project:

### 1. Clone or Download the Repository

```bash
git clone <your-repo-url>
cd <cloned-folder>
````

Alternatively, download the ZIP archive from GitHub, extract it, and navigate into the folder:

```bash
cd <extracted-folder>
```

### 2. Create a Virtual Environment

```bash
python -m venv menv
```

### 3. Activate the Virtual Environment

* **Linux/macOS**:

  ```bash
  source menv/bin/activate
  ```

* **Windows**:

  ```bash
  menv\Scripts\activate
  ```

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 5. Choose the Number of Epochs

Open `index.py` and modify the `NUM_EPOCHS` variable to set your desired number of training epochs:

```python
NUM_EPOCHS = <your_desired_number>
```

### 6. Run the Script

```bash
python index.py
```

##  Output

* A **Pareto Front** plot will be generated, illustrating the trade-off between **Model Size (before training)** and **Accuracy**.
* All training metrics are logged to:

  ```text
  metrics_log.csv
  ```

##  File Structure

```text
├── dataset/                 # Dataset used for training
├── index.py                 # Main script to run training and visualization
├── requirements.txt         # Python dependencies
├── metrics_log.csv          # Output metrics file
└── README.md                # Project documentation (this file)
```

##  Notes

* Ensure you have **Python 3.7** or above installed.
* We recommend using a virtual environment to avoid conflicts.

##  Troubleshooting

If you run into issues:

1. Verify your Python and `pip` versions.
2. Confirm that all dependencies in `requirements.txt` are installed.
3. Make sure the `dataset/` directory is present and correctly structured.


