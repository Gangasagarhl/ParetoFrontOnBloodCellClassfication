# Multi-objective Optimization for Blood Cell Classification

- **Kaggle Notebook**: [Blood Cell Pareto Front](https://www.kaggle.com/code/harshithgangasgarhl/blood-cell-pareto-front)
- **Code**: [GitHub Repository](https://github.com/Gangasagarhl/ParetoFrontOnBloodCellClassfication)
- **Dataset**: [Blood Cell Images on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data)

## Project Setup

### Dataset
The dataset required for training and evaluation is already included in the repository.

### Getting Started
Follow the steps below to set up and run the project:

> Notes:
> - Ensure you have **Python 3.7** or above installed.
> - We recommend using a virtual environment to avoid conflicts.

1. Clone or Download the Repository
2. Create a Virtual Environment
    - `python -m venv myenv`
3. Activate the Virtual Environment
    - `source myenv/bin/activate`
4. Install Required Libraries
    - `pip install -r requirements.txt`
5. Choose the Number of Epochs
    ```py
    # edit "index.py" pass argument, how many epochs
    run.build_model_and_fit(1)
    ```
6. Run the Script
    - `python3 index.py`

### Output
- A **Pareto Front** plot will be generated, illustrating the trade-off between **Model Size (before training)** and **Accuracy**.
- All training metrics are logged to:

**`metrics_log.csv`**



## Summary

Blood cell classification is a critical task in medical diagnostics, enabling early detection of diseases such as leukemia, anemia, and infections. Traditional methods rely on manual microscopy, which is time-consuming and prone to human error. **Convolutional Neural Networks (CNNs)** have emerged as powerful tools for automating this process, offering high accuracy in classifying cell subtypes like Eosinophils, Lymphocytes, Monocytes, and Neutrophils.

However, deploying such models in real-world medical settings introduces practical constraints. Clinics and hospitals, particularly in resource-limited regions, often rely on edge devices (e.g., portable diagnostic tools) with limited computational power and storage. This necessitates a trade-off between **model accuracy** and **deployment feasibility**.

## Approach: Grid Search
We employed grid search to explore hyperparameter combinations exhaustively. This method involves:
- Defining discrete values for hyperparameters (e.g., dense layers, convolutional filters).
- Training models for all possible combinations.
- Evaluating performance metrics (accuracy, model size).


## Hyperparameter Space (CNN Architecture)
Edit `hyper_paremetes.xlsx` as per your requirements.

| Parameter | Values  |
|-----------|---------|
| **Dense Layers** | 2, 3, 4, 5, 6, 7 |
| **Neurons per Dense Layer** | [250, 300], [380, 430, 477], [200, 210, 300, 350], [70, 95, 120, 100, 4], [150, 208, 300, 350, 405, 450], [300, 250, 200, 180, 175, 160, 150] |
| **Convolutional Filters** | [70, 80, 100, 140, 180, 220], [80, 100, 160, 200, 210, 220], [120, 140, 160, 180, 200, 256], [128, 150, 180, 200, 256] |
| **Kernel Sizes** | 7, 5, 5, 3, 3, 3 |

## Numerical Results

### Experimental Setup
- **Dataset**: [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) (12,500 images, 4 classes).
- **Training Protocols**:  
  - **10 Epochs**: Simulates resource-constrained training.  
  - **30 Epochs**: Allows better convergence.  
- **GPU**: NVIDIA P100 on Kaggle Notebooks.

## Pareto Frontiers:

### 10-Epoch Training

![](/pareto_outputs/train_accuracy_vs_model_size_epochs_10.png)

**Key Observations**:

- **Pareto Front**: 12 non-dominated points.  
- **Optimal Models**:  
  - **4.36 MB**: 82.3% accuracy (suitable for edge deployment).  
  - **10.11 MB**: 95.2% accuracy (theoretical upper bound).  
- **Diminishing Returns**: Accuracy plateaus beyond 8 MB.

### 30-Epoch Training

![](/pareto_outputs/train_accuracy_vs_model_size_epochs_30.png)

**Key Observations**:

- **Pareto Front**: 15 non-dominated points.  
- **Optimal Models**:  
  - **6.2 MB**: 91.8% accuracy (balanced choice).  
  - **8.42 MB**: 94.7% accuracy (high-resource settings).  
- **Improved Convergence**: Longer training reduces accuracy variance.


### Why Training Accuracy?
Due to computational constraints, validation accuracy was not used for Pareto analysis. This is because achieving reliable validation metrics would require training all candidate models for an impractically large number of epochs. As a proof of concept, we selected a Pareto-optimal model from our grid search results and conducted extended training with significantly more epochs. This experiment achieved **86.27% validation accuracy**, demonstrating that our training-optimized models retain the potential for robust generalization when given sufficient computational resources.

## Future Work

- Incorporate **validation accuracy**}** and **inference latency**}** as objectives.
- Explore **Bayesian Optimization**}** for efficient hyperparameter search.
- Apply **quantization**}** and **pruning**}** to further compress models.

