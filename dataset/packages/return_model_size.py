class model_size: 
    def __init__(self):
        pass

    def get_model_size_mb(model):
        """Returns the size of the model in megabytes."""
        total_bytes = 0
        for layer in model.layers:
            for weight in layer.get_weights():
                total_bytes += weight.size * weight.itemsize
        return total_bytes / (1024 ** 2)  
