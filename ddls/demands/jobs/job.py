class Job:
    def __init__(self,
                 job_id: int,
                 num_layers: int,
                 num_dims_per_layer: int,
                 weight_size: int,
                 num_weights: int,
                 batch_size: int,
                 sample_size: int, # memory per data set sample
                 num_samples: int, # number of samples in data set
                 num_epochs: int,
                 details: dict = {},
                 job_type: str = 'DNN'):
        
        self.job_id = job_id
        self.weight_size = weight_size
        self.num_weights = num_weights
        self.num_layers = num_layers
        self.num_dims_per_layer = num_dims_per_layer
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.details = details
        self.job_type = job_type
    
    def __str__(self):
        descr = f'Job ID: {self.job_id}'
        descr += f' | Job type: {self.job_type}'
        descr += f' | Per-weight memory: {self.weight_size}'
        descr += f' | # weights: {self.num_weights:.3e}'
        descr += f' | Model memory: {self.get_model_size():.3e}'
        descr += f' | # layers: {self.num_layers}'
        descr += f' | Per-layer # dims: {self.num_dims_per_layer}'
        descr += f' | Batch size: {self.batch_size}'
        descr += f' | Per-sample memory: {self.sample_size}'
        descr += f' | # samples: {self.num_samples:.3e}'
        descr += f' | Data set size: {self.get_dataset_size():.3e}'
        descr += f' | # epochs: {self.num_epochs}'
        return descr
    
    def __eq__(self, other):
        return self.job_id == other.job_id
    
    def get_model_size(self):
        return self.weight_size * self.num_weights

    def get_dataset_size(self):
        return self.sample_size * self.num_samples
