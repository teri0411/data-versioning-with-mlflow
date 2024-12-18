import mlflow
from .base_train import BaseTrain
from config import MLFLOW_TRACKING_URI
from utils import get_git_commit_info, get_dvc_paths

class MLflowTrain(BaseTrain):
    """MLflow-based training for wine quality prediction"""
    
    def __init__(self):
        super().__init__()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
    def train(self, data):
        """Train model and track with MLflow"""
        with mlflow.start_run():
            # Get DVC paths and Git info
            dvc_paths = get_dvc_paths()
            git_commit = get_git_commit_info()
            
            # Log DVC paths
            mlflow.log_params({
                'data_path': dvc_paths['data_path'],
                'model_path': dvc_paths['model_path']
            })
            
            # Log Git commit as tag
            if git_commit:
                mlflow.set_tag('git_commit', git_commit)
            
            # Log model parameters
            mlflow.log_params({
                'model_type': 'ElasticNet',
                'input_features': data.drop('quality', axis=1).columns.tolist()
            })
            
            # Preprocess and train
            X_scaled, y = self.preprocess_data(data)
            self.metrics = self.model.fit(X_scaled, y)
            
            # Log metrics
            mlflow.log_metrics(self.metrics)
            
            # Save model
            self.save_model()
            
        return self.metrics
        
    def save_model(self):
        """Save model to MLflow"""
        mlflow.pytorch.log_model(self.model, 'model')
