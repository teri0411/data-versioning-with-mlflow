import mlflow
from .base_train import BaseTrain
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from utils import get_git_commit_info, get_dvc_paths

class MLflowTrain(BaseTrain):
    """MLflow-based training for wine quality prediction"""
    
    def __init__(self):
        super().__init__()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
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
        self.log_model(self.model, self.metrics)
        
    def log_model(self, model, metrics, params=None, tags=None):
        """Log model metadata to MLflow"""
        with mlflow.start_run():
            # Log parameters
            if params:
                mlflow.log_params(params)
                
            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)
                
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
                    
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
    def get_latest_model(self):
        """Get latest model from MLflow"""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            raise ValueError(f"Experiment {EXPERIMENT_NAME} not found")
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("No runs found for experiment")
            
        return runs[0]
        
    def load_model(self, run_id=None):
        """Load model from MLflow"""
        if run_id is None:
            # Get latest run
            run = self.get_latest_model()
            run_id = run.info.run_id
            
        model_uri = f"runs:/{run_id}/model"
        return mlflow.sklearn.load_model(model_uri)
