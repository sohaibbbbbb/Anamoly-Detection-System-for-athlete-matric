import pandas as pd
import logging
import joblib
import time
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from anamoly_detection_nn import AthleteAnomalyDataset, AnomalyDetectorANN, train_anomaly_model
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DateFeatureExtractor':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy: pd.DataFrame = X.copy()
        for col in X_copy.columns:
            X_copy[col] = pd.to_datetime(X_copy[col])
            X_copy[f'{col}_year'] = X_copy[col].dt.year
            X_copy[f'{col}_month'] = X_copy[col].dt.month
            X_copy[f'{col}_dayofweek'] = X_copy[col].dt.dayofweek
        return X_copy.drop(columns=X.columns)


class DataPipelineProcessor:
    def __init__(
        self, 
        file_path: str, 
        target_column: str, 
        date_columns: Optional[List[str]] = None,
        group_col: Optional[str] = None
    ) -> None:
        self.file_path: str = file_path
        self.target_column: str = target_column
        self.date_columns: List[str] = date_columns if date_columns else []
        self.group_col: Optional[str] = group_col
        self.df: pd.DataFrame = pd.DataFrame()
        self.model_pipeline: Optional[Pipeline] = None

    def load_data(self) -> None:
        logging.info("Loading CSV into DataFrame...")
        self.df = pd.read_csv(self.file_path)

    def clean_commas(self) -> None:
        logging.info("Removing commas from string data...")
        # Restrict regex replacement strictly to object/string columns
        str_cols: List[str] = self.df.select_dtypes(include=['object']).columns.tolist()
        if str_cols:
            self.df[str_cols] = self.df[str_cols].replace(',', '', regex=True)

    def engineer_group_features(self) -> None:
        if not self.group_col or self.group_col not in self.df.columns:
            return
            
        logging.info(f"Executing complex grouping on '{self.group_col}'...")
        
        num_cols: List[str] = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if self.target_column in num_cols:
            num_cols.remove(self.target_column)

        if num_cols:
            agg_funcs: dict = {col: ['mean', 'max'] for col in num_cols}
            
            grouped_data: pd.DataFrame = self.df.groupby(self.group_col).agg(agg_funcs)
            grouped_data.columns = [f"{col}_{stat}_by_{self.group_col}" for col, stat in grouped_data.columns]
            
            self.df = self.df.merge(grouped_data, on=self.group_col, how='left')
            logging.info(f"Successfully added {len(grouped_data.columns)} aggregated group features.")

    def build_and_train_pipeline(self) -> Pipeline:
        logging.info("Building scikit-learn preprocessor and model pipeline...")
        
        X: pd.DataFrame = self.df.drop(columns=[self.target_column])
        y: pd.Series = self.df[self.target_column]

        numeric_features: List[str] = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        categorical_features: List[str] = [
            col for col in X.select_dtypes(include=['object']).columns 
            if col not in self.date_columns
        ]
        
        date_features: List[str] = [col for col in self.date_columns if col in X.columns]

        numeric_transformer: Pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer: Pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        date_transformer: Pipeline = Pipeline(steps=[
            ('extractor', DateFeatureExtractor()),
            ('scaler', StandardScaler()) 
        ])

        transformers: list = [
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
        
        if date_features:
            transformers.append(('date', date_transformer, date_features))

        preprocessor: ColumnTransformer = ColumnTransformer(transformers=transformers)
        
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])

        logging.info("Executing fit: Preprocessing data and training Random Forest...")
        self.model_pipeline.fit(X, y)
        logging.info("Pipeline executed and model trained successfully.")
        
        return self.model_pipeline

    def save_pipeline(self, save_path: str) -> None:
        """Saves the fitted pipeline to a binary file using joblib."""
        if self.model_pipeline:
            joblib.dump(self.model_pipeline, save_path)
            logging.info(f"Pipeline successfully saved to {save_path}")
        else:
            logging.error("No pipeline exists to save. Must build and train first.")

    def run(self, save_path: Optional[str] = "model_pipeline.pkl") -> tuple:
        logging.info("Starting pipeline execution...")
        self.load_data()
        self.clean_commas()
        self.engineer_group_features()
        
        # Build pipeline and transform the data simultaneously
        self.build_and_train_pipeline()
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column].values
        
        # Extract the finalized mathematical matrix
        X_processed = self.model_pipeline.fit_transform(X) 
        
        if save_path:
            self.save_pipeline(save_path)
            
        return X_processed, y

if __name__ == "__main__":
    processor = DataPipelineProcessor(
        file_path='raw_athlete_data.csv', 
        target_column='anomaly_flag', 
        date_columns=[], 
        group_col='athlete_id' 
    )
    
    X_matrix, y_vector = processor.run(save_path="production_pipeline.pkl")
    
    print("Converting to PyTorch Tensors...")
    dataset = AthleteAnomalyDataset(X_matrix, y_vector)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = X_matrix.shape[1]
    model = AnomalyDetectorANN(input_dim=input_dim, hidden_dim=16)
    
    print("Initiating PyTorch Deep Learning Core...")
    trained_model = train_anomaly_model(model=model, dataloader=loader, epochs=15, lr=0.001)