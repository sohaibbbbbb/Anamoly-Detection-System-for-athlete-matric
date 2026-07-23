import torch
from torch.utils.data import DataLoader

# Import your custom classes from the previous files
from data_preprocessing import DataPipelineProcessor
from anamoly_detection_nn import AthleteAnomalyDataset, AnomalyDetectorANN, train_anomaly_model

if __name__ == "__main__":
    print("1. Running Data Pipeline & Saving Preprocessor...")
    processor = DataPipelineProcessor(
        file_path='raw_athlete_data.csv', 
        target_column='anomaly_flag', 
        date_columns=[], 
        group_col='athlete_id' 
    )
    # This generates 'production_pipeline.pkl'
    X_matrix, y_vector = processor.run(save_path="production_pipeline.pkl")
    
    print("2. Converting Tensors & Training Neural Network...")
    dataset = AthleteAnomalyDataset(X_matrix, y_vector)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = X_matrix.shape[1]
    model = AnomalyDetectorANN(input_dim=input_dim, hidden_dim=16)
    trained_model = train_anomaly_model(model=model, dataloader=loader, epochs=15, lr=0.001)
    
    print("3. Saving Neural Network Weights...")
    # This generates 'anomaly_weights.pth'
    torch.save(trained_model.state_dict(), "anomaly_weights.pth")
    
    print("System binaries generated successfully. Ready for deployment.")