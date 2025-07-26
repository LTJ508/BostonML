import joblib
import pandas as pd
import os
import json
from datetime import datetime

# Model tracking configuration
MODEL_REGISTRY_FILE = "history_registry/model_registry_regression.csv"
MODELS_DIR = "saved_models_regression"

# Model tracking configuration for classification
MODEL_REGISTRY_FILE_CLASSIFICATION = "history_registry/model_registry_classification.csv"
MODELS_DIR_CLASSIFICATION = "saved_models_classification"

def load_or_create_registry_regression():
    """Load existing model registry or create new one"""
    try:
        return pd.read_csv(MODEL_REGISTRY_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            'model_name', 'scaler_type', 'mse', 'mae', 'rmse', 'r2', 'timestamp', 
            'model_path', 'scaler_path', 'parameters'
        ])
    
def load_or_create_registry_classification():
    """Load existing model registry or create new one for classification"""
    try:
        return pd.read_csv(MODEL_REGISTRY_FILE_CLASSIFICATION)
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            'model_name', 'scaler_type', 'accuracy', 'precision', 'recall', 'f1', 'timestamp',
            'model_path', 'scaler_path', 'parameters'
        ])

def save_model_if_better_regression(model_name, model, scaler, scaler_name, mse, mae, rmse, r2, parameters=None):
    """Save model only if it performs better than existing version for the same scaler"""
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_REGISTRY_FILE), exist_ok=True)
    
    # Load registry
    registry = load_or_create_registry_regression()
    
    
    # Check if model exists for the same scaler
    existing_model = registry[(registry['model_name'] == model_name) & 
                             (registry['scaler_type'] == scaler_name)]
    
    should_save = False
    old_model_path = None
    old_scaler_path = None
    
    if existing_model.empty:
        print(f"{model_name} ({scaler_name}): First time training - saving model")
        should_save = True
    else:
        existing_mse = existing_model['mse'].iloc[0]
        existing_r2 = existing_model['r2'].iloc[0]
        
        # Store old file paths for deletion
        old_model_path = existing_model['model_path'].iloc[0]
        old_scaler_path = existing_model['scaler_path'].iloc[0]
        
        # Check if current model is better (lower MSE and higher R2)
        if mse < existing_mse and r2 > existing_r2:
            print(f"{model_name} ({scaler_name}): Better performance - updating model")
            print(f"   MSE: {existing_mse:.4f} → {mse:.4f} (↓)")
            print(f"   R2:  {existing_r2:.4f} → {r2:.4f} (↑)")
            should_save = True
        else:
            print(f"{model_name} ({scaler_name}): No improvement - keeping existing model")
            print(f"   Current MSE: {mse:.4f} vs Best: {existing_mse:.4f}")
            print(f"   Current R2:  {r2:.4f} vs Best: {existing_r2:.4f}")
    
    if should_save:
        # Delete old model files if they exist
        if old_model_path and os.path.exists(old_model_path):
            try:
                os.remove(old_model_path)
                print(f"    Deleted old model: {old_model_path}")
            except Exception as e:
                print(f"    Warning: Could not delete old model file: {e}")
                
        if old_scaler_path and os.path.exists(old_scaler_path):
            try:
                os.remove(old_scaler_path)
                print(f"    Deleted old scaler: {old_scaler_path}")
            except Exception as e:
                print(f"    Warning: Could not delete old scaler file: {e}")
        
        # Save new model and scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name.lower().replace(' ', '_')}_{scaler_name}_{timestamp}.pkl"
        scaler_filename = f"{model_name.lower().replace(' ', '_')}_scaler_{scaler_name}_{timestamp}.pkl"
        
        model_path = os.path.join(MODELS_DIR, model_filename)
        scaler_path = os.path.join(MODELS_DIR, scaler_filename)
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Remove old entry from registry (if exists)
        registry = registry[~((registry['model_name'] == model_name) & 
                             (registry['scaler_type'] == scaler_name))]
        
        # Process parameters to make them JSON serializable
        json_serializable_params = None
        if parameters:
            try:
                # Create a copy of parameters and convert non-serializable objects to strings
                serializable_params = {}
                for key, value in parameters.items():
                    try:
                        # Try to serialize the value
                        json.dumps(value)
                        serializable_params[key] = value
                    except (TypeError, ValueError):
                        # If not serializable, convert to string representation
                        serializable_params[key] = str(value)
                
                json_serializable_params = json.dumps(serializable_params)
            except Exception as e:
                print(f"    Warning: Could not serialize parameters: {e}")
                json_serializable_params = json.dumps({"error": "Could not serialize parameters"})
        
        # Add new entry to registry
        new_entry = pd.DataFrame([{
            'model_name': model_name,
            'scaler_type': scaler_name,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'timestamp': timestamp,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'parameters': json_serializable_params
        }])
        
        registry = pd.concat([registry, new_entry], ignore_index=True)
        registry.to_csv(MODEL_REGISTRY_FILE, index=False)
        
        print(f"    New model saved to: {model_path}")
        print(f"    New scaler saved to: {scaler_path}")
    
    return should_save

def save_model_if_better_classification(model_name, model, scaler, scaler_name, accuracy, precision, recall, f1, parameters=None):
    """Save classification model only if it performs better than existing version for the same scaler"""
    
    # Create directories
    os.makedirs(MODELS_DIR_CLASSIFICATION, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_REGISTRY_FILE_CLASSIFICATION), exist_ok=True)
    
    # Load registry
    registry = load_or_create_registry_classification()
    
    # Check if model exists for the same scaler
    existing_model = registry[(registry['model_name'] == model_name) & 
                             (registry['scaler_type'] == scaler_name)]
    
    should_save = False
    old_model_path = None
    old_scaler_path = None
    
    if existing_model.empty:
        print(f"{model_name} ({scaler_name}): First time training - saving model")
        should_save = True
    else:
        existing_accuracy = existing_model['accuracy'].iloc[0]
        
        # Store old file paths for deletion
        old_model_path = existing_model['model_path'].iloc[0]
        old_scaler_path = existing_model['scaler_path'].iloc[0]
        
        # Check if current model is better (higher accuracy)
        if accuracy > existing_accuracy:
            print(f"{model_name} ({scaler_name}): Better performance - updating model")
            print(f"   Accuracy: {existing_accuracy:.4f} → {accuracy:.4f} (↑)")
            should_save = True
        else:
            print(f"{model_name} ({scaler_name}): No improvement - keeping existing model")
            print(f"   Current Accuracy: {accuracy:.4f} vs Best: {existing_accuracy:.4f}")
    
    if should_save:
        # Delete old model files if they exist
        if old_model_path and os.path.exists(old_model_path):
            try:
                os.remove(old_model_path)
                print(f"    Deleted old model: {old_model_path}")
            except Exception as e:
                print(f"    Warning: Could not delete old model file: {e}")
                
        if old_scaler_path and os.path.exists(old_scaler_path):
            try:
                os.remove(old_scaler_path)
                print(f"    Deleted old scaler: {old_scaler_path}")
            except Exception as e:
                print(f"    Warning: Could not delete old scaler file: {e}")
        
        # Save new model and scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name.lower().replace(' ', '_')}_{scaler_name}_{timestamp}.pkl"
        scaler_filename = f"{model_name.lower().replace(' ', '_')}_scaler_{scaler_name}_{timestamp}.pkl"
        
        model_path = os.path.join(MODELS_DIR_CLASSIFICATION, model_filename)
        scaler_path = os.path.join(MODELS_DIR_CLASSIFICATION, scaler_filename)
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Remove old entry from registry (if exists)
        registry = registry[~((registry['model_name'] == model_name) & 
                             (registry['scaler_type'] == scaler_name))]
        
        # Process parameters to make them JSON serializable
        json_serializable_params = None
        if parameters:
            try:
                # Create a copy of parameters and convert non-serializable objects to strings
                serializable_params = {}
                for key, value in parameters.items():
                    try:
                        # Try to serialize the value
                        json.dumps(value)
                        serializable_params[key] = value
                    except (TypeError, ValueError):
                        # If not serializable, convert to string representation
                        serializable_params[key] = str(value)
                
                json_serializable_params = json.dumps(serializable_params)
            except Exception as e:
                print(f"    Warning: Could not serialize parameters: {e}")
                json_serializable_params = json.dumps({"error": "Could not serialize parameters"})
        
        # Add new entry to registry
        new_entry = pd.DataFrame([{
            'model_name': model_name,
            'scaler_type': scaler_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'timestamp': timestamp,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'parameters': json_serializable_params
        }])
        
        registry = pd.concat([registry, new_entry], ignore_index=True)
        registry.to_csv(MODEL_REGISTRY_FILE_CLASSIFICATION, index=False)
        
        print(f"    New model saved to: {model_path}")
        print(f"    New scaler saved to: {scaler_path}")
    
    return should_save

def show_model_registry_regression():
    """Display current model registry in descending order of R2"""
    try:
        registry = pd.read_csv(MODEL_REGISTRY_FILE)
        if not registry.empty:
            # Sort by R2 in descending order (best models first)
            registry_sorted = registry.sort_values('r2', ascending=False)
            
            print("\n" + "="*90)
            print("MODEL REGISTRY - BEST PERFORMING MODELS (Sorted by R2)")
            print("="*90)
            display_cols = ['model_name', 'scaler_type', 'mse', 'mae', 'rmse', 'r2', 'timestamp']
            print(registry_sorted[display_cols].to_string(index=False))
        else:
            print("No models saved yet.")
    except FileNotFoundError:
        print("No model registry found.")

def show_model_registry_classification():
    """Display current classification model registry in descending order of F1"""
    try:
        registry = pd.read_csv(MODEL_REGISTRY_FILE_CLASSIFICATION)
        if not registry.empty:
            # Sort by F1 in descending order (best models first)
            registry_sorted = registry.sort_values('f1', ascending=False)
            
            print("\n" + "="*90)
            print("CLASSIFICATION MODEL REGISTRY - BEST PERFORMING MODELS (Sorted by F1)")
            print("="*90)
            display_cols = ['model_name', 'scaler_type', 'accuracy', 'precision', 'recall', 'f1', 'timestamp']
            print(registry_sorted[display_cols].to_string(index=False))
        else:
            print("No classification models saved yet.")
    except FileNotFoundError:
        print("No classification model registry found.")

def view_model_parameters_regression(model_name=None, scaler_type=None):
    """View parameters of saved models"""
    try:
        registry = pd.read_csv(MODEL_REGISTRY_FILE)
        
        if model_name:
            if scaler_type:
                model_info = registry[(registry['model_name'] == model_name) & 
                                     (registry['scaler_type'] == scaler_type)]
                if model_info.empty:
                    print(f"No model found with name: {model_name} and scaler: {scaler_type}")
                    return
            else:
                model_info = registry[registry['model_name'] == model_name]
                if model_info.empty:
                    print(f"No model found with name: {model_name}")
                    return
            
            for _, row in model_info.iterrows():
                params_json = row['parameters']
                print(f"\n{row['model_name']} ({row['scaler_type']}) Parameters:")
                print("="*60)
                
                if params_json:
                    params = json.loads(params_json)
                    
                    if 'best_params' in params:
                        print("OPTIMIZED PARAMETERS:")
                        for key, value in params['best_params'].items():
                            print(f"   {key}: {value}")
                        print(f"\nCross-validation score: {params.get('cv_score', 'N/A')}")
                        
                        if 'all_params' in params:
                            print(f"\nCOMPLETE PARAMETER SET:")
                            all_params = params['all_params']
                            for key, value in sorted(all_params.items()):
                                print(f"   {key}: {value}")
                    else:
                        print("MODEL PARAMETERS:")
                        for key, value in sorted(params.items()):
                            print(f"   {key}: {value}")
                else:
                    print("No parameters saved for this model")
                print(f"Performance: MSE={row['mse']:.4f}, R2={row['r2']:.4f}")
        else:
            print("\n" + "="*90)
            print("ALL SAVED MODELS AND THEIR PARAMETERS")
            print("="*90)
            
            # Sort by R2 score descending (best first)
            registry_sorted = registry.sort_values('r2', ascending=False)
            
            for _, row in registry_sorted.iterrows():
                print(f"\n{row['model_name']} ({row['scaler_type']})")
                print(f"   Performance: MSE={row['mse']:.4f}, R2={row['r2']:.4f}")
                
                if row['parameters']:
                    params = json.loads(row['parameters'])
                    
                    if 'best_params' in params:
                        print("   Optimized Parameters:")
                        for key, value in params['best_params'].items():
                            print(f"      {key}: {value}")
                        print(f"   CV Score: {params.get('cv_score', 'N/A'):.4f}")
                    else:
                        print("   All Parameters:")
                        for key, value in sorted(params.items()):
                            if key not in ['algorithm']:
                                print(f"      {key}: {value}")
                else:
                    print("   No parameters saved")
                print(f"   Saved: {row['timestamp']}")
                
    except Exception as e:
        print(f"Error viewing parameters: {e}")

def view_model_parameters_classification(model_name=None, scaler_type=None):
    """View parameters of saved classification models"""
    try:
        registry = pd.read_csv(MODEL_REGISTRY_FILE_CLASSIFICATION)
        
        if model_name:
            if scaler_type:
                model_info = registry[(registry['model_name'] == model_name) & 
                                     (registry['scaler_type'] == scaler_type)]
                if model_info.empty:
                    print(f"No model found with name: {model_name} and scaler: {scaler_type}")
                    return
            else:
                model_info = registry[registry['model_name'] == model_name]
                if model_info.empty:
                    print(f"No model found with name: {model_name}")
                    return
            
            for _, row in model_info.iterrows():
                params_json = row['parameters']
                print(f"\n{row['model_name']} ({row['scaler_type']}) Parameters:")
                print("="*60)
                
                if params_json:
                    params = json.loads(params_json)
                    
                    if 'best_params' in params:
                        print("OPTIMIZED PARAMETERS:")
                        for key, value in params['best_params'].items():
                            print(f"   {key}: {value}")
                        print(f"\nCross-validation score: {params.get('cv_score', 'N/A')}")
                        
                        if 'all_params' in params:
                            print(f"\nCOMPLETE PARAMETER SET:")
                            all_params = params['all_params']
                            for key, value in sorted(all_params.items()):
                                print(f"   {key}: {value}")
                    else:
                        print("MODEL PARAMETERS:")
                        for key, value in sorted(params.items()):
                            print(f"   {key}: {value}")
                else:
                    print("No parameters saved for this model")
                print(f"Performance: Accuracy={row['accuracy']:.4f}, F1={row['f1']:.4f}")
        else:
            print("\n" + "="*90)
            print("ALL SAVED CLASSIFICATION MODELS AND THEIR PARAMETERS")
            print("="*90)
            
            # Sort by F1 score descending (best first)
            registry_sorted = registry.sort_values('f1', ascending=False)
            
            for _, row in registry_sorted.iterrows():
                print(f"\n{row['model_name']} ({row['scaler_type']})")
                print(f"   Performance: Accuracy={row['accuracy']:.4f}, F1={row['f1']:.4f}")
                
                if row['parameters']:
                    params = json.loads(row['parameters'])
                    
                    if 'best_params' in params:
                        print("   Optimized Parameters:")
                        for key, value in params['best_params'].items():
                            print(f"      {key}: {value}")
                        print(f"   CV Score: {params.get('cv_score', 'N/A'):.4f}")
                    else:
                        print("   All Parameters:")
                        for key, value in sorted(params.items()):
                            if key not in ['algorithm']:
                                print(f"      {key}: {value}")
                else:
                    print("   No parameters saved")
                print(f"   Saved: {row['timestamp']}")
                
    except Exception as e:
        print(f"Error viewing parameters: {e}")

# Usage examples:
# view_model_parameters_classification()  # View all classification models with their parameters
# view_model_parameters_classification("Random Forest Classifier")  # View specific model parameters  
# view_model_parameters_classification("Random Forest Classifier", "minmax")  # View specific model with specific scaler

# Function to load the best model
def load_best_model_regression(model_name, scaler_type=None):
    """Load the best saved model and scaler"""
    try:
        registry = pd.read_csv(MODEL_REGISTRY_FILE)
        
        if scaler_type:
            model_info = registry[(registry['model_name'] == model_name) & 
                                 (registry['scaler_type'] == scaler_type)]
            if model_info.empty:
                print(f"No saved model found for: {model_name} with {scaler_type} scaler")
                return None, None
        else:
            model_info = registry[registry['model_name'] == model_name]
            if model_info.empty:
                print(f"No saved model found for: {model_name}")
                return None, None
            # Get the best performing model (highest R2)
            model_info = model_info.loc[model_info['r2'].idxmax():model_info['r2'].idxmax()]
            
        model_path = model_info['model_path'].iloc[0]
        scaler_path = model_info['scaler_path'].iloc[0]
        used_scaler = model_info['scaler_type'].iloc[0]
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print(f"Loaded {model_name} ({used_scaler}) from {model_path}")
        return model, scaler
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Example usage:
# print("\n" + "="*50)
# print("USAGE EXAMPLES")
# print("="*50)
# print("# To load a specific model:")
# print("model, scaler = load_best_model_regression('Random Forest Tuned')")
# print("\n# To make predictions:")
# print("# scaled_data = scaler.transform(new_data)")
# print("# predictions = model.predict(scaled_data)")

def load_best_model_classification(model_name, scaler_type=None):
    """Load the best saved classification model and scaler"""
    try:
        registry = pd.read_csv(MODEL_REGISTRY_FILE_CLASSIFICATION)
        
        if scaler_type:
            model_info = registry[(registry['model_name'] == model_name) & 
                                 (registry['scaler_type'] == scaler_type)]
            if model_info.empty:
                print(f"No saved classification model found for: {model_name} with {scaler_type} scaler")
                return None, None
        else:
            model_info = registry[registry['model_name'] == model_name]
            if model_info.empty:
                print(f"No saved classification model found for: {model_name}")
                return None, None
            # Get the best performing model (highest F1)
            model_info = model_info.loc[model_info['f1'].idxmax():model_info['f1'].idxmax()]
            
        model_path = model_info['model_path'].iloc[0]
        scaler_path = model_info['scaler_path'].iloc[0]
        used_scaler = model_info['scaler_type'].iloc[0]
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print(f"Loaded {model_name} ({used_scaler}) from {model_path}")
        return model, scaler
        
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None, None

# Example usage:
# print("\n" + "="*50)
# print("USAGE EXAMPLES")
# print("="*50)
# print("# To load a specific model:")
# print("model, scaler = load_best_model_classification('Random Forest')")
# print("\n# To make predictions:")
# print("# scaled_data = scaler.transform(new_data)")
# print("# predictions = model.predict(scaled_data)")

