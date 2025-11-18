"""
BOOSTING ALGORITHMS PIPELINE - COMPLETE VERSION
Modular pipeline for classification and regression with various boosting algorithms
"""

import numpy as np
import pandas as pd
import pickle
import json
import base64
import io
import requests
import urllib.parse
import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Boosting algorithm imports
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
except ImportError:
    GradientBoostingClassifier = GradientBoostingRegressor = None
    AdaBoostClassifier = AdaBoostRegressor = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

# =============================================================================
# GLOBAL PIPELINE CONFIGURATION
# =============================================================================

DEFAULT_BOOSTING_CONFIG = {
    'pipeline_name': 'boosting_algorithms_pipeline',
    'version': '1.0.0',
    'created_at': str(datetime.now()),
    
    'dataset': {
        'test_split': 0.2,
        'validation_split': 0.1,
        'random_state': 42,
        'shuffle': True,
        'problem_type': None,  # 'classification' or 'regression'
        'target_column': 'target'
    },
    
    'preprocessing': {
        'handle_missing': True,
        'encode_categorical': True,
        'scale_features': False,  # Boosting algorithms typically don't need scaling
        'feature_selection': False
    },
    
    'model': {
        'algorithm': 'xgboost',  # xgboost, gradient_boosting, adaboost, lightgbm, catboost
        'task_type': 'classification',  # 'classification' or 'regression'
        'random_state': 42,
        'n_jobs': -1,
        
        # Common parameters
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        
        # Algorithm-specific parameters
        'xgboost_params': {},
        'gradient_boosting_params': {},
        'adaboost_params': {},
        'lightgbm_params': {},
        'catboost_params': {}
    },
    
    'training': {
        'early_stopping_rounds': 50,
        'verbose': True,
        'eval_metric': None,  # Auto-set based on task
        'use_gpu': False
    },
    
    'paths': {
        'base_path': "boosting_pipeline",
        'processed_data': "data/processed",
        'trained_models': "models/trained",
        'training_logs': "models/logs",
        'configs': "models/configs",
        'results': "results"
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_url(url: str) -> str:
    """Normalize URL by decoding special characters"""
    url = urllib.parse.unquote(url)
    url = url.replace("_$_", "_$$").replace("_-", "_$$").replace("__DOLLARS__", "$$")
    return url

def save_metrics(metrics: dict):
    """Save metrics for Katib - using appropriate metric for task type"""
    global_step = 1
    trial_id = "0"
    timestamp = time.time()
    
    print("=== SAVING BOOSTING METRICS ===")
    print(f"Input metrics: {metrics}")
    
    # Determine the primary metric based on task type
    if 'task_type' in metrics:
        task_type = metrics['task_type']
        if task_type == 'classification':
            # For classification, use accuracy or negative log loss
            if 'accuracy' in metrics:
                metric_name = "accuracy"
                metric_value = f"{metrics['accuracy']:.6f}"
            elif 'val_accuracy' in metrics:
                metric_name = "val_accuracy"
                metric_value = f"{metrics['val_accuracy']:.6f}"
            else:
                metric_name = "accuracy"
                metric_value = "0.000000"
        else:  # regression
            # For regression, use negative MSE or R2 score
            if 'neg_mse' in metrics:
                metric_name = "neg_mse"
                metric_value = f"{metrics['neg_mse']:.6f}"
            elif 'r2_score' in metrics:
                metric_name = "r2_score"
                metric_value = f"{metrics['r2_score']:.6f}"
            else:
                metric_name = "neg_mse"
                metric_value = "-10.000000"  # Poor performance
    else:
        # Default to accuracy for classification
        metric_name = "accuracy"
        metric_value = "0.000000"
    
    record = {
        metric_name: metric_value,
        "checkpoint_path": "",
        "global_step": str(global_step),
        "timestamp": timestamp,
        "trial": trial_id,
    }
    print(f"Record being saved: {record}")
    
    # Create directory if it doesn't exist
    katib_dir = Path("/katib")
    katib_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = katib_dir / "mnist.json"
    with open(metrics_file, "a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")
        
    print("=== BOOSTING METRICS SAVING COMPLETE ===")

def build_config(model_type: str, model_name: str, unknown_args: list, 
                process_data_url: str = None, config_json: dict = None, 
                data: Any = None) -> dict:
    """
    Build configuration for boosting algorithms
    """
    merged_config = DEFAULT_BOOSTING_CONFIG.copy()
    
    # Parse model_type to extract algorithm and task
    # Expected format: <algorithm>_<task> (e.g., xgboost_classification, gradientboosting_regression)
    parts = model_type.lower().split('_')
    algorithm = None
    task_type = None
    
    # Identify algorithm and task type
    algorithm_keywords = ['xgboost', 'gradientboosting', 'adaboost', 'lightgbm', 'catboost', 'gradient_boosting']
    task_keywords = ['classification', 'regression']
    
    for part in parts:
        if part in algorithm_keywords:
            algorithm = part
        elif part in task_keywords:
            task_type = part
    
    # Set defaults if not found
    if not algorithm:
        algorithm = 'xgboost'
    if not task_type:
        task_type = 'classification'
    
    # Map gradientboosting to gradient_boosting for internal consistency
    if algorithm == 'gradientboosting':
        algorithm = 'gradient_boosting'
    
    merged_config['model']['algorithm'] = algorithm
    merged_config['model']['task_type'] = task_type
    merged_config['model']['model_name'] = model_name
    
    # Apply config_json
    if config_json:
        try:
            for key, value in config_json.items():
                if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            print(f"[CONFIG] Applied config_json with {len(config_json)} parameters")
        except Exception as e:
            print(f"[CONFIG] Error applying config_json: {e}")
    
    # Apply CLI overrides
    cli_overrides = {}
    i = 0
    while i < len(unknown_args):
        tok = unknown_args[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            if "=" in key:
                key, sval = key.split("=", 1)
                try: 
                    val = json.loads(sval)
                except: 
                    val = sval
                cli_overrides[key] = val
                i += 1
            else:
                if i+1 < len(unknown_args) and not unknown_args[i+1].startswith("--"):
                    sval = unknown_args[i+1]
                    try: 
                        val = json.loads(sval)
                    except: 
                        val = sval
                    cli_overrides[key] = val
                    i += 2
                else:
                    cli_overrides[key] = True
                    i += 1
        else: 
            i += 1
    
    # Apply CLI overrides
    for key, value in cli_overrides.items():
        if '.' in key:
            parts = key.split('.')
            current = merged_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            merged_config[key] = value
    
    # Auto-set evaluation metric based on task type
    if merged_config['training'].get('eval_metric') is None:
        if task_type == 'classification':
            merged_config['training']['eval_metric'] = 'logloss'
        else:  # regression
            merged_config['training']['eval_metric'] = 'rmse'
    
    print(f"🔧 Algorithm: {algorithm}, Task: {task_type}")
    return merged_config

# =============================================================================
# DATA LOADER AND PROCESSOR
# =============================================================================

class LabeledDataset:
    """Compatible dataset class for boosting algorithms"""
    def __init__(self, dataset=None, label_mapping=None):
        self.dataset = dataset or []
        self.label_mapping = label_mapping or {}
        
    def __len__(self):
        try:
            if hasattr(self.dataset, '__len__'):
                return len(self.dataset)
            return 100
        except:
            return 100
        
    def __getitem__(self, idx):
        try:
            if hasattr(self.dataset, '__getitem__'):
                item = self.dataset[idx]
                if isinstance(item, tuple) and len(item) == 2:
                    data, label = item
                elif isinstance(item, dict):
                    data = item.get('features', item.get('image_data'))
                    label = item.get('label', 0)
                    return data, label
                else:
                    return item, 0
        except:
            pass
        return np.random.rand(10), 0

class DataWrapper:
    """Data wrapper for pickled data"""
    def __init__(self, data_dict=None):
        if data_dict:
            self.__dict__.update(data_dict)

def load_pickle_url(url: str):
    """Load pickled data from URL"""
    print(f"Downloading from: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'LabeledDataset':
                return LabeledDataset
            elif name == 'DataWrapper':
                return DataWrapper
            try:
                return super().find_class(module, name)
            except:
                class GenericObject:
                    def __init__(self, *args, **kwargs):
                        pass
                return GenericObject
    
    try:
        data = CustomUnpickler(io.BytesIO(resp.content)).load()
        print(f"✅ Data loaded successfully: {type(data)}")
        
        # Debug info
        if hasattr(data, '__dict__'):
            print(f"   Data attributes: {list(data.__dict__.keys())}")
        
        return data
    except Exception as e:
        print(f"❌ Pickle loading failed: {e}")
        
        # Create fallback data
        class FallbackData:
            def __init__(self):
                self.X_train = np.random.rand(100, 10)
                self.y_train = np.random.randint(0, 2, 100)
                self.X_test = np.random.rand(20, 10)
                self.y_test = np.random.randint(0, 2, 20)
                self.feature_names = [f'feature_{i}' for i in range(10)]
                self.target_names = ['class_0', 'class_1']
                print("   Created FallbackData with random data")
        
        return FallbackData()

class DataProcessor:
    """Process data for boosting algorithms"""
    
    def __init__(self, model_config):
        self.model_config = model_config
    
    def extract_data_from_pickle(self, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract features and labels from various data formats"""
        print("🔧 Extracting data from loaded object...")
        
        X_train, y_train, X_test, y_test = None, None, None, None
        
        try:
            # Case 1: Data has train_loader and test_loader (PyTorch style)
            if hasattr(data, 'train_loader') and data.train_loader is not None:
                print("   Extracting from PyTorch data loaders...")
                X_train, y_train = self._extract_from_dataloader(data.train_loader)
                X_test, y_test = self._extract_from_dataloader(data.test_loader)
            
            # Case 2: Data has X_train, y_train, X_test, y_test attributes
            elif hasattr(data, 'X_train') and hasattr(data, 'y_train'):
                print("   Extracting from sklearn-style attributes...")
                X_train, y_train = data.X_train, data.y_train
                X_test, y_test = data.X_test, data.y_test
            
            # Case 3: Data has train_dataset and test_dataset
            elif hasattr(data, 'train_dataset') and data.train_dataset is not None:
                print("   Extracting from dataset objects...")
                X_train, y_train = self._extract_from_dataset(data.train_dataset)
                X_test, y_test = self._extract_from_dataset(data.test_dataset)
            
            # Case 4: Data is a dictionary-like object
            elif hasattr(data, '__dict__'):
                data_dict = data.__dict__
                for key in ['X_train', 'x_train', 'train_features']:
                    if key in data_dict:
                        X_train = data_dict[key]
                        break
                for key in ['y_train', 'train_labels', 'train_target']:
                    if key in data_dict:
                        y_train = data_dict[key]
                        break
                for key in ['X_test', 'x_test', 'test_features']:
                    if key in data_dict:
                        X_test = data_dict[key]
                        break
                for key in ['y_test', 'test_labels', 'test_target']:
                    if key in data_dict:
                        y_test = data_dict[key]
                        break
            
            # Convert to numpy arrays if they are tensors
            if X_train is not None:
                if hasattr(X_train, 'numpy'):
                    X_train = X_train.numpy()
                if hasattr(y_train, 'numpy'):
                    y_train = y_train.numpy()
                if hasattr(X_test, 'numpy'):
                    X_test = X_test.numpy()
                if hasattr(y_test, 'numpy'):
                    y_test = y_test.numpy()
                    
        except Exception as e:
            print(f"❌ Error extracting data: {e}")
        
        # Create fallback data if extraction failed
        if X_train is None:
            print("⚠️  Using fallback random data")
            X_train = np.random.rand(100, 10)
            y_train = np.random.randint(0, 2, 100)
            X_test = np.random.rand(20, 10)
            y_test = np.random.randint(0, 2, 20)
        
        print(f"✅ Data extracted - Train: {X_train.shape}, {y_train.shape}, Test: {X_test.shape}, {y_test.shape}")
        return X_train, y_train, X_test, y_test
    
    def _extract_from_dataloader(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract data from PyTorch DataLoader"""
        all_features = []
        all_labels = []
        
        try:
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    features, labels = batch[0], batch[1]
                    
                    # Convert tensors to numpy
                    if hasattr(features, 'numpy'):
                        features = features.numpy()
                    if hasattr(labels, 'numpy'):
                        labels = labels.numpy()
                    
                    # Handle image data by flattening
                    if len(features.shape) > 2:
                        features = features.reshape(features.shape[0], -1)
                    
                    all_features.append(features)
                    all_labels.append(labels)
        except Exception as e:
            print(f"❌ Error extracting from dataloader: {e}")
        
        if all_features:
            return np.vstack(all_features), np.concatenate(all_labels)
        else:
            return np.random.rand(10, 10), np.random.randint(0, 2, 10)
    
    def _extract_from_dataset(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract data from Dataset object"""
        all_features = []
        all_labels = []
        
        try:
            for i in range(min(len(dataset), 1000)):  # Limit to first 1000 samples
                item = dataset[i]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    features, label = item[0], item[1]
                    
                    if hasattr(features, 'numpy'):
                        features = features.numpy()
                    if hasattr(label, 'numpy'):
                        label = label.numpy()
                    
                    # Handle image data
                    if len(features.shape) > 1:
                        features = features.flatten()
                    
                    all_features.append(features)
                    all_labels.append(label)
        except Exception as e:
            print(f"❌ Error extracting from dataset: {e}")
        
        if all_features:
            return np.array(all_features), np.array(all_labels)
        else:
            return np.random.rand(10, 10), np.random.randint(0, 2, 10)

# =============================================================================
# BOOSTING MODEL FACTORY
# =============================================================================

class BoostingModelFactory:
    """Factory for creating boosting models"""
    
    @staticmethod
    def create_model(algorithm: str, task_type: str, model_config: dict):
        """Create a boosting model based on algorithm and task type"""
        print(f"🔧 Creating {algorithm} model for {task_type}")
        
        common_params = {
            'n_estimators': model_config.get('n_estimators', 100),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'random_state': model_config.get('random_state', 42)
        }
        
        if algorithm == 'xgboost':
            return BoostingModelFactory._create_xgboost_model(task_type, common_params, model_config)
        elif algorithm == 'gradient_boosting':
            return BoostingModelFactory._create_gradient_boosting_model(task_type, common_params, model_config)
        elif algorithm == 'adaboost':
            return BoostingModelFactory._create_adaboost_model(task_type, common_params, model_config)
        elif algorithm == 'lightgbm':
            return BoostingModelFactory._create_lightgbm_model(task_type, common_params, model_config)
        elif algorithm == 'catboost':
            return BoostingModelFactory._create_catboost_model(task_type, common_params, model_config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def _create_xgboost_model(task_type: str, common_params: dict, model_config: dict):
        """Create XGBoost model"""
        if xgb is None:
            raise ImportError("XGBoost is not installed")
        
        algorithm_specific = model_config.get('xgboost_params', {})
        params = {**common_params, **algorithm_specific}
        
        if task_type == 'classification':
            params['objective'] = 'binary:logistic' if model_config.get('num_classes', 2) == 2 else 'multi:softprob'
            model = xgb.XGBClassifier(**params)
        else:  # regression
            params['objective'] = 'reg:squarederror'
            model = xgb.XGBRegressor(**params)
        
        print(f"✅ Created XGBoost {task_type} model")
        return model
    
    @staticmethod
    def _create_gradient_boosting_model(task_type: str, common_params: dict, model_config: dict):
        """Create sklearn Gradient Boosting model"""
        if GradientBoostingClassifier is None:
            raise ImportError("scikit-learn is not installed")
        
        algorithm_specific = model_config.get('gradient_boosting_params', {})
        params = {**common_params, **algorithm_specific}
        
        # Remove parameters not supported by sklearn
        params.pop('random_state', None)  # sklearn uses random_state, not seed
        
        if task_type == 'classification':
            model = GradientBoostingClassifier(**params)
        else:  # regression
            model = GradientBoostingRegressor(**params)
        
        print(f"✅ Created Gradient Boosting {task_type} model")
        return model
    
    @staticmethod
    def _create_adaboost_model(task_type: str, common_params: dict, model_config: dict):
        """Create AdaBoost model"""
        if AdaBoostClassifier is None:
            raise ImportError("scikit-learn is not installed")
        
        algorithm_specific = model_config.get('adaboost_params', {})
        params = {**common_params, **algorithm_specific}
        
        # sklearn AdaBoost uses base_estimator, but we'll use default
        if 'base_estimator' not in params:
            params.pop('base_estimator', None)
        
        if task_type == 'classification':
            model = AdaBoostClassifier(**params)
        else:  # regression
            model = AdaBoostRegressor(**params)
        
        print(f"✅ Created AdaBoost {task_type} model")
        return model
    
    @staticmethod
    def _create_lightgbm_model(task_type: str, common_params: dict, model_config: dict):
        """Create LightGBM model"""
        if lgb is None:
            raise ImportError("LightGBM is not installed")
        
        algorithm_specific = model_config.get('lightgbm_params', {})
        params = {**common_params, **algorithm_specific}
        
        if task_type == 'classification':
            params['objective'] = 'binary' if model_config.get('num_classes', 2) == 2 else 'multiclass'
            model = lgb.LGBMClassifier(**params)
        else:  # regression
            params['objective'] = 'regression'
            model = lgb.LGBMRegressor(**params)
        
        print(f"✅ Created LightGBM {task_type} model")
        return model
    
    @staticmethod
    def _create_catboost_model(task_type: str, common_params: dict, model_config: dict):
        """Create CatBoost model"""
        if cb is None:
            raise ImportError("CatBoost is not installed")
        
        algorithm_specific = model_config.get('catboost_params', {})
        params = {**common_params, **algorithm_specific}
        
        if task_type == 'classification':
            model = cb.CatBoostClassifier(**params, verbose=False)
        else:  # regression
            model = cb.CatBoostRegressor(**params, verbose=False)
        
        print(f"✅ Created CatBoost {task_type} model")
        return model

# =============================================================================
# MODEL TRAINER
# =============================================================================

class BoostingModelTrainer:
    """Trainer for boosting models"""
    
    def __init__(self, model_config):
        self.model_config = model_config
    
    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> Any:
        """Train the boosting model"""
        print(f"🚀 Training {model_name}...")
        
        training_config = self.model_config['training']
        algorithm = self.model_config['model']['algorithm']
        task_type = self.model_config['model']['task_type']
        
        try:
            # Handle different training approaches for different algorithms
            if algorithm == 'xgboost':
                return self._train_xgboost(model, X_train, y_train, X_val, y_val, training_config)
            elif algorithm == 'lightgbm':
                return self._train_lightgbm(model, X_train, y_train, X_val, y_val, training_config)
            elif algorithm == 'catboost':
                return self._train_catboost(model, X_train, y_train, X_val, y_val, training_config)
            else:
                # For sklearn models (GradientBoosting, AdaBoost)
                return self._train_sklearn(model, X_train, y_train, training_config)
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def _train_xgboost(self, model, X_train, y_train, X_val, y_val, training_config):
        """Train XGBoost with compatibility for different versions"""
        try:
            if X_val is not None:
                # Try different parameter combinations for XGBoost
                eval_set = [(X_val, y_val)]
                early_stopping = training_config.get('early_stopping_rounds', 50)
                
                # Try with early_stopping_rounds
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=early_stopping,
                        verbose=training_config.get('verbose', True)
                    )
                    print("✅ XGBoost trained with early_stopping_rounds")
                except TypeError:
                    # Try with early_stopping
                    try:
                        model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            early_stopping=early_stopping,
                            verbose=training_config.get('verbose', True)
                        )
                        print("✅ XGBoost trained with early_stopping")
                    except TypeError:
                        # Train without early stopping
                        print("⚠️  XGBoost: Training without early stopping")
                        model.fit(X_train, y_train, verbose=training_config.get('verbose', True))
            else:
                model.fit(X_train, y_train, verbose=training_config.get('verbose', True))
                
        except Exception as e:
            print(f"⚠️  XGBoost training fallback: {e}")
            model.fit(X_train, y_train)
            
        return model
    
    def _train_lightgbm(self, model, X_train, y_train, X_val, y_val, training_config):
        """Train LightGBM with compatibility for different versions"""
        try:
            if X_val is not None:
                eval_set = [(X_val, y_val)]
                early_stopping = training_config.get('early_stopping_rounds', 50)
                
                # LightGBM uses eval_set and early_stopping (not early_stopping_rounds)
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping=early_stopping,
                        verbose=training_config.get('verbose', True)
                    )
                    print("✅ LightGBM trained with early_stopping")
                except TypeError as e:
                    # Try alternative parameter names
                    try:
                        model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            early_stopping_rounds=early_stopping,
                            verbose=training_config.get('verbose', True)
                        )
                        print("✅ LightGBM trained with early_stopping_rounds")
                    except TypeError:
                        # Train without early stopping
                        print("⚠️  LightGBM: Training without early stopping")
                        model.fit(X_train, y_train, verbose=training_config.get('verbose', True))
            else:
                model.fit(X_train, y_train, verbose=training_config.get('verbose', True))
                
        except Exception as e:
            print(f"⚠️  LightGBM training fallback: {e}")
            model.fit(X_train, y_train)
            
        return model
    
    def _train_catboost(self, model, X_train, y_train, X_val, y_val, training_config):
        """Train CatBoost with early stopping"""
        try:
            if X_val is not None:
                eval_set = [(X_val, y_val)]
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=training_config.get('early_stopping_rounds', 50),
                    verbose=training_config.get('verbose', False)
                )
            else:
                model.fit(X_train, y_train, verbose=training_config.get('verbose', False))
        except Exception as e:
            print(f"⚠️  CatBoost training fallback: {e}")
            model.fit(X_train, y_train)
            
        return model
    
    def _train_sklearn(self, model, X_train, y_train, training_config):
        """Train sklearn models (no early stopping)"""
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"⚠️  Sklearn training fallback: {e}")
            model.fit(X_train, y_train)
        return model
    
# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class BoostingModelEvaluator:
    """Evaluator for boosting models"""
    
    def __init__(self, model_config):
        self.model_config = model_config
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, Any]:
        """Evaluate the trained model"""
        print(f"📊 Evaluating {model_name}...")
        
        task_type = self.model_config['model']['task_type']
        
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # Get probabilities for classification tasks
            if task_type == 'classification' and hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics based on task type
            if task_type == 'classification':
                return self._evaluate_classification(y_test, y_pred, y_pred_proba, model_name)
            else:  # regression
                return self._evaluate_regression(y_test, y_pred, model_name)
                
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_classification(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate classification model"""
        accuracy = accuracy_score(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'task_type': 'classification',
            'accuracy': accuracy,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add additional metrics if probabilities are available
        if y_pred_proba is not None:
            from sklearn.metrics import log_loss, roc_auc_score
            try:
                if len(np.unique(y_true)) == 2:  # binary classification
                    results['log_loss'] = log_loss(y_true, y_pred_proba[:, 1])
                    results['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # multiclass
                    results['log_loss'] = log_loss(y_true, y_pred_proba)
            except:
                pass
        
        print(f"🎯 Classification Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        if 'log_loss' in results:
            print(f"   Log Loss: {results['log_loss']:.4f}")
        if 'roc_auc' in results:
            print(f"   ROC AUC: {results['roc_auc']:.4f}")
        
        # Save metrics for Katib
        save_metrics({
            'task_type': 'classification',
            'accuracy': accuracy,
            'val_accuracy': accuracy
        })
        
        return results
    
    def _evaluate_regression(self, y_true, y_pred, model_name):
        """Evaluate regression model"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'task_type': 'regression',
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'neg_mse': -mse  # Negative MSE for hyperparameter tuning (minimization)
        }
        
        print(f"🎯 Regression Results:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")
        
        # Save metrics for Katib
        save_metrics({
            'task_type': 'regression',
            'neg_mse': -mse,
            'r2_score': r2
        })
        
        return results

# =============================================================================
# MAIN PIPELINE
# =============================================================================

class BoostingPipeline:
    """Main pipeline orchestrator for boosting algorithms"""
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.data_processor = DataProcessor(model_config)
        self.model_trainer = BoostingModelTrainer(model_config)
        self.model_evaluator = BoostingModelEvaluator(model_config)
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        paths_config = self.model_config['paths']
        base_path = Path(paths_config['base_path'])
        
        directories = [
            paths_config['processed_data'],
            paths_config['trained_models'],
            paths_config['training_logs'],
            paths_config['configs'],
            paths_config['results']
        ]
        
        for directory in directories:
            dir_path = base_path / directory
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
            except Exception as e:
                print(f"⚠️  Could not create directory {dir_path}: {e}")
        
        # Also create the base models directory if it doesn't exist
        models_dir = Path("models/trained")
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Ensured models directory exists: {models_dir}")
    
    def run_complete_pipeline(self, process_data_url: str, model_type: str, 
                             model_name: str, unknown_args: list = None) -> Dict[str, Any]:
        """
        Run complete boosting pipeline
        """
        print("🚀 BOOSTING ALGORITHMS PIPELINE")
        print("=" * 50)
        print(f"Model Type: {model_type}")
        print(f"Model Name: {model_name}")
        print(f"Data URL: {process_data_url}")
        print("=" * 50)
        
        # Step 1: Load data
        print("\n1. 📦 LOADING DATA")
        raw_data = load_pickle_url(normalize_url(process_data_url))
        
        # Step 2: Extract and process data
        print("\n2. 🔧 PROCESSING DATA")
        X_train, y_train, X_test, y_test = self.data_processor.extract_data_from_pickle(raw_data)
        
        # Create validation split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=self.model_config['dataset'].get('validation_split', 0.1),
            random_state=self.model_config['dataset'].get('random_state', 42),
            shuffle=self.model_config['dataset'].get('shuffle', True)
        )
        
        print(f"   Final data shapes:")
        print(f"   - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   - X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"   - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Step 3: Create model
        print("\n3. 🏗️ BUILDING MODEL")
        algorithm = self.model_config['model']['algorithm']
        task_type = self.model_config['model']['task_type']
        
        model = BoostingModelFactory.create_model(algorithm, task_type, self.model_config['model'])
        
        # Step 4: Train model
        print("\n4. 🚀 TRAINING MODEL")
        full_model_name = f"{algorithm}_{task_type}_{model_name}"
        trained_model = self.model_trainer.train_model(model, X_train, y_train, X_val, y_val, full_model_name)
        
        # Step 5: Save model
        print("\n5. 💾 SAVING MODEL")
        model_path = self._save_model(trained_model, full_model_name)
        
        # Step 6: Evaluate model
        print("\n6. 📊 EVALUATING MODEL")
        evaluation_results = self.model_evaluator.evaluate_model(trained_model, X_test, y_test, full_model_name)
        
        print(f"\n🎉 BOOSTING PIPELINE COMPLETE!")
        return {
            'model_path': model_path,
            'results': evaluation_results,
            'data_shapes': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }
    
    def _save_model(self, model, model_name: str) -> str:
        """Save trained model"""
        try:
            # Try the configured path first
            models_path = Path(self.model_config['paths']['trained_models'])
            model_path = models_path / f"{model_name}_model.pkl"
            
            # Ensure directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"✅ Model saved to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            print(f"⚠️  Failed to save to configured path {model_path}: {e}")
            
            # Fallback: save to current directory
            try:
                fallback_path = Path(f"{model_name}_model.pkl")
                with open(fallback_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✅ Model saved to fallback location: {fallback_path}")
                return str(fallback_path)
            except Exception as e2:
                print(f"❌ Failed to save model: {e2}")
                return ""

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Boosting Pipeline for Classification and Regression')
    parser.add_argument('--model_type', type=str, required=True, 
                       help='Type of boosting algorithm and task (e.g., xgboost_classification, gradientboosting_regression)')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Specific model name/identifier')
    parser.add_argument('--process_data_url', type=str, required=True, 
                       help='URL to preprocessed dataset')
    parser.add_argument('--config', type=str, required=True, 
                       help='Base64 encoded JSON configuration')
    
    args, unknown = parser.parse_known_args()
    
    try:
        # Decode config
        print("=== DECODING CONFIG ===")
        try:
            decoded_config = base64.b64decode(args.config).decode('utf-8')
            config_json = json.loads(decoded_config)
            print("✅ Config decoded successfully")
        except Exception as e:
            print(f"❌ Failed to decode config: {e}")
            config_json = {}
        
        # Build configuration
        print("=== BUILDING CONFIGURATION ===")
        initial_config = build_config(
            model_type=args.model_type,
            model_name=args.model_name,
            unknown_args=unknown,
            process_data_url=args.process_data_url,
            config_json=config_json
        )
        
        initial_config['config_base64'] = args.config
        
        # Check algorithm availability
        algorithm = initial_config['model']['algorithm']
        algorithm_checks = {
            'xgboost': xgb is not None,
            'gradient_boosting': GradientBoostingClassifier is not None,
            'adaboost': AdaBoostClassifier is not None,
            'lightgbm': lgb is not None,
            'catboost': cb is not None
        }
        
        if not algorithm_checks.get(algorithm, False):
            available = [alg for alg, available in algorithm_checks.items() if available]
            raise ImportError(f"Algorithm '{algorithm}' not available. Available algorithms: {available}")
        
        # Run pipeline
        pipeline = BoostingPipeline(initial_config)
        results = pipeline.run_complete_pipeline(
            process_data_url=args.process_data_url,
            model_type=args.model_type,
            model_name=args.model_name,
            unknown_args=unknown
        )
        
        print(f"\n✅ Boosting pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error metrics
        save_metrics({'accuracy': 0.0, 'task_type': 'classification'})
        sys.exit(1)
    
    print("=== BOOSTING PIPELINE COMPLETED ===")
    sys.exit(0)

if __name__ == "__main__":
    main()
