import yaml
import optuna
import tempfile
import mlflow
from ai.walk_forward_trainer import WalkForwardTrainer
from data_manager import DataManager

class Optimizer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_manager = DataManager(self.config['data_manager'])
        # 合并策略和模型超参数空间
        self.hyperparameter_space = {
            **self.config['optimizer']['strategy_hyperparameter_space'],
            **self.config['optimizer']['model_hyperparameter_space']
        }

    def _objective(self, trial):
        # [新] 启动一个MLflow运行来跟踪实验
        with mlflow.start_run():
            params = {}
            for param, bounds in self.hyperparameter_space.items():
                if bounds['type'] == 'int':
                    params[param] = trial.suggest_int(param, bounds['min'], bounds['max'])
                elif bounds['type'] == 'float':
                    params[param] = trial.suggest_float(param, bounds['min'], bounds['max'], log=bounds.get('log', False))
            
            # [新] 记录本次试验的所有参数
            mlflow.log_params(params)

            # 为本次试验创建一个临时的模型配置
            model_config = self._create_trial_model_config(params)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml', dir='ai') as tmp_file:
                yaml.dump(model_config, tmp_file)
                model_config_path = tmp_file.name

            historical_data = self.data_manager.load_historical_data('btcusdt')
            
            trainer = WalkForwardTrainer(self.config, params, model_config_path=model_config_path)
            results = trainer.run(historical_data)

            # [新] 记录本次试验的主要指标
            sharpe_ratio = results.get('sharpe_ratio', 0.0)
            mlflow.log_metric("sharpe_ratio", sharpe_ratio)
            
            return sharpe_ratio

    def _create_trial_model_config(self, params):
        """[新] 从试验参数创建一个模型配置字典。"""
        return {
            'level_one_cnn': {
                'filters': params['cnn_filters'],
                'kernel_size': 2 # 暂时保持固定
            },
            'level_two_transformer': {
                'head_size': 256, # 固定
                'num_heads': 4, # 固定
                'ff_dim': 4, # 固定
                'num_transformer_blocks': params['transformer_blocks'],
                'dropout': params['transformer_dropout']
            }
        }

    def run(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.config['optimizer']['n_trials'])
        print("最佳试验:", study.best_trial.params)
        return study.best_trial.params
