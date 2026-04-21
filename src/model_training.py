import joblib
import logging
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def train_model_with_grid_search(pipeline, X_train, y_train, param_grid):
    """Executa o tuning (GridSearch) focado em RECALL."""
    try:
        logging.info("Iniciando Grid Search (Otimizando para Recall)...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='recall', 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Erro no treinamento: {e}")
        raise

def save_final_pipeline(model_pipeline, path="models/modelo_saude_mental_v1.pkl"):
    """Garante a criação da pasta e salva o artefato final."""
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        joblib.dump(model_pipeline, path)
        logging.info(f"Modelo salvo com sucesso em: {path}")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo: {e}")