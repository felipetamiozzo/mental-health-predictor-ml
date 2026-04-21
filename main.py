print("Iniciando o script...")
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from src.preprocessing import get_preprocessor
from src.model_training import train_model_with_grid_search, save_final_pipeline
from src.evaluation import plot_threshold_comparison, plot_feature_importance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run():
    # 1. Carga e Salvamento de Processados
    os.makedirs('data/processed', exist_ok=True)
    df = pd.read_csv('data/raw/survey.csv')
    df.to_csv('data/processed/mental_health_cleaned.csv', index=False)
    
    features = ['work_interfere', 'family_history', 'leave', 'care_options', 
                'benefits', 'Country', 'phys_health_consequence', 
                'mental_health_interview', 'anonymity', 'no_employees', 'Gender']
    
    X = df[features].copy()
    y = LabelEncoder().fit_transform(df['treatment'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2, stratify=y
    )
    
    # 2. Pipeline e Treinamento
    preprocessor = get_preprocessor(features)
    pipeline_base = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
    
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.005, 0.1],
        'model__max_depth': [3, 4],
        'model__subsample': [0.8, 1.0]
    }
    
    best_model = train_model_with_grid_search(pipeline_base, X_train, y_train, param_grid)
    
    # 3. Avaliação de Threshold
    y_proba = best_model.predict_proba(X_test)[:, 1]
    plot_threshold_comparison(y_test, y_proba, [0.40, 0.45, 0.50, 0.55])
    
    # Relatório final (Threshold 0.40)
    print("\n" + "="*30)
    print("RELATÓRIO FINAL (Threshold 0.40)")
    print("="*30)
    print(classification_report(y_test, (y_proba >= 0.40).astype(int)))
    
    # 4. Importância das Variáveis (Mapeamento)
    rename_map = {
        'x1_No': 'Histórico Familiar: Não', 'x0_Never': 'Interfere: Nunca', 
        'x1_Yes': 'Histórico Familiar: Sim', 'x3_Yes': 'Opções de Cuidado: Sim',
        'x0_Often': 'Interfere: Frequente', 'x10_Male': 'Gênero: Masculino'
    }
    plot_feature_importance(best_model, rename_map)
    
    # 5. Salvamento do Modelo Final
    save_final_pipeline(best_model)

if __name__ == "__main__":
    run()