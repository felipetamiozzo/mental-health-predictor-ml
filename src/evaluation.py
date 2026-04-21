import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, classification_report

def plot_threshold_comparison(y_test, y_proba, thresholds):
    """Visualiza o impacto do threshold no Recall."""
    plt.figure(figsize=(16, 4))
    for i, thr in enumerate(thresholds):
        y_pred_thr = (y_proba >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred_thr)
        rec_1 = recall_score(y_test, y_pred_thr)
        
        plt.subplot(1, 4, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Threshold: {thr}\nRecall: {rec_1:.2f}')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model_pipeline, rename_map):
    """Plota a importância das variáveis traduzida para o negócio."""
    model = model_pipeline.named_steps['model']
    preprocessor = model_pipeline.named_steps['preprocessor']
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    
    feature_names = encoder.get_feature_names_out()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    importance_df['Readable'] = importance_df['Feature'].map(rename_map).fillna(importance_df['Feature'])

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Readable', data=importance_df.head(15), palette='viridis')
    plt.title('Top 15 Fatores de Impacto na Saúde Mental')
    plt.xlabel('Importância (Gini)')
    plt.ylabel(None)
    plt.show()