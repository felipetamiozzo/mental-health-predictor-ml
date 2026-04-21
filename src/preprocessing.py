import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def get_preprocessor(categorical_features):
    """Cria o objeto de pré-processamento (Pipeline de encoders)."""
    try:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ],
            remainder='passthrough'
        )
        logging.info("Preprocessor criado com sucesso.")
        return preprocessor
    except Exception as e:
        logging.error(f"Erro ao criar o preprocessor: {e}")
        raise