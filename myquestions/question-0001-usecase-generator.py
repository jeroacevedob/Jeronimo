import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import random

def generar_caso_de_uso_entrenar_clasificador():
    """
    Genera un caso de prueba aleatorio para la función entrenar_clasificador(df, label_col).
    Retorna: (input_data, output_data)
      - input_data: dict con claves 'df' y 'label_col'
      - output_data: KNeighborsClassifier entrenado
    """
    random.seed(None)
    n_rows = random.randint(20, 50)
    n_features = random.randint(2, 5)
    n_classes = random.randint(2, 4)

    feature_cols = [f'feature_{i}' for i in range(n_features)]
    data = np.random.randn(n_rows, n_features)

    df = pd.DataFrame(data, columns=feature_cols)

    # Introducir ~10% de NaN en features
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan

    label_col = 'clase'
    df[label_col] = np.random.randint(0, n_classes, size=n_rows)

    input_data = {
        'df': df.copy(),
        'label_col': label_col
    }

    # Ground truth: replicar la lógica de entrenar_clasificador
    df_clean = df.dropna(subset=[label_col]).copy()
    X = df_clean.drop(columns=[label_col])
    y = df_clean[label_col].values

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    modelo = KNeighborsClassifier(n_neighbors=3)
    modelo.fit(X_imputed, y)

    # Verificamos el output comparando predicciones sobre los mismos datos de entrenamiento
    output_data = modelo.predict(X_imputed)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_entrenar_clasificador()
    print("=== INPUT ===")
    print(f"label_col: {entrada['label_col']}")
    print(entrada['df'].head())
    print("\n=== OUTPUT (predicciones del modelo sobre X_train) ===")
    print(salida)
