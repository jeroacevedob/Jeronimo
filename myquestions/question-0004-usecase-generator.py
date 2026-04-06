import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

def generar_caso_de_uso_reducir_dimensiones():
    """
    Genera un caso de prueba aleatorio para la función reducir_dimensiones(df, n_components).
    Retorna: (input_data, output_data)
      - input_data: dict con claves 'df' y 'n_components'
      - output_data: tupla (df_pca, varianza_explicada)
    """
    random.seed(None)
    n_rows = random.randint(30, 80)
    n_features = random.randint(4, 8)
    n_components = random.randint(2, min(n_features, 4))

    cols = [f'var_{i}' for i in range(n_features)]
    data = np.random.randn(n_rows, n_features)
    df = pd.DataFrame(data, columns=cols)

    # Introducir ~8% de NaN
    mask = np.random.choice([True, False], size=df.shape, p=[0.08, 0.92])
    df[mask] = np.nan

    input_data = {
        'df': df.copy(),
        'n_components': n_components
    }

    # Ground truth
    df_clean = df.dropna().reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols)

    varianza_explicada = pca.explained_variance_ratio_

    output_data = (df_pca, varianza_explicada)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_reducir_dimensiones()
    print("=== INPUT ===")
    print(f"n_components: {entrada['n_components']}")
    print(entrada['df'].head())
    print("\n=== OUTPUT ===")
    df_pca, varianza = salida
    print("DataFrame PCA:")
    print(df_pca.head())
    print("Varianza explicada:", varianza)
