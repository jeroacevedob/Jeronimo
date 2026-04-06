import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_limpiar_y_resumir():
    """
    Genera un caso de prueba aleatorio para la función limpiar_y_resumir(df, umbral_nan).
    Retorna: (input_data, output_data)
      - input_data: dict con claves 'df' y 'umbral_nan'
      - output_data: tupla (df_limpio, dict_medianas)
    """
    random.seed(None)
    n_rows = random.randint(15, 40)
    n_cols = random.randint(3, 6)

    cols = [f'col_{i}' for i in range(n_cols)]
    data = np.random.randn(n_rows, n_cols) * random.randint(1, 10)
    df = pd.DataFrame(data, columns=cols)

    # Introducir NaN con distintas tasas por columna
    for col in cols:
        nan_rate = random.uniform(0.0, 0.6)
        mask = np.random.choice([True, False], size=n_rows, p=[nan_rate, 1 - nan_rate])
        df.loc[mask, col] = np.nan

    # Duplicar algunas filas (~15%)
    n_dupes = max(1, int(n_rows * 0.15))
    dupe_indices = np.random.choice(df.index, size=n_dupes, replace=False)
    df = pd.concat([df, df.loc[dupe_indices]], ignore_index=True)

    umbral_nan = round(random.uniform(0.2, 0.5), 2)

    input_data = {
        'df': df.copy(),
        'umbral_nan': umbral_nan
    }

    # Ground truth
    df_gt = df.copy()
    df_gt = df_gt.drop_duplicates()

    nan_rate_cols = df_gt.isna().mean()
    cols_to_keep = nan_rate_cols[nan_rate_cols < umbral_nan].index.tolist()
    df_gt = df_gt[cols_to_keep]

    medianas = {col: df_gt[col].median() for col in df_gt.columns}
    df_gt = df_gt.fillna(medianas)

    output_data = (df_gt.reset_index(drop=True), medianas)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_limpiar_y_resumir()
    print("=== INPUT ===")
    print(f"umbral_nan: {entrada['umbral_nan']}")
    print(entrada['df'].head())
    print("\n=== OUTPUT ===")
    df_limpio, medianas = salida
    print("DataFrame limpio:")
    print(df_limpio.head())
    print("Medianas:", medianas)
