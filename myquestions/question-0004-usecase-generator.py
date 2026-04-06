import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

def generar_caso_de_uso_segmentar_productos_tienda():
    random.seed(None)
    n = random.randint(40, 100)
    n_clusters = random.randint(2, 4)
    categorias = ['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Libros']

    df = pd.DataFrame({
        'producto_id': range(1, n + 1),
        'categoria': [random.choice(categorias) for _ in range(n)],
        'precio': np.round(np.random.uniform(5, 500, n), 2),
        'n_ventas': np.random.randint(0, 1000, n).astype(float),
        'calificacion': np.round(np.random.uniform(1, 5, n), 1)
    })
    # Introducir NaN
    nan_idx_cal = np.random.choice(n, size=int(n * 0.1), replace=False)
    nan_idx_ven = np.random.choice(n, size=int(n * 0.08), replace=False)
    df.loc[nan_idx_cal, 'calificacion'] = np.nan
    df.loc[nan_idx_ven, 'n_ventas'] = np.nan
    # Algunos precios inválidos
    df.loc[np.random.choice(n, 2, replace=False), 'precio'] = np.nan

    input_data = {'df': df.copy(), 'n_clusters': n_clusters}

    # Ground truth
    df_gt = df.copy()
    df_gt = df_gt[df_gt['precio'].notna() & (df_gt['precio'] > 0)].copy()
    df_gt['calificacion'] = df_gt['calificacion'].fillna(df_gt['calificacion'].median())
    df_gt['n_ventas'] = df_gt['n_ventas'].fillna(0)
    df_gt['ingreso_estimado'] = df_gt['precio'] * df_gt['n_ventas']

    features = df_gt[['precio', 'n_ventas', 'calificacion', 'ingreso_estimado']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_gt['segmento'] = kmeans.fit_predict(features_scaled)
    df_gt = df_gt.reset_index(drop=True)

    perfil = df_gt.groupby('segmento')[['precio', 'n_ventas', 'calificacion', 'ingreso_estimado']]\
                  .mean().round(2)

    output_data = (df_gt, perfil)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_segmentar_productos_tienda()
    print("=== INPUT ===")
    print(f"n_clusters: {entrada['n_clusters']}")
    print(entrada['df'].head())
    print("\n=== OUTPUT ===")
    df_out, perfil = salida
    print(df_out[['producto_id', 'segmento']].head())
    print("\nPerfil por segmento:")
    print(perfil)
