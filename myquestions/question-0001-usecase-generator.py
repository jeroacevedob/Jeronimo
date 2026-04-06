import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_antiguedad_salarial():
    random.seed(None)
    n = random.randint(20, 60)
    departamentos = ['Ingeniería', 'Ventas', 'RRHH', 'Finanzas', 'Operaciones']
    
    fechas = pd.date_range(end='2024-01-01', periods=n, freq='90D')
    fechas_str = [f.strftime('%Y-%m-%d') for f in fechas]
    random.shuffle(fechas_str)

    df = pd.DataFrame({
        'empleado_id': range(1, n + 1),
        'departamento': [random.choice(departamentos) for _ in range(n)],
        'fecha_ingreso': fechas_str,
        'salario': np.round(np.random.uniform(2000, 12000, n), 2),
        'activo': np.random.choice([True, False], n, p=[0.75, 0.25])
    })

    # Ground truth
    df_gt = df.copy()
    df_gt['fecha_ingreso'] = pd.to_datetime(df_gt['fecha_ingreso'])
    df_gt['antiguedad_anios'] = ((pd.Timestamp.today() - df_gt['fecha_ingreso']).dt.days / 365.25).round(2)
    df_gt = df_gt[df_gt['activo'] == True]

    resultado = df_gt.groupby('departamento').agg(
        salario_promedio=('salario', lambda x: round(x.mean(), 2)),
        antiguedad_promedio=('antiguedad_anios', lambda x: round(x.mean(), 2)),
        n_empleados=('empleado_id', 'count')
    ).reset_index().sort_values('salario_promedio', ascending=False).reset_index(drop=True)

    input_data = {'df': df.copy()}
    output_data = resultado

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_calcular_antiguedad_salarial()
    print("=== INPUT ===")
    print(entrada['df'].head())
    print("\n=== OUTPUT ===")
    print(salida)
    print(f"label_col: {entrada['label_col']}")
    print(entrada['df'].head())
    print("\n=== OUTPUT (predicciones del modelo sobre X_train) ===")
    print(salida)
