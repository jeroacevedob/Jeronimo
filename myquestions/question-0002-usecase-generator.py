import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_saldo_acumulado():
    random.seed(None)
    n_cuentas = random.randint(2, 4)
    cuentas = [f'CUENTA-{i:03d}' for i in range(1, n_cuentas + 1)]
    cuenta_elegida = random.choice(cuentas)

    n_total = random.randint(20, 50)
    fechas = pd.date_range(start='2023-01-01', periods=n_total, freq='5D')
    fechas_shuffle = list(fechas)
    random.shuffle(fechas_shuffle)

    tipos = np.random.choice(['debito', 'credito'], n_total)
    montos = np.round(np.random.uniform(10, 5000, n_total), 2)
    cuentas_col = [random.choice(cuentas) for _ in range(n_total)]
    # Garantizar que la cuenta elegida tenga al menos 5 transacciones
    for i in range(5):
        cuentas_col[i] = cuenta_elegida

    df = pd.DataFrame({
        'cuenta_id': cuentas_col,
        'fecha': [f.strftime('%Y-%m-%d') for f in fechas_shuffle],
        'tipo': tipos,
        'monto': montos
    })

    input_data = {'df': df.copy(), 'cuenta_id': cuenta_elegida}

    # Ground truth
    df_gt = df.copy()
    df_gt['fecha'] = pd.to_datetime(df_gt['fecha'])
    df_gt = df_gt[df_gt['cuenta_id'] == cuenta_elegida].sort_values('fecha').reset_index(drop=True)
    df_gt['monto_neto'] = np.where(df_gt['tipo'] == 'credito', df_gt['monto'], -df_gt['monto'])
    df_gt['saldo_acumulado'] = df_gt['monto_neto'].cumsum()
    output_data = df_gt[['fecha', 'tipo', 'monto', 'monto_neto', 'saldo_acumulado']].reset_index(drop=True)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_calcular_saldo_acumulado()
    print("=== INPUT ===")
    print(f"cuenta_id: {entrada['cuenta_id']}")
    print(entrada['df'].head())
    print("\n=== OUTPUT ===")
    print(salida)
