import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import random

def generar_caso_de_uso_evaluar_regresion():
    """
    Genera un caso de prueba aleatorio para la función evaluar_regresion(X, y, test_size).
    Retorna: (input_data, output_data)
      - input_data: dict con claves 'X', 'y', 'test_size'
      - output_data: tupla (r2, mae) de floats redondeados a 4 decimales
    """
    random.seed(None)
    n_samples = random.randint(50, 150)
    n_features = random.randint(1, 5)
    test_size = round(random.uniform(0.15, 0.4), 2)

    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features)
    noise = np.random.randn(n_samples) * random.uniform(0.1, 2.0)
    y = X @ coef + noise

    input_data = {
        'X': X.copy(),
        'y': y.copy(),
        'test_size': test_size
    }

    # Ground truth
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = round(float(r2_score(y_test, y_pred)), 4)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 4)

    output_data = (r2, mae)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_evaluar_regresion()
    print("=== INPUT ===")
    print(f"X shape: {entrada['X'].shape}")
    print(f"y shape: {entrada['y'].shape}")
    print(f"test_size: {entrada['test_size']}")
    print("\n=== OUTPUT ===")
    r2, mae = salida
    print(f"R2: {r2}")
    print(f"MAE: {mae}")
