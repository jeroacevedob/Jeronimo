import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluar_con_confusion_acumulada(X, y, n_folds):
    # Validación cruzada estratificada
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    n_classes = len(np.unique(y))
    confusion_total = np.zeros((n_classes, n_classes), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Escalado
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Modelo
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train_s, y_train)

        # Predicción
        y_pred = clf.predict(X_test_s)

        # Métricas
        accuracies.append(accuracy_score(y_test, y_pred))
        confusion_total += confusion_matrix(
            y_test, y_pred, labels=np.arange(n_classes)
        )

    return {
        'accuracy_promedio': round(float(np.mean(accuracies)), 4),
        'confusion_acumulada': confusion_total
    }


# 🔽 BLOQUE DE PRUEBA CON PRINTS (como te piden)
if __name__ == "__main__":
    # Generar datos de prueba (solo para testear)
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    n_folds = 4

    resultado = evaluar_con_confusion_acumulada(X, y, n_folds)

    print("=== INPUT ===")
    print(f"X shape: {X.shape}")
    print(f"n_folds: {n_folds}")

    print("\n=== OUTPUT ===")
    print(f"Accuracy promedio: {resultado['accuracy_promedio']}")
    print("Confusion acumulada:")
    print(resultado['confusion_acumulada'])
