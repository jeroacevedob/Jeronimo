import numpy as np
import random

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluar_con_confusion_acumulada(X, y, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    n_classes = len(np.unique(y))
    confusion_total = np.zeros((n_classes, n_classes), dtype=int)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)

        accuracies.append(accuracy_score(y_test, y_pred))
        confusion_total += confusion_matrix(
            y_test, y_pred, labels=np.arange(n_classes)
        )

    return {
        'accuracy_promedio': round(float(np.mean(accuracies)), 4),
        'confusion_acumulada': confusion_total
    }


def generar_caso_de_uso_evaluar_con_confusion_acumulada():
    random.seed(None)

    n_samples = random.randint(80, 200)
    n_classes = random.randint(2, 4)
    n_folds = random.randint(3, 5)
    n_features = random.randint(4, 8)

    n_informative = min(
        n_features - 1,
        max(2, int(np.ceil(np.log2(n_classes))))
    )

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random.randint(0, 100)
    )

    entrada = {
        "X": X,
        "y": y,
        "n_folds": n_folds
    }

    salida = evaluar_con_confusion_acumulada(X, y, n_folds)

    return entrada, salida


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_evaluar_con_confusion_acumulada()

    print("=== INPUT ===")
    print(f"X shape: {entrada['X'].shape}")
    print(f"n_folds: {entrada['n_folds']}")

    print("\n=== OUTPUT ===")
    print(f"Accuracy promedio: {salida['accuracy_promedio']}")
    print("Confusion acumulada:")
    print(salida['confusion_acumulada'])
