
import numpy as np
import pandas as pd
import time
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Função de avaliação e salvamento de relatórios
def avaliar_modelo_multiclasse(modelo, X_test, y_test, nome_modelo='Modelo'):
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print(f"\nAvaliação do modelo: {nome_modelo}")
    print(f"Acurácia         : {acc:.4f}")
    print(f"F1-score (macro) : {f1:.4f}")
    print(f"Precisão (macro) : {precision:.4f}")
    print(f"Revocação (macro): {recall:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'matriz_confusao/matriz_confusao_{nome_modelo}.png')
    plt.close()

    # Relatório
    with open(f"relatorios_classificacao/classification_report_{nome_modelo}.txt", "w") as f:
        f.write(f"Relatório do modelo: {nome_modelo}\n\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))

    return {
        'Modelo': nome_modelo,
        'Acuracia': acc,
        'F1_macro': f1,
        'Precisao_macro': precision,
        'Recall_macro': recall
    }

# Pipeline com organização de arquivos e treinamento
def NCV_treinar_modelos_robustos(df, nome_alvo, colunas_numericas, colunas_onehot,
                                test_size=0.2, random_state=42, n_iter_nn=20):

    # Criar pastas
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("matriz_confusao", exist_ok=True)
    os.makedirs("relatorios_classificacao", exist_ok=True)
    os.makedirs("modelo_vencedor", exist_ok=True)

    # Preparar dados
    X = df.drop(columns=[nome_alvo])
    y = df[nome_alvo]
    X[colunas_onehot] = X[colunas_onehot].astype(str)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), colunas_numericas),
        ('cat', 'passthrough', colunas_onehot)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state)

    # Definição dos modelos
    modelos = {
        'Regressao_Logistica': (
            LogisticRegression(solver='saga', max_iter=1000, random_state=random_state),
            [
                {
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__C': [0.001, 0.01, 0.1, 1],
                    'classifier__class_weight': [None, 'balanced']
                },
                {
                    'classifier__penalty': ['elasticnet'],
                    'classifier__C': [0.001, 0.01, 0.1, 1],
                    'classifier__l1_ratio': [0.3, 0.5, 0.7],
                    'classifier__class_weight': [None, 'balanced']
                }
            ],
            GridSearchCV
        ),
        'Random_Forest': (
            RandomForestClassifier(random_state=random_state),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 20],
                'classifier__min_samples_split': [3, 5],
                'classifier__min_samples_leaf': [1, 3],
                'classifier__max_features': ['sqrt', 'log2', 0.5],
                'classifier__bootstrap': [True, False]
            },
            RandomizedSearchCV
        ),
        'LightGBM': (
            LGBMClassifier(objective='multiclass', random_state=random_state),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [-1, 5, 10],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__num_leaves': [31, 63],
                'classifier__min_child_samples': [5, 10],
                'classifier__subsample': [0.6, 0.8],
                'classifier__colsample_bytree': [0.6, 0.8]
            },
            RandomizedSearchCV
        ),
        'CatBoost': (
            CatBoostClassifier(verbose=0, random_state=random_state),
            {
                'classifier__iterations': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__depth': [4, 6]
            },
            RandomizedSearchCV
        ),
        'ExtraTrees': (
            ExtraTreesClassifier(random_state=random_state),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20]
            },
            RandomizedSearchCV
        ),
        'Rede_Neural': (
            MLPClassifier(max_iter=1000, random_state=random_state),
            {
                'classifier__hidden_layer_sizes': [(100, 50), (128, 64), (64, 32, 16)],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__solver': ['adam', 'sgd'],
                'classifier__alpha': [0.01, 0.001],
                'classifier__batch_size': [128,64],
                'classifier__learning_rate_init': [0.05, 0.001],
                'classifier__momentum': [0.9, 0.95],
            },
            lambda *args, **kwargs: RandomizedSearchCV(*args, n_iter=n_iter_nn, **kwargs)
        )
    }

    resultados = []
    best_score = -np.inf
    best_model_name, best_model_base, best_model_params = None, None, None

    for nome, (modelo, hiperparams, buscador) in modelos.items():
        print(f"\nTreinando modelo: {nome}")
        try:
            pipe = Pipeline([
                ('smote', SMOTE(random_state=random_state)),
                ('preprocessor', preprocessor),
                ('classifier', modelo)
            ])

            start = time.time()
            search = buscador(pipe, hiperparams, cv=5, scoring='f1_macro', n_jobs=-1)
            search.fit(X_train, y_train)
            tempo = time.time() - start

            avaliacao = avaliar_modelo_multiclasse(search.best_estimator_, X_test, y_test, nome)

            resultados.append({
                'Modelo': nome,
                'Melhores Parametros': search.best_params_,
                'Melhor Score (Treino CV)': search.best_score_,
                'Tempo': tempo,
                **avaliacao
            })

            if avaliacao['F1_macro'] > best_score:
                best_score = avaliacao['F1_macro']
                best_model_name = nome
                best_model_base = modelo
                best_model_params = search.best_params_

        except Exception as e:
            print(f"Erro no modelo {nome}: {e}")

    # Salvar resultados
    df_resultados = pd.DataFrame(resultados).sort_values(by='F1_macro', ascending=False)
    df_resultados.to_csv("resultados/resultados_modelos.csv", index=False)

    print(f"\nRetreinando o melhor modelo: {best_model_name}")
    final_pipeline = Pipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('preprocessor', preprocessor),
        ('classifier', best_model_base)
    ])
    final_pipeline.set_params(**best_model_params)
    final_pipeline.fit(X_train, y_train)

    avaliar_modelo_multiclasse(final_pipeline, X_test, y_test, f"modelo_FINAL")
    joblib.dump(final_pipeline, f"modelo_vencedor/melhor_modelo.pkl")
    print(f"Modelo final salvo como modelo_vencedor/melhor_modelo.pkl")

    return df_resultados, final_pipeline, best_model_name
        

def main():
    df = pd.read_csv("dados/dataset_filmes_class.csv") 

    colunas_numericas = ['ano_lancamento',
                         'mes_seno',
                         'mes_cos',
                         'duracao',
                         'orcamento',
                         'media_elenco',
                         'media_direcao',
                         'mediana_elenco',
                         'mediana_direcao'
    ]

    colunas_onehot = [col for col in df.columns if col not in colunas_numericas + ['categoria_lucro']]

    resultados, modelo_final, nome_modelo = NCV_treinar_modelos_robustos(
        df,
        nome_alvo='categoria_lucro',
        colunas_numericas=colunas_numericas,
        colunas_onehot=colunas_onehot
    )

if __name__ == "__main__":
    main()
