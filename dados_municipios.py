import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)

st.title("Municipios do Brasil")

url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

dados = requests.get(url).json()

lista = []

for municipio in dados:

    codigo = municipio["id"]

    nome = municipio["nome"]

    uf = municipio["microrregiao"]["mesorregiao"]["UF"]["sigla"]

    lista.append({
        "codigo": codigo,
        "nome": nome,
        "uf": uf
    })

df = pd.DataFrame(lista)

df["codigo_reduzido"] = df["codigo"] % 1000

df["indice"] = (
    (df["codigo"] % 1000)
    +
    (df["codigo"] % 100)
)

df["sul"] = (
    df["uf"].isin(["PR", "SC", "RS"])
).astype(int)

st.write("Tabela de Municipios")

st.dataframe(df)

st.write("Quantidade de municipios")

st.write(len(df))

st.write("Regressao Linear")

X = df[["codigo_reduzido"]]

y = df["indice"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo_linear = LinearRegression()

modelo_linear.fit(X_treino, y_treino)

previsoes = modelo_linear.predict(X_teste)

mae = mean_absolute_error(
    y_teste,
    previsoes
)

r2 = r2_score(
    y_teste,
    previsoes
)

st.write("MAE")

st.write(mae)

st.write("R2")

st.write(r2)

plt.figure()

plt.scatter(
    y_teste,
    previsoes
)

plt.xlabel("Real")

plt.ylabel("Previsto")

plt.title("Regressao Linear")

st.pyplot(plt)

st.write("Regressao Logistica")

X = df[["codigo_reduzido"]]

y = df["sul"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo_logistico = LogisticRegression()

modelo_logistico.fit(
    X_treino,
    y_treino
)

previsoes = modelo_logistico.predict(
    X_teste
)

acuracia = accuracy_score(
    y_teste,
    previsoes
)

st.write("Acuracia")

st.write(acuracia)

matriz = confusion_matrix(
    y_teste,
    previsoes
)

plt.figure()

plt.imshow(matriz)

plt.title("Matriz de Confusao")

st.pyplot(plt)

st.write("Teste de Predicao")

valor = st.number_input(
    "Digite um codigo reduzido",
    min_value=0,
    max_value=999
)

if st.button("Prever"):

    resultado = modelo_logistico.predict(
        [[valor]]
    )

    if resultado[0] == 1:

        st.write(
            "O modelo acredita que pertence a Regiao Sul"
        )

    else:

        st.write(
            "O modelo acredita que nao pertence a Regiao Sul"
        )