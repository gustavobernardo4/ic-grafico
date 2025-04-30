import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import streamlit as st

st.set_page_config(layout="wide")
np.set_printoptions(precision=2, suppress=True)

def simplificar_autovetores(autovetores):
    autovetores = np.real_if_close(autovetores, tol=1e-6)
    for i in range(autovetores.shape[1]):
        v = autovetores[:, i]
        elementos_nao_nulos = np.abs(v[v != 0])
        if len(elementos_nao_nulos) == 0:
            continue
        menor_valor = elementos_nao_nulos[np.argmin(elementos_nao_nulos)]
        v_normalizado = np.round(v / menor_valor, 5)
        autovetores[:, i] = v_normalizado
    return autovetores

def verificar_diagonalizavel(A, autovalores):
    lambda_ = autovalores[0]
    matriz_aux = A - lambda_ * np.eye(2)
    return np.linalg.matrix_rank(matriz_aux) == 0

def analisar_sistema(a, b, c):
    matriz_A = np.array([[0, 1], [-c/a, -b/a]])
    discriminante = b**2 - 4*a*c

    if np.isclose(discriminante, 0):
        lambda1 = -b / (2*a)
        autovalores = np.array([lambda1, lambda1])
        if verificar_diagonalizavel(matriz_A, autovalores):
            autovetores = np.eye(2)
        else:
            v1 = np.array([1, -b/(2*a)])
            autovetores = np.column_stack((v1, v1))
    else:
        autovalores, autovetores = np.linalg.eig(matriz_A)
        autovalores = np.real_if_close(autovalores, tol=1e-6)
        autovetores = simplificar_autovetores(autovetores)

    lambda1, lambda2 = autovalores

    if np.allclose(lambda1, lambda2):
        if verificar_diagonalizavel(matriz_A, autovalores):
            solucao = f"x(t) = C1·e^({lambda1:.2f}t)·{autovetores[:, 0]} + C2·e^({lambda2:.2f}t)·{autovetores[:, 1]}"
        else:
            solucao = f"x(t) = (C1 + C2·t)·e^({lambda1:.2f}t)·{autovetores[:, 0]}"
    elif np.iscomplex(autovalores).any():
        alpha = np.real(lambda1)
        beta = np.abs(np.imag(lambda1))
        solucao = f"x(t) = e^({alpha:.2f}t) * [C1·cos({beta:.2f}t) + C2·sin({beta:.2f}t)]"
    else:
        solucao = f"x(t) = C1·e^({lambda1:.2f}t)·{autovetores[:, 0]} + C2·e^({lambda2:.2f}t)·{autovetores[:, 1]}"

    if np.iscomplex(autovalores).any():
        tipo = "Foco " + ("estável" if np.real(autovalores[0]) < 0 else "instável")
    elif np.allclose(lambda1, lambda2):
        if verificar_diagonalizavel(matriz_A, autovalores):
            tipo = "Nó " + ("estável" if lambda1 < 0 else "instável") + " próprio"
        else:
            tipo = "Nó " + ("estável" if lambda1 < 0 else "instável") + " degenerado"
    elif lambda1 * lambda2 > 0:
        tipo = "Nó " + ("estável" if lambda1 < 0 else "instável")
    else:
        tipo = "Ponto de sela"

    t = np.linspace(0, 3, 1000)
    condicoes_iniciais = [[1,0], [0,1], [-1,0], [0,-1], [2,1], [-2,-1], [1,-2], [-1,2]]
    todas_solucoes = []

    for x0 in condicoes_iniciais:
        solucao_numerica = odeint(lambda x, t: matriz_A @ x, x0, t)
        todas_solucoes.append(solucao_numerica)

    todas_solucoes = np.vstack(todas_solucoes)
    x_min, x_max = np.min(todas_solucoes[:, 0]), np.max(todas_solucoes[:, 0])
    y_min, y_max = np.min(todas_solucoes[:, 1]), np.max(todas_solucoes[:, 1])
    margem = 0.2
    dx = (x_max - x_min) * margem
    dy = (y_max - y_min) * margem

    x, x_ponto = np.meshgrid(
        np.linspace(x_min - dx, x_max + dx, 20),
        np.linspace(y_min - dy, y_max + dy, 20)
    )
    dx_campo = matriz_A[0,0] * x + matriz_A[0,1] * x_ponto
    dx_ponto_campo = matriz_A[1,0] * x + matriz_A[1,1] * x_ponto

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.streamplot(x, x_ponto, dx_campo, dx_ponto_campo, color='blue', density=1.5, linewidth=0.8)

    for x0 in condicoes_iniciais:
        solucao_numerica = odeint(lambda x, t: matriz_A @ x, x0, t)
        ax.plot(solucao_numerica[:, 0], solucao_numerica[:, 1], 'r-', alpha=0.7)

    if not np.iscomplex(autovalores).any():
        escala = 3 / np.max(np.abs(autovetores))
        ax.quiver(0, 0, *(escala * autovetores[:, 0]), color='black', angles='xy', scale_units='xy', scale=1, alpha=0.8)
        if not np.allclose(lambda1, lambda2) or verificar_diagonalizavel(matriz_A, autovalores):
            ax.quiver(0, 0, *(escala * autovetores[:, 1]), color='green', angles='xy', scale_units='xy', scale=1, alpha=0.8)

    ax.set_title(f"Sistema: {tipo}\nAutovalores: {lambda1:.2f}, {lambda2:.2f}")
    ax.set_xlabel('x')
    ax.set_ylabel("x'")
    ax.grid(True)
    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    return fig, matriz_A, autovalores, autovetores, solucao, tipo

# ========== INTERFACE STREAMLIT ==========
st.title("Analisador de EDO de 2ª ordem: Interface Web")

col1, col2 = st.columns(2)
with col1:
    a = st.number_input("Coeficiente a", value=1.0)
    b = st.number_input("Coeficiente b", value=2.0)
    c = st.number_input("Coeficiente c", value=3.0)

if a != 0:
    fig, matriz_A, autovalores, autovetores, solucao, tipo = analisar_sistema(a, b, c)
    st.pyplot(fig)
    with st.expander("📋 Detalhes da Análise"):
        st.markdown(f"**Matriz do sistema:**\n```\n{matriz_A}\n```")
        st.markdown(f"**Autovalores:** `{autovalores}`")
        st.markdown(f"**Autovetores:**\n```\n{autovetores}\n```")
        st.markdown(f"**Solução geral:**\n```\n{solucao}\n```")
        st.markdown(f"**Classificação do sistema:** `{tipo}`")
else:
    st.warning("O coeficiente 'a' deve ser diferente de zero.")
