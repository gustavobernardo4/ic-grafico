import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import streamlit as st
from io import BytesIO

# Configuração visual
st.set_page_config(page_title="Simulador de Campo de Fase de EDO de 2ª Ordem", layout="wide")
st.markdown("""
<style>
    header { visibility: hidden; }
    .block-container {
        padding-top: 0.2rem;
        padding-bottom: 0rem;
        max-width: none;
    }
    .main > div {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    img {
        max-width: 600px !important;
        height: auto;
        margin: auto;
        display: block;
    }
    .st-emotion-cache-13k62yr {
        padding-top: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

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
    A = np.array([[0, 1], [-c/a, -b/a]])
    autovalores, autovetores = np.linalg.eig(A)
    autovalores = np.real_if_close(autovalores, tol=1e-6)
    autovetores = simplificar_autovetores(autovetores)
    lambda1, lambda2 = autovalores

    if np.allclose(lambda1, lambda2):
        if verificar_diagonalizavel(A, autovalores):
            solucao = f"x(t) = C_1 e^{{{lambda1:.2f}t}} {autovetores[:, 0]} + C_2 e^{{{lambda2:.2f}t}} {autovetores[:, 1]}"
        else:
            solucao = f"x(t) = (C_1 + C_2 t) e^{{{lambda1:.2f}t}} {autovetores[:, 0]}"
    elif np.iscomplex(autovalores).any():
        alpha = np.real(lambda1)
        beta = np.abs(np.imag(lambda1))
        solucao = f"x(t) = e^{{{alpha:.2f}t}} [C_1 \cos({beta:.2f}t) + C_2 \sin({beta:.2f}t)]"
    else:
        solucao = f"x(t) = C_1 e^{{{lambda1:.2f}t}} {autovetores[:, 0]} + C_2 e^{{{lambda2:.2f}t}} {autovetores[:, 1]}"

    if np.iscomplex(autovalores).any():
        tipo = "Foco estável" if np.real(lambda1) < 0 else "Foco instável"
    elif lambda1 * lambda2 > 0:
        tipo = "Nó estável" if lambda1 < 0 else "Nó instável"
    elif np.allclose(lambda1, lambda2):
        tipo = "Nó estável próprio" if lambda1 < 0 else "Nó instável próprio"
        if not verificar_diagonalizavel(A, autovalores):
            tipo = tipo.replace("próprio", "degenerado")
    else:
        tipo = "Ponto de sela"

    t = np.linspace(0, 3, 1000)
    conds = [[1,0], [0,1], [-1,0], [0,-1], [2,1], [-2,-1], [1,-2], [-1,2]]
    sols = [odeint(lambda x, t: A @ x, x0, t) for x0 in conds]
    all_sols = np.vstack(sols)
    x_min, x_max = np.min(all_sols[:, 0]), np.max(all_sols[:, 0])
    y_min, y_max = np.min(all_sols[:, 1]), np.max(all_sols[:, 1])
    dx = (x_max - x_min) * 0.1
    dy = (y_max - y_min) * 0.1

    x, y = np.meshgrid(np.linspace(x_min - dx, x_max + dx, 20), np.linspace(y_min - dy, y_max + dy, 20))
    dx_field = A[0, 0]*x + A[0, 1]*y
    dy_field = A[1, 0]*x + A[1, 1]*y

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=100)
    ax.streamplot(x, y, dx_field, dy_field, color='blue', density=1.5, linewidth=0.8)
    for x0 in conds:
        sol = odeint(lambda x, t: A @ x, x0, t)
        ax.plot(sol[:, 0], sol[:, 1], 'r-', alpha=0.7)

    if not np.iscomplex(autovalores).any():
        escala = 3 / np.max(np.abs(autovetores))
        ax.quiver(0, 0, *(escala * autovetores[:, 0]), color='black', angles='xy', scale_units='xy', scale=1)
        if not np.allclose(lambda1, lambda2) or verificar_diagonalizavel(A, autovalores):
            ax.quiver(0, 0, *(escala * autovetores[:, 1]), color='green', angles='xy', scale_units='xy', scale=1)

    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    return buf, A, autovalores, autovetores, solucao, tipo

explicacoes = {
    "Foco estável": "As soluções giram em espiral para o ponto de equilíbrio.",
    "Foco instável": "As soluções giram em espiral para longe do ponto de equilíbrio.",
    "Nó estável próprio": "As soluções convergem diretamente pela base de autovetores reais distintos.",
    "Nó instável próprio": "As soluções divergem diretamente pela base de autovetores reais distintos.",
    "Nó estável degenerado": "As soluções convergem com autovalores repetidos e base incompleta.",
    "Nó instável degenerado": "As soluções divergem com autovalores repetidos e base incompleta.",
    "Ponto de sela": "As soluções divergem em uma direção e convergem em outra."
}

st.markdown("## Simulador de Campo de Fase de EDO de 2ª Ordem")

default_ranges = {
    "a": {"min_value": 0.1, "max_value": 5.0, "step": 0.1},
    "b": {"min_value": -10.0, "max_value": 10.0, "step": 0.5},
    "c": {"min_value": -10.0, "max_value": 10.0, "step": 0.5},
}

tab_simulacao, tab_config = st.tabs(["📊 Simulação", "⚙️ Intervalos"])

with tab_config:
    for p in ["a", "b", "c"]:
        st.subheader(f"Intervalo para '{p}'")
        default_ranges[p]["min_value"] = st.number_input(f"Min de {p}", value=default_ranges[p]["min_value"], key=f"{p}_min")
        default_ranges[p]["max_value"] = st.number_input(f"Max de {p}", value=default_ranges[p]["max_value"], key=f"{p}_max")
        default_ranges[p]["step"] = st.number_input(f"Passo de {p}", value=default_ranges[p]["step"], key=f"{p}_step")

with tab_simulacao:
    st.sidebar.header("🎛️ Coeficientes")
    a = st.sidebar.slider("a", **default_ranges["a"], value=1.0, format="%.2f")
    b = st.sidebar.slider("b", **default_ranges["b"], value=2.0)
    c = st.sidebar.slider("c", **default_ranges["c"], value=3.0)

    if a != 0:
        buf, A, autovalores, autovetores, solucao, tipo = analisar_sistema(a, b, c)
        col1, col2 = st.columns([1.7, 1.3], gap="large")

        with col1:
            st.markdown("### 🧮 Equação diferencial analisada:")
            st.latex(f"{a:.2f} \\cdot x''(t) + {b:.2f} \\cdot x'(t) + {c:.2f} \\cdot x(t) = 0")
            st.image(buf, use_container_width=True)
            st.markdown("<p style='text-align:center; font-size:0.85rem; margin-top:-0.5rem;'>📈 Eixo horizontal: posição (x)    |   Eixo vertical: velocidade (x′)</p>", unsafe_allow_html=True)
            st.download_button("📥 Baixar gráfico", data=buf, file_name="grafico_EDO.png", mime="image/png")

        with col2:
            st.markdown("### Detalhes da Análise")
            st.markdown("**Classificação:**")
            st.markdown(f"*{tipo}*")
            st.markdown(explicacoes.get(tipo, ""))

            st.markdown("**Matriz do sistema:**")
            st.latex(f"A = \\begin{{bmatrix}} 0 & 1 \\ {A[1,0]:.2f} & {A[1,1]:.2f} \\end{{bmatrix}}")

            st.markdown("**Autovalores:**")
            st.latex(f"\\lambda_1 = {autovalores[0]:.2f},\\quad \\lambda_2 = {autovalores[1]:.2f}")

            st.markdown("**Autovetores:**")
            for i in range(autovetores.shape[1]):
                st.latex(f"\\vec{{v}}_{{{i+1}}} = \\begin{{bmatrix}} {autovetores[0,i]:.2f} \\ {autovetores[1,i]:.2f} \\end{{bmatrix}}")

            st.markdown("**Solução geral:**")
            st.latex(solucao)
    else:
        st.warning("O coeficiente 'a' deve ser diferente de zero.")
