import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Apprendre Dijkstra", layout="wide")
st.markdown("""
    <style>
    
    section[data-testid="stSidebar"] {
        background-color: #A5D6A7;   
    }

    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1B5E20;
        font-weight: bold;
    }

    
    section[data-testid="stSidebar"] button {
        background-color: white;
        color: #1B5E20;
        border-radius: 12px;
        margin-bottom: 8px;
        border: none;
        font-weight: bold;
    }

    
    section[data-testid="stSidebar"] button:hover {
        background-color: #E8F5E9;
        color: #1B5E20;
    }
    </style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "Accueil"

def set_page(name):
    st.session_state.page = name

st.sidebar.title("🚀 Menu")
menu_items = [
    ("🏠 Accueil", "Accueil"),
    ("📖 À propos", "À propos"),
    ("📚 Théorie", "Théorie"),
    ("🧪 Exemples", "Exemples"),
    ("🔢 Dijkstra", "Dijkstra"),
    
]

with st.sidebar:
    for label, name in menu_items:
        if st.button(label, key=name):
            st.session_state.page = name

page = st.session_state.page



if page == "Accueil":
    st.title("🎓 Apprendre l'Algorithme de Dijkstra")
    st.write("Une application interactive pour visualiser et comprendre le plus court chemin.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Commencer"):
            st.session_state.page = "Dijkstra"
            st.rerun()

    with col2:
        if st.button("🧪 Voir un exemple"):
            st.session_state.page = "Exemples"
            st.rerun()

    st.subheader("🌟 Pourquoi cette application ?")
    c1, c2, c3 = st.columns(3)
    c1.info("📘 Théorie illustrée:\nComprendre Dijkstra étape par étape.")
    c2.success("🧮 Calcul manuel:\nTester vos propres matrices.")
    c3.warning("💡 Exemples prêts:\nExplorer différents graphes.")

    st.subheader("🛠️ Comment utiliser cette application ?")
    st.write("""
    1. Allez dans **Dijkstra**.  
    2. Entrez le nombre des sommets.
    3. Entrez les **sommets** et la **matrice des poids**. 
    4. Choisissez le **départ** et l’**arrivée**. 
    5. Cliquez sur **Calculer**.
    6. Visualisez les étapes et le graphe.
    """)





elif page == "À propos":
    st.title("👩‍💻 À propos")

    st.subheader("📌 Présentation")
    st.write("""
    Cette application a été réalisée dans le cadre d’un projet académique
    visant à comprendre, appliquer et visualiser l’algorithme de Dijkstra.
    """)

    st.subheader("🎯 Objectif du projet")
    st.write("""
    L’objectif principal de cette application est de permettre à l’utilisateur :
    - de saisir un graphe pondéré sous forme de matrice de poids,
    - d’appliquer l’algorithme de Dijkstra,
    - de calculer les plus courts chemins depuis un sommet source
      vers **tous les autres sommets du graphe**,
    - d’identifier le plus court chemin entre deux sommets choisis,
    - de suivre les différentes étapes de l’algorithme.
    """)

    st.subheader("🧠 Principe")
    st.write("""
    À partir d’un sommet de départ, l’algorithme de Dijkstra calcule
    progressivement la distance minimale vers chaque sommet du graphe,
    en garantissant l’optimalité des chemins lorsque les poids sont positifs.
    """)

    st.subheader("🛠️ Technologies utilisées")
    st.write("""
    - **Python** pour l’implémentation de l’algorithme  
    - **Streamlit** pour l’interface utilisateur  
    - **Pandas** pour la manipulation des matrices  
    - **NetworkX** et **Matplotlib** pour la visualisation des graphes
    """)

    st.subheader("🎓 Contexte académique")
    st.write("""
    Ce projet a été réalisé dans le cadre du module **Optimisation Mathématique**
    à l’ENSA Oujda.
    """)

    
    


elif page == "Théorie":
    st.title("📚 Théorie : Algorithme de Dijkstra–Moore")

    st.subheader("🎯 Objectif")
    st.write("""
    L’algorithme de **Dijkstra–Moore** permet de calculer les **plus courts chemins**
    à partir d’un sommet source **s** vers tous les autres sommets d’un graphe pondéré,
    à condition que les poids soient **positifs ou nuls**.
    """)

    st.subheader("📌 Notations")
    st.write("""
    - **G = (X, U)** : graphe orienté  
    - **X = {1, 2, ..., n}** : ensemble des sommets  
    - **U** : ensemble des arcs  
    - **l(i, j)** : longueur (poids) de l’arc (i, j)  
    - **s** : sommet de départ  
    - **D(i)** : longueur du plus court chemin de *s* vers *i*  
    """)

    st.subheader("🧠 Principe")
    st.write("""
    L’algorithme construit progressivement l’ensemble **Y** des sommets
    dont la distance minimale depuis le sommet source est définitivement connue.
    """)

    st.subheader("🔁 Étapes de l’algorithme")

    st.markdown("""
    ### (a) Initialisation
    - **Y = {s}**  
    - **Ȳ = X \\ Y**  
    - **D(s) = 0**

    Pour tout sommet *i* appartenant à *X* :
    - **D(i) = l(s, i)** si *i* est un successeur de *s*
    - **D(i) = +∞** sinon
    """)

    st.markdown("""
    ### (b) Sélection
    Choisir un sommet **j ∈ Ȳ** tel que :
    
    **D(j) = min{ D(i) | i ∈ Ȳ }**

    Ajouter **j** à l’ensemble **Y** :
    - **Y ← Y ∪ {j}**
    - **Ȳ ← Ȳ \\ {j}**

    Si **Ȳ = ∅**, l’algorithme s’arrête.
    """)

    st.markdown("""
    ### (c) Mise à jour (relaxation)
    Pour tout sommet **i ∈ Γ⁺(j) ∩ Ȳ** :

    **D(i) ← min( D(i), D(j) + l(j, i) )**

    Puis retourner à l’étape **(b)**.
    """)

    st.subheader("⚠️ Condition d’application")
    st.warning("""
    L’algorithme de Dijkstra–Moore ne fonctionne que si tous les poids des arcs
    sont **positifs ou nuls**.
    """)
    

    st.divider()
    st.subheader("📊 Exemple de graphe")

   
    G = nx.Graph()
    G.add_edge("A", "B", weight=2)
    G.add_edge("A", "C", weight=5)
    G.add_edge("B", "C", weight=1)
    G.add_edge("B", "D", weight=4)
    G.add_edge("C", "D", weight=3)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=900)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    st.pyplot(plt)

    


elif page == "Exemples":
    st.title("🧪 Exemple illustré — Algorithme de Dijkstra (pas à pas)")

    
    def format_distances(D):
        result = {}
        for k, v in D.items():
            if v == math.inf:
                result[k] = "+∞"
            else:
                result[k] = int(v) if v == int(v) else v
        return result

  
    X = ["A", "B", "C", "D"]
    s = "A"  
    t = "D"   

    st.subheader("🎯 Paramètres")
    st.write(f"Sommet de départ s = **{s}**")
    st.write(f"Sommet d’arrivée = **{t}**")

   
    st.subheader("🔗 Graphe de départ")

    edges = [
        ("A", "B", 1),
        ("A", "C", 4),
        ("B", "C", 2),
        ("B", "D", 5),
        ("C", "D", 1),
    ]

    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(5, 4))
    nx.draw(G, pos, with_labels=True, node_size=900)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=nx.get_edge_attributes(G, "weight")
    )
    st.pyplot(plt)

   
    st.subheader("📊 Matrice des poids l(i, j)")

    matrix = pd.DataFrame(0, index=X, columns=X)
    for u, v, w in edges:
        matrix.loc[u, v] = w

    st.table(matrix)

    st.subheader("(a) Initialisation")

    Y = {s}
    Y_bar = set(X) - Y

    D = {}
    parent = {}

    for i in X:
        if matrix.loc[s, i] > 0:
            D[i] = matrix.loc[s, i]
            parent[i] = s
        else:
            D[i] = math.inf
            parent[i] = None

    D[s] = 0

    st.write(f"Y = {Y}")
    st.write(f"Ȳ = {Y_bar}")
    st.write("Distances initiales D(i) :")

    st.table(
        pd.DataFrame.from_dict(
            format_distances(D),
            orient="index",
            columns=["D(i)"]
        )
    )

   
    step = 1
    while Y_bar:
        st.subheader(f"(b) Sélection — Étape {step}")

        j = min(Y_bar, key=lambda i: D[i])
        st.write(f"Sommet sélectionné j = **{j}** (min D(i))")

        Y.add(j)
        Y_bar.remove(j)

        st.write(f"Y = {Y}")
        st.write(f"Ȳ = {Y_bar}")

        st.subheader("(c) Mise à jour (relaxation)")

        for i in Y_bar:
            if matrix.loc[j, i] > 0:
                nouvelle_distance = D[j] + matrix.loc[j, i]
                if nouvelle_distance < D[i]:
                    D[i] = nouvelle_distance
                    parent[i] = j

        st.write("Distances après mise à jour :")

        st.table(
            pd.DataFrame.from_dict(
                format_distances(D),
                orient="index",
                columns=["D(i)"]
            )
        )

        step += 1

    
    st.subheader("✅ Résultat final")

    chemin = []
    cur = t
    while cur:
        chemin.append(cur)
        cur = parent[cur]
    chemin.reverse()

    st.success(
        f"Plus court chemin de **{s}** vers **{t}** : "
        f"{' → '.join(chemin)} | Distance = {format_distances(D)[t]}"
    )









elif page == "Dijkstra":

    st.title("🔢 Dijkstra")

    n = st.number_input("Nombre de sommets", min_value=2, max_value=15, value=4)
    nodes = [chr(ord('A') + i) for i in range(n)]
    st.write("Sommets :", nodes)

    matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    edited = st.data_editor(matrix, use_container_width=True)

    col1, col2 = st.columns(2)
    start = col1.selectbox("Sommet de départ", nodes)

    mode = st.radio(
        "Mode de calcul",
        ("Un seul sommet d’arrivée", "Tous les sommets / plusieurs sommets")
    )

    if mode == "Un seul sommet d’arrivée":
        end = col2.selectbox("Sommet d’arrivée", nodes)
    else:
        ends = col2.multiselect(
            "Sommets d’arrivée",
            nodes,
            default=[n for n in nodes if n != start]
        )

    if st.button("Calculer le plus court chemin"):

        graph = {}
        for i in nodes:
            graph[i] = {}
            for j in nodes:
                w = edited.at[i, j]
                if i != j and w > 0:
                    graph[i][j] = w

       
        def dijkstra(g, start):
            D = {node: math.inf for node in g}
            parent = {node: None for node in g}
            D[start] = 0
            visited = set()

            while len(visited) < len(g):
                u = min((n for n in g if n not in visited), key=lambda x: D[x])
                visited.add(u)
                for v, w in g[u].items():
                    if D[u] + w < D[v]:
                        D[v] = D[u] + w
                        parent[v] = u
            return D, parent

        distances, parent = dijkstra(graph, start)

       
        st.subheader("📍 Résultats")

        if mode == "Un seul sommet d’arrivée":
            path = []
            cur = end
            while cur:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            st.success(f"{start} → {end} : {' → '.join(path)} | Distance = {distances[end]}")

       
        st.subheader("📈 Visualisation du graphe")

        G = nx.DiGraph()
        for u in graph:
            for v, w in graph[u].items():
                G.add_edge(u, v, weight=w)

        st.write("Arêtes du graphe :", list(G.edges(data=True)))

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(6, 5))
        nx.draw(G, pos, with_labels=True, node_size=900)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=nx.get_edge_attributes(G, 'weight')
        )

        st.pyplot(plt)
