# ===================== EN-TÊTE / IMPORTS & CHEMINS =====================
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import json

# (optionnel si tu utilises Holt) :
try:
    from statsmodels.tsa.holtwinters import Holt
except Exception:
    Holt = None  # évite un crash si non utilisé

# Chemins ancrés sur le dossier du script
BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

# Création des dossiers persistants
for d in (ASSETS, DATA_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Config de la page Streamlit
st.set_page_config(page_title="Portfolio Data Analyst", page_icon="📊", layout="wide")

# ===================== HELPERS (chemins, images, pdf, upload) =====================
def resolve_asset(name: str) -> Path | None:
    """
    Résout un nom/chemin de fichier vers le dossier assets/.
    Accepte :
      - "pareto.png"
      - "assets/pareto.png"
      - chemin absolu
    Corrige aussi l'erreur fréquente "assets-xxx.png" -> "xxx.png".
    """
    if not name:
        return None
    s = str(name).strip()

    # Auto-fix erreurs fréquentes
    if s.startswith("assets-"):
        s = s.replace("assets-", "", 1)

    p = Path(s)

    # Chemin absolu directement
    if p.is_absolute():
        return p if p.exists() else None

    # Si on t'a mis "assets/xxx", garde juste le nom
    parts = p.parts
    if len(parts) >= 2 and parts[0].lower() == "assets":
        p = Path(parts[-1])  # garde le dernier segment (nom du fichier)

    cand = ASSETS / p.name
    return cand if cand.exists() else None


def show_image_safe(img: str) -> None:
    """
    Affiche une image depuis URL ou fichier local (résolu via resolve_asset).
    Évite les crashs si le fichier est manquant.
    """
    if not img:
        return
    s = str(img)
    if s.startswith(("http://", "https://")):
        st.image(s, use_container_width=True)
        return

    p = resolve_asset(s) or (BASE_DIR / s if (BASE_DIR / s).exists() else None)
    if p and p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning(f"Image introuvable : {img} (cherché dans {ASSETS})")


def pdf_download_button(label: str, report: str, key: str | None = None) -> None:
    """
    Crée un bouton de téléchargement pour un PDF local (ou un lien si URL).
    """
    if not report:
        return
    s = str(report)
    if s.startswith(("http://", "https://")):
        st.link_button(label, s, use_container_width=True)
        return

    p = resolve_asset(s) or (BASE_DIR / s if (BASE_DIR / s).exists() else None)
    if p and p.exists():
        st.download_button(
            label,
            data=p.read_bytes(),
            file_name=p.name,
            mime="application/pdf",
            use_container_width=True,
            key=key,
        )
    else:
        st.caption(f"PDF introuvable : {report}")


def save_uploaded_file(dirpath: Path, uploaded_file) -> Path:
    """
    Sauvegarde un fichier uploadé dans dirpath en évitant d'écraser (suffixe _1, _2, ...).
    Retourne le chemin final.
    """
    safe_name = Path(uploaded_file.name).name
    dest = dirpath / safe_name
    i = 1
    while dest.exists():
        dest = dirpath / f"{dest.stem}_{i}{dest.suffix}"
        i += 1
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


# ===================== (OPTION) DEBUG RAPIDE DANS LA SIDEBAR =====================
if st.sidebar.checkbox("🔧 Debug chemins", value=False):
    st.sidebar.write("BASE_DIR:", str(BASE_DIR))
    st.sidebar.write("ASSETS:", str(ASSETS))
    st.sidebar.write("UPLOADS_DIR:", str(UPLOADS_DIR))
    try:
        st.sidebar.write("Contenu assets:", [p.name for p in ASSETS.glob("*")])
    except Exception:
        pass

# ===================== (OPTION) PLACEHOLDER DES PROJETS =====================
# Tu peux garder vide ou mettre tes projets ici.
PROJECTS: list[dict] = []


# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(page_title="Portfolio Data Analyst", page_icon="📊", layout="wide")

# ------------------------------------------------------------
# Dossiers (persistants) utilisés par l'app
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"   # CSV / Excel
ASSETS = BASE_DIR / "assets"          # Images, CV, etc.
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Sidebar (profil + liens)
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## À propos")
    portrait = ASSETS / "portrait.jpg"  # remplace par ton fichier
    if portrait.exists():
        st.image(str(portrait), caption="Ton nom", use_container_width=True)
    else:
        st.info("Ajoute une photo dans assets/portrait.jpg")

    st.write("Je suis Data Analyst spécialisé en data cleaning, data viz et time series.")
    st.write("Basé à Lyon — Disponible en freelance.")

    st.link_button("GitHub", "https://github.com/TON_COMPTE", use_container_width=True)
    st.link_button("LinkedIn", "https://www.linkedin.com/in/TON_PROFIL", use_container_width=True)

    cv_path = ASSETS / "CV.pdf"
    if cv_path.exists():
        with open(cv_path, "rb") as f:
            st.download_button("📄 Télécharger mon CV", f, file_name="CV_TonNom.pdf", use_container_width=True)
    else:
        st.caption("Place ton CV dans assets/CV.pdf pour activer le bouton.")

# ------------------------------------------------------------
# En-tête
# ------------------------------------------------------------
st.title("Portfolio — Data Analyst")
st.caption(f"Dossier des données persistantes : {UPLOADS_DIR}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Années d'expérience", "2")
c2.metric("Projets publiés", "8")
c3.metric("Outils", "Python • SQL • Power BI")
c4.metric("Disponibilité", "Sept. 2025")

# ------------------------------------------------------------
# Définition des onglets
# ------------------------------------------------------------
tab_home, tab_projects, tab_skills, tab_contact, tab_smooth, tab_gallery = st.tabs(
    ["Accueil", "Projets", "Compétences", "Contact", "Lissage exp.", "Galerie"]
)

# ------------------------------------------------------------
# Données des projets (exemples — à personnaliser)
# ------------------------------------------------------------
PROJECTS = []
    
        

# Données des projets (exemples — à personnaliser)
# --------------------------------------------


# ------------------------------------------------------------
# Données des projets (exemples — à personnaliser)
# ------------------------------------------------------------
PROJECTS = [
    {
        "title": "Analyse Pareto 20/80",
        "summary": "Segmentation catégories, KPI, recommandations.",
        "skills": ["Python", "Pandas", "Matplotlib"],
        "image": "assets-pareto.png",
        "report": "assets-pareto.pdf",
        "repo": "",
        "demo": ""
    },
    {
        "title": "Ventes mensuelles",
        "summary": "Tendance, saisonnalité, détection des pics.",
        "skills": ["Python", "Pandas"],
        "image": "dash.png",
        "report": "ventes.mensuel.pdf",
        "repo": "",
        "demo": ""
    }
]

# ------------------------------------------------------------
# Onglet Accueil
# ------------------------------------------------------------
with tab_home:
    st.subheader("Bienvenue ✨")
    st.markdown(
        """
        Ce portfolio présente une sélection de projets de data analyse :
        - Data cleaning & feature engineering
        - Data viz (Matplotlib/Plotly) et BI (Power BI)
        - Séries temporelles (Holt, Holt-Winters) et indicateurs (MAE, RMSE, MAPE)
        """
    )

# ------------------------------------------------------------
# Onglet Projets
# ------------------------------------------------------------
    # --- Onglet PROJETS (bloc complet) ---
from pathlib import Path

# Chemins locaux (au cas où non définis plus haut)
BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"

def _resolve(name: str) -> Path | None:
    """Résout un nom/chemin vers un Path existant, en priorisant le dossier assets/."""
    if not name:
        return None
    s = str(name).strip()
    p = Path(s)
    # 1) si absolu et existe
    if p.is_absolute() and p.exists():
        return p
    # 2) chercher par nom de fichier dans assets/
    cand = ASSETS / Path(s).name
    if cand.exists():
        return cand
    # 3) tenter relatif au script
    rel = BASE_DIR / s
    if rel.exists():
        return rel
    return None

# Données du projet (adapter les textes si besoin)
PROJECTS = [
    {
        "title": "Analyse Pareto 20/80",
        "summary": "Analyse des ventes : distribution 20/80, identification des catégories majeures et recommandations d’assortiment.",
        "skills": ["Python", "Pandas", "Matplotlib"],
        "image": "capture.png",            # image principale (ex: capture d’écran)
        "report": "pareto.pdf",            # rapport PDF
        "extras": ["assets-pareto.png"],   # images additionnelles à afficher (ex: visuel Pareto)
        "repo": "",
        "demo": ""
    }
]

with tab_projects:
    import streamlit as st

    st.subheader("📦 Projets")

    if not PROJECTS:
        st.info("Aucun projet listé.")
    else:
        for p in PROJECTS:
            with st.container(border=True):
                cols = st.columns([1, 2])

                # -------- Colonne image(s) --------
                with cols[0]:
                    # image principale
                    main_img = _resolve(p.get("image", ""))
                    if main_img:
                        st.image(str(main_img), use_container_width=True, caption=p.get("title", ""))
                    else:
                        st.warning(f"Image introuvable : {p.get('image','')} (cherché dans {ASSETS})")

                    # images supplémentaires
                    for extra in p.get("extras", []):
                        extra_path = _resolve(extra)
                        if extra_path:
                            st.image(str(extra_path), use_container_width=True, caption=Path(extra).name)
                        else:
                            st.caption(f"Image additionnelle introuvable : {extra}")

                # -------- Colonne contenu --------
                with cols[1]:
                    st.subheader(p.get("title", "Projet"))
                    st.markdown(p.get("summary", ""))

                    skills = p.get("skills", [])
                    if skills:
                        st.caption("Compétences : " + ", ".join(skills))

                    btn_cols = st.columns(3)

                    # Lien code (si tu as un repo)
                    with btn_cols[0]:
                        if p.get("repo"):
                            st.link_button("Code", p["repo"], use_container_width=True)

                    # Bouton PDF (rapport)
                    with btn_cols[1]:
                        pdf_path = _resolve(p.get("report", ""))
                        if pdf_path:
                            st.download_button(
                                "📄 Rapport (PDF)",
                                data=pdf_path.read_bytes(),
                                file_name=pdf_path.name,
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"pdf_{p.get('title','')}"
                            )
                        else:
                            st.caption(f"PDF introuvable : {p.get('report','')}")

                    # Démo (si lien)
                    with btn_cols[2]:
                        if p.get("demo"):
                            st.link_button("Démo", p["demo"], use_container_width=True)


        

# ------------------------------------------------------------
# Onglet Compétences
# ------------------------------------------------------------
with tab_skills:
    st.subheader("🛠️ Compétences")
    c1, c2, c3 , c4 = st.columns(4)
    with c1:
        st.markdown("**Data Cleaning**  \nPandas, règles métiers, qualité des données")
    with c2:
        st.markdown("**Data Viz & BI**  \nMatplotlib, Plotly, Power BI")
    with c3:
        st.markdown("**Time Series & ML**  \nHolt, Holt-Winters, régression, diagnostic")
    with c4:
        st.markdown("**Excel , Python , Power Bi desktop")

# ------------------------------------------------------------
# Onglet Contact
# ------------------------------------------------------------
with tab_contact:
    st.subheader("📬 Contact")
    st.markdown("Pour me contacter : **prenom.nom@email.com**")
    st.caption("Astuce : ajoute un bouton mailto si tu veux.")

# ------------------------------------------------------------
# Onglet Lissage exponentiel (persistant)
# ------------------------------------------------------------
with tab_smooth:
    st.subheader("📈 Lissage exponentiel double (méthode de Holt)")
    st.caption(f"Dossier des données persistantes : {UPLOADS_DIR}")

    # --- Upload + enregistrement persistant ---
    up = st.file_uploader("Charger un CSV ou Excel", type=["csv", "xlsx"])
    if up is not None:
        raw_name = Path(up.name).stem
        ext = Path(up.name).suffix.lower()
        dest = UPLOADS_DIR / f"{raw_name}{ext}"
        i = 1
        while dest.exists():
            dest = UPLOADS_DIR / f"{raw_name}_{i}{ext}"
            i += 1
        dest.write_bytes(up.getbuffer())
        st.success(f"Fichier enregistré : {dest.name}")
        st.session_state["last_data_file"] = str(dest)

    # --- Choisir un fichier déjà enregistré ---
    saved_files = sorted(
        list(UPLOADS_DIR.glob("*.csv")) + list(UPLOADS_DIR.glob("*.xlsx")),
        key=lambda p: p.name.lower(),
    )
    saved_names = [p.name for p in saved_files]

    default_idx = 0
    if "last_data_file" in st.session_state:
        last = Path(st.session_state["last_data_file"]).name
        if last in saved_names:
            default_idx = saved_names.index(last)

    if saved_files:
        selected_name = st.selectbox("Ou choisir un fichier enregistré", saved_names, index=default_idx)
        load_path = UPLOADS_DIR / selected_name
        st.caption(f"✅ Fichier sélectionné : {load_path.name}")
    else:
        load_path = None
        st.info("Aucun fichier enregistré pour l’instant. Charge un CSV/Excel ci-dessus ou utilise l’exemple.")

    # --- Exemple si rien n’est dispo ---
    use_example = st.checkbox("Utiliser un jeu d'exemple", value=(load_path is None and up is None))

    # --- Lecture dataset -> df ---
    if load_path is not None and load_path.exists():
        try:
            if load_path.suffix.lower() == ".csv":
                df = pd.read_csv(load_path)
            else:
                df = pd.read_excel(load_path)  # nécessite openpyxl
        except Exception as e:
            st.error(f"Impossible de lire {load_path.name} : {e}")
            st.stop()
    elif use_example:
        dates = pd.date_range("2022-01-01", periods=36, freq="MS")
        values = 200 + np.arange(36) * 3 + np.sin(np.arange(36) / 6) * 10 + np.random.normal(0, 5, 36)
        df = pd.DataFrame({"date": dates, "valeur": values})
        st.caption("📦 Exemple 36 mois. Charge un fichier pour utiliser tes données.")
    else:
        st.stop()

    # --- Contrôle minimal ---
    if df.shape[1] < 2:
        st.error("Ton fichier doit contenir au moins une colonne date et une colonne numérique.")
        st.dataframe(df.head())
        st.stop()

    # ===================== PIPELINE ANALYSE =====================
    st.write("### Sélection des colonnes")
    col_date = st.selectbox("Colonne de date", df.columns.tolist(), index=0)
    numeric_cols = [c for c in df.columns if c != col_date and pd.api.types.is_numeric_dtype(df[c])]
    default_val_idx = df.columns.tolist().index(numeric_cols[0]) if numeric_cols else (1 if df.shape[1] > 1 else 0)
    col_value = st.selectbox("Colonne de valeurs", df.columns.tolist(), index=default_val_idx)

    # Série temporelle propre
    s = df[[col_date, col_value]].copy()
    s[col_date] = pd.to_datetime(s[col_date], errors="coerce")
    s = s.dropna(subset=[col_date]).sort_values(col_date)
    s = s.groupby(col_date, as_index=False)[col_value].sum().set_index(col_date)[col_value]

    st.write("### Fréquence temporelle")
    freq_option = st.selectbox("Fréquence", ["Auto", "Jour", "Semaine", "Mois", "Trimestre", "Année"], index=0)
    freq_map = {"Jour": "D", "Semaine": "W", "Mois": "MS", "Trimestre": "QS", "Année": "YS"}

    if freq_option == "Auto":
        inferred = pd.infer_freq(s.index)
        if inferred is None:
            median_delta = s.index.to_series().diff().median()
            inferred = "D" if (pd.notna(median_delta) and getattr(median_delta, "days", None) == 1) else "MS"
        freq = inferred
    else:
        freq = freq_map[freq_option]

    s = s.asfreq(freq)
    if s.isna().any():
        s = s.interpolate(limit_direction="both")

    st.line_chart(s.rename("Observé"))

    st.write("### Paramètres du modèle")
    c1, c2, c3 = st.columns(3)
    with c1:
        optimize = st.checkbox("Optimiser α et β automatiquement", value=True, help="Statsmodels choisit les meilleurs paramètres.")
    with c2:
        alpha = st.slider("α (niveau)", 0.01, 0.99, 0.5, 0.01, disabled=optimize)
    with c3:
        beta = st.slider("β (tendance)", 0.01, 0.99, 0.2, 0.01, disabled=optimize)

    h = st.slider("Horizon de prévision (pas)", 1, 36, 12, help="Nombre de périodes à prévoir (selon la fréquence)")

    # Ajustement Holt
    try:
        model = Holt(s, initialization_method="estimated")
        if optimize:
            fit = model.fit(optimized=True)
            alpha = float(fit.params.get("smoothing_level", np.nan))
            beta = float(fit.params.get("smoothing_trend", np.nan))
        else:
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)

        fitted = fit.fittedvalues.rename("Ajusté")
        fcst = fit.forecast(h).rename("Prévision")

        # Métriques simples in-sample
        err = (s - fitted).dropna()
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err**2).mean()))
        mape = float((err.abs() / s.loc[err.index]).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100)

        st.success(f"Paramètres: α = {alpha:.3f}, β = {beta:.3f}  |  MAE = {mae:.2f}  RMSE = {rmse:.2f}  MAPE = {mape:.1f}%")

        plot_df = pd.concat([s.rename("Observé"), fitted, fcst], axis=1)
        st.line_chart(plot_df)

        # Export des résultats
        result = pd.concat([s.rename("y"), fitted.rename("y_fitted")], axis=1)
        result = pd.concat([result, fcst.rename("y_forecast")], axis=1)
        st.dataframe(result.tail(20))
        st.download_button(
            "💾 Télécharger les résultats (CSV)",
            result.to_csv(index=True).encode("utf-8"),
            file_name="resultats_lissage_double.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Échec de l'ajustement Holt : {e}")
        st.stop()

# ------------------------------------------------------------
# Onglet Galerie (images persistantes)
# ------------------------------------------------------------
with tab_gallery:
    from pathlib import Path
    import base64

    st.subheader("🖼️ Galerie")

    # --- Upload d'images ---
    img_up = st.file_uploader(
        "Ajouter une image (png/jpg/jpeg/webp/gif)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="img_gallery",
    )
    if img_up is not None:
        dest = ASSETS / img_up.name
        i = 1
        while dest.exists():
            dest = ASSETS / f"{dest.stem}_{i}{dest.suffix}"
            i += 1
        dest.write_bytes(img_up.getbuffer())
        st.success(f"Ajouté : {dest.name}")
        st.image(str(dest), use_container_width=True)

    # --- Grille d'images ---
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    images = sorted([p for p in ASSETS.glob("*") if p.suffix.lower() in exts])
    if not images:
        st.info("Aucune image trouvée. Place des fichiers dans le dossier assets/ ou utilise l’upload ci-dessus.")
    else:
        cols = st.columns(3)
        for i, p in enumerate(images):
            with cols[i % 3]:
                st.image(str(p), caption=p.name, use_container_width=True)

    st.write("---")

    # --- Image spécifique : grille Holt + interprétation ---
    st.subheader("📸 Grille de recherche Holt (α–β)")
    grid_path = ASSETS / "opti-coeff.png"   # place ce fichier dans assets/
    if grid_path.exists():
        st.image(str(grid_path), use_container_width=True,
                 caption="Erreur par couple (α, β) — plus bas = mieux")
        st.markdown("""
**Interprétation (opti par RMSE).**  
Évaluation d'une grille de couples \\((\\alpha,\\beta)\\) avec calcul de la **RMSE** pour chacun.  
La **RMSE la plus faible (≈ 1,38)** est obtenue pour **\\(\\alpha=0{,}9\\)** et **\\(\\beta=0{,}1\\)** : **on retient donc ces coefficients**.  
\\(\\alpha\\) élevé = niveau plus réactif ; \\(\\beta\\) faible = tendance plus lissée.  
*Validation recommandée sur un jeu de test pour confirmer la performance hors-échantillon.*
""")
    else:
        st.info(f"Place l'image de la grille dans {ASSETS} sous le nom 'opti-coeff.png' pour l'afficher ici.")

    st.write("---")

    # --- PDF (upload + aperçu + téléchargement) ---
    st.subheader("📄 Rapports (PDF)")
    pdf_up = st.file_uploader("Ajouter un PDF", type=["pdf"], key="pdf_uploader")
    if pdf_up is not None:
        dest = ASSETS / pdf_up.name
        i = 1
        while dest.exists():
            dest = ASSETS / f"{Path(pdf_up.name).stem}_{i}.pdf"
            i += 1
        dest.write_bytes(pdf_up.getbuffer())
        st.success(f"Ajouté : {dest.name}")

    pdfs = sorted(ASSETS.glob("*.pdf"), key=lambda p: p.name.lower())
    if not pdfs:
        st.info("Aucun PDF trouvé. Dépose des fichiers ici ou place-les dans le dossier assets/.")
    else:
        for p in pdfs:
            with st.container(border=True):
                st.write(f"**{p.name}**")
                st.download_button(
                    "💾 Télécharger",
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="application/pdf",
                    key=f"dl_{p.name}",
                    use_container_width=True
                )
                if st.toggle("🔍 Aperçu", key=f"prev_{p.name}"):
                    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600"></iframe>',
                        unsafe_allow_html=True
                    )

                
