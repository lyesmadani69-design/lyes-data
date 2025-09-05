# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import json
from typing import Optional
import fitz  # PyMuPDF

# ===================== CONFIG APP =====================
st.set_page_config(page_title="Portfolio Data Analyst", page_icon="📊", layout="wide")

# statsmodels (optionnel)
try:
    from statsmodels.tsa.holtwinters import Holt
except Exception:
    Holt = None  # évite un crash si non installé

# Dossiers ancrés sur le script
BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"
st.sidebar.write("📂 Contenu du dossier assets :")
st.sidebar.write([p.name for p in ASSETS.glob("*")])
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

for d in (ASSETS, DATA_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# petit état utile pour invalider du cache d'images si besoin
if "assets_version" not in st.session_state:
    st.session_state["assets_version"] = 0


# ===================== HELPERS =====================
def resolve_asset(name: str) -> Path | None:
    """
    Résout un nom/chemin de fichier vers le dossier assets/.
    Accepte:
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

    # Chemin absolu
    if p.is_absolute():
        return p if p.exists() else None

    # Si "assets/xxx", garde juste le nom
    parts = p.parts
    if len(parts) >= 2 and parts[0].lower() == "assets":
        p = Path(parts[-1])

    cand = ASSETS / p.name
    return cand if cand.exists() else None


def show_image_safe(img: str) -> None:
    """Affiche une image depuis URL ou fichier local (sécurisé)."""
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
    """Bouton de téléchargement pour un PDF local (ou lien si URL)."""
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
    """Sauvegarde un fichier uploadé sans écraser (suffixe _1, _2, ...)."""
    safe_name = Path(uploaded_file.name).name
    dest = dirpath / safe_name
    i = 1
    while dest.exists():
        dest = dirpath / f"{dest.stem}_{i}{dest.suffix}"
        i += 1
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## À propos")
    portrait = ASSETS / "dash.png"  # remplace par ton fichier si dispo
    if portrait.exists():
        st.image(str(portrait), caption="Lyes", use_container_width=True)
    else:
        st.info("Place une image '' dans assets/ pour l'afficher ici.")

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

    # Debug chemins (optionnel)
    if st.checkbox("🔧 Debug chemins", value=False):
        st.write("BASE_DIR:", str(BASE_DIR))
        st.write("ASSETS:", str(ASSETS))
        st.write("UPLOADS_DIR:", str(UPLOADS_DIR))
        try:
            st.write("Contenu assets:", [p.name for p in ASSETS.glob("*")])
        except Exception:
            pass


# ===================== HEADER =====================
st.title("Portfolio — Data Analyst")
st.caption(f"Dossier des données persistantes : {UPLOADS_DIR}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Années d'expérience", "2")
c2.metric("Projets publiés", "8")
c3.metric("Outils", "Python • SQL • Power BI")
c4.metric("Disponibilité", "Sept. 2025")


# ===================== TABS =====================
tab_home, tab_projects, tab_skills, tab_contact, tab_smooth, tab_gallery = st.tabs(
    ["Accueil", "Projets", "Compétences", "Contact", "Lissage exp.", "Galerie"]
)


# ===================== HOME =====================
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
    st.header("👋 Data Analyst — orientation business")
    st.write(
        """
Je transforme des données brutes en recommandations actionnables.
Secteurs : retail / e-commerce. Outils : **Python (pandas, statsmodels)**, **SQL**, **Excel/Power BI**.
Je privilégie des livrables clairs : **tableaux de bord**, **rapports courts**, et **métriques** qui permettent de décider.
"""
    )
    st.markdown("**Ce que je fais bien :** Prévision courte/moyenne période • Analyses Pareto • Segmentation clients (RFM) • A/B tests.")


# ===================== PROJECTS =====================
PROJECTS = [
    {
        "title": "Analyse Pareto 20/80",
        "summary": "Analyse des ventes : distribution 20/80, identification des catégories majeures et recommandations d’assortiment.",
        "skills": ["Python", "Pandas", "Matplotlib"],
        "image": "kpi-four.png",           # dans assets/ si dispo
        "report": "pareto.pdf",           # dans assets/ si dispo
        "extras": ["assets-pareto.png"],  # images additionnelles
        "repo": "",
        "demo": "pareto.pdf"
    },
    {
        "title": "",
        "summary": "Tendance, saisonnalité, détection des pics.",
        "skills": ["Python", "Pandas"],
        "image": "dash.png",
        "report": "ventes.mensuel.pdf",
        "extras": [],
        "repo": "",
        "demo": ""
    }
]

def _resolve(name: str) -> Path | None:
    """Résout un nom/chemin vers un Path existant, priorise assets/."""
    if not name:
        return None
    s = str(name).strip()
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p
    cand = ASSETS / Path(s).name
    if cand.exists():
        return cand
    rel = BASE_DIR / s
    return rel if rel.exists() else None

with tab_projects:
    st.subheader("📦 Projets")
    if not PROJECTS:
        st.info("Aucun projet listé.")
    else:
        for p in PROJECTS:
            with st.container(border=True):
                cols = st.columns([1, 2])
                with cols[0]:
                    main_img = _resolve(p.get("image", ""))
                    if main_img:
                        st.image(str(main_img), use_container_width=True, caption=p.get("title", ""))
                    else:
                        st.warning(f"Image introuvable : {p.get('image','')} (cherché dans {ASSETS})")
                    for extra in p.get("extras", []):
                        extra_path = _resolve(extra)
                        if extra_path:
                            st.image(str(extra_path), use_container_width=True, caption=Path(extra).name)
                        else:
                            st.caption(f"Image additionnelle introuvable : {extra}")
                with cols[1]:
                    st.subheader(p.get("title", "Projet"))
                    st.markdown(p.get("summary", ""))
                    skills = p.get("skills", [])
                    if skills:
                        st.caption("Compétences : " + ", ".join(skills))
                    btn_cols = st.columns(3)
                    with btn_cols[0]:
                        if p.get("repo"):
                            st.link_button("Code", p["repo"], use_container_width=True)
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
                    with btn_cols[2]:
                        if p.get("demo"):
                            st.link_button("Démo", p["demo"], use_container_width=True)


# ===================== SKILLS (placeholder simple) =====================
with tab_skills:
    st.subheader("🧰 Compétences")
    st.write("- Python (pandas, numpy, statsmodels)")
    st.write("- SQL (requêtes analytiques)")
    st.write("- Data viz (Matplotlib/Plotly) • BI (Power BI)")
    st.write("- Time series (Holt, Holt-Winters) • KPI (MAE, RMSE, MAPE)")


# ===================== CONTACT (placeholder) =====================
with tab_contact:
    st.subheader("📮 Contact")
    st.write("Envoyez-moi un message sur LinkedIn ou GitHub (liens dans la sidebar).")
    st.write("Possibilité d'intervenir en mission freelance, sur devis.")


# ===================== TAB SMOOTH (Holt) =====================
with tab_smooth:
    st.subheader("📈 Lissage exponentiel double (méthode de Holt)")
    st.caption(f"Dossier des données persistantes : {UPLOADS_DIR.resolve()}")

    # Upload
    up = st.file_uploader(
        "Charger un CSV ou Excel",
        type=["csv", "xlsx"],
        key="smooth_up",
        help="Charge un fichier .csv ou .xlsx ; il sera copié dans le dossier de persistance."
    )
    load_path = None
    if up is not None:
        dest = save_uploaded_file(UPLOADS_DIR, up)
        st.success(f"Fichier enregistré : {dest.name}")
        st.session_state["last_data_file"] = str(dest)

    # Fichiers existants
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
        selected_name = st.selectbox(
            "Ou choisir un fichier enregistré",
            saved_names,
            index=default_idx,
            key="smooth_pick_saved",
        )
        load_path = UPLOADS_DIR / selected_name
        st.caption(f"✅ Fichier sélectionné : {load_path.name}")
    else:
        st.info("Aucun fichier enregistré pour l’instant. Charge un CSV/Excel ci-dessus ou utilise l’exemple.")

    # Exemple
    use_example = st.checkbox(
        "Utiliser un jeu d'exemple",
        value=(load_path is None and up is None),
        key="smooth_use_example",
    )

    # ===== Lecture dataset -> df (2 if séparés, pas d'elif) =====
    df = None

    # 1) Lecture du fichier sélectionné
    if (load_path is not None) and load_path.exists():
        try:
            if load_path.suffix.lower() == ".csv":
                df = pd.read_csv(load_path)
            else:
                df = pd.read_excel(load_path)  # nécessite openpyxl
        except Exception as e:
            st.error(f"Impossible de lire {load_path.name} : {e}")
            df = None

    # 2) Exemple si df encore vide
    if (df is None) and use_example:
        dates = pd.date_range("2022-01-01", periods=36, freq="MS")
        values = 200 + np.arange(36) * 3 + np.sin(np.arange(36) / 6) * 10 + np.random.normal(0, 5, 36)
        df = pd.DataFrame({"date": dates, "valeur": values})
        st.caption("📦 Exemple 36 mois. Charge un fichier pour utiliser tes données.")

    # Info si toujours rien
    if df is None and not use_example:
        st.info("Charge un fichier ou coche 'Utiliser un jeu d'exemple'.")

    # ===== Garde =====
    if df is None:
        pass
    else:
        # Sélection colonnes
        cols = list(df.columns)
        default_date_col = "date" if "date" in cols else cols[0]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        default_val_col = "valeur" if "valeur" in cols else (numeric_cols[0] if numeric_cols else cols[-1])

        st.markdown("### ⚙️ Paramètres & colonnes")
        c1, c2 = st.columns(2)
        with c1:
            col_date = st.selectbox("Colonne date / temps", cols, index=cols.index(default_date_col))
            parse_date = st.checkbox("Convertir en datetime (to_datetime)", value=True)
        with c2:
            col_val = st.selectbox("Colonne valeur", cols, index=cols.index(default_val_col))

        # Préparation série
        s_df = df[[col_date, col_val]].copy()
        if parse_date:
            s_df[col_date] = pd.to_datetime(s_df[col_date], errors="coerce")
        s_df = s_df.dropna(subset=[col_date, col_val]).sort_values(col_date).reset_index(drop=True)

        # Fréquence (pour horizon futur)
        try:
            inferred = pd.infer_freq(s_df[col_date])
        except Exception:
            inferred = None
        freq_choices = ["D", "W", "MS", "M", "Q", "Y"]
        default_freq_idx = freq_choices.index("MS")
        st.caption(f"Fréquence inférée : **{inferred}**" if inferred else "Fréquence non inférée.")
        freq = st.selectbox(
            "Fréquence pour la prévision future",
            freq_choices,
            index=(freq_choices.index(inferred) if inferred in freq_choices else default_freq_idx),
            help="Utilisée pour générer les dates futures (prévisions)."
        )

        st.markdown("### 🔧 Holt (double lissage)")
        c3, c4, c5 = st.columns(3)
        with c3:
            optimized = st.toggle("Optimiser automatiquement", value=True, help="Si activé, ignore alpha/beta.")
        with c4:
            alpha = st.slider("alpha (niveau)", 0.01, 0.99, 0.50, 0.01)
        with c5:
            beta = st.slider("beta (tendance)", 0.01, 0.99, 0.10, 0.01)

        c6, c7 = st.columns(2)
        with c6:
            damped = st.checkbox("Tendance amortie (damped_trend)", value=False)
        with c7:
            horizon = st.number_input("Horizon de prévision (pas de temps)", min_value=1, max_value=120, value=12, step=1)

        st.markdown("### 👀 Aperçu des données")
        st.dataframe(s_df.head(10))

        if Holt is None:
            st.warning("Le module Holt (statsmodels) n'est pas disponible. Installe-le : `pip install statsmodels`")
        else:
            if st.button("🚀 Appliquer Holt et tracer"):
                try:
                    y = pd.Series(
                        s_df[col_val].astype(float).values,
                        index=pd.Index(s_df[col_date].values, name=col_date)
                    )

                    model = Holt(y, exponential=False, damped_trend=damped)
                    if optimized:
                        fit = model.fit(optimized=True)
                    else:
                        fit = model.fit(smoothing_level=float(alpha), smoothing_slope=float(beta), optimized=False)

                    # Prévisions futures
                    last_dt = s_df[col_date].iloc[-1]
                    future_index = pd.date_range(start=last_dt, periods=horizon + 1, freq=freq)[1:]
                    fcst = fit.forecast(horizon)
                    try:
                        fcst.index = future_index
                    except Exception:
                        fcst = pd.Series(fcst.values, index=future_index, name="forecast")

                    observed = y.rename("observé")
                    fitted = fit.fittedvalues.rename("lissé")
                    forecast = pd.Series(fcst.values, index=fcst.index, name="prévision")

                    df_plot = pd.concat([observed, fitted], axis=1)
                    df_plot = pd.concat([df_plot, forecast], axis=1)

                    st.markdown("### 📊 Série, lissé et prévisions")
                    st.line_chart(df_plot)

                    # Métriques
                    sse = getattr(fit, "sse", np.nan)
                    aic = getattr(fit, "aic", np.nan)
                    bic = getattr(fit, "bic", np.nan)
                    llf = getattr(fit, "llf", np.nan)

                    c8, c9, c10, c11 = st.columns(4)
                    c8.metric("SSE", f"{sse:,.2f}")
                    c9.metric("AIC", f"{aic:,.2f}" if pd.notna(aic) else "n/a")
                    c10.metric("BIC", f"{bic:,.2f}" if pd.notna(bic) else "n/a")
                    c11.metric("Log-Likelihood", f"{llf:,.2f}" if pd.notna(llf) else "n/a")

                    # Export CSV
                    exp = pd.DataFrame({
                        col_date: list(observed.index) + list(forecast.index),
                        "observé": list(observed.values) + [np.nan] * len(forecast),
                        "lissé": list(fitted.reindex(observed.index).values) + [np.nan] * len(forecast),
                        "prévision": [np.nan] * len(observed) + list(forecast.values),
                    })
                    st.download_button(
                        "💾 Télécharger (observé/lissé/prévision .csv)",
                        exp.to_csv(index=False).encode("utf-8"),
                        file_name="holt_resultats.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Erreur pendant l'entraînement Holt ou le tracé : {e}")

# --- fin de l'onglet Lissage exp. ---

st.subheader("📸 Schéma du double lissage")
img_path = ASSETS / "doubleliss.png"
if img_path.exists():
    st.image(str(img_path), caption="Méthode du double lissage", use_container_width=True)
else:
    st.warning("Image 'doubleliss.png' manquante dans assets/")


# ===================== GALLERY =====================
with tab_gallery:
    st.subheader("🖼️ Galerie")

    # Upload d'images
    img_up = st.file_uploader(
        "Ajouter une image (png/jpg/jpeg/webp/gif)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="img_gallery",
    )
    if img_up is not None:
        dest = save_uploaded_file(ASSETS, img_up)
        st.success(f"Ajouté : {dest.name}")
        st.image(str(dest), use_container_width=True)

    # Grille d'images
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    images = sorted([p for p in ASSETS.glob("*") if p.suffix.lower() in exts])
    if not images:
        st.info("Aucune image trouvée. Place des fichiers dans assets/ ou utilise l’upload ci-dessus.")
    else:
        cols = st.columns(3)
        for i, p in enumerate(images):
            with cols[i % 3]:
                st.image(str(p), caption=p.name, use_container_width=True)

    st.write("---")

    # Image spécifique : grille Holt + interprétation (optionnel)
    st.subheader("📸 Grille de recherche Holt (α–β)")
    grid_path = ASSETS / "opti-coeff.png"   # place ce fichier dans assets/
    if grid_path.exists():
        st.image(str(grid_path), use_container_width=True,
                 caption="Erreur par couple (α, β) — plus bas = mieux")
        st.markdown(r"""
**Interprétation (opti par RMSE).**
Évaluation d'une grille de couples \((\alpha,\beta)\) avec calcul de la **RMSE** pour chacun.
La **RMSE la plus faible** est obtenue pour un couple \(\alpha\) élevé et \(\beta\) faible : niveau réactif, tendance lissée.
*Valider sur un jeu de test pour confirmer la performance hors-échantillon.*
""")
    else:
        st.info(f"Place l'image 'opti-coeff.png' dans {ASSETS} pour l'afficher ici.")

    st.write("---")

    # PDF (upload + téléchargement + aperçu)
    # --- PDF (upload + téléchargement + aperçu robuste) ---
import io

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
    st.info("Aucun PDF trouvé. Dépose des fichiers ici ou place-les dans assets/.")
else:
    for p in pdfs:
        with st.container(border=True):
            st.write(f"**{p.name}**")

            # Bouton télécharger (100% fiable)
            st.download_button(
                "💾 Télécharger",
                data=p.read_bytes(),
                file_name=p.name,
                mime="application/pdf",
                key=f"dl_{p.name}",
                use_container_width=True
            )

            # Aperçu image (1ʳᵉ page) pour contourner les blocages iframe
            show_preview = st.toggle("🔍 Aperçu", key=f"prev_{p.name}")
            if show_preview:
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(p)  # peut ouvrir via Path
                    page = doc.load_page(0)  # première page
                    pix = page.get_pixmap(dpi=144)  # résolution correcte
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, use_container_width=True, caption=f"Prévisualisation — {p.name}")
                    doc.close()
                except Exception as e:
                    # Fallback: tenter l'iframe base64 (peut être bloqué par le navigateur)
                    st.info("Prévisualisation image indisponible, tentative en iframe (peut être bloquée par votre navigateur).")
                    import base64
                    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600"></iframe>',
                        unsafe_allow_html=True
                    )
