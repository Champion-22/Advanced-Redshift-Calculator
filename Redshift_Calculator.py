# Erforderliche Bibliotheken importieren
import streamlit as st
import numpy as np
from scipy.integrate import quad
import math

# --- Konstanten (unverändert) ---
C_KM_PER_S = 299792.458
KM_PER_MPC = 3.085677581491367e+19
KM_PER_AU = 1.495978707e+8
KM_PER_LY = 9.4607304725808e+12
KM_PER_LS = C_KM_PER_S
GYR_PER_YR = 1e9

# --- Standard Kosmologische Parameter (unverändert) ---
H0_DEFAULT = 67.4
OMEGA_M_DEFAULT = 0.315
OMEGA_LAMBDA_DEFAULT = 0.685

# --- Übersetzungsdaten ---
# Schlüssel sind die englischen Originaltexte (oder Bezeichner)
translations = {
    'DE': {
        "lang_select": "Sprache wählen",
        "input_params": "Eingabeparameter",
        "redshift_z": "Rotverschiebung (z)",
        "cosmo_params": "Kosmologische Parameter",
        "hubble_h0": "Hubble-Konstante (H₀) [km/s/Mpc]",
        "omega_m": "Materiedichte (Ωm)",
        "omega_lambda": "Dunkle Energie (ΩΛ)",
        "flat_universe_warning": "Ωm + ΩΛ ≉ 1. Berechnungen gehen von flachem Universum aus (Ωk=0).",
        "results_for": "Ergebnisse für z = {z:.5f}",
        "error_invalid_input": "Ungültige Eingabe. Bitte Zahlen verwenden.",
        "error_h0_positive": "Hubble-Konstante muss positiv sein.",
        "error_omega_negative": "Omega-Parameter dürfen nicht negativ sein.",
        "warn_blueshift": "Warnung: Rotverschiebung ist negativ (Blueshift). Kosmologische Distanzen sind hier 0 oder nicht direkt anwendbar.",
        "error_dep_scipy": "Abhängigkeit 'scipy' nicht gefunden. Bitte installieren.",
        "error_calc_failed": "Berechnung fehlgeschlagen: {e}",
        "warn_integration_accuracy": "Warnung: Relative Integrationsgenauigkeit möglicherweise nicht erreicht (Fehler: DC={err_dc:.2e}, LT={err_lt:.2e}).",
        "lookback_time": "Rückblickzeit (Lookback Time)",
        "cosmo_distances": "Kosmologische Distanzen",
        "comoving_distance_title": "**Mitbewegte Distanz (Comoving Distance):**",
        "luminosity_distance_title": "**Leuchtkraftdistanz (Luminosity Distance):**",
        "angular_diameter_distance_title": "**Winkeldurchmesserdistanz (Angular Diameter Distance):**",
        "unit_Gyr": "Gyr (Milliarden Jahre)",
        "unit_Mpc": "Mpc",
        "unit_Gly": "Gly (Milliarden Lichtjahre)",
        "unit_km": "km",
        "unit_km_sci": "km (wiss.)",
        "unit_km_full": "km (ausgeschr.)",
        "unit_LJ": "LJ",
        "unit_AE": "AE",
        "unit_Ls": "Ls",
        "calculation_note": "Berechnung basiert auf dem flachen ΛCDM-Modell unter Vernachlässigung der Strahlungsdichte.",
        "donate_text": "Gefällt Ihnen dieser Rechner? Unterstützen Sie die Entwicklung mit einer kleinen Spende!",
        "donate_button": "Spenden via Ko-fi",
        "bug_report": "Fehler gefunden?",
        "bug_report_button": "Problem melden",
        "glossary": "Glossar",
        # Glossarbegriffe werden separat übersetzt
    },
    'EN': {
        "lang_select": "Select Language",
        "input_params": "Input Parameters",
        "redshift_z": "Redshift (z)",
        "cosmo_params": "Cosmological Parameters",
        "hubble_h0": "Hubble Constant (H₀) [km/s/Mpc]",
        "omega_m": "Matter Density (Ωm)",
        "omega_lambda": "Dark Energy Density (ΩΛ)",
        "flat_universe_warning": "Ωm + ΩΛ ≉ 1. Calculations assume a flat universe (Ωk=0).",
        "results_for": "Results for z = {z:.5f}",
        "error_invalid_input": "Invalid input. Please use numbers.",
        "error_h0_positive": "Hubble constant must be positive.",
        "error_omega_negative": "Omega parameters cannot be negative.",
        "warn_blueshift": "Warning: Redshift is negative (Blueshift). Cosmological distances are 0 or not directly applicable here.",
        "error_dep_scipy": "Dependency 'scipy' not found. Please install.",
        "error_calc_failed": "Calculation failed: {e}",
        "warn_integration_accuracy": "Warning: Relative integration accuracy might not be achieved (Error: DC={err_dc:.2e}, LT={err_lt:.2e}).",
        "lookback_time": "Lookback Time",
        "cosmo_distances": "Cosmological Distances",
        "comoving_distance_title": "**Comoving Distance:**",
        "luminosity_distance_title": "**Luminosity Distance:**",
        "angular_diameter_distance_title": "**Angular Diameter Distance:**",
        "unit_Gyr": "Gyr (Billion Years)",
        "unit_Mpc": "Mpc",
        "unit_Gly": "Gly (Billion Lightyears)",
        "unit_km": "km",
        "unit_km_sci": "km (sci.)",
        "unit_km_full": "km (full)",
        "unit_LJ": "ly",
        "unit_AE": "AU",
        "unit_Ls": "Ls",
        "calculation_note": "Calculation based on the flat ΛCDM model, neglecting radiation density.",
        "donate_text": "Like this calculator? Support its development with a small donation!",
        "donate_button": "Donate via Ko-fi",
        "bug_report": "Found a bug?",
        "bug_report_button": "Report Issue",
        "glossary": "Glossary",
    },
    'FR': {
        "lang_select": "Choisir la langue",
        "input_params": "Paramètres d'entrée",
        "redshift_z": "Décalage vers le rouge (z)",
        "cosmo_params": "Paramètres Cosmologiques",
        "hubble_h0": "Constante de Hubble (H₀) [km/s/Mpc]",
        "omega_m": "Densité de matière (Ωm)",
        "omega_lambda": "Densité d'énergie noire (ΩΛ)",
        "flat_universe_warning": "Ωm + ΩΛ ≉ 1. Les calculs supposent un univers plat (Ωk=0).",
        "results_for": "Résultats pour z = {z:.5f}",
        "error_invalid_input": "Entrée invalide. Veuillez utiliser des chiffres.",
        "error_h0_positive": "La constante de Hubble doit être positive.",
        "error_omega_negative": "Les paramètres Omega ne peuvent pas être négatifs.",
        "warn_blueshift": "Avertissement : Décalage vers le rouge négatif (Blueshift). Les distances cosmologiques sont 0 ou non directement applicables ici.",
        "error_dep_scipy": "Dépendance 'scipy' introuvable. Veuillez l'installer.",
        "error_calc_failed": "Le calcul a échoué : {e}",
        "warn_integration_accuracy": "Avertissement : La précision relative de l'intégration pourrait ne pas être atteinte (Erreur : DC={err_dc:.2e}, LT={err_lt:.2e}).",
        "lookback_time": "Temps de regard en arrière",
        "cosmo_distances": "Distances Cosmologiques",
        "comoving_distance_title": "**Distance comobile :**",
        "luminosity_distance_title": "**Distance de luminosité :**",
        "angular_diameter_distance_title": "**Distance de diamètre angulaire :**",
        "unit_Gyr": "Ga (Milliards d'années)",
        "unit_Mpc": "Mpc",
        "unit_Gly": "Gal (Milliards d'années-lumière)",
        "unit_km": "km",
        "unit_km_sci": "km (sci.)",
        "unit_km_full": "km (complet)",
        "unit_LJ": "al",
        "unit_AE": "UA",
        "unit_Ls": "sl",
        "calculation_note": "Calcul basé sur le modèle ΛCDM plat, négligeant la densité de rayonnement.",
        "donate_text": "Vous aimez ce calculateur ? Soutenez son développement avec un petit don !",
        "donate_button": "Faire un don via Ko-fi",
        "bug_report": "Trouvé un bug ?",
        "bug_report_button": "Signaler un problème",
        "glossary": "Glossaire",
    }
}

# --- Glossar Daten ---
glossary_data = {
    'DE': {
        "Rotverschiebung (z)": "Ein Maß dafür, wie stark sich das Licht von entfernten Objekten aufgrund der Expansion des Universums zum roten Ende des Spektrums verschoben hat. Höhere z-Werte bedeuten größere Entfernungen und frühere Zeiten im Universum.",
        "Hubble-Konstante (H₀)": "Die Rate, mit der das Universum heute expandiert, typischerweise angegeben in km/s pro Megaparsec (Mpc). Sie verknüpft die Entfernung eines Objekts mit seiner scheinbaren Rückzugsgeschwindigkeit.",
        "Materiedichte (Ωm)": "Der Anteil der Gesamtenergiedichte des Universums, der auf Materie (sowohl normale baryonische Materie als auch Dunkle Materie) entfällt.",
        "Dunkle Energie (ΩΛ)": "Der Anteil der Gesamtenergiedichte des Universums, der auf Dunkle Energie entfällt, die für die beschleunigte Expansion des Universums verantwortlich gemacht wird.",
        "Mitbewegte Distanz": "Eine Distanzmessung, die den Effekt der Expansion des Universums herausrechnet. Sie repräsentiert die Entfernung zwischen zwei Objekten zu einem bestimmten Zeitpunkt (z.B. heute), wenn sie sich nur aufgrund der Hubble-Expansion bewegen.",
        "Leuchtkraftdistanz": "Eine Distanzmessung, die verwendet wird, um die beobachtete Helligkeit eines Objekts mit seiner tatsächlichen (intrinsischen) Leuchtkraft in Beziehung zu setzen. Sie ist größer als die mitbewegte Distanz bei z > 0.",
        "Winkeldurchmesserdistanz": "Eine Distanzmessung, die verwendet wird, um die beobachtete Winkelgröße eines Objekts mit seiner tatsächlichen physikalischen Größe in Beziehung zu setzen. Interessanterweise nimmt sie ab z ≈ 1.6 wieder zu.",
        "Rückblickzeit": "Die Zeitspanne, die das Licht von einem entfernten Objekt benötigt hat, um uns zu erreichen. Es ist das Alter des Universums, als das Licht ausgesandt wurde, abgezogen vom heutigen Alter des Universums.",
        "Mpc (Megaparsec)": "Eine astronomische Entfernungseinheit, die etwa 3.26 Millionen Lichtjahren entspricht.",
        "Gly/Gyr (Gigalichtjahr/Gigajahr)": "Eine Milliarde (10⁹) Lichtjahre bzw. Jahre.",
    },
    'EN': {
        "Redshift (z)": "A measure of how much the light from distant objects has been stretched towards the red end of the spectrum due to the expansion of the universe. Higher z values mean greater distances and earlier times in the universe.",
        "Hubble Constant (H₀)": "The rate at which the universe is expanding today, typically given in km/s per Megaparsec (Mpc). It relates an object's distance to its apparent recession velocity.",
        "Matter Density (Ωm)": "The fraction of the total energy density of the universe attributed to matter (both normal baryonic matter and dark matter).",
        "Dark Energy Density (ΩΛ)": "The fraction of the total energy density of the universe attributed to dark energy, which is responsible for the accelerated expansion of the universe.",
        "Comoving Distance": "A distance measure that factors out the expansion of the universe. It represents the distance between two objects at a specific time (e.g., today) if they were only moving due to Hubble expansion.",
        "Luminosity Distance": "A distance measure used to relate the observed brightness (flux) of an object to its actual (intrinsic) luminosity. It is larger than the comoving distance for z > 0.",
        "Angular Diameter Distance": "A distance measure used to relate the observed angular size of an object to its actual physical size. Interestingly, it decreases again beyond z ≈ 1.6.",
        "Lookback Time": "The amount of time the light from a distant object has traveled to reach us. It's the age of the universe when the light was emitted subtracted from the age of the universe today.",
        "Mpc (Megaparsec)": "An astronomical unit of distance equal to about 3.26 million light-years.",
        "Gly/Gyr (Gigalightyear/Gigayear)": "One billion (10⁹) light-years or years, respectively.",
    },
    'FR': {
        "Décalage vers le rouge (z)": "Mesure de l'étirement de la lumière des objets distants vers l'extrémité rouge du spectre en raison de l'expansion de l'univers. Des valeurs de z plus élevées signifient des distances plus grandes et des temps plus reculés dans l'univers.",
        "Constante de Hubble (H₀)": "Le taux d'expansion actuel de l'univers, généralement exprimé en km/s par Mégaparsec (Mpc). Elle relie la distance d'un objet à sa vitesse de récession apparente.",
        "Densité de matière (Ωm)": "La fraction de la densité d'énergie totale de l'univers attribuée à la matière (matière baryonique normale et matière noire).",
        "Densité d'énergie noire (ΩΛ)": "La fraction de la densité d'énergie totale de l'univers attribuée à l'énergie noire, responsable de l'expansion accélérée de l'univers.",
        "Distance comobile": "Mesure de distance qui élimine l'effet de l'expansion de l'univers. Elle représente la distance entre deux objets à un moment précis (par exemple, aujourd'hui) s'ils ne se déplaçaient qu'en raison de l'expansion de Hubble.",
        "Distance de luminosité": "Mesure de distance utilisée pour relier la luminosité observée (flux) d'un objet à sa luminosité réelle (intrinsèque). Elle est plus grande que la distance comobile pour z > 0.",
        "Distance de diamètre angulaire": "Mesure de distance utilisée pour relier la taille angulaire observée d'un objet à sa taille physique réelle. Curieusement, elle diminue à nouveau au-delà de z ≈ 1.6.",
        "Temps de regard en arrière": "Le temps que la lumière d'un objet distant a mis pour nous parvenir. C'est l'âge de l'univers au moment où la lumière a été émise, soustrait de l'âge actuel de l'univers.",
        "Mpc (Mégaparsec)": "Unité de distance astronomique équivalant à environ 3,26 millions d'années-lumière.",
        "Gal/Ga (Giga-année-lumière/Giga-année)": "Un milliard (10⁹) d'années-lumière ou d'années, respectivement.",
    }
}

# --- Hilfsfunktionen für Integration & Berechnung (unverändert) ---
def hubble_parameter_inv_integrand(z, omega_m, omega_lambda):
  epsilon = 1e-15
  denominator = np.sqrt(omega_m * (1 + z)**3 + omega_lambda + epsilon)
  if denominator < epsilon: return 0.0
  return 1.0 / denominator

def lookback_time_integrand(z, omega_m, omega_lambda):
  epsilon = 1e-15
  term_in_sqrt = omega_m * (1 + z)**3 + omega_lambda
  term_in_sqrt = max(term_in_sqrt, 0)
  denominator = (1 + z) * np.sqrt(term_in_sqrt + epsilon)
  if math.isclose(z, 0):
      denom_at_zero = np.sqrt(omega_m + omega_lambda + epsilon)
      if denom_at_zero < epsilon: return 0.0
      return 1.0 / denom_at_zero
  if abs(denominator) < epsilon: return 0.0
  return 1.0 / denominator

@st.cache_data # Ergebnisse cachen für gleiche Eingaben
def calculate_lcdm_distances(redshift, h0, omega_m, omega_lambda):
  """Berechnet Distanzen & Zeit, gibt Dict zurück."""
  # Eingabevalidierung
  if not isinstance(redshift, (int, float)) or \
     not isinstance(h0, (int, float)) or \
     not isinstance(omega_m, (int, float)) or \
     not isinstance(omega_lambda, (int, float)):
       return {'error_msg': "error_invalid_input"} # Schlüssel für Übersetzung verwenden

  if redshift < 0:
     return {
        'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0,
        'error_msg': "warn_blueshift" # Schlüssel für Übersetzung
    }
  if math.isclose(redshift, 0):
      return {
        'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0,
        'error_msg': None
    }
  if h0 <= 0:
    return {'error_msg': "error_h0_positive"} # Schlüssel
  if omega_m < 0 or omega_lambda < 0:
      return {'error_msg': "error_omega_negative"} # Schlüssel

  dh = C_KM_PER_S / h0
  try:
    integral_dc, err_dc = quad(hubble_parameter_inv_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
    comoving_distance_mpc = dh * integral_dc
    hubble_time_gyr = 977.8 / h0
    integral_lt, err_lt = quad(lookback_time_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
    lookback_time_gyr = hubble_time_gyr * integral_lt
    luminosity_distance_mpc = comoving_distance_mpc * (1 + redshift)
    angular_diameter_distance_mpc = comoving_distance_mpc / (1 + redshift)

    warning_msg_key = None
    warning_msg_args = {}
    integration_warning_threshold = 1e-5
    if err_dc > integration_warning_threshold or err_lt > integration_warning_threshold:
       warning_msg_key = "warn_integration_accuracy" # Schlüssel
       warning_msg_args = {'err_dc': err_dc, 'err_lt': err_lt}

    return {
        'comoving_mpc': comoving_distance_mpc, 'luminosity_mpc': luminosity_distance_mpc,
        'ang_diam_mpc': angular_diameter_distance_mpc, 'lookback_gyr': lookback_time_gyr,
        'error_msg': None, 'integration_warning_key': warning_msg_key, 'integration_warning_args': warning_msg_args
    }
  except ImportError: return {'error_msg': "error_dep_scipy"} # Schlüssel
  except Exception as e:
        st.exception(e)
        return {'error_msg': "error_calc_failed", 'error_args': {'e': e}} # Schlüssel + Argumente

# --- Einheitenumrechnungsfunktionen (unverändert) ---
def convert_mpc_to_km(d): return d * KM_PER_MPC
def convert_km_to_au(d): return 0.0 if d == 0 else d / KM_PER_AU
def convert_km_to_ly(d): return 0.0 if d == 0 else d / KM_PER_LY
def convert_km_to_ls(d): return 0.0 if d == 0 else d / KM_PER_LS
def convert_mpc_to_gly(d):
    if d == 0: return 0.0
    km_per_gly = KM_PER_LY * 1e9
    distance_km = convert_mpc_to_km(d)
    return 0.0 if km_per_gly == 0 else distance_km / km_per_gly

# --- Formatierungsfunktion (unverändert) ---
def format_large_number(number):
    if number == 0: return "0"
    if not np.isfinite(number): return str(number)
    try:
        formatted = f"{number:,.0f}".replace(",", " ")
        return formatted
    except (ValueError, TypeError): return str(number)

# --- Übersetzungshelfer ---
# Initialisiere Sprache im Session State, falls nicht vorhanden
if 'lang' not in st.session_state:
    st.session_state.lang = 'DE' # Standard auf Deutsch

# Funktion zum Abrufen von Übersetzungen
def t(key, **kwargs):
    """Holt die Übersetzung für einen Schlüssel in der aktuellen Sprache."""
    lang = st.session_state.lang
    translation = translations.get(lang, translations['EN']) # Fallback auf Englisch
    text = translation.get(key, key) # Fallback auf den Schlüssel selbst
    # Argumente formatieren, falls vorhanden
    try:
        return text.format(**kwargs)
    except KeyError as e:
        print(f"Warnung: Fehlender Formatierungsschlüssel {e} für Text '{key}' in Sprache {lang}")
        return text # Gib den unformatierten Text zurück

# --- Streamlit UI Aufbau ---

# Seitenkonfiguration (Titel ohne Icon)
st.set_page_config(page_title="Advanced Redshift Calculator", layout="wide")

# Haupttitel (nicht übersetzt)
st.title("Advanced Redshift Calculator")

# --- Sidebar ---
with st.sidebar:
    st.header(t("input_params"))

    # Sprachauswahl oben in der Sidebar
    selected_lang = st.selectbox(
        label=t("lang_select"), # Übersetztes Label
        options=['DE', 'EN', 'FR'],
        index=['DE', 'EN', 'FR'].index(st.session_state.lang), # Aktuelle Sprache vorauswählen
        key='lang_selector' # Eindeutiger Schlüssel
    )
    # Update session state wenn Auswahl geändert wird
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun() # App neu laden, um alle Texte zu aktualisieren

    # Eingaben mit übersetzten Labels
    z_input = st.number_input(
        label=t("redshift_z"),
        min_value=-0.99, # Physikalisches Limit nahe -1
        value=0.03403,
        step=0.1,
        format="%.5f",
        help="Geben Sie die kosmologische Rotverschiebung ein." # Tooltip
    )

    st.markdown("---")
    st.subheader(t("cosmo_params"))
    h0_input = st.number_input(
        label=t("hubble_h0"),
        min_value=1.0, value=H0_DEFAULT, step=0.1, format="%.1f"
    )
    omega_m_input = st.number_input(
        label=t("omega_m"),
        min_value=0.0, max_value=2.0, value=OMEGA_M_DEFAULT, step=0.01, format="%.3f"
    )
    omega_lambda_input = st.number_input(
        label=t("omega_lambda"),
        min_value=0.0, max_value=2.0, value=OMEGA_LAMBDA_DEFAULT, step=0.01, format="%.3f"
    )

    # Optional: Hinweis auf flaches Universum
    if not math.isclose(omega_m_input + omega_lambda_input, 1.0, abs_tol=1e-3):
        st.warning(t("flat_universe_warning"))

    # Bug Report Button
    st.markdown("---")
    st.markdown(f"**{t('bug_report')}**")
    report_mail = "debrun2005@gmail.com"
    report_subject = "Bug Report: Advanced Redshift Calculator"
    report_body = f"Hallo,\n\nich habe einen Fehler im Advanced Redshift Calculator gefunden:\n\n[Bitte beschreiben Sie den Fehler hier]\n\nParameter:\nz={z_input}\nH0={h0_input}\nOmega_m={omega_m_input}\nOmega_Lambda={omega_lambda_input}\n\nDanke!"
    # URL-Encoding für Mailto Body (vereinfacht)
    report_body_encoded = report_body.replace("\n", "%0A").replace(" ", "%20")
    st.link_button(t("bug_report_button"), f"mailto:{report_mail}?subject={report_subject}&body={report_body_encoded}")


# --- Hauptbereich ---

# Berechnung durchführen
results = calculate_lcdm_distances(z_input, h0_input, omega_m_input, omega_lambda_input)

# Ergebnisse anzeigen
st.header(t("results_for", z=z_input))

# Fehler oder Warnungen zuerst behandeln
error_key = results.get('error_msg')
if error_key:
    error_args = results.get('error_args', {})
    error_text = t(error_key, **error_args) # Übersetzung holen
    if "Warnung:" in error_text or "Warning:" in error_text or "Avertissement:" in error_text: # Prüfe übersetzten Text
        st.warning(error_text)
    else:
        st.error(error_text)
        # Bei echtem Fehler hier anhalten, außer bei Blueshift-Warnung
        if error_key != "warn_blueshift":
            st.stop()

# Optionale Integrationswarnung anzeigen
warning_key = results.get('integration_warning_key')
if warning_key:
    warning_args = results.get('integration_warning_args', {})
    st.info(t(warning_key, **warning_args)) # Übersetzung holen

# Ergebnisse extrahieren (nach Fehlerprüfung)
comoving_mpc = results['comoving_mpc']
luminosity_mpc = results['luminosity_mpc']
ang_diam_mpc = results['ang_diam_mpc']
lookback_gyr = results['lookback_gyr']

# Umrechnungen
comoving_gly = convert_mpc_to_gly(comoving_mpc)
luminosity_gly = convert_mpc_to_gly(luminosity_mpc)
ang_diam_gly = convert_mpc_to_gly(ang_diam_mpc)
comoving_km = convert_mpc_to_km(comoving_mpc)
comoving_ly = convert_km_to_ly(comoving_km)
comoving_au = convert_km_to_au(comoving_km)
comoving_ls = convert_km_to_ls(comoving_km)
comoving_km_ausgeschrieben = format_large_number(comoving_km)

# Ergebnisse modern anzeigen (z.B. mit Metriken und Spalten)
st.metric(label=t("lookback_time"), value=f"{lookback_gyr:.4f}", delta=t("unit_Gyr"))
st.markdown("---")

st.subheader(t("cosmo_distances"))

col1, col2 = st.columns(2) # Spalten für bessere Darstellung

with col1:
    st.markdown(t("comoving_distance_title"))
    st.text(f"  {comoving_mpc:,.4f} {t('unit_Mpc')}")
    st.text(f"  {comoving_gly:,.4f} {t('unit_Gly')}")
    st.text(f"  {comoving_km:,.3e} {t('unit_km_sci')}")
    st.text(f"  {comoving_km_ausgeschrieben} {t('unit_km_full')}") # Angepasstes Label
    st.text(f"  {comoving_ly:,.3e} {t('unit_LJ')}")
    st.text(f"  {comoving_au:,.3e} {t('unit_AE')}")
    st.text(f"  {comoving_ls:,.3e} {t('unit_Ls')}")

with col2:
    st.markdown(t("luminosity_distance_title"))
    st.text(f"  {luminosity_mpc:,.4f} {t('unit_Mpc')}")
    st.text(f"  {luminosity_gly:,.4f} {t('unit_Gly')}")

    st.markdown(t("angular_diameter_distance_title"), unsafe_allow_html=True) # Leerzeile einfügen
    st.text(f"  {ang_diam_mpc:,.4f} {t('unit_Mpc')}")
    st.text(f"  {ang_diam_gly:,.4f} {t('unit_Gly')}")


st.markdown("---")

# Glossar
with st.expander(t("glossary")):
    current_glossary = glossary_data.get(st.session_state.lang, glossary_data['EN']) # Fallback EN
    for term, definition in current_glossary.items():
        st.markdown(f"**{term}:** {definition}")

st.markdown("---")

# Spendenlink
st.markdown(f"{t('donate_text')}")
st.link_button(t("donate_button"), "https://ko-fi.com/advanceddsofinder")


st.caption(t("calculation_note"))

# Hinweis zum Theme (kann nicht direkt aus dem Skript erzwungen werden)
# st.info("Hinweis: Das helle/dunkle Theme passt sich normalerweise an Ihre Browsereinstellungen an oder kann im Streamlit-Menü (oben rechts) geändert werden.")

