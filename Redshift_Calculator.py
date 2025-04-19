# Erforderliche Bibliotheken importieren
import streamlit as st
import numpy as np
from scipy.integrate import quad
import math

# --- Konstanten ---
C_KM_PER_S = 299792.458  # Lichtgeschwindigkeit in km/s
KM_PER_MPC = 3.085677581491367e+19  # Kilometer pro Megaparsec
KM_PER_AU = 1.495978707e+8   # Kilometer pro Astronomische Einheit (AE)
KM_PER_LY = 9.4607304725808e+12  # Kilometer pro Lichtjahr (LJ)
KM_PER_LS = C_KM_PER_S             # Kilometer pro Lichtsekunde (Ls)
GYR_PER_YR = 1e9 # Gigajahre pro Jahr

# --- Standard Kosmologische Parameter (Planck 2018, gerundet) ---
H0_DEFAULT = 67.4
OMEGA_M_DEFAULT = 0.315
OMEGA_LAMBDA_DEFAULT = 0.685

# --- Hilfsfunktionen für die Integration (unverändert) ---
def hubble_parameter_inv_integrand(z, omega_m, omega_lambda):
  """
  Berechnet den Integranden 1/E(z) für die mitbewegte Distanz in einem flachen LCDM-Universum.
  E(z) = H(z)/H0 = sqrt(omega_m*(1+z)^3 + omega_lambda)
  """
  epsilon = 1e-15
  denominator = np.sqrt(omega_m * (1 + z)**3 + omega_lambda + epsilon)
  if denominator < epsilon:
      return 0.0
  return 1.0 / denominator

def lookback_time_integrand(z, omega_m, omega_lambda):
  """
  Berechnet den Integranden für die Rückblickzeit in einem flachen LCDM-Universum.
  Integrand = 1 / (E(z) * (1+z))
  """
  epsilon = 1e-15
  term_in_sqrt = omega_m * (1 + z)**3 + omega_lambda
  term_in_sqrt = max(term_in_sqrt, 0)
  denominator = (1 + z) * np.sqrt(term_in_sqrt + epsilon)
  if math.isclose(z, 0):
      denom_at_zero = np.sqrt(omega_m + omega_lambda + epsilon)
      if denom_at_zero < epsilon: return 0.0
      return 1.0 / denom_at_zero
  if abs(denominator) < epsilon:
       return 0.0
  return 1.0 / denominator

# --- Kernberechnungsfunktion (unverändert) ---
# @st.cache_data # Streamlit Caching für schnellere Neuberechnungen bei gleichen Eingaben
def calculate_lcdm_distances(redshift, h0, omega_m, omega_lambda):
  """
  Berechnet verschiedene kosmologische Distanzen und die Rückblickzeit
  unter Verwendung des flachen Lambda-CDM-Modells.
  Gibt ein Dictionary mit Ergebnissen oder Fehlermeldung zurück.
  """
  # Eingabevalidierung
  if not isinstance(redshift, (int, float)) or \
     not isinstance(h0, (int, float)) or \
     not isinstance(omega_m, (int, float)) or \
     not isinstance(omega_lambda, (int, float)):
       return {'error_msg': "Ungültige Eingabe. Bitte Zahlen verwenden."}

  if redshift < 0:
     return {
        'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0,
        'error_msg': "Warnung: Rotverschiebung ist negativ (Blueshift). Kosmologische Distanzen sind hier 0 oder nicht direkt anwendbar."
    }
  if math.isclose(redshift, 0):
      return {
        'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0,
        'error_msg': None
    }
  if h0 <= 0:
    return {'error_msg': "Hubble-Konstante muss positiv sein."}
  if omega_m < 0 or omega_lambda < 0:
      return {'error_msg': "Omega-Parameter dürfen nicht negativ sein."}

  # Hubble-Distanz in Mpc
  dh = C_KM_PER_S / h0

  try:
    # Integrationen
    integral_dc, err_dc = quad(hubble_parameter_inv_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
    comoving_distance_mpc = dh * integral_dc

    hubble_time_gyr = 977.8 / h0
    integral_lt, err_lt = quad(lookback_time_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
    lookback_time_gyr = hubble_time_gyr * integral_lt

    # Andere Distanzen
    luminosity_distance_mpc = comoving_distance_mpc * (1 + redshift)
    angular_diameter_distance_mpc = comoving_distance_mpc / (1 + redshift)

    # Integrationswarnung (optional, kann im UI angezeigt werden)
    warning_msg = None
    integration_warning_threshold = 1e-5
    if err_dc > integration_warning_threshold or err_lt > integration_warning_threshold:
       warning_msg = f"Warnung: Relative Integrationsgenauigkeit möglicherweise nicht erreicht (Fehler: DC={err_dc:.2e}, LT={err_lt:.2e})."

    return {
        'comoving_mpc': comoving_distance_mpc,
        'luminosity_mpc': luminosity_distance_mpc,
        'ang_diam_mpc': angular_diameter_distance_mpc,
        'lookback_gyr': lookback_time_gyr,
        'error_msg': None, # Kein Fehler, aber evtl. eine Warnung
        'integration_warning': warning_msg
    }

  except ImportError:
        return {'error_msg': "Abhängigkeit 'scipy' nicht gefunden. Bitte installieren."}
  except Exception as e:
        st.exception(e) # Zeigt den vollen Traceback in der App an
        return {'error_msg': f"Berechnung fehlgeschlagen: {e}"}

# --- Einheitenumrechnungsfunktionen (unverändert) ---
def convert_mpc_to_km(distance_mpc): return distance_mpc * KM_PER_MPC
def convert_km_to_au(distance_km): return 0.0 if distance_km == 0 else distance_km / KM_PER_AU
def convert_km_to_ly(distance_km): return 0.0 if distance_km == 0 else distance_km / KM_PER_LY
def convert_km_to_ls(distance_km): return 0.0 if distance_km == 0 else distance_km / KM_PER_LS
def convert_mpc_to_gly(distance_mpc):
    if distance_mpc == 0: return 0.0
    km_per_gly = KM_PER_LY * 1e9
    distance_km = convert_mpc_to_km(distance_mpc)
    return 0.0 if km_per_gly == 0 else distance_km / km_per_gly

# --- Formatierungsfunktion (unverändert) ---
def format_large_number(number):
    if number == 0: return "0"
    if not np.isfinite(number): return str(number)
    try:
        formatted = f"{number:,.0f}".replace(",", " ")
        return formatted
    except (ValueError, TypeError): return str(number)

# --- Streamlit UI Aufbau ---

st.set_page_config(page_title="Kosmo Rechner", layout="wide")

st.title("🌌 Kosmologischer Rechner (ΛCDM)")
st.markdown("Berechnen Sie kosmologische Distanzen und Zeiten basierend auf der Rotverschiebung.")

# --- Eingabeparameter in der Sidebar ---
st.sidebar.header("Eingabeparameter")

z_input = st.sidebar.number_input(
    label="Rotverschiebung (z)",
    min_value=0.0, # Erlaube auch negative Werte für Blueshift, Behandlung in Funktion
    value=0.03403, # Standardwert
    step=0.1,
    format="%.5f" # Format für die Anzeige im Input-Feld
)

st.sidebar.markdown("---")
st.sidebar.subheader("Kosmologische Parameter")
h0_input = st.sidebar.number_input(
    label="Hubble-Konstante (H₀) [km/s/Mpc]",
    min_value=1.0,
    value=H0_DEFAULT,
    step=0.1,
    format="%.1f"
)
omega_m_input = st.sidebar.number_input(
    label="Materiedichte (Ωm)",
    min_value=0.0,
    max_value=2.0, # Erlaube etwas mehr Spielraum
    value=OMEGA_M_DEFAULT,
    step=0.01,
    format="%.3f"
)
omega_lambda_input = st.sidebar.number_input(
    label="Dunkle Energie (ΩΛ)",
    min_value=0.0,
    max_value=2.0, # Erlaube etwas mehr Spielraum
    value=OMEGA_LAMBDA_DEFAULT,
    step=0.01,
    format="%.3f"
)

# Optional: Hinweis auf flaches Universum
if not math.isclose(omega_m_input + omega_lambda_input, 1.0, abs_tol=1e-3):
    st.sidebar.warning("Ωm + ΩΛ ≉ 1. Die Berechnungen gehen von einem flachen Universum aus (Ωk=0).")


# --- Berechnung durchführen ---
results = calculate_lcdm_distances(z_input, h0_input, omega_m_input, omega_lambda_input)

# --- Ergebnisse anzeigen ---
st.header(f"Ergebnisse für z = {z_input:.5f}")

# Fehler oder Warnungen zuerst behandeln
if results.get('error_msg'):
    if "Warnung:" in results['error_msg']:
        st.warning(results['error_msg'])
    else:
        st.error(results['error_msg'])
        st.stop() # Bei echtem Fehler hier anhalten

# Optionale Integrationswarnung anzeigen
if results.get('integration_warning'):
    st.info(results['integration_warning'])


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

# Ergebnisse in Spalten oder mit st.metric anzeigen
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Rückblickzeit (Lookback Time)", value=f"{lookback_gyr:.4f} Gyr")

# Detaillierte Distanzen
st.subheader("Kosmologische Distanzen")

st.markdown("**Mitbewegte Distanz (Comoving Distance):**")
st.text(f"  {comoving_mpc:,.4f} Mpc")
st.text(f"  {comoving_gly:,.4f} Gly (Milliarden Lichtjahre)")
st.text(f"  {comoving_km:,.3e} km")
st.text(f"  {comoving_km_ausgeschrieben} km (ausgeschrieben)")
st.text(f"  {comoving_ly:,.3e} LJ")
st.text(f"  {comoving_au:,.3e} AE")
st.text(f"  {comoving_ls:,.3e} Ls")

st.markdown("**Leuchtkraftdistanz (Luminosity Distance):**")
st.text(f"  {luminosity_mpc:,.4f} Mpc")
st.text(f"  {luminosity_gly:,.4f} Gly")

st.markdown("**Winkeldurchmesserdistanz (Angular Diameter Distance):**")
st.text(f"  {ang_diam_mpc:,.4f} Mpc")
st.text(f"  {ang_diam_gly:,.4f} Gly")

st.markdown("---")
st.caption("Berechnung basiert auf dem flachen ΛCDM-Modell unter Vernachlässigung der Strahlungsdichte.")

