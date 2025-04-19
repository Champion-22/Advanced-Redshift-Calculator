# Advanced Redshift Calculator

An interactive web application for calculating various cosmological distances and the lookback time based on redshift (z) using the standard ΛCDM model. Built with Streamlit.

Intended for students, amateur astronomers, educators, and anyone interested in cosmology.

**➡️ Try the live application here: [https://advanced-redshift-calculator.streamlit.app/](https://advanced-redshift-calculator.streamlit.app/)**

## ✨ Features

* Calculations based on the **ΛCDM model**.
* Computes key cosmological measures:
    * Comoving Distance
    * Luminosity Distance
    * Angular Diameter Distance
    * Lookback Time
* Displays distances in various units (Mpc, Gly, km, ly, AU, Ls), including the full kilometer value written out.
* **Interactive input** of redshift (z) and cosmological parameters (H₀, Ωm, ΩΛ) via the sidebar.
* **Multilingual User Interface:** German (DE), English (EN), French (FR).
* **Contextual Examples:** Tangible comparisons for lookback time and comoving distance to better understand the scales involved.
* Brief **explanations** of the meaning of luminosity and angular diameter distance.
* Integrated, translated **Glossary** explaining key terms.
* Option to report errors via a **Bug Report button**.
* Link to support development via **Ko-fi**.
* Modern, responsive design thanks to Streamlit.

## 🚀 Installation

Ensure you have Python 3.9 or higher installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Champion-22/Advanced-Redshift-Calculator.git](https://github.com/Champion-22/Advanced-Redshift-Calculator.git)
    cd Advanced-Redshift-Calculator
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The required packages are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing, create it with the content `streamlit`, `numpy`, `scipy`, or generate it using `pip freeze > requirements.txt` in your environment after manually installing the packages: `pip install streamlit numpy scipy`)*

## ▶️ Usage

After installation, you can start the application locally:

1.  Open your terminal or command prompt.
2.  Navigate to the project directory.
3.  Ensure your virtual environment (if created) is activated.
4.  Run the following command (replace `redshift_app.py` if your script has a different name):
    ```bash
    streamlit run redshift_app.py
    ```
5.  Streamlit will start a local server and automatically open the application in your default web browser.
6.  Change the input parameters in the sidebar to update the results live.

## ⚙️ Configuration

* The cosmological parameters (H₀, Ωm, ΩΛ) can be adjusted directly in the application's sidebar. The default values are based on the Planck 2018 results.
* The theme (light/dark mode) can usually be controlled via the Streamlit menu (≡ icon in the top right corner of the app) or through your browser/OS system settings.

## ❤️ Support / Donation

If you like this calculator and find it useful, please consider supporting its development:

* [Donate via Ko-fi](https://ko-fi.com/advanceddsofinder)

Thank you for your support!

## 🤝 Contributing & Bug Reports

Bug reports and suggestions are welcome!

* Use the **"Report Issue"** button in the app's sidebar to create an email with the current parameters.
* Alternatively, you can create an [Issue on GitHub](https://github.com/Champion-22/Advanced-Redshift-Calculator/issues).

## 🙏 Acknowledgements

* Based on the standard ΛCDM model of cosmology.
* Uses the libraries [Streamlit](https://streamlit.io/), [NumPy](https://numpy.org/), and [SciPy](https://scipy.org/).
* Default parameters are based on the [Planck 2018 results](https://www.cosmos.esa.int/web/planck/publications).

