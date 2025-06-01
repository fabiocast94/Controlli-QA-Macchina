import numpy as np
import pydicom
import math
from pylinac.core.image import DicomImage
import streamlit as st
from pylinac import DRMLC, DRGS, PicketFence, Starshot, CatPhan504, WinstonLutz, FieldAnalysis
from pylinac.field_analysis import Interpolation, Normalization, Centering
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from PIL import Image
import warnings
import datetime
import os
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt
import base64   # <-- questa riga serve


warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# CONFIGURAZIONE PAGINA
st.set_page_config(page_title="Controlli Qualità LINAC", layout="wide")

# PATH LOGO
logo_file_path = "logo.png"

# --- FUNZIONI ---

def mostra_logo_e_titolo(logo_path, titolo):
    logo = Image.open(logo_path)
    buffered = BytesIO()
    logo.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_str}" style="max-width: 300px; height: auto; margin-bottom: 10px;" />
            <h1 style="margin: 0;">{titolo}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

def inserisci_logo_pdf(c, logo_path, page_width, page_height):
    logo = Image.open(logo_path)
    max_width = page_width * 0.6
    wpercent = max_width / float(logo.size[0])
    hsize = int(float(logo.size[1]) * wpercent)
    logo = logo.resize((int(max_width), hsize), Image.Resampling.LANCZOS)

    img_io = BytesIO()
    logo.save(img_io, format="PNG")
    img_io.seek(0)
    x = (page_width - max_width) / 2
    y = page_height - hsize - 50
    c.drawImage(ImageReader(img_io), x, y, width=max_width, height=hsize, mask='auto')

def crea_report_pdf(titolo, risultati, pylinac_obj, utente, linac, energia):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    inserisci_logo_pdf(c, logo_file_path, width, height)

    y_start = height - 180
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_start, f"Controlli Qualità LINAC - {titolo}")
    c.setFont("Helvetica", 12)
    c.drawString(50, y_start - 20, f"Utente: {utente}")
    c.drawString(50, y_start - 40, f"Linac: {linac}")
    c.drawString(50, y_start - 60, f"Energia: {energia}")
    c.drawString(50, y_start - 80, f"Data: {datetime.date.today()}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_start - 110, "Risultati Analisi:")
    c.setFont("Courier", 10)
    text_obj = c.beginText(50, y_start - 130)

    for line in risultati.splitlines():
        if text_obj.getY() < 50:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText(50, height - 50)
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.save()
    buffer.seek(0)
    return buffer

def get_profile(ds):
    img = ds.pixel_array.astype(float)
    center_row = img.shape[0] // 2
    return img[center_row, :]

def find_dose_at_distance(profile, pixel_spacing_cm, wdistL):
    center_pixel = len(profile) // 2
    half_dist_pix = int((wdistL / 2) / pixel_spacing_cm)

    left_index = max(center_pixel - half_dist_pix, 0)
    right_index = min(center_pixel + half_dist_pix, len(profile) - 1)

    D1 = profile[left_index]
    D2 = profile[right_index]
    return D1, D2, left_index, right_index

def calculate_theta(D1, D2, u, wdistL):
    if D1 <= 0 or D2 <= 0:
        raise ValueError("Dose D1 e D2 devono essere positivi")
    ln_ratio = math.log(D1 / D2)
    theta_rad = math.atan(ln_ratio / (u * wdistL))
    return math.degrees(theta_rad)

def plot_profile(profile, left_idx, right_idx):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(profile, label='Profilo dose')
    ax.scatter([left_idx, right_idx], [profile[left_idx], profile[right_idx]], color='red', label='D1 e D2')
    ax.axvline(x=len(profile)//2, color='gray', linestyle='--', label='Centro')
    ax.set_title('Profilo dose EPID')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Dose (counts)')
    ax.legend()
    ax.grid(True)
    return fig
    
def number_input_with_key(label, key_suffix, **kwargs):
    return st.number_input(label, key=f"{label}_{key_suffix}", **kwargs)

# --- INIZIO UI ---

mostra_logo_e_titolo(logo_file_path, "Controlli Qualità LINAC")

utente = st.text_input("Nome Utente")
linac = st.selectbox("Seleziona Linac", ["Linac1", "Edge", "Linac3", "Linac4", "STx"])
energia = st.selectbox("Seleziona Energia", ["6 MV", "10 MV", "15 MV", "6 FFF", "10 FFF"])

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Dose Rate Gantry Speed",
    "Dose Rate Leaf Speed",
    "Picket Fence",
    "Star Shot",
    "CBCT CatPhan",
    "Winston Lutz",
    "Field Analysis",
    "Wedge Analysis"
])

# -- TAB 1: Dose Rate Gantry Speed --
with tab1:
    st.header("Dose Rate Gantry Speed (DRGS)")
    open_img = st.file_uploader("Carica immagine Open.dcm", type=["dcm"], key="drgs_open")
    dmlc_img = st.file_uploader("Carica immagine Field.dcm", type=["dcm"], key="drgs_field")
    tolerance = number_input_with_key("Tolleranza (%)", "drgs", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    roi_width = number_input_with_key("Larghezza ROI (mm)", "drgs", min_value=5, max_value=50, value=10, step=1)
    roi_height = number_input_with_key("Altezza ROI (mm)", "drgs", min_value=50, max_value=300, value=150, step=5)
    shift_roi = number_input_with_key("Shift globale ROI (mm)", "drgs", min_value=-100, max_value=100, value=0, step=1)
    # resto del codice invariato ...
    n_roi = 7
    spacing = 15
    base_offsets = [spacing * (i - n_roi // 2) for i in range(n_roi)]
    custom_roi_config = {f"ROI {i+1}": {"offset_mm": base_offsets[i] + shift_roi} for i in range(n_roi)}

    if open_img and dmlc_img and st.button("Esegui analisi DRGS"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_open, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_field:
            f_open.write(open_img.getbuffer())
            f_field.write(dmlc_img.getbuffer())
            f_open.flush()
            f_field.flush()

            drgs = DRGS(image_paths=(f_open.name, f_field.name))
            drgs.default_roi_config = custom_roi_config
            drgs.analyze(tolerance=tolerance, segment_size_mm=(roi_width, roi_height))

            risultati = drgs.results()
            st.text(risultati)
            drgs.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf("Dose Rate Gantry Speed", risultati, drgs, utente, linac, energia)
                st.download_button("📥 Scarica Report DRGS PDF", data=report_pdf,
                                   file_name="QA_Report_DRGS.pdf", mime="application/pdf")
            else:
                st.warning("Inserisci il nome utente per generare il report.")

# -- TAB 2: Dose Rate Leaf Speed --
with tab2:
    st.header("Dose Rate Leaf Speed (DRMLC)")
    open_img = st.file_uploader("Carica immagine Open.dcm", type=["dcm"], key="drmlc_open")
    dmlc_img = st.file_uploader("Carica immagine Field.dcm", type=["dcm"], key="drmlc_field")
    tolerance = number_input_with_key("Tolleranza (%)", "drmlc", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    roi_width = number_input_with_key("Larghezza ROI (mm)", "drmlc", min_value=5, max_value=50, value=15, step=1)
    roi_height = number_input_with_key("Altezza ROI (mm)", "drmlc", min_value=50, max_value=300, value=150, step=5)
    shift_roi = number_input_with_key("Shift globale ROI (mm)", "drmlc", min_value=-100, max_value=100, value=0, step=1)
    # resto invariato ...
    n_roi = 9
    spacing = 15
    base_offsets = [spacing * (i - n_roi // 2) for i in range(n_roi)]
    custom_roi_config = {f"ROI {i+1}": {"offset_mm": base_offsets[i] + shift_roi} for i in range(n_roi)}

    if open_img and dmlc_img and st.button("Esegui analisi DRMLC"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_open, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_field:
            f_open.write(open_img.getbuffer())
            f_field.write(dmlc_img.getbuffer())
            f_open.flush()
            f_field.flush()

            drmlc = DRMLC(image_paths=(f_open.name, f_field.name))
            drmlc.default_roi_config = custom_roi_config
            drmlc.analyze(tolerance=tolerance, segment_size_mm=(roi_width, roi_height))

            risultati = drmlc.results()
            st.text(risultati)
            drmlc.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf("Dose Rate Leaf Speed", risultati, drmlc, utente, linac, energia)
                st.download_button("📥 Scarica Report DRMLC PDF", data=report_pdf,
                                   file_name="QA_Report_DRMLC.pdf", mime="application/pdf")
            else:
                st.warning("Inserisci il nome utente per generare il report.")

# -- TAB 3: Picket Fence --
with tab3:
    st.header("Picket Fence")
    dmlc_img = st.file_uploader("Carica immagine PicketFence.dcm", type=["dcm"], key="picket")
    tolerance = number_input_with_key("Tolleranza (%)", "picket", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    # resto invariato ...
    if dmlc_img and st.button("Esegui analisi Picket Fence"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_picket:
            f_picket.write(dmlc_img.getbuffer())
            f_picket.flush()

            pf = PicketFence(f_picket.name)
            pf.analyze(tolerance=tolerance)

            risultati = pf.results()
            st.text(risultati)
            pf.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf("Picket Fence", risultati, pf, utente, linac, energia)
                st.download_button("📥 Scarica Report Picket Fence PDF", data=report_pdf,
                                   file_name="QA_Report_PicketFence.pdf", mime="application/pdf")
            else:
                st.warning("Inserisci il nome utente per generare il report.")

# -- TAB 4: Star Shot --
with tab4:
    st.header("Star Shot")
    starshot_img = st.file_uploader("Carica immagine StarShot.dcm", type=["dcm"], key="starshot")
    tolerance = number_input_with_key("Tolleranza (mm)", "starshot", min_value=0.01, max_value=5.0, value=0.3, step=0.01)
    # resto invariato ...
    if starshot_img and st.button("Esegui analisi Star Shot"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_star:
            f_star.write(starshot_img.getbuffer())
            f_star.flush()

            ss = Starshot(f_star.name)
            ss.analyze(tolerance=tolerance)

            risultati = ss.results()
            st.text(risultati)
            ss.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf("Star Shot", risultati, ss, utente, linac, energia)
                st.download_button("📥 Scarica Report Star Shot PDF", data=report_pdf,
                                   file_name="QA_Report_StarShot.pdf", mime="application/pdf")
            else:
                st.warning("Inserisci il nome utente per generare il report.")

# -- TAB 5: CatPhan504 (CBCT) --
with tab5:
    st.header("CBCT CatPhan504")
    catphan_img = st.file_uploader("Carica immagine CatPhan504.dcm", type=["dcm"], key="catphan")
    tolerance_contrast = number_input_with_key("Tolleranza Contrasto (%)", "catphan", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    tolerance_uniformity = number_input_with_key("Tolleranza Uniformità (%)", "catphan", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    # resto invariato ...
    if catphan_img and st.button("Esegui analisi CatPhan"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_catphan:
            f_catphan.write(catphan_img.getbuffer())
            f_catphan.flush()

            cp = CatPhan504(f_catphan.name)
            cp.analyze()
            risultati = cp.results()
            st.text(risultati)
            cp.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf("CatPhan504", risultati, cp, utente, linac, energia)
                st.download_button("📥 Scarica Report CatPhan504 PDF", data=report_pdf,
                                   file_name="QA_Report_CatPhan504.pdf", mime="application/pdf")
            else:
                st.warning("Inserisci il nome utente per generare il report.")

# -- TAB 6: Winston Lutz --
with tab6:
    st.header("Winston Lutz")
    wl_img = st.file_uploader("Carica immagine WinstonLutz.dcm", type=["dcm"], key="wl")
    tolerance = st.number_input("Tolleranza (mm)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)

    if wl_img and st.button("Esegui analisi Winston Lutz"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_wl:
            f_wl.write(wl_img.getbuffer())
            f_wl.flush()

            wl = WinstonLutz(f_wl.name)
            wl.analyze(tolerance=tolerance)
            risultati = wl.results()
            st.text(risultati)
            wl.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf("Winston Lutz", risultati, wl, utente, linac, energia)
                st.download_button("📥 Scarica Report Winston Lutz PDF", data=report_pdf,
                                   file_name="QA_Report_WinstonLutz.pdf", mime="application/pdf")
            else:
                st.warning("Inserisci il nome utente per generare il report.")


# -- TAB 7: Field Analysis --
with tab7:
    st.header("Field Analysis")
    fa_img = st.file_uploader("Carica immagine FieldAnalysis.dcm", type=["dcm"], key="fa")
    # qui non hai tolerance quindi no change

    if fa_img and st.button("Esegui analisi Field Analysis"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_fa:
            f_fa.write(fa_img.getbuffer())
            f_fa.flush()

            fa = FieldAnalysis(f_fa.name)
            fa.analyze()
            risultati = fa.results()
            st.text(risultati)
            fa.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf("Field Analysis", risultati, fa, utente, linac, energia)
                st.download_button("📥 Scarica Report Field Analysis PDF", data=report_pdf,
                                   file_name="QA_Report_FieldAnalysis.pdf", mime="application/pdf")
            else:
                st.warning("Inserisci il nome utente per generare il report.")

# -- TAB 8: Wedge Analysis --
with tab8:
    st.header("Wedge Analysis")
    wedge_img = st.file_uploader("Carica immagine EPID.dcm", type=["dcm"], key="wedge")
    
    # Questi due input devono essere fuori dal button perché se no non li vedi subito
    u = number_input_with_key("Parametro u (default 0.5)", "wedge", value=0.5, step=0.1)
    wdistL = number_input_with_key("Distanza wdistL (mm)", "wedge", min_value=1.0, value=50.0, step=1.0)

    if wedge_img and st.button("Calcola angolo EPID"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_wedge:
            f_wedge.write(wedge_img.getbuffer())
            f_wedge.flush()
            ds = pydicom.dcmread(f_wedge.name)
            pixel_spacing = ds.PixelSpacing[0]  # assuming square pixels
            profile = get_profile(ds)

            try:
                D1, D2, left_idx, right_idx = find_dose_at_distance(profile, pixel_spacing, wdistL)
                theta = calculate_theta(D1, D2, u, wdistL / 10)  # convert mm to cm if necessario

                st.write(f"D1: {D1:.2f}")
                st.write(f"D2: {D2:.2f}")
                st.write(f"Angolo θ (gradi): {theta:.3f}")

                fig = plot_profile(profile, left_idx, right_idx)
                st.pyplot(fig)
                plt.clf()

                if utente.strip():
                    risultati = f"D1 = {D1:.2f}\nD2 = {D2:.2f}\nAngolo θ = {theta:.3f}°"
                    report_pdf = crea_report_pdf("Wedge Analysis", risultati, None, utente, linac, energia)
                    st.download_button("📥 Scarica Report Wedge PDF", data=report_pdf,
                                       file_name="QA_Report_Wedge.pdf", mime="application/pdf")
                else:
                    st.warning("Inserisci il nome utente per generare il report.")
            except Exception as e:
                st.error(f"Errore nel calcolo: {e}")

