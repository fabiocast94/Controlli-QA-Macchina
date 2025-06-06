# IMPORT
import numpy as np
from pylinac.core.image import DicomImage
import streamlit as st
from pylinac import DRMLC
import matplotlib.pyplot as plt
from pylinac import DRGS, PicketFence, Starshot, CatPhan504, WinstonLutz, FieldAnalysis
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
import pydicom
import math


warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# CONFIGURAZIONE PAGINA
st.set_page_config(page_title="Controlli Qualità LINAC", layout="wide")

# LOGO + TITOLO
logo_file_path = "logo.png"

def mostra_logo_e_titolo(logo_path, titolo):
    logo = Image.open(logo_path)
    import base64
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

mostra_logo_e_titolo(logo_file_path, "Controlli Qualità LINAC")

# DATI GENERALI
utente = st.text_input("Nome Utente")
linac = st.selectbox("Seleziona Linac", ["Linac1", "Edge", "Linac3", "Linac4", "STx"])
energia = st.selectbox("Seleziona Energia", ["6 MV", "10 MV", "15 MV", "6 FFF", "10 FFF"])

# FUNZIONE PDF
def inserisci_logo_pdf(c, logo_path, page_width, page_height):
    logo = Image.open(logo_path)
    max_width = page_width * 0.6
    wpercent = max_width / float(logo.size[0])
    hsize = int((float(logo.size[1]) * float(wpercent)))
    logo = logo.resize((int(max_width), hsize), Image.Resampling.LANCZOS)

    img_io = BytesIO()
    logo.save(img_io, format="PNG")
    img_io.seek(0)
    x = (page_width - max_width) / 2
    y = page_height - hsize - 50
    c.drawImage(ImageReader(img_io), x, y, width=max_width, height=hsize, mask='auto')

def crea_report_pdf_senza_immagini(titolo, risultati, pylinac_obj, utente, linac, energia):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    import datetime
    from io import BytesIO

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Inserisci logo centrato in alto
    try:
        logo = Image.open(logo_file_path)
        max_width = width * 0.6
        wpercent = max_width / float(logo.size[0])
        hsize = int((float(logo.size[1]) * float(wpercent)))
        logo = logo.resize((int(max_width), hsize), Image.Resampling.LANCZOS)
        img_io = BytesIO()
        logo.save(img_io, format="PNG")
        img_io.seek(0)
        x = (width - max_width) / 2
        y = height - hsize - 50
        c.drawImage(ImageReader(img_io), x, y, width=max_width, height=hsize, mask='auto')
    except Exception as e:
        # Se fallisce il logo, non bloccare tutto
        pass

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




tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Dose Rate Gantry Speed",
    "Dose Rate Leaf Speed",
    "Picket Fence",
    "Star Shot",
    "CBCT CatPhan",
    "Wiston Lutz",
    "Field Analysis",
    "Wedge Angle"
])


with tab1:
    st.header("Dose Rate Gantry Speed (DRGS)")

    open_img = st.file_uploader("Carica immagine Open.dcm", type=["dcm"], key="drgs_open")
    dmlc_img = st.file_uploader("Carica immagine Field.dcm", type=["dcm"], key="drgs_field")

    tolerance = st.number_input("Tolleranza (%)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    roi_width = st.number_input("Larghezza ROI (mm)", min_value=5, max_value=50, value=10, step=1)
    roi_height = st.number_input("Altezza ROI (mm)", min_value=50, max_value=300, value=150, step=5)

    shift_roi = st.number_input("Shift globale ROI (mm)", min_value=-100, max_value=100, value=0, step=1)

    # Definisci posizione base delle ROI, ad esempio posizioni relative (in mm)
    base_offsets = [-50, -38, -25, -10, 3, 17, 30]  # Oppure un metodo per calcolare la posizione base rispetto all’immagine
    # O puoi definire in modo più dinamico tipo equidistanti intorno allo zero
    # Ad esempio:
    n_roi = 7
    spacing = 15
    base_offsets = [spacing * (i - n_roi // 2) for i in range(n_roi)]

    # Calcola l'offset totale sommando shift globale
    custom_roi_config = {f"ROI {i+1}": {"offset_mm": base_offsets[i] + shift_roi} for i in range(n_roi)}

    if open_img and dmlc_img and st.button("Esegui analisi DRGS"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_open, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_field:
            f_open.write(open_img.getbuffer())
            f_field.write(dmlc_img.getbuffer())

        drgs = DRGS(image_paths=(f_open.name, f_field.name))
        drgs.default_roi_config = custom_roi_config

        segment_size_mm = (roi_width, roi_height)
        drgs.analyze(tolerance=tolerance, segment_size_mm=segment_size_mm)

        risultati = drgs.results()
        st.text(risultati)
        drgs.plot_analyzed_image()
        st.pyplot(plt.gcf())
        plt.clf()

        if utente:
            report_pdf = crea_report_pdf_senza_immagini("Dose Rate Gantry Speed", risultati, drgs, utente, linac, energia)
            st.download_button(
                "📥 Scarica Report DRGS PDF",
                data=report_pdf,
                file_name="QA_Report_DRGS.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Inserisci il nome utente per generare il report.")




with tab2:
    st.header("Dose Rate Leaf Speed (DRMLC)")

    open_img = st.file_uploader("Carica immagine Open Field (DICOM)", type=["dcm"], key="drmlc_open")
    mlc_img = st.file_uploader("Carica immagine MLC Field (DICOM)", type=["dcm"], key="drmlc_mlc")

    tolerance = st.number_input("Tolleranza (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    if open_img and mlc_img and st.button("Esegui analisi DRLS"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_open, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f_mlc:
            f_open.write(open_img.getbuffer())
            f_mlc.write(mlc_img.getbuffer())

        try:
            drmlc = DRMLC((f_open.name, f_mlc.name))
            drmlc.analyze(tolerance=tolerance)
            risultati = drmlc.results()

            st.text(risultati)
            drmlc.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf_senza_immagini("Dose Rate Leaf Speed", risultati, drmlc, utente, linac, energia)
                st.download_button(
                    "📥 Scarica Report DRLS PDF",
                    data=report_pdf,
                    file_name="QA_Report_DRLS.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Inserisci il nome utente per generare il report.")
        except Exception as e:
            st.error(f"Errore durante l'analisi DRLS: {e}")



with tab3:
    from pylinac.picketfence import MLC

    st.header("Picket Fence")
    pf_img = st.file_uploader("Carica immagine PicketFence.dcm", type=["dcm"])

    # ✅ Aggiunta selezione tipo di MLC
    mlc_type_label = st.selectbox("Seleziona tipo di MLC", ["Millennium", "HD Millennium"])
    mlc_type = MLC.MILLENNIUM if mlc_type_label == "Millennium" else MLC.HD_MILLENNIUM

    tolerance = st.number_input("Tolleranza (mm)", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    action_tolerance = st.number_input("Action tolerance (mm)", min_value=0.01, max_value=1.0, value=0.03, step=0.01)

    if pf_img and st.button("Esegui analisi Picket Fence"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f:
            f.write(pf_img.getbuffer())

        try:
            pf = PicketFence(f.name, mlc=mlc_type)
            pf.analyze(tolerance=tolerance, action_tolerance=action_tolerance)
            risultati = pf.results()

            st.text(risultati)
            pf.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf_senza_immagini("Picket Fence", risultati, pf, utente, linac, energia)
                st.download_button(
                    "📥 Scarica Report Picket Fence PDF",
                    data=report_pdf,
                    file_name="QA_Report_PicketFence.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Inserisci il nome utente per generare il report.")
        except Exception as e:
            st.error(f"Errore durante l'analisi Picket Fence: {e}")

with tab4:
    st.header("Star Shot")
    star_img = st.file_uploader("Carica immagine Starshot (TIFF)", type=["tif", "tiff"])
    sid_value = st.number_input("SID (mm)", min_value=100, max_value=2000, value=1000)
    tolerance = st.number_input("Tolleranza", min_value=0.1, max_value=5.0, value=0.8, step=0.1)

    if star_img and st.button("Esegui analisi Star Shot"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
            f.write(star_img.getbuffer())

        ss = Starshot(f.name, sid=sid_value)
        ss.analyze(tolerance=tolerance)
        risultati = ss.results()

        st.text(risultati)
        ss.plot_analyzed_image()
        st.pyplot(plt.gcf())
        plt.clf()

        if utente:
            report_pdf = crea_report_pdf_senza_immagini("Star Shot", risultati, ss, utente, linac, energia)
            st.download_button(
                "📥 Scarica Report Star Shot PDF",
                data=report_pdf,
                file_name="QA_Report_StarShot.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Inserisci il nome utente per generare il report.")

with tab5:
    st.header("CBCT CatPhan")

    uploaded_file = st.file_uploader("Carica un file ZIP contenente immagini DICOM", type="zip")

    if uploaded_file and st.button("Esegui analisi CatPhan"):
        import tempfile
        import zipfile
        import os
        import pydicom

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            except zipfile.BadZipFile:
                st.error("Il file caricato non è un archivio ZIP valido.")
            else:
                dicom_files = []
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith('.dcm'):
                            dicom_files.append(os.path.join(root, file))

                if not dicom_files:
                    st.error("Nessun file DICOM trovato nello ZIP.")
                else:
                    st.success(f"Trovati {len(dicom_files)} file DICOM.")

                    try:
                        ds = pydicom.dcmread(dicom_files[0])
                        st.write(f"**Patient Name:** {ds.get('PatientName', 'N/A')}")
                        st.write(f"**Study Date:** {ds.get('StudyDate', 'N/A')}")
                        st.write(f"**Modality:** {ds.get('Modality', 'N/A')}")
                        # Qui puoi aggiungere ulteriori analisi con pylinac CatPhan504
                        # Esempio base di inizializzazione CatPhan:
                        catphan = CatPhan504(dicom_files)
                        catphan.analyze()
                        risultati = catphan.results()

                        st.text(risultati)
                        catphan.plot_analyzed_image()
                        st.pyplot(plt.gcf())
                        plt.clf()

                        if utente.strip():
                            report_pdf = crea_report_pdf_senza_immagini("CBCT CatPhan", risultati, catphan, utente, linac, energia)
                            st.download_button(
                                "📥 Scarica Report CBCT CatPhan PDF",
                                data=report_pdf,
                                file_name="QA_Report_CBCT_CatPhan.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.warning("Inserisci il nome utente per generare il report.")

                    except Exception as e:
                        st.error(f"Errore durante l'analisi CBCT CatPhan: {e}")




with tab6:
    st.header("Winston Lutz")
    wl_zip = st.file_uploader("Carica file ZIP contenente immagini per Winston-Lutz", type=["zip"])

    if wl_zip and st.button("Esegui analisi Winston Lutz"):
        import zipfile
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "winston_lutz.zip")
            with open(zip_path, "wb") as f:
                f.write(wl_zip.getbuffer())

            try:
                # Estrai i file dallo ZIP nella cartella temporanea
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Filtra solo i file DICOM (.dcm)
                dicom_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                               if f.lower().endswith('.dcm')]

                if not dicom_files:
                    st.error("Nessun file DICOM trovato nello ZIP.")
                else:
                    # Esegui analisi WinstonLutz con la lista di file DICOM
                    wl = WinstonLutz(dicom_files)
                    wl.analyze()
                    risultati = wl.results()

                    st.text(risultati)
                    wl.plot_images()
                    st.pyplot(plt.gcf())
                    plt.clf()

                    if utente.strip():
                        report_pdf = crea_report_pdf_senza_immagini("Winston Lutz", risultati, wl, utente, linac, energia)
                        st.download_button(
                            "📥 Scarica Report Winston Lutz PDF",
                            data=report_pdf,
                            file_name="QA_Report_WinstonLutz.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("Inserisci il nome utente per generare il report.")

            except zipfile.BadZipFile:
                st.error("Il file caricato non è un file ZIP valido.")
            except Exception as e:
                st.error(f"Errore durante l'analisi Winston Lutz: {e}")



with tab7:
    import matplotlib.pyplot as plt

    st.header("Field Analysis")

    fa_file = st.file_uploader("Carica immagine FieldAnalysis (DICOM)", type=["dcm"])

    interpolation = st.selectbox("Interpolazione", options=[i.name for i in Interpolation])
    normalization = st.selectbox("Normalizzazione", options=[n.name for n in Normalization])
    centering = st.selectbox("Centering method", options=[c.name for c in Centering])

    # Usa il nome utente già inserito all’inizio (non ridichiararlo)
    # utente è già definito globalmente

    if fa_file and st.button("Esegui analisi Field Analysis"):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as f:
            f.write(fa_file.getbuffer())
            temp_path = f.name

        try:
            fa = FieldAnalysis(temp_path)
            fa.interpolation = Interpolation[interpolation]
            fa.normalization = Normalization[normalization]
            fa.centering = Centering[centering]

            fa.analyze()
            risultati = fa.results()

            st.text(risultati)
            fa.plot_analyzed_image()
            st.pyplot(plt.gcf())
            plt.clf()

            if utente.strip():
                report_pdf = crea_report_pdf_senza_immagini("Field Analysis", risultati, fa, utente, linac, energia)
                st.download_button(
                    "📥 Scarica Report Field Analysis PDF",
                    data=report_pdf,
                    file_name="QA_Report_FieldAnalysis.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Inserisci il nome utente per generare il report.")

        except Exception as e:
            st.error(f"Errore durante l'analisi Field Analysis: {e}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
with tab8:
    st.header("Wedge Angle")

    wedge_img = st.file_uploader("Carica immagine EPID con wedge (DICOM)", type=["dcm"])

    wdistL = st.number_input("Distanza campo (cm, es. 5 per 10x10)", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    u = st.number_input("Valore u (costante da misure water tank)", min_value=0.001, max_value=1.0, value=0.036, step=0.001)
    nominal_angle = st.number_input("Angolo nominale wedge (°)", min_value=0.0, max_value=90.0, value=60.0, step=0.1)
    tolerance_percent = st.number_input("Tolleranza percentuale (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

    def get_profile(ds):
        img = ds.pixel_array.astype(float)
        center_col = img.shape[1] // 2   # colonna centrale per asse y
        profile = img[:, center_col]     # profilo lungo asse y
        return profile

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
        theta_deg = math.degrees(theta_rad)
        return abs(theta_deg)  # valore assoluto

    def plot_profile(profile, left_idx, right_idx):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(profile, label='Profilo dose')
        ax.scatter([left_idx, right_idx], [profile[left_idx], profile[right_idx]], color='red', label='D1 e D2')
        ax.axvline(x=len(profile)//2, color='gray', linestyle='--', label='Centro')
        ax.set_title('Profilo dose EPID (asse Y)')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Dose (counts)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    if wedge_img and st.button("Calcola Wedge Angle"):
        try:
            ds = pydicom.dcmread(wedge_img)
            tag = (0x3002, 0x0011)

            if tag in ds:
                pixel_spacing_mm = [float(x) for x in ds[tag].value]
                pixel_spacing_cm = pixel_spacing_mm[0] / 10
            else:
                pixel_spacing_cm = 0.025  # fallback

            profile = get_profile(ds)
            D1, D2, left_idx, right_idx = find_dose_at_distance(profile, pixel_spacing_cm, wdistL)

            theta = calculate_theta(D1, D2, u, wdistL)
            diff_percent = abs(theta - nominal_angle) / nominal_angle * 100

            st.write(f"D1 (dose a -{wdistL/2} cm dal centro): **{D1:.2f}**")
            st.write(f"D2 (dose a +{wdistL/2} cm dal centro): **{D2:.2f}**")
            st.write(f"Angolo θ calcolato (valore assoluto): **{theta:.2f}°**")
            st.write(f"Differenza percentuale dall'angolo nominale: **{diff_percent:.2f}%**")

            if diff_percent <= tolerance_percent:
                st.success(f"RISULTATO: PASS (differenza entro ±{tolerance_percent}%)")
            else:
                st.error(f"RISULTATO: FAIL (differenza fuori tolleranza ±{tolerance_percent}%)")

            plot_profile(profile, left_idx, right_idx)

            # Generazione report PDF
            if utente.strip():
                risultati_wedge = (
                    f"D1 (dose a -{wdistL/2} cm dal centro): {D1:.2f}\n"
                    f"D2 (dose a +{wdistL/2} cm dal centro): {D2:.2f}\n"
                    f"Angolo wedge calcolato: {theta:.2f}°\n"
                    f"Differenza percentuale dall'angolo nominale: {diff_percent:.2f}%\n"
                    f"Risultato: {'PASS' if diff_percent <= tolerance_percent else 'FAIL'}"
                )



