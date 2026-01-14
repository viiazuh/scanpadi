import os
# ================== KONFIGURASI TENSORFLOW ==================
# Memaksa menggunakan implementasi Keras legacy untuk kompatibilitas
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================== IMPORT ==================
import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import tensorflow as tf

# --- BAGIAN INI DIPERBAIKI (SIMPLIFIED IMPORTS) ---
# Kita gunakan jalur import standar untuk TF 2.15.0
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import InputLayer
    # Coba import img_to_array dari utils (lokasi baru) atau preprocessing (lokasi lama)
    try:
        from tensorflow.keras.utils import img_to_array
    except ImportError:
        from tensorflow.keras.preprocessing.image import img_to_array
except ImportError as e:
    st.error(f"Error Import TensorFlow: {e}")
    st.stop()

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="AI Kesehatan Padi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== PERBAIKAN KOMPATIBILITAS MODEL (ANTI-ERROR) ==================
# Class ini dimodifikasi untuk membuang SEMUA argumen config yang tidak dikenali
class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Daftar argumen yang sering bikin error di TensorFlow versi baru
        ignore_keys = ['batch_shape', 'batch_input_shape', 'sparse', 'ragged', 'dtype']
        for k in ignore_keys:
            if k in kwargs:
                kwargs.pop(k)
        super().__init__(*args, **kwargs)

# ================== LOAD MODEL (CACHE) ==================
@st.cache_resource
def load_ai_model():
    with st.spinner("⏳ Memuat model AI, mohon tunggu..."):
        try:
            # Load model dengan custom object FixedInputLayer
            return load_model("best_model.h5", compile=False, custom_objects={'InputLayer': FixedInputLayer})
        except Exception as e:
            st.error(f"Gagal memuat model. Pastikan requirements.txt berisi 'tensorflow==2.15.0'. Error: {e}")
            return None

model = load_ai_model()

# ================== DATABASE LABELS & PENYAKIT ==================

# LABEL ASLI MODEL (Hanya 4 ini yang bisa diprediksi AI saat ini)
MODEL_LABELS = ["Blas", "Hawar Daun", "Tungro", "Sehat"]

# DATABASE LENGKAP (Termasuk Dummy Data untuk penyakit yang belum didukung AI)
DISEASE_KB = {
    # --- DETEKSI AI AKTIF ---
    "Blas": {
        "title": "Penyakit Blas (Leaf Blast)",
        "cause": "Jamur Pyricularia oryzae. Bercak berbentuk belah ketupat.",
        "prevention": ["Gunakan varietas tahan", "Hindari pupuk N berlebih", "Jaga jarak tanam"],
        "treatment": ["Fungisida Tricyclazole", "Bakar jerami sisa"],
        "color": "red"
    },
    "Hawar Daun": {
        "title": "Hawar Daun Bakteri (Bacterial Leaf Blight)",
        "cause": "Bakteri Xanthomonas oryzae. Daun mengering dari ujung (kresek).",
        "prevention": ["Atur pengairan (jangan tergenang terus)", "Kurangi Urea"],
        "treatment": ["Bakterisida tembaga", "Keringkan sawah berkala"],
        "color": "orange"
    },
    "Tungro": {
        "title": "Penyakit Tungro",
        "cause": "Virus dari wereng hijau. Tanaman kerdil dan daun kuning-oranye.",
        "prevention": ["Tanam serempak", "Kendalikan wereng hijau"],
        "treatment": ["Cabut tanaman sakit (eradikasi)", "Insektisida sistemik"],
        "color": "red"
    },
    "Sehat": {
        "title": "Tanaman Sehat (Healthy)",
        "cause": "Kondisi optimal, tidak ada serangan OPT.",
        "prevention": ["Lanjutkan pemupukan berimbang", "Monitoring rutin"],
        "treatment": ["Tidak perlu tindakan"],
        "color": "green"
    },

    # --- DUMMY DATA (Manual Input / Belum ada di Model AI) ---
    "Brown Spot": {
        "title": "Bercak Coklat (Brown Spot)",
        "cause": "Jamur Helminthosporium oryzae. Bercak coklat lonjong seperti biji wijen.",
        "prevention": ["Pemupukan Kalium (K) yang cukup", "Gunakan benih sehat"],
        "treatment": ["Fungisida berbahan aktif Difenokonazol"],
        "color": "brown"
    },
    "Leaf Scald": {
        "title": "Hawar Daun (Leaf Scald)",
        "cause": "Jamur Microdochium oryzae. Pola zonasi pada ujung daun.",
        "prevention": ["Hindari kepadatan tanam tinggi", "Sanitasi lahan"],
        "treatment": ["Fungisida Benomyl atau Carbendazim"],
        "color": "orange"
    },
    "Narrow Brown Spot": {
        "title": "Bercak Coklat Sempit (Narrow Brown Spot)",
        "cause": "Jamur Cercospora janseana. Bercak coklat kemerahan sempit memanjang.",
        "prevention": ["Gunakan varietas tahan", "Pemupukan KCL"],
        "treatment": ["Fungisida Propikonazol"],
        "color": "brown"
    },
    "Neck Blast": {
        "title": "Busuk Leher (Neck Blast)",
        "cause": "Fase lanjut dari Blas yang menyerang leher malai (patah leher).",
        "prevention": ["Semprot fungisida saat bunting dan berbunga penuh"],
        "treatment": ["Fungisida Isoprothiolane atau Tricyclazole"],
        "color": "red"
    },
    "Rice Hispa": {
        "title": "Hama Putih Palsu (Rice Hispa)",
        "cause": "Kumbang Dicladispa armigera. Daun tampak putih transparan bergaris.",
        "prevention": ["Pangkas daun yang bertelur", "Jebakan cahaya"],
        "treatment": ["Insektisida Klorpirifos"],
        "color": "gray"
    },
    "Sheath Blight": {
        "title": "Hawar Pelepah (Sheath Blight)",
        "cause": "Jamur Rhizoctonia solani. Bercak pada pelepah dekat air.",
        "prevention": ["Atur jarak tanam (legowo)", "Kurangi kelembaban"],
        "treatment": ["Fungisida Validamycin atau Azoxystrobin"],
        "color": "orange"
    }
}

# ================== FUNGSI UTAMA ==================
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

def predict_image(image):
    # Jika model gagal load, kembalikan error
    if model is None:
        return {"hasil": "Error", "confidence": 0, "detail": {"title": "Model Gagal Load", "cause": "Cek logs", "prevention": [], "treatment": [], "color": "black"}}
    
    img = preprocess_image(image)
    pred = model.predict(img)
    idx = np.argmax(pred[0])
    confidence = float(np.max(pred[0])) * 100
    
    # Pastikan index sesuai dengan LABEL MODEL (Cuma 4)
    if idx < len(MODEL_LABELS):
        label = MODEL_LABELS[idx]
    else:
        label = MODEL_LABELS[0] # Fallback jika index aneh
    
    return {
        "hasil": label,
        "confidence": round(confidence, 1),
        "detail": DISEASE_KB.get(label, DISEASE_KB["Sehat"])
    }

# ================== SESSION STATE ==================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None

# ================== SIDEBAR (DEBUG & SIMULASI) ==================
with st.sidebar:
    st.header("🛠 Mode Developer")
    st.info("Gunakan mode ini untuk melihat data penyakit yang belum disupport AI.")
    
    # Dropdown untuk memilih SEMUA penyakit (Termasuk Dummy)
    sim_disease = st.selectbox("Pilih Penyakit (Simulasi)", list(DISEASE_KB.keys()))
    
    if st.button("Tampilkan Info Dummy"):
        # Buat dummy result seolah-olah hasil scan
        dummy_result = {
            "hasil": sim_disease,
            "confidence": 100.0,
            "detail": DISEASE_KB[sim_disease],
            # Pakai gambar placeholder atau null jika simulasi
            "image": Image.new('RGB', (200, 200), color=DISEASE_KB[sim_disease]['color'])
        }
        st.session_state.result = dummy_result
        st.session_state.page = "result"
        st.rerun()

# ================== HALAMAN HOME ==================
def home_page():
    st.markdown("## 🌾 Dashboard Kesehatan Tanaman")
    
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("➕ Scan Baru", use_container_width=True):
            st.session_state.page = "scan"

    if not st.session_state.history:
        st.info("Belum ada data scan.")
    else:
        for i, h in enumerate(st.session_state.history):
            with st.container():
                c1, c2 = st.columns([1,3])
                with c1:
                    st.image(h["image"], use_column_width=True)
                with c2:
                    st.markdown(f"**{h['title']}**")
                    st.write(f"Akurasi: {h['confidence']}%")
                    st.caption(h["date"])
                    if st.button("🗑 Hapus", key=f"del_{i}"):
                        st.session_state.history.pop(i)
                        st.rerun()
            st.divider()

# ================== HALAMAN SCAN ==================
def scan_page():
    st.markdown("## 📸 Scan Tanaman")
    
    if model is None:
        st.error("Model AI gagal dimuat. Cek logs di Streamlit Cloud.")
    
    col1, col2 = st.columns(2)
    image = None

    with col1:
        cam = st.camera_input("Ambil foto")
        if cam: image = Image.open(cam)
    with col2:
        file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])
        if file: image = Image.open(file)

    if image:
        st.image(image, caption="Preview", use_column_width=True)
        # Tombol disable jika model error
        if st.button("🔍 Analisis AI", use_container_width=True, disabled=(model is None)):
            with st.spinner("Menganalisis..."):
                result = predict_image(image)
                result["image"] = image
                st.session_state.result = result
                st.session_state.page = "result"
                st.rerun()

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"

# ================== HALAMAN RESULT ==================
def result_page():
    if st.session_state.result is None:
        st.session_state.page = "home"
        st.rerun()
        return

    r = st.session_state.result
    info = r["detail"]

    st.image(r["image"], use_column_width=True)
    
    # Warna badge berdasarkan confidence/tipe
    st.markdown(f"## {info['title']}")
    st.caption(f"Hasil Analisis: {r['hasil']} | Kepercayaan: {r['confidence']}%")
    
    st.info(f"**Penyebab:** {info['cause']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛡 Pencegahan")
        for p in info["prevention"]:
            st.write(f"✅ {p}")

    with col2:
        st.markdown("### 💊 Pengobatan")
        for t in info["treatment"]:
            st.write(f"💊 {t}")

    # Tombol simpan
    if st.button("💾 Simpan ke Riwayat", use_container_width=True):
        st.session_state.history.insert(0, {
            "title": info["title"],
            "confidence": r["confidence"],
            "image": r["image"],
            "date": datetime.now().strftime("%d %B %Y %H:%M")
        })
        st.session_state.page = "home"
        st.rerun()

    if st.button("🔄 Scan Lagi"):
        st.session_state.page = "scan"
        st.rerun()

# ================== ROUTING ==================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "scan":
    scan_page()
elif st.session_state.page == "result":
    result_page()
