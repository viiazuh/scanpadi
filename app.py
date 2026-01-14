import os
# ================== KONFIGURASI TENSORFLOW ==================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================== IMPORT ==================
import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

# --- IMPORT TENSORFLOW YANG LEBIH AMAN ---
import tensorflow as tf

# Kita definisikan shortcut agar code di bawah tidak perlu diubah banyak
# Ini menghindari error "No module named tensorflow.keras"
try:
    models = tf.keras.models
    layers = tf.keras.layers
    utils = tf.keras.utils
    preprocessing = tf.keras.preprocessing
except AttributeError:
    # Fallback untuk versi TF yang sangat baru/sangat lama
    import keras
    models = keras.models
    layers = keras.layers
    utils = keras.utils
    preprocessing = keras.preprocessing

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="AI Kesehatan Padi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== PERBAIKAN KOMPATIBILITAS MODEL ==================
# Class ini dimodifikasi untuk membuang SEMUA argumen config yang tidak dikenali
class FixedInputLayer(layers.InputLayer):
    def __init__(self, *args, **kwargs):
        # Daftar argumen yang sering bikin error
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
            # Menggunakan tf.keras.models.load_model
            return models.load_model("best_model.h5", compile=False, custom_objects={'InputLayer': FixedInputLayer})
        except Exception as e:
            st.error(f"Gagal memuat model. Error detail: {e}")
            return None

model = load_ai_model()

# ================== DATABASE LABELS & PENYAKIT ==================
MODEL_LABELS = ["Blas", "Hawar Daun", "Tungro", "Sehat"]

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
    # --- DUMMY DATA ---
    "Brown Spot": {
        "title": "Bercak Coklat (Brown Spot)",
        "cause": "Jamur Helminthosporium oryzae. Bercak coklat lonjong.",
        "prevention": ["Pemupukan Kalium (K)", "Benih sehat"],
        "treatment": ["Fungisida Difenokonazol"],
        "color": "brown"
    },
    "Leaf Scald": {
        "title": "Hawar Daun (Leaf Scald)",
        "cause": "Jamur Microdochium oryzae. Pola zonasi pada ujung daun.",
        "prevention": ["Hindari kepadatan tanam tinggi"],
        "treatment": ["Fungisida Benomyl"],
        "color": "orange"
    },
    "Narrow Brown Spot": {
        "title": "Bercak Coklat Sempit",
        "cause": "Jamur Cercospora janseana. Bercak coklat kemerahan sempit.",
        "prevention": ["Gunakan varietas tahan"],
        "treatment": ["Fungisida Propikonazol"],
        "color": "brown"
    },
    "Neck Blast": {
        "title": "Busuk Leher (Neck Blast)",
        "cause": "Fase lanjut dari Blas yang menyerang leher malai.",
        "prevention": ["Semprot fungisida saat bunting"],
        "treatment": ["Fungisida Isoprothiolane"],
        "color": "red"
    },
    "Rice Hispa": {
        "title": "Hama Putih Palsu (Rice Hispa)",
        "cause": "Kumbang Dicladispa armigera. Daun tampak putih transparan.",
        "prevention": ["Pangkas daun bertelur"],
        "treatment": ["Insektisida Klorpirifos"],
        "color": "gray"
    },
    "Sheath Blight": {
        "title": "Hawar Pelepah",
        "cause": "Jamur Rhizoctonia solani. Bercak pada pelepah.",
        "prevention": ["Atur jarak tanam (legowo)"],
        "treatment": ["Fungisida Validamycin"],
        "color": "orange"
    }
}

# ================== FUNGSI UTAMA ==================
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    # Menggunakan tf.keras.utils atau preprocessing
    try:
        x = utils.img_to_array(image)
    except AttributeError:
        x = preprocessing.image.img_to_array(image)
        
    x = np.expand_dims(x, axis=0)
    return x / 255.0

def predict_image(image):
    if model is None:
        return {"hasil": "Error", "confidence": 0, "detail": DISEASE_KB["Sehat"]}
    
    img = preprocess_image(image)
    pred = model.predict(img)
    idx = np.argmax(pred[0])
    confidence = float(np.max(pred[0])) * 100
    
    label = MODEL_LABELS[idx] if idx < len(MODEL_LABELS) else MODEL_LABELS[0]
    
    return {
        "hasil": label,
        "confidence": round(confidence, 1),
        "detail": DISEASE_KB.get(label, DISEASE_KB["Sehat"])
    }

# ================== SESSION STATE ==================
if "page" not in st.session_state: st.session_state.page = "home"
if "history" not in st.session_state: st.session_state.history = []
if "result" not in st.session_state: st.session_state.result = None

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("🛠 Mode Developer")
    st.info("Simulasi data untuk presentasi.")
    sim_disease = st.selectbox("Pilih Penyakit (Simulasi)", list(DISEASE_KB.keys()))
    if st.button("Tampilkan Info Dummy"):
        dummy_result = {
            "hasil": sim_disease,
            "confidence": 100.0,
            "detail": DISEASE_KB[sim_disease],
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
        st.warning("⚠️ Model AI sedang bermasalah. Silakan gunakan 'Mode Developer' di sidebar.")
    
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
    st.markdown(f"## {info['title']}")
    st.caption(f"Hasil Analisis: {r['hasil']} | Kepercayaan: {r['confidence']}%")
    st.info(f"**Penyebab:** {info['cause']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛡 Pencegahan")
        for p in info["prevention"]: st.write(f"✅ {p}")
    with col2:
        st.markdown("### 💊 Pengobatan")
        for t in info["treatment"]: st.write(f"💊 {t}")

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
if st.session_state.page == "home": home_page()
elif st.session_state.page == "scan": scan_page()
elif st.session_state.page == "result": result_page()
