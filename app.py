import os
# ================== KONFIGURASI ENV ==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

# ================== IMPORT TENSORFLOW (Versi 2.13.0 Aman) ==================
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="AI Kesehatan Padi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_ai_model():
    with st.spinner("⏳ Memuat model AI..."):
        try:
            # Di TF 2.13.0, load_model sangat stabil untuk .h5
            model = load_model("best_model.h5", compile=False)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None

model = load_ai_model()

# ================== DATABASE DATABASE ==================
MODEL_LABELS = ["Blas", "Hawar Daun", "Tungro", "Sehat"]

DISEASE_KB = {
    # --- DETEKSI AI AKTIF ---
    "Blas": {
        "title": "Penyakit Blas (Leaf Blast)",
        "cause": "Jamur Pyricularia oryzae. Bercak berbentuk belah ketupat.",
        "prevention": ["Gunakan varietas tahan", "Hindari pupuk N berlebih"],
        "treatment": ["Fungisida Tricyclazole", "Bakar jerami sisa"],
        "color": "red"
    },
    "Hawar Daun": {
        "title": "Hawar Daun Bakteri (Bacterial Leaf Blight)",
        "cause": "Bakteri Xanthomonas oryzae. Daun mengering dari ujung (kresek).",
        "prevention": ["Atur pengairan", "Kurangi Urea"],
        "treatment": ["Bakterisida tembaga", "Keringkan sawah berkala"],
        "color": "orange"
    },
    "Tungro": {
        "title": "Penyakit Tungro",
        "cause": "Virus dari wereng hijau. Tanaman kerdil dan kuning.",
        "prevention": ["Tanam serempak", "Kendalikan wereng hijau"],
        "treatment": ["Cabut tanaman sakit", "Insektisida sistemik"],
        "color": "red"
    },
    "Sehat": {
        "title": "Tanaman Sehat (Healthy)",
        "cause": "Kondisi optimal.",
        "prevention": ["Lanjutkan perawatan rutin"],
        "treatment": ["-"],
        "color": "green"
    },
    # --- DUMMY DATA ---
    "Brown Spot": {
        "title": "Bercak Coklat (Brown Spot)",
        "cause": "Jamur Helminthosporium oryzae.",
        "prevention": ["Pemupukan Kalium", "Benih sehat"],
        "treatment": ["Fungisida Difenokonazol"],
        "color": "brown"
    },
    "Rice Hispa": {
        "title": "Hama Putih Palsu (Rice Hispa)",
        "cause": "Kumbang Dicladispa armigera.",
        "prevention": ["Pangkas daun bertelur"],
        "treatment": ["Insektisida Klorpirifos"],
        "color": "gray"
    }
}

# ================== FUNGSI PREDIKSI ==================
def predict_image(image):
    if model is None:
        return None
    
    # Preprocess
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    # Predict
    pred = model.predict(x)
    idx = np.argmax(pred[0])
    confidence = float(np.max(pred[0])) * 100
    
    # Mapping label
    if idx < len(MODEL_LABELS):
        label = MODEL_LABELS[idx]
    else:
        label = "Sehat"
    
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
    sim_disease = st.selectbox("Pilih Penyakit", list(DISEASE_KB.keys()))
    if st.button("Tampilkan Info Dummy"):
        st.session_state.result = {
            "hasil": sim_disease,
            "confidence": 100.0,
            "detail": DISEASE_KB[sim_disease],
            "image": Image.new('RGB', (200, 200), color=DISEASE_KB[sim_disease]['color'])
        }
        st.session_state.page = "result"
        st.rerun()

# ================== HALAMAN ==================
def home_page():
    st.markdown("## 🌾 Dashboard Kesehatan Padi")
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button("➕ Scan Baru", use_container_width=True):
            st.session_state.page = "scan"
    
    if not st.session_state.history:
        st.info("Belum ada riwayat.")
    else:
        for i, h in enumerate(st.session_state.history):
            with st.container():
                c1, c2 = st.columns([1,3])
                with c1: st.image(h["image"], use_column_width=True)
                with c2:
                    st.markdown(f"**{h['title']}**")
                    st.caption(f"{h['confidence']}% - {h['date']}")
                    if st.button("Hapus", key=f"del_{i}"):
                        st.session_state.history.pop(i)
                        st.rerun()
            st.divider()

def scan_page():
    st.markdown("## 📸 Scan Tanaman")
    
    img_file = st.camera_input("Ambil Foto")
    upl_file = st.file_uploader("Upload Foto", type=["jpg","png","jpeg"])
    
    image = None
    if img_file: image = Image.open(img_file)
    elif upl_file: image = Image.open(upl_file)
    
    if image:
        st.image(image, caption="Preview", use_column_width=True)
        if st.button("🔍 Analisis AI", use_container_width=True):
            if model is None:
                st.error("Model gagal dimuat. Cek logs.")
            else:
                with st.spinner("Menganalisis..."):
                    res = predict_image(image)
                    res["image"] = image
                    st.session_state.result = res
                    st.session_state.page = "result"
                    st.rerun()
    
    if st.button("⬅ Kembali"): st.session_state.page = "home"

def result_page():
    if not st.session_state.result:
        st.session_state.page = "home"
        st.rerun()
        return

    r = st.session_state.result
    info = r["detail"]
    
    st.image(r["image"], use_column_width=True)
    st.markdown(f"## {info['title']}")
    st.success(f"Confidence: {r['confidence']}%")
    st.info(f"Penyebab: {info['cause']}")
    
    c1, c2 = st.columns(2)
    with c1: 
        st.write("### 🛡 Pencegahan")
        for p in info["prevention"]: st.write(f"- {p}")
    with c2:
        st.write("### 💊 Pengobatan")
        for t in info["treatment"]: st.write(f"- {t}")

    if st.button("Simpan", use_container_width=True):
        st.session_state.history.insert(0, {
            "title": info["title"],
            "confidence": r["confidence"],
            "image": r["image"],
            "date": datetime.now().strftime("%d-%m %H:%M")
        })
        st.session_state.page = "home"
        st.rerun()

    if st.button("Scan Lagi"):
        st.session_state.page = "scan"
        st.rerun()

# ================== ROUTING ==================
if st.session_state.page == "home": home_page()
elif st.session_state.page == "scan": scan_page()
elif st.session_state.page == "result": result_page()
