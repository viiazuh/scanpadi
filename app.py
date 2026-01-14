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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="AI Kesehatan Padi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== LOAD MODEL (CACHE) ==================
@st.cache_resource
def load_ai_model():
    with st.spinner("⏳ Memuat model AI, mohon tunggu..."):
        return load_model("best_model.h5", compile=False)

model = load_ai_model()


LABELS = ["Blas", "Hawar Daun", "Tungro", "Sehat"]

# ================== DATABASE PENYAKIT ==================
DISEASE_KB = {
    "Blas": {
        "title": "Penyakit Blas (Rice Blast)",
        "cause": "Disebabkan oleh jamur Pyricularia oryzae.",
        "prevention": [
            "Gunakan varietas tahan blas",
            "Hindari pupuk nitrogen berlebihan",
            "Jaga jarak tanam",
            "Bersihkan gulma sekitar sawah"
        ],
        "treatment": [
            "Gunakan fungisida Tricyclazole",
            "Bakar jerami terinfeksi",
            "Rotasi varietas"
        ],
        "color": "red"
    },
    "Hawar Daun": {
        "title": "Hawar Daun Bakteri (Kresek)",
        "cause": "Disebabkan oleh bakteri Xanthomonas oryzae.",
        "prevention": [
            "Atur pengairan",
            "Kurangi pupuk urea",
            "Gunakan bibit sehat"
        ],
        "treatment": [
            "Gunakan bakterisida berbahan tembaga",
            "Keringkan sawah berkala"
        ],
        "color": "orange"
    },
    "Tungro": {
        "title": "Penyakit Tungro",
        "cause": "Virus yang ditularkan wereng hijau.",
        "prevention": [
            "Tanam serempak",
            "Kendalikan wereng",
            "Gunakan varietas tahan"
        ],
        "treatment": [
            "Cabut tanaman sakit",
            "Gunakan insektisida sistemik"
        ],
        "color": "red"
    },
    "Sehat": {
        "title": "Tanaman Padi Sehat",
        "cause": "Tanaman tumbuh optimal tanpa gejala penyakit.",
        "prevention": [
            "Pemupukan berimbang",
            "Pantau OPT rutin"
        ],
        "treatment": [
            "Tidak perlu pengobatan",
            "Lanjutkan perawatan rutin"
        ],
        "color": "green"
    }
}

# ================== PREPROCESS & PREDICT ==================
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

def predict_image(image):
    img = preprocess_image(image)
    pred = model.predict(img)
    idx = np.argmax(pred[0])
    confidence = float(np.max(pred[0])) * 100
    label = LABELS[idx]

    return {
        "hasil": label,
        "confidence": round(confidence, 1),
        "detail": DISEASE_KB[label]
    }

# ================== SESSION STATE ==================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

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
                    st.image(h["image"], use_container_width=True)
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

    col1, col2 = st.columns(2)
    image = None

    with col1:
        cam = st.camera_input("Ambil foto dari kamera")
        if cam:
            image = Image.open(cam)

    with col2:
        file = st.file_uploader("Atau upload gambar", type=["jpg","jpeg","png"])
        if file:
            image = Image.open(file)

    if image:
        st.image(image, caption="Preview", use_container_width=True)
        if st.button("🔍 Analisis", use_container_width=True):
            with st.spinner("Menganalisis gambar..."):
                result = predict_image(image)
                result["image"] = image
                st.session_state.result = result
                st.session_state.page = "result"

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"

# ================== HALAMAN RESULT ==================
def result_page():
    r = st.session_state.result
    info = r["detail"]

    st.image(r["image"], use_container_width=True)
    st.markdown(f"## {info['title']}")
    st.success(f"Akurasi: {r['confidence']}%")
    st.write(info["cause"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pencegahan")
        for p in info["prevention"]:
            st.write("•", p)

    with col2:
        st.markdown("### Pengobatan")
        for t in info["treatment"]:
            st.write("•", t)

    if st.button("💾 Simpan ke Riwayat", use_container_width=True):
        st.session_state.history.insert(0, {
            "title": info["title"],
            "confidence": r["confidence"],
            "image": r["image"],
            "date": datetime.now().strftime("%d %B %Y %H:%M")
        })
        st.session_state.page = "home"

    if st.button("🔄 Scan Lagi"):
        st.session_state.page = "scan"

# ================== ROUTER ==================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "scan":
    scan_page()
elif st.session_state.page == "result":
    result_page()
