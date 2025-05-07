import streamlit as st
import os

# -----------------------------
# 1. Konfigurasi Direktori Upload
# -----------------------------
UPLOAD_DIR = 'uploaded_images'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# -----------------------------
# 2. Inisialisasi Session State
# -----------------------------
if 'images' not in st.session_state:
    st.session_state.images = []

# -----------------------------
# 3. Styling Uploader jadi Tombol
# -----------------------------
css = """
<style>
/* Sembunyikan seluruh area drag & drop */
div[data-testid="stFileUploaderDropzone"] {
    border: none !important;
    padding: 0 !important;
    background: none !important;
}

/* Styling tombol <label> default */
div[data-testid="stFileUploaderDropzone"] > label {
    display: inline-block;
    padding: 0.5em 1.2em;
    margin: 0.5em 0;
    font-size: 1rem;
    font-weight: 500;
    color: white;
    background-color: #007bff;
    border-radius: 0.25em;
    cursor: pointer;
    text-align: center;
}

/* Hover effect */
div[data-testid="stFileUploaderDropzone"] > label:hover {
    background-color: #0056b3;
}

/* Sembunyikan teks drag & drop bawaan */
div[data-testid="stFileUploaderDropzone"] span[data-testid="stFileUploaderDropzoneText"] {
    display: none !important;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -----------------------------
# 4. Judul Aplikasi
# -----------------------------
st.title('Image Uploader & Manager')

# -----------------------------
# 5. File Uploader sebagai Tombol
# -----------------------------
uploaded_files = st.file_uploader(
    label='Pilih Gambar',
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
    accept_multiple_files=True,
    key='custom_uploader'
)

# -----------------------------
# 6. Proses Penyimpanan
# -----------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        # Jika belum tersimpan, simpan dan catat path-nya
        if file_path not in st.session_state.images:
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.images.append(file_path)
            st.success(f"✔️ Berhasil menyimpan: {uploaded_file.name}")

# -----------------------------
# 7. Tampilkan Daftar & Preview
# -----------------------------
st.header('Daftar Gambar yang Tersimpan')
if st.session_state.images:
    selected = st.selectbox('Pilih gambar untuk dilihat', st.session_state.images)
    st.image(selected, caption=os.path.basename(selected), use_column_width=True)
else:
    st.write('Belum ada gambar yang diupload.')
