import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Klasifikasi Hipertensi | LightGBM-SHAP",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "model_bundle.joblib"

st.title("Klasifikasi Hipertensi")
st.caption("Prototipe penelitian berbasis LightGBM — bukan alat diagnosis medis.")

if not MODEL_PATH.exists():
    st.error(
        "File model_bundle.joblib belum ditemukan. "
        "Jalankan kode penyimpanan model di notebook terlebih dahulu, "
        "lalu letakkan file tersebut satu folder dengan app.py."
    )
    st.stop()

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

bundle = load_bundle(MODEL_PATH)
model = bundle["model"]
mapping = bundle["mapping"]
feature_names = bundle["feature_names"]
target_mapping = bundle.get("target_mapping", {"No": 0, "Yes": 1})

# Encoding kategori mengikuti mapping yang dibuat pada notebook.
def original_categories(feature):
    enc = mapping.get(feature, {})
    return [label for label, code in enc.items() if label != "Medication" and code is not None]

st.info(
    "Masukkan nilai sesuai definisi dan satuan pada dataset penelitian. "
    "Prediksi ini hanya menunjukkan keluaran model pada data yang dimasukkan; "
    "hasilnya tidak menggantikan pemeriksaan atau saran tenaga kesehatan."
)

with st.form("prediction_form"):
    st.subheader("Masukan variabel")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45, step=1)
        salt = st.number_input("Asupan garam (ikuti satuan dataset)", min_value=0.0, value=5.0, step=0.5)
        stress = st.number_input("Stress score", min_value=0, max_value=10, value=5, step=1)
        sleep = st.number_input("Durasi tidur (jam)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.5)

    with col2:
        bp_options = list(mapping["BP_History"].keys())
        bp_history = st.selectbox("Riwayat tekanan darah", bp_options)

        med_mapping = mapping.get("Medication", {})
        medication_options = list(med_mapping.keys())
        medication = st.selectbox("Pengobatan", medication_options)

        family_options = list(mapping["Family_History"].keys())
        family_history = st.selectbox("Riwayat keluarga", family_options)

        exercise_options = list(mapping["Exercise_Level"].keys())
        exercise = st.selectbox("Tingkat olahraga", exercise_options)

        smoking_options = list(mapping["Smoking_Status"].keys())
        smoking = st.selectbox("Status merokok", smoking_options)

    submitted = st.form_submit_button("Jalankan klasifikasi", type="primary")

if submitted:
    input_values = {
        "Age": age,
        "Salt_Intake": salt,
        "Stress_Score": stress,
        "BP_History": mapping["BP_History"][bp_history],
        "Sleep_Duration": sleep,
        "BMI": bmi,
        "Medication": mapping["Medication"][medication],
        "Family_History": mapping["Family_History"][family_history],
        "Exercise_Level": mapping["Exercise_Level"][exercise],
        "Smoking_Status": mapping["Smoking_Status"][smoking],
    }

    # Jaga urutan dan nama fitur persis seperti saat model dilatih.
    input_df = pd.DataFrame(
        [[input_values[col] for col in feature_names]],
        columns=feature_names,
    )

    prediction = model.predict(input_df)[0]
    reverse_target = {value: key for key, value in target_mapping.items()}
    result_label = reverse_target.get(int(prediction), str(prediction))

    st.subheader("Hasil klasifikasi model")
    if str(result_label).lower() in {"yes", "1", "hipertensi"}:
        st.warning("Model mengklasifikasikan masukan sebagai: HIPERTENSI")
    else:
        st.success("Model mengklasifikasikan masukan sebagai: TIDAK HIPERTENSI")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_labels = [reverse_target.get(int(c), str(c)) for c in model.classes_]
        probability_df = pd.DataFrame({
            "Kelas": class_labels,
            "Probabilitas keluaran model": probabilities,
        })
        st.write("Probabilitas keluaran model (bukan probabilitas diagnosis klinis):")
        st.dataframe(
            probability_df.style.format({"Probabilitas keluaran model": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Peringatan: aplikasi ini adalah prototipe akademik. "
        "Keluaran model tidak boleh digunakan sebagai diagnosis atau dasar perubahan pengobatan."
    )

with st.expander("Informasi model"):
    st.write(f"Skenario model: {bundle.get('scenario', 'tidak dicantumkan')}")
    st.write(f"Jumlah fitur: {len(feature_names)}")
    st.write("Fitur yang digunakan:")
    st.code(", ".join(feature_names))
