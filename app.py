
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="Klasifikasi Hipertensi | LightGBM-SHAP",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "model_bundle.joblib"

st.title("Klasifikasi Hipertensi")
st.caption(
    "Prototipe penelitian berbasis LightGBM dan SHAP. "
    "Bukan alat diagnosis medis."
)

if not MODEL_PATH.exists():
    st.error("File model_bundle.joblib tidak ditemukan.")
    st.stop()


@st.cache_resource
def load_bundle(path, modified_time):
    return joblib.load(path)


bundle = load_bundle(
    str(MODEL_PATH),
    MODEL_PATH.stat().st_mtime,
)

model = bundle["model"]
mapping = bundle["mapping"]
feature_names = bundle["feature_names"]
target_mapping = bundle.get(
    "target_mapping",
    {"No": 0, "Yes": 1},
)

reverse_target = {
    value: key for key, value in target_mapping.items()
}


def get_label(value):
    """Mengubah kode target menjadi label kelas."""
    for code, label in reverse_target.items():
        if value == code:
            return label
    return str(value)


def get_yes_value():
    """Mencari kode target untuk kelas Yes."""
    for label, code in target_mapping.items():
        if str(label).strip().lower() == "yes":
            return code

    return 1


def get_categories(feature):
    """Mengambil kategori asli dari mapping."""
    return list(mapping.get(feature, {}).keys())


# Konfigurasi input fitur numerik.
numeric_features = {
    "Age": {
        "label": "Usia (tahun)",
        "min": 1.0,
        "max": 120.0,
        "value": 45.0,
        "step": 1.0,
    },
    "Salt_Intake": {
        "label": "Asupan garam (ikuti satuan dataset)",
        "min": 0.0,
        "max": 1000.0,
        "value": 5.0,
        "step": 0.5,
    },
    "Stress_Score": {
        "label": "Stress score",
        "min": 0.0,
        "max": 10.0,
        "value": 5.0,
        "step": 1.0,
    },
    "Sleep_Duration": {
        "label": "Durasi tidur (jam)",
        "min": 0.0,
        "max": 24.0,
        "value": 7.0,
        "step": 0.5,
    },
    "BMI": {
        "label": "BMI",
        "min": 0.0,
        "max": 100.0,
        "value": 25.0,
        "step": 0.5,
    },
}

categorical_features = {
    "BP_History": "Riwayat tekanan darah",
    "Medication": "Pengobatan",
    "Family_History": "Riwayat keluarga",
    "Exercise_Level": "Tingkat olahraga",
    "Smoking_Status": "Status merokok",
}


# =========================================================
# 2. MENU APLIKASI
# =========================================================

page = st.sidebar.radio(
    "Menu aplikasi",
    [
        "Prediksi Hipertensi",
        "Evaluasi Model",
        "Interpretasi SHAP Global",
    ],
)

st.sidebar.divider()
st.sidebar.write(
    f"Skenario: {bundle.get('scenario', 'Tidak dicantumkan')}"
)
st.sidebar.write(f"Jumlah fitur: {len(feature_names)}")


# =========================================================
# 3. HALAMAN PREDIKSI HIPERTENSI
# =========================================================

if page == "Prediksi Hipertensi":

    st.header("Masukan variabel")

    st.info(
        "Masukkan nilai sesuai definisi dan satuan dataset "
        "penelitian. Hasil model bukan diagnosis klinis."
    )

    input_values = {}

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_names):

            target_col = col1 if i % 2 == 0 else col2

            with target_col:

                if feature in numeric_features:

                    config = numeric_features[feature]

                    input_values[feature] = st.number_input(
                        config["label"],
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["value"],
                        step=config["step"],
                        key=f"input_{feature}",
                    )

                elif feature in categorical_features:

                    categories = get_categories(feature)

                    if not categories:
                        st.error(
                            f"Mapping kategori untuk {feature} "
                            "tidak ditemukan."
                        )
                        st.stop()

                    selected = st.selectbox(
                        categorical_features[feature],
                        categories,
                        key=f"input_{feature}",
                    )

                    input_values[feature] = mapping[feature][selected]

                else:
                    st.error(
                        f"Fitur {feature} belum memiliki "
                        "konfigurasi input. Periksa feature_names "
                        "dan mapping pada model bundle."
                    )
                    st.stop()

        submitted = st.form_submit_button(
            "Jalankan klasifikasi",
            type="primary",
        )

    if submitted:

        # Pastikan urutan fitur sama dengan saat training.
        input_df = pd.DataFrame(
            [[input_values[f] for f in feature_names]],
            columns=feature_names,
        )

        prediction = model.predict(input_df)[0]
        result_label = get_label(prediction)

        # -------------------------------------------------
        # HASIL PREDIKSI
        # -------------------------------------------------

        st.subheader("Hasil klasifikasi model")

        if str(result_label).strip().lower() in {
            "yes", "1", "hipertensi"
        }:
            st.warning(
                "Model mengklasifikasikan masukan sebagai: "
                "HIPERTENSI (YES)"
            )
        else:
            st.success(
                "Model mengklasifikasikan masukan sebagai: "
                "TIDAK HIPERTENSI (NO)"
            )

        # -------------------------------------------------
        # PROBABILITAS MODEL
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            probability_df = pd.DataFrame({
                "Kelas": [
                    get_label(c) for c in model.classes_
                ],
                "Probabilitas keluaran model": probabilities,
            })

            st.write(
                "Probabilitas keluaran model "
                "(bukan probabilitas diagnosis klinis):"
            )

            st.dataframe(
                probability_df.style.format({
                    "Probabilitas keluaran model": "{:.2%}"
                }),
                use_container_width=True,
                hide_index=True,
            )

        # =================================================
        # SHAP INDIVIDUAL
        # =================================================

        st.divider()
        st.subheader("SHAP Individual")
        st.write(
            "Grafik ini menjelaskan kontribusi fitur terhadap "
            "keluaran model untuk input yang baru saja diprediksi."
        )

        st.caption(
            "Nilai SHAP positif menunjukkan dorongan ke kelas "
            "YES pada skala keluaran yang dijelaskan. Nilai negatif "
            "menunjukkan dorongan ke arah sebaliknya."
        )

        try:

            # Buat explainer dari model LightGBM.
            explainer = shap.TreeExplainer(model)

            # Hitung SHAP untuk satu data input.
            try:
                shap_output = explainer.shap_values(
                    input_df,
                    check_additivity=False,
                )
            except TypeError:
                shap_output = explainer.shap_values(input_df)

            yes_value = get_yes_value()
            classes = list(model.classes_)

            if yes_value in classes:
                yes_index = classes.index(yes_value)
            else:
                raise ValueError(
                    "Kelas YES tidak ditemukan pada model.classes_. "
                    "Periksa target_mapping pada model bundle."
                )

            # Menangani beberapa format keluaran SHAP.
            if isinstance(shap_output, list):

                # Format SHAP lama: satu array per kelas.
                shap_array = np.asarray(
                    shap_output[yes_index]
                )[0]

            else:

                # Format SHAP baru dapat berupa Explanation.
                if hasattr(shap_output, "values"):
                    shap_array = np.asarray(shap_output.values)
                else:
                    shap_array = np.asarray(shap_output)

                if shap_array.ndim == 3:

                    # Bentuk: sampel x fitur x kelas.
                    shap_array = shap_array[0, :, yes_index]

                elif shap_array.ndim == 2:

                    # Bentuk: sampel x fitur.
                    shap_array = shap_array[0]

                    # Untuk output biner 2D, TreeExplainer
                    # LightGBM umumnya menjelaskan kelas positif
                    # (kelas kedua) pada skala margin.
                    if len(classes) == 2 and yes_index == 0:
                        shap_array = -shap_array

                elif shap_array.ndim == 1:
                    pass

                else:
                    raise ValueError(
                        "Format keluaran SHAP tidak dikenali."
                    )

            shap_array = np.asarray(shap_array).reshape(-1)

            if len(shap_array) != len(feature_names):
                raise ValueError(
                    "Jumlah nilai SHAP tidak sama dengan jumlah "
                    "fitur. Periksa versi SHAP dan model bundle."
                )

            # Susun tabel kontribusi individual.
            individual_df = pd.DataFrame({
                "Fitur": feature_names,
                "Nilai input": [
                    input_values[f] for f in feature_names
                ],
                "Nilai SHAP": shap_array,
            })

            individual_df["Arah kontribusi"] = np.where(
                individual_df["Nilai SHAP"] > 0,
                "Mendorong ke YES",
                np.where(
                    individual_df["Nilai SHAP"] < 0,
                    "Mendorong menjauhi YES",
                    "Kontribusi nol",
                ),
            )

            individual_df["Absolut SHAP"] = (
                individual_df["Nilai SHAP"].abs()
            )

            # Urutkan berdasarkan besarnya kontribusi.
            individual_df = individual_df.sort_values(
                "Absolut SHAP",
                ascending=False,
            )

            top_n = min(10, len(individual_df))

            plot_df = individual_df.head(top_n).sort_values(
                "Nilai SHAP",
                ascending=True,
            )

            # Grafik kontribusi SHAP.
            fig, ax = plt.subplots(
                figsize=(10, max(4, 0.5 * len(plot_df)))
            )

            colors = [
                "#D95F5F" if value > 0 else "#4285B4"
                for value in plot_df["Nilai SHAP"]
            ]

            ax.barh(
                plot_df["Fitur"],
                plot_df["Nilai SHAP"],
                color=colors,
            )

            ax.axvline(
                0,
                color="black",
                linewidth=0.8,
            )

            ax.set_xlabel(
                "Nilai SHAP (skala keluaran model)"
            )
            ax.set_ylabel("Fitur")
            ax.set_title(
                "Kontribusi Fitur pada Prediksi Individual"
            )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Ringkasan arah kontribusi.
            st.markdown("**Ringkasan kontribusi fitur**")

            positive_count = int(
                (individual_df["Nilai SHAP"] > 0).sum()
            )
            negative_count = int(
                (individual_df["Nilai SHAP"] < 0).sum()
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Fitur mendorong ke YES",
                positive_count,
            )
            c2.metric(
                "Fitur mendorong menjauhi YES",
                negative_count,
            )

            st.caption(
                "Jumlah fitur menunjukkan arah kontribusi, bukan "
                "jumlah faktor risiko klinis."
            )

            # Tabel rincian.
            st.markdown("**Rincian nilai SHAP individual**")

            st.dataframe(
                individual_df[
                    [
                        "Fitur",
                        "Nilai input",
                        "Nilai SHAP",
                        "Arah kontribusi",
                    ]
                ].style.format({
                    "Nilai SHAP": "{:.6f}"
                }),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:

            st.error(
                "SHAP Individual belum dapat dihitung. "
                "Periksa detail error berikut:"
            )
            st.code(str(e))

        st.caption(
            "SHAP menjelaskan kontribusi fitur terhadap keluaran "
            "model, bukan membuktikan hubungan sebab-akibat atau "
            "menentukan diagnosis klinis."
        )

    # -----------------------------------------------------
    # INFORMASI MODEL
    # -----------------------------------------------------

    with st.expander("Informasi model"):

        st.write(
            "Skenario model:",
            bundle.get("scenario", "Tidak dicantumkan"),
        )
        st.write("Jumlah fitur:", len(feature_names))
        st.write("Fitur yang digunakan:")
        st.code(", ".join(feature_names))


# =========================================================
# 4. HALAMAN EVALUASI MODEL
# =========================================================

elif page == "Evaluasi Model":

    st.header("Evaluasi Model LightGBM")

    metrics = bundle.get("metrics")
    cm = bundle.get("confusion_matrix")

    if not metrics or cm is None:

        st.warning(
            "Metrik evaluasi atau confusion matrix belum "
            "tersimpan dalam model_bundle.joblib. "
            "Tambahkan hasil evaluasi aktual dari notebook "
            "ke dalam bundle, lalu unggah ulang file tersebut."
        )

    else:

        st.caption(
            "Metrik di bawah harus berasal dari evaluasi "
            "model pada data uji penelitian."
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}",
        )
        c2.metric(
            "Precision",
            f"{metrics['precision']:.2%}",
        )
        c3.metric(
            "Recall",
            f"{metrics['recall']:.2%}",
        )
        c4.metric(
            "F1-score",
            f"{metrics['f1_score']:.2%}",
        )

        st.subheader("Confusion Matrix")

        cm = np.asarray(cm)

        if cm.shape != (2, 2):
            st.error(
                "Confusion matrix harus berukuran 2 x 2 "
                "untuk klasifikasi biner."
            )
        else:

            fig, ax = plt.subplots(figsize=(6, 4))

            image = ax.imshow(cm, cmap="Blues")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Prediksi No", "Prediksi Yes"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Aktual No", "Aktual Yes"])

            ax.set_xlabel("Kelas prediksi")
            ax.set_ylabel("Kelas aktual")
            ax.set_title("Confusion Matrix LightGBM")

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=12,
                    )

            fig.colorbar(image, ax=ax)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Pastikan urutan kelas pada confusion matrix "
                "sesuai dengan urutan kelas saat evaluasi notebook."
            )


# =========================================================
# 5. HALAMAN SHAP GLOBAL
# =========================================================

elif page == "Interpretasi SHAP Global":

    st.header("Interpretasi Fitur dengan SHAP Global")

    shap_importance = bundle.get("shap_importance")

    if not shap_importance:

        st.warning(
            "Ringkasan SHAP global belum ditemukan dalam "
            "model_bundle.joblib. Simpan hasil perhitungan "
            "SHAP global dari notebook ke dalam bundle."
        )

    else:

        shap_df = pd.DataFrame(
            list(shap_importance.items()),
            columns=["Fitur", "Mean Absolute SHAP"],
        )

        shap_df["Mean Absolute SHAP"] = pd.to_numeric(
            shap_df["Mean Absolute SHAP"],
            errors="coerce",
        )

        shap_df = (
            shap_df.dropna()
            .sort_values(
                "Mean Absolute SHAP",
                ascending=False,
            )
        )

        if shap_df.empty:

            st.warning("Nilai SHAP global tidak tersedia.")

        else:

            st.write(
                "Peringkat kepentingan fitur berdasarkan rata-rata "
                "nilai absolut SHAP pada data yang dianalisis."
            )

            top_n = st.slider(
                "Jumlah fitur yang ditampilkan",
                min_value=1,
                max_value=len(shap_df),
                value=min(10, len(shap_df)),
            )

            plot_df = shap_df.head(top_n).sort_values(
                "Mean Absolute SHAP",
                ascending=True,
            )

            fig, ax = plt.subplots(
                figsize=(10, max(4, 0.45 * len(plot_df)))
            )

            ax.barh(
                plot_df["Fitur"],
                plot_df["Mean Absolute SHAP"],
            )

            ax.set_xlabel("Mean absolute SHAP value")
            ax.set_ylabel("Fitur")
            ax.set_title("Peringkat Kontribusi Fitur Global")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Tabel nilai SHAP global")

            st.dataframe(
                shap_df.style.format({
                    "Mean Absolute SHAP": "{:.6f}"
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "SHAP global menunjukkan kepentingan fitur secara "
                "keseluruhan, bukan arah pengaruh pada satu individu."
            )


# =========================================================
# 6. PERINGATAN
# =========================================================

st.divider()

st.caption(
    "Aplikasi ini merupakan prototipe akademik. "
    "Hasil klasifikasi, probabilitas model, dan interpretasi SHAP "
    "tidak menggantikan pemeriksaan tenaga kesehatan serta "
    "tidak boleh digunakan sebagai dasar perubahan pengobatan."
)
