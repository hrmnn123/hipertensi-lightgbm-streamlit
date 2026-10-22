
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Klasifikasi Hipertensi | LightGBM-SHAP",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "model_bundle.joblib"

st.title("Klasifikasi Hipertensi")
st.caption(
    "Prototipe penelitian berbasis LightGBM dan SHAP — "
    "bukan alat diagnosis medis."
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
    "target_mapping", {"No": 0, "Yes": 1}
)

reverse_target = {
    value: key for key, value in target_mapping.items()
}


def get_categories(feature):
    return list(mapping.get(feature, {}).keys())


def get_label(value):
    return reverse_target.get(value, str(value))


page = st.sidebar.radio(
    "Menu aplikasi",
    [
        "Prediksi Hipertensi",
        "Evaluasi Model",
        "Interpretasi SHAP",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    f"Skenario model: {bundle.get('scenario', 'Tidak dicantumkan')}"
)
st.sidebar.caption(f"Jumlah fitur: {len(feature_names)}")


# =========================================================
# HALAMAN 1: PREDIKSI
# =========================================================

if page == "Prediksi Hipertensi":

    st.header("Prediksi Hipertensi")

    st.info(
        "Masukkan nilai sesuai definisi dan satuan pada dataset "
        "penelitian. Hasil model bukan diagnosis klinis."
    )

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

    input_values = {}

    with st.form("prediction_form"):
        st.subheader("Masukan variabel")

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_names):
            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                if feature in numeric_features:
                    config = numeric_features[feature]

                    value = st.number_input(
                        config["label"],
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["value"],
                        step=config["step"],
                        key=f"input_{feature}",
                    )

                    input_values[feature] = value

                elif feature in categorical_features:
                    categories = get_categories(feature)

                    if not categories:
                        st.warning(
                            f"Kategori untuk {feature} "
                            "tidak ditemukan dalam mapping."
                        )
                        continue

                    selected = st.selectbox(
                        categorical_features[feature],
                        categories,
                        key=f"input_{feature}",
                    )

                    input_values[feature] = mapping[feature][selected]

                else:
                    st.warning(
                        f"Fitur {feature} belum memiliki "
                        "konfigurasi input."
                    )

        submitted = st.form_submit_button(
            "Jalankan klasifikasi",
            type="primary",
        )

    if submitted:
        missing = [
            feature for feature in feature_names
            if feature not in input_values
        ]

        if missing:
            st.error(
                "Input belum tersedia untuk fitur: "
                + ", ".join(missing)
            )
            st.stop()

        input_df = pd.DataFrame(
            [[input_values[f] for f in feature_names]],
            columns=feature_names,
        )

        prediction = model.predict(input_df)[0]
        result_label = get_label(prediction)

        st.subheader("Hasil klasifikasi model")

        if str(result_label).lower() in {
            "yes", "1", "hipertensi"
        }:
            st.warning(
                "Model mengklasifikasikan masukan sebagai: HIPERTENSI"
            )
        else:
            st.success(
                "Model mengklasifikasikan masukan sebagai: "
                "TIDAK HIPERTENSI"
            )

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

    with st.expander("Informasi model"):
        st.write(
            "Skenario:",
            bundle.get("scenario", "Tidak dicantumkan"),
        )
        st.write("Jumlah fitur:", len(feature_names))
        st.write("Fitur yang digunakan:")
        st.code(", ".join(feature_names))


# =========================================================
# HALAMAN 2: EVALUASI MODEL
# =========================================================

elif page == "Evaluasi Model":

    st.header("Evaluasi Model LightGBM")

    metrics = bundle.get("metrics")
    cm = bundle.get("confusion_matrix")

    if not metrics or cm is None:
        st.warning(
            "Hasil evaluasi belum ditemukan di model_bundle.joblib. "
            "Pastikan file model terbaru yang memuat metrics dan "
            "confusion_matrix sudah diunggah ke GitHub."
        )
        st.stop()

    st.caption(
        "Metrik berikut berasal dari evaluasi yang disimpan "
        "dalam model bundle penelitian."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1-score", f"{metrics['f1_score']:.2%}")

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Aktual No", "Aktual Yes"],
        columns=["Prediksi No", "Prediksi Yes"],
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)

    ax.set_xlabel("Kelas prediksi")
    ax.set_ylabel("Kelas aktual")
    ax.set_title("Confusion Matrix LightGBM")

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(
                j, i, str(cm[i][j]),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Pastikan urutan label kelas 0 dan 1 sesuai dengan "
        "pemetaan target yang digunakan saat evaluasi notebook."
    )


# =========================================================
# HALAMAN 3: INTERPRETASI SHAP
# =========================================================

elif page == "Interpretasi SHAP":

    st.header("Interpretasi Fitur dengan SHAP")

    shap_importance = bundle.get("shap_importance")

    if not shap_importance:
        st.warning(
            "Ringkasan SHAP belum ditemukan dalam model bundle. "
            "Pastikan file terbaru yang memuat shap_importance "
            "sudah diunggah ke GitHub."
        )
        st.stop()

    shap_df = pd.DataFrame(
        list(shap_importance.items()),
        columns=["Fitur", "Mean Absolute SHAP"],
    )

    shap_df = shap_df.sort_values(
        "Mean Absolute SHAP",
        ascending=False,
    )

    st.write(
        "Peringkat fitur berdasarkan rata-rata nilai absolut SHAP. "
        "Nilai lebih besar menunjukkan kontribusi rata-rata "
        "yang lebih besar terhadap keluaran model."
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
        figsize=(9, max(4, 0.45 * len(plot_df)))
    )

    ax.barh(
        plot_df["Fitur"],
        plot_df["Mean Absolute SHAP"],
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Fitur")
    ax.set_title("Peringkat Kontribusi Fitur")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Tabel nilai SHAP")
    st.dataframe(
        shap_df.style.format({
            "Mean Absolute SHAP": "{:.6f}"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Grafik ini menunjukkan kepentingan fitur secara global, "
        "bukan arah pengaruh untuk setiap individu. SHAP menjelaskan "
        "keluaran model dan bukan hubungan sebab-akibat."
    )


st.divider()
st.caption(
    "Peringatan: aplikasi ini merupakan prototipe akademik. "
    "Hasil klasifikasi dan probabilitas model bukan diagnosis medis "
    "dan tidak boleh dijadikan dasar perubahan pengobatan."
)
