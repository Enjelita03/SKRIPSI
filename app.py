from flask import Flask, render_template, request, send_file, abort
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import time
import logging

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
CHART_FOLDER = "static/charts"
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


def create_charts(df_clean, df_full, ts):
    """
    Membuat 5 chart sesuai dengan penjelasan skripsi:
    1. Distribusi Nilai Penjualan (right-skewed)
    2. Tren Penjualan Harian (time series)
    3. Distribusi Channel Penjualan (pie chart)
    4. Produk Terlaris (bar chart)
    5. Visualisasi Deteksi Anomali (scatter plot)
    """
    charts = {}

    try:
        # ===== CHART 1: Distribusi Nilai Penjualan =====
        plt.figure(figsize=(12, 6))
        plt.hist(
            df_clean["total_sales"],
            bins=50,
            color="#3498db",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title(
            "Gambar 4.2 Distribusi Nilai Penjualan",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Total Penjualan", fontsize=11)
        plt.ylabel("Frekuensi", fontsize=11)
        plt.grid(axis="y", alpha=0.3)

        # Tambahkan anotasi untuk menunjukkan right-skewed
        mean_val = df_clean["total_sales"].mean()
        median_val = df_clean["total_sales"].median()
        plt.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:,.0f}",
        )
        plt.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:,.0f}",
        )
        plt.legend(fontsize=10)

        plt.tight_layout()
        chart_path = os.path.join(CHART_FOLDER, f"distribusi_penjualan_{ts}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["chart1"] = f"distribusi_penjualan_{ts}.png"
        logging.info(f"Chart 1 (Distribusi Penjualan) dibuat: {chart_path}")

        # ===== CHART 2: Tren Penjualan Harian =====
        df_sorted = df_clean.sort_values("date")
        daily_sales = df_sorted.groupby(df_sorted["date"].dt.date)["total_sales"].sum()

        plt.figure(figsize=(14, 6))
        plt.plot(
            daily_sales.index,
            daily_sales.values,
            color="#2ecc71",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.title(
            "Gambar 4.3 Tren Penjualan Harian", fontsize=14, fontweight="bold", pad=20
        )
        plt.xlabel("Tanggal", fontsize=11)
        plt.ylabel("Total Penjualan", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Highlight lonjakan
        max_sale_idx = daily_sales.argmax()
        max_sale_date = daily_sales.index[max_sale_idx]
        max_sale_value = daily_sales.values[max_sale_idx]
        plt.scatter(
            [max_sale_date], [max_sale_value], color="red", s=200, zorder=5, marker="*"
        )
        plt.annotate(
            f"Puncak: {max_sale_value:,.0f}",
            xy=(max_sale_date, max_sale_value),
            xytext=(10, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        plt.tight_layout()
        chart_path = os.path.join(CHART_FOLDER, f"tren_penjualan_harian_{ts}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["chart2"] = f"tren_penjualan_harian_{ts}.png"
        logging.info(f"Chart 2 (Tren Penjualan Harian) dibuat: {chart_path}")

        # ===== CHART 3: Distribusi Channel Penjualan =====
        channel_counts = df_clean["distribution_channel"].value_counts()
        colors = sns.color_palette("Set2", len(channel_counts))

        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(
            channel_counts.values,
            labels=channel_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 11},
        )

        # Highlight autotexts
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

        plt.title(
            "Gambar 4.4 Distribusi Channel Penjualan",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        chart_path = os.path.join(CHART_FOLDER, f"distribusi_channel_{ts}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["chart3"] = f"distribusi_channel_{ts}.png"
        logging.info(f"Chart 3 (Distribusi Channel) dibuat: {chart_path}")

        # ===== CHART 4: Produk Terlaris =====
        product_sales = (
            df_clean.groupby("product_type")["total_sales"]
            .sum()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(12, 6))
        bars = plt.barh(
            range(len(product_sales)),
            product_sales.values,
            color="#9b59b6",
            edgecolor="black",
            alpha=0.8,
        )
        plt.yticks(range(len(product_sales)), product_sales.index)
        plt.title("Gambar 4.5 Produk Terlaris", fontsize=14, fontweight="bold", pad=20)
        plt.xlabel("Total Penjualan", fontsize=11)
        plt.ylabel("Tipe Produk", fontsize=11)
        plt.grid(axis="x", alpha=0.3)

        # Tambahkan value labels pada bar
        for idx, (bar, value) in enumerate(zip(bars, product_sales.values)):
            plt.text(
                value,
                idx,
                f" {value:,.0f}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        chart_path = os.path.join(CHART_FOLDER, f"produk_terlaris_{ts}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["chart4"] = f"produk_terlaris_{ts}.png"
        logging.info(f"Chart 4 (Produk Terlaris) dibuat: {chart_path}")

        # ===== CHART 5: Visualisasi Deteksi Anomali =====
        plt.figure(figsize=(14, 7))

        # Urutkan data berdasarkan tanggal agar sumbu X (waktu) berurutan dengan benar
        df_sorted = df_full.sort_values("date").dropna(subset=["date", "total_sales"])

        # Pisahkan data normal dan anomali
        df_normal = df_sorted[df_sorted["anomaly"] == 0]
        df_anomali = df_sorted[df_sorted["anomaly"] == 1]

        # Plot normal data (Menggunakan tanggal di sumbu X)
        plt.scatter(
            df_normal["date"],
            df_normal["total_sales"],
            c="#3498db",
            label="Data Normal",
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

        # Plot anomali data (Menggunakan tanggal di sumbu X)
        plt.scatter(
            df_anomali["date"],
            df_anomali["total_sales"],
            c="#e74c3c",
            label="Data Anomali (Isolation Forest)",
            alpha=0.9,
            s=100,
            marker="X",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=5,  # Memastikan anomali selalu digambar di atas data normal
        )

        plt.title(
            "Gambar 4.8 Visualisasi Deteksi Anomali Penjualan",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Tanggal Transaksi", fontsize=11)
        plt.ylabel("Total Penjualan", fontsize=11)
        plt.legend(fontsize=11, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)  # Memutar teks tanggal agar tidak bertabrakan

        plt.tight_layout()
        chart_path = os.path.join(CHART_FOLDER, f"deteksi_anomali_{ts}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts["chart5"] = f"deteksi_anomali_{ts}.png"
        logging.info(f"Chart 5 (Deteksi Anomali) dibuat: {chart_path}")

        return charts

    except Exception as e:
        logging.error(f"Error saat membuat chart: {str(e)}")
        raise


@app.route("/")
def index():
    logging.info("Halaman utama diakses.")
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        logging.info("=== PROSES ANALISIS DIMULAI ===")

        if "file" not in request.files:
            logging.error("File tidak ditemukan")
            abort(400, description="File tidak ditemukan.")

        file = request.files["file"]

        if file.filename == "":
            logging.error("Nama file kosong")
            abort(400, description="Nama file kosong.")

        if not allowed_file(file.filename):
            logging.error("Format file tidak valid")
            abort(400, description="Format harus CSV/Excel.")

        safe_name = secure_filename(file.filename)
        ts = int(time.time())
        filepath = os.path.join(UPLOAD_FOLDER, f"{ts}_{safe_name}")
        file.save(filepath)

        logging.info(f"File berhasil diupload: {filepath}")

        # Load data
        ext = os.path.splitext(filepath.lower())[1]
        df = pd.read_csv(filepath) if ext == ".csv" else pd.read_excel(filepath)

        logging.info(f"Jumlah data awal: {len(df)}")

        # Validasi kolom
        required = {
            "date",
            "quantity_sold",
            "unit_price",
            "total_sales",
            "product_type",
            "distribution_channel",
        }
        missing = required - set(df.columns)

        if missing:
            logging.error(f"Kolom kurang: {missing}")
            abort(400, description=f"Kolom kurang: {sorted(missing)}")

        # Cleaning
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ["quantity_sold", "unit_price", "total_sales"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df_clean = df.dropna(
            subset=["date", "quantity_sold", "unit_price", "total_sales"]
        ).copy()

        logging.info(f"Data setelah drop NA: {len(df_clean)}")

        # Validasi nilai
        df_clean = df_clean[
            (df_clean["quantity_sold"] > 0)
            & (df_clean["unit_price"] > 0)
            & (df_clean["total_sales"] > 0)
        ]

        logging.info(f"Data setelah filter nilai valid: {len(df_clean)}")

        if len(df_clean) == 0:
            logging.error("Data kosong setelah cleaning")
            abort(400, description="Data kosong setelah cleaning.")

        # Feature Engineering
        df_clean["month"] = df_clean["date"].dt.month
        df_clean["day_of_week"] = df_clean["date"].dt.dayofweek
        df_clean["avg_price"] = df_clean["total_sales"] / df_clean["quantity_sold"]

        features = df_clean[
            [
                "quantity_sold",
                "unit_price",
                "total_sales",
                "month",
                "day_of_week",
                "avg_price",
            ]
        ]

        logging.info(f"Fitur yang digunakan: {list(features.columns)}")

        # Scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        logging.info("Scaling selesai")

        # Model
        # Model
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(features_scaled)

        logging.info("Model Isolation Forest selesai training")

        # Prediction
        df_clean["anomaly"] = model.predict(features_scaled)
        df_clean["anomaly"] = df_clean["anomaly"].map({1: 0, -1: 1})
        df_clean["anomaly_score"] = model.decision_function(features_scaled)

        logging.info("Prediksi anomaly selesai")

        # Gabung hasil ke dataframe asli
        df["anomaly"] = 0
        df["anomaly_score"] = None
        df.loc[df_clean.index, "anomaly"] = df_clean["anomaly"]
        df.loc[df_clean.index, "anomaly_score"] = df_clean["anomaly_score"]

        anomaly_df = df[df["anomaly"] == 1]

        logging.info(f"Jumlah anomaly ditemukan: {len(anomaly_df)}")

        # Buat Charts (5 chart sesuai skripsi)
        charts = create_charts(df_clean, df, ts)
        logging.info("Semua 5 chart berhasil dibuat")

        # Save hasil ke Excel
        result_path = os.path.join(RESULT_FOLDER, f"hasil_{ts}.xlsx")
        df.to_excel(result_path, index=False)

        logging.info(f"Hasil disimpan: {result_path}")

        # Statistik
        total = len(df)
        anomaly = len(anomaly_df)
        percent = round((anomaly / total) * 100, 2)

        logging.info(f"Persentase anomaly: {percent}%")
        logging.info("=== PROSES ANALISIS SELESAI ===")

        return render_template(
            "dashboard.html",
            total=total,
            anomaly=anomaly,
            percent=percent,
            charts=charts,
            result_file=f"hasil_{ts}.xlsx",
        )

    except Exception as e:
        logging.exception("ERROR TERJADI:")
        abort(500, description=str(e))


@app.route("/download")
def download():
    result_file = request.args.get("file")
    if not result_file:
        abort(400, description="File tidak dispesifikasikan.")

    file_path = os.path.join(RESULT_FOLDER, result_file)

    if not os.path.exists(file_path):
        abort(404, description="File tidak ditemukan.")

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
