# Submission Analisis Data Bike Sharing

Proyek analisis data menggunakan **Bike Sharing Dataset (Capital Bikeshare, Washington D.C., 2011–2012)**.

## Struktur Submission

submission/
├── dashboard/
│   └── dashboard.py
├── data/
│   ├── day.csv
│   └── hour.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt

## Isi Analisis

Notebook mendokumentasikan seluruh tahapan analisis: business understanding, data gathering, data assessment, cleaning, exploratory data analysis, visualization, analisis lanjutan, dan conclusion & recommendation.

### Pertanyaan Bisnis
1. Bagaimana pola rata-rata jumlah rental sepeda berdasarkan bulan dan jam selama 2011–2012, serta bagaimana perbedaannya antara working day dan non-working day untuk mendukung perencanaan ketersediaan sepeda?
2. Sejauh mana kondisi cuaca dan lingkungan—temperatur, kelembapan, kecepatan angin, dan kondisi cuaca—berhubungan dengan jumlah rental sepeda selama 2011–2012?

### Analisis Lanjutan Non-ML
Karena dataset tidak memiliki customer ID, nilai transaksi, maupun koordinat lokasi, RFM dan geospatial analysis tidak relevan untuk dataset ini. Sebagai teknik analisis lanjutan digunakan **binning/manual grouping** untuk segmentasi demand:
- Rendah
- Sedang
- Tinggi
- Sangat Tinggi

Selain itu, jam dibagi menjadi empat kelompok waktu untuk membantu interpretasi kebutuhan operasional.

## Menjalankan Dashboard Lokal

```bash
pip install -r requirements.txt
streamlit run dashboard/dashboard.py
```

Dashboard membaca data dari folder `data/`, sehingga struktur direktori harus dipertahankan.

## Deploy ke Streamlit Community Cloud

1. Upload seluruh isi folder `submission` ke repository GitHub.
2. Di Streamlit Community Cloud pilih repository tersebut.
3. Pilih branch yang digunakan.
4. Set main file path menjadi `dashboard/dashboard.py`.
5. Deploy aplikasi.
6. Setelah mendapatkan URL `*.streamlit.app`, masukkan URL tersebut ke `url.txt`.

## Catatan Reproducibility

Dataset yang digunakan adalah `day.csv` dan `hour.csv`. Kolom `casual` dan `registered` tidak digunakan sebagai prediktor model karena keduanya merupakan komponen langsung dari `cnt` dan dapat menyebabkan target leakage.