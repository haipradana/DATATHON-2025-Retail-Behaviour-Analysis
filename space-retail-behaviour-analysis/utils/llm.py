import os
import base64
import pandas as pd
import google.generativeai as genai
from IPython.display import display, Markdown

def encode_image_to_base64(image_path):
    """Convert image to base64 encoding for Gemini"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_llm_insights(metrics_data, archetype_df, output_dir):
    """Generate AI insights based on analysis results including charts and CSV data"""
    try:
        # Ambil API key dari environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "API key belum dikonfigurasi. Tambahkan GEMINI_API_KEY di pengaturan Space."
        
        # Inisialisasi Gemini dengan model vision
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Baca file grafik
        heatmap_path = os.path.join(output_dir, 'heatmap.png')
        dwell_path = os.path.join(output_dir, 'dwell_time_chart.png')
        journey_path = os.path.join(output_dir, 'journey_chart.png')
        
        # Baca file CSV
        try:
            action_summary = pd.read_csv(os.path.join(output_dir, 'action_summary.csv'))
            journey_analysis = pd.read_csv(os.path.join(output_dir, 'journey_analysis.csv'))
            dwell_time = pd.read_csv(os.path.join(output_dir, 'average_dwell_time.csv'))
        except Exception as e:
            print(f"Warning: Tidak bisa membaca beberapa file CSV: {e}")
            action_summary = None
            journey_analysis = None
            dwell_time = None
        
        # Siapkan prompt teks dengan semua data CSV
        text_prompt = f"""
        Sebagai pakar analitik retail, analisis data perilaku supermarket berikut dan berikan insight strategis:
        
        Arketype Perilaku Pelanggan:
        {archetype_df.to_string()}
        
        """
        
        # Tambahkan data dari CSV jika tersedia
        if action_summary is not None:
            text_prompt += f"\nAksi Pelanggan (Jumlah):\n{action_summary.to_string()}\n"
        
        if journey_analysis is not None:
            text_prompt += f"\nAnalisis Perilaku Pengunjung:\n{journey_analysis.to_string()}\n"
        
        if dwell_time is not None:
            text_prompt += f"\nRata-rata Waktu Tinggal di Rak:\n{dwell_time.to_string()}\n"
            
        text_prompt += """
        Berdasarkan semua data di atas dan grafik yang diberikan (heatmap pergerakan pelanggan, 
        grafik waktu tinggal, dan analisis journey/perilaku pengunjung), berikan:
        
        1. Analisis komprehensif tentang perilaku pelanggan di toko
        2. Identifikasi area atau rak yang berkinerja baik dan yang bermasalah
        3. 3-5 rekomendasi spesifik dan dapat ditindaklanjuti untuk meningkatkan tata letak toko
        4. Strategi untuk meningkatkan konversi pada rak dengan minat tinggi
        5. Saran untuk meningkatkan customer experience secara keseluruhan
        
        PENTING: Berikan jawaban dalam Bahasa Indonesia yang profesional dan mudah dipahami, namun jangan telalu kaku.
        JANGAN gunakan tanda bintang (*) atau tanda pagar (#) untuk formatting.
        Gunakan nomor (1, 2, 3) atau huruf (a, b, c) untuk daftar.
        Hindari format Markdown..
        Anda juga disarankan menjawab langsung ke pointnya saja tanpa basa-basi.
        """
        
        # Siapkan input multi-modal (teks + gambar)
        contents = [text_prompt]
        
        # Tambahkan gambar jika ada
        try:
            contents.append({
                "mime_type": "image/png",
                "data": encode_image_to_base64(heatmap_path)
            })
            contents.append({
                "mime_type": "image/png", 
                "data": encode_image_to_base64(dwell_path)
            })
            contents.append({
                "mime_type": "image/png",
                "data": encode_image_to_base64(journey_path)
            })
        except Exception as e:
            print(f"Warning: Tidak bisa membaca beberapa file gambar: {e}")
        
        # Hasilkan respons
        response = model.generate_content(contents)
        return response.text
        
    except Exception as e:
        return f"Error menghasilkan insights: {str(e)}"