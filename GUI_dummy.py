import gradio as gr
from PIL import Image, ImageDraw

def predict_xray_dummy(image_path):
    if image_path is None:
        return "Mohon unggah gambar X-ray.", 0.0, None, "Belum Ada Prediksi"

    # 1. Load gambar
    original_image = Image.open(image_path).convert("RGB") # Pastikan RGB untuk pewarnaan

    # 2. Bounding Box(nantinya diganti dari output ML)
    img_width, img_height = original_image.size
    dummy_bbox = [img_width * 0.3, img_height * 0.3, img_width * 0.7, img_height * 0.7]

    # Gambar bounding box di atas gambar asli
    draw = ImageDraw.Draw(original_image)
    draw.rectangle(dummy_bbox, outline="red", width=3)

    # 3. Persentase Keyakinan(nanti dari output ML)
    dummy_confidence = 0.85

    # 4. Klasifikasi(nanti dari output ML)
    dummy_classification = "Terdeteksi TBC (Dummy)"

    return (
        f"Gambar diproses!",
        dummy_confidence,
        original_image, # Gambar dengan bounding box
        dummy_classification
    )

demo = gr.Interface(
    fn=predict_xray_dummy,
    inputs=gr.Image(type="filepath", label="Unggah Gambar X-ray Paru-paru (PNG)"),
    outputs=[
        "text",
        gr.Number(label="Persentase Keyakinan (%)"),
        gr.Image(type="pil", label="Hasil Analisis X-ray (dengan Bounding Box)"),
        gr.Textbox(label="Klasifikasi") \
    ],
    title="Sistem Prediksi TBC dari X-ray Paru-paru",
    description="Unggah citra X-ray paru-paru Anda untuk mendapatkan prediksi TBC, persentase keyakinan, dan lokasi lesi."
)

demo.launch()
