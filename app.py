import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import requests
from io import BytesIO

# ==============================
# 🌈 ตั้งค่าหน้าเพจ
# ==============================
st.set_page_config(
    page_title="🧫 Parasitic Egg Detection",
    page_icon="🧬",
    layout="wide",
)

# ==============================
# 🎨 CSS ตกแต่ง
# ==============================
page_bg = """
<style>
.stApp { background: linear-gradient(135deg, #e3f2fd, #e8f5e9); font-family: 'Segoe UI', sans-serif; }
.main { background-color: #ffffffcc; padding: 2rem; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); backdrop-filter: blur(8px); }
h1, h2, h3 { color: #1b5e20; font-weight: 600; }
.stFileUploader label { color: white !important; background: linear-gradient(90deg, #4CAF50, #2E7D32); padding: 10px 20px; border-radius: 8px; text-align: center; font-weight: bold; cursor: pointer; }
[data-testid="stSidebar"] { background-color: #e8f5e9 !important; border-right: 2px solid #c8e6c9; }
.result-card { background: white; padding: 1rem 1.5rem; border-radius: 12px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); margin-bottom: 1rem; border-left: 5px solid #4CAF50; }
footer {visibility: hidden;}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==============================
# ⚙️ โหลดโมเดล YOLO
# ==============================
model = YOLO("best (1).pt")  # เปลี่ยนเป็น path โมเดลของคุณ

# ==============================
# 🧠 Header
# ==============================
st.markdown("<h1 style='text-align:center;'>🧬 Parasitic Egg Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#2e7d32;'>ระบบตรวจจับสัตว์น้ำจากภาพด้วย YOLOv8</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# 🎛 Sidebar
# ==============================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80)
st.sidebar.title("⚙️ การตั้งค่า")

# เพิ่ม key ให้ slider เพื่อป้องกัน duplicate element error
conf_threshold = st.sidebar.slider(
    "ระดับความมั่นใจ (Confidence)",
    0.1, 1.0, 0.3, 0.05,
    key="conf_slider"
)

# เลือก source ของภาพ
input_mode = st.sidebar.radio("เลือกแหล่งข้อมูล", ["Upload", "Folder", "URL"], key="input_mode")
st.sidebar.info("ปรับระดับความมั่นใจเพื่อกรองผลลัพธ์")

# ==============================
# 📤 รับภาพ
# ==============================
col1, col2 = st.columns([1, 2])
images_to_check = []

with col1:
    if input_mode == "Upload":
        uploaded_file = st.file_uploader("📸 อัปโหลดภาพตรวจสอบ", type=["jpg", "jpeg", "png"], key="upload_file")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"ภาพต้นฉบับ: {uploaded_file.name}", use_column_width=True)
            images_to_check.append((uploaded_file.name, image))
    
    elif input_mode == "Folder":
        folder_path = st.text_input("ใส่ path ของโฟลเดอร์ภาพ", key="folder_path")
        if folder_path and os.path.exists(folder_path):
            img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            st.write(f"พบ {len(img_files)} ภาพในโฟลเดอร์")
            for f in img_files:
                img_path = os.path.join(folder_path, f)
                image = Image.open(img_path).convert("RGB")
                st.image(image, caption=f"ภาพจากโฟลเดอร์: {f}", use_column_width=True)
                images_to_check.append((f, image))
    
    elif input_mode == "URL":
        url_input = st.text_input("ใส่ URL ของภาพ", key="url_input")
        if url_input:
            try:
                response = requests.get(url_input)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                st.image(image, caption="ภาพจาก URL", use_column_width=True)
                images_to_check.append(("URL Image", image))
            except:
                st.error("โหลดภาพจาก URL ไม่สำเร็จ")

# ==============================
# 🔍 ตรวจจับภาพ
# ==============================
with col2:
    st.subheader("🔍 ผลการตรวจจับ")
    if images_to_check:
        for img_name, img in images_to_check:
            try:
                # Resize ภาพให้ YOLOv8
                img_resized = img.resize((640, 640))
                
                pred = model.predict(source=img_resized, conf=conf_threshold)
                result = pred[0]
                boxes = result.boxes
                names = result.names
                result_image = result.plot()
                
                if boxes is not None and boxes.cls.numel() > 0:
                    st.image(result_image, caption=f"📸 ผลตรวจ: {img_name}", use_column_width=True)
                    
                    for i in range(len(boxes.cls)):
                        class_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
                        label = names[class_id]
                        
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <h4>✅ พบ: {label}</h4>
                                <p>ความมั่นใจ: <b>{conf:.2%}</b></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.warning(f"🪱 ไม่พบสัตว์น้ำในภาพ: {img_name}")
            except Exception as e:
                st.error(f"❌ ตรวจภาพ {img_name} ไม่สำเร็จ: {e}")
    else:
        st.info("⬅️ กรุณาอัปโหลดภาพ / ใส่โฟลเดอร์ / URL ก่อนตรวจจับ")

# ==============================
# 📌 Footer
# ==============================
st.markdown("<br><hr><p style='text-align:center;color:gray;'>© 2025 Parasitic Detection Dashboard | YOLOv8 + Streamlit</p>", unsafe_allow_html=True)
