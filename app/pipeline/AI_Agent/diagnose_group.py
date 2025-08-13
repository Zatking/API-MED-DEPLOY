import google.generativeai as genai
from PIL import Image
from typing import Optional
import logging
import os
from dotenv import load_dotenv
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def extract_label_name(label):
    if isinstance(label, tuple) and isinstance(label[0], (str, np.str_)):
        return str(label[0])
    elif isinstance(label, str):
        return label
    return str(label)  

def generate_diagnosis_with_gemini(description, sorted_labels):
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')  
        labels_only = [extract_label_name(label) for label in sorted_labels[:30]]
        labels_text = "\n".join([f"- {label}" for label in labels_only])
        print(f"label text: {labels_text} \n description: {description}")
        prompt = f"""
Bạn là **chuyên gia da liễu**.

---
## QUY TẮC TRẢ LỜI (BẮT BUỘC):
1. Chỉ trả về **một dòng duy nhất** chứa đúng tên nhóm bệnh:
   - fungal_infections
   - virus
   - bacterial_infections
   - parasitic_infections
2. Không viết hoa, không dấu câu, không giải thích, không từ gần nghĩa.
3. Trước khi trả lời, hãy phân tích nội bộ (ẩn) theo hướng dẫn. **Không in ra phần phân tích**.

---
## VÍ DỤ MẪU (FEW-SHOT):
Mô tả: Mảng da bong vảy trắng, bờ rõ, trung tâm lành, ngứa nhẹ ở thân mình.
Output: fungal_infections

Mô tả: Cụm mụn nước nhỏ, đau rát, phân bố theo dải, không có mủ vàng.
Output: virus

Mô tả: Nốt đỏ, sưng, đau, có mủ vàng, đóng mày, ở gần nang lông vùng cằm.
Output: bacterial_infections

Mô tả: Ngứa dữ dội về đêm, nhiều sẩn nhỏ và vết xước ở kẽ ngón tay.
Output: parasitic_infections

---
## DỮ LIỆU ĐẦU VÀO:
**Mô tả tổn thương da:**
\"\"\"{description}\"\"\"

**Danh sách bệnh AI dự đoán (ưu tiên từ trên xuống):**
{labels_text}

---
## NHIỆM VỤ:
1. Đọc kỹ mô tả và danh sách bệnh dự đoán.
2. So sánh từng đặc điểm với tiêu chí của 4 nhóm bệnh sau:

- fungal_infections: da khô, bong vảy mịn như phấn, bờ rõ/hình vòng, trung tâm lành hơn ngoại vi, không mủ, không sưng nóng đỏ, ngứa nhẹ/vừa, vị trí: da đầu, thân mình, chi, bẹn.
- virus: mụn nước/bóng nước/sẩn/loét nông, đau rát hoặc ngứa, phân bố dọc dây thần kinh hoặc đối xứng, không mủ vàng/vảy tiết dày.
- bacterial_infections: sưng nóng đỏ đau, có mủ, đóng mày vàng, hoại tử nhẹ, vảy trắng vàng, bề mặt sần sùi, bờ rõ/không đều, lan nhanh. Nếu tổn thương da đầu/mặt/cổ gần nang lông → ưu tiên nhóm này.
- parasitic_infections: ngứa nhiều (đặc biệt ban đêm), sẩn nhỏ/rãnh/vết xước, vị trí: kẽ ngón tay/bẹn/quanh rốn/mông, lây qua tiếp xúc.

---
## LUẬT LOẠI TRỪ:
- Không mủ/sưng/đỏ → loại bacterial_infections.
- Không ngứa dữ dội/đường hầm → loại parasitic_infections.
- Không ban đỏ dạng mụn nước/không phân bố đối xứng/dải → loại virus.
- Có vảy rõ, bờ rõ, không dịch → ưu tiên fungal_infections.

---
## LƯU Ý:
- Nếu tổn thương nhỏ (<5mm), rải rác, không vảy/mủ/sưng, không đối xứng → cân nhắc parasitic_infections.
- Không nhầm fungal nếu không có vảy rõ hoặc mảng lớn lan tỏa.
- Không nhầm virus nếu không có cụm sẩn/ban/mụn nước.

---
## CHỈ DẪN CUỐI:
Hãy suy luận nội bộ từng bước để chọn nhóm bệnh duy nhất phù hợp nhất.
**Chỉ in ra kết quả cuối cùng đúng định dạng quy định ở trên.**
"""       
        response = model.generate_content(prompt)
        try:
            caption = response.text.replace("\n", " ").strip().lower()
            return caption
        except Exception as e:
            return response
    except Exception as e:
        logging.error(f"❌ Lỗi khi tạo chẩn đoán với Gemini: {e}")
        return "unknown"