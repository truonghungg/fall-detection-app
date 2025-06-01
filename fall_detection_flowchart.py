import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Thiết lập kích thước và màu sắc
plt.figure(figsize=(10, 14))
ax = plt.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Định nghĩa các màu
colors = {
    'process': '#e2f0cb',       # Xanh lá nhạt
    'decision': '#ffcccc',      # Đỏ nhạt
    'input': '#cce5ff',         # Xanh dương nhạt
    'output': '#ffedcc',        # Cam nhạt
    'border': '#555555',        # Xám đậm
    'arrow': '#333333',         # Đen nhạt
    'highlight': '#ff6b6b',     # Đỏ nổi bật
    'special': '#c5b3e6'        # Tím nhạt
}

# Hàm vẽ hình chữ nhật
def draw_rect(x, y, width, height, text, color, fontsize=9):
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, 
                             edgecolor=colors['border'], facecolor=color, alpha=0.9)
    ax.add_patch(rect)
    
    # Xử lý văn bản nhiều dòng
    lines = text.split('\n')
    line_height = height / (len(lines) + 1)
    
    for i, line in enumerate(lines):
        ax.text(x + width/2, y + height - (i+1)*line_height, line, 
                ha='center', va='center', fontsize=fontsize, 
                fontweight='bold' if i==0 else 'normal')
    
    return (x + width/2, y + height/2)  # Trả về điểm trung tâm

# Hàm vẽ hình thoi
def draw_diamond(x, y, width, height, text, color, fontsize=9):
    diamond_path = Path([
        (x + width/2, y),            # Đỉnh trên
        (x + width, y + height/2),   # Đỉnh phải
        (x + width/2, y + height),   # Đỉnh dưới
        (x, y + height/2),           # Đỉnh trái
        (x + width/2, y)             # Trở lại đỉnh trên
    ])
    
    patch = patches.PathPatch(diamond_path, facecolor=color, 
                              edgecolor=colors['border'], linewidth=1.5, alpha=0.9)
    ax.add_patch(patch)
    
    # Xử lý văn bản nhiều dòng
    lines = text.split('\n')
    line_height = height / (len(lines) + 1)
    
    for i, line in enumerate(lines):
        ax.text(x + width/2, y + height/2 - (i - len(lines)/2) * line_height, line, 
                ha='center', va='center', fontsize=fontsize, 
                fontweight='bold' if i==0 else 'normal')
    
    return (x + width/2, y + height/2)  # Trả về điểm trung tâm

# Hàm vẽ hình bầu dục
def draw_oval(x, y, width, height, text, color, fontsize=9):
    ellipse = patches.Ellipse((x + width/2, y + height/2), width, height, 
                              facecolor=color, edgecolor=colors['border'], linewidth=1.5, alpha=0.9)
    ax.add_patch(ellipse)
    
    # Xử lý văn bản nhiều dòng
    lines = text.split('\n')
    line_height = height / (len(lines) + 1.5)
    
    for i, line in enumerate(lines):
        ax.text(x + width/2, y + height/2 - (i - len(lines)/2) * line_height, line, 
                ha='center', va='center', fontsize=fontsize, 
                fontweight='bold' if i==0 else 'normal')
    
    return (x + width/2, y + height/2)  # Trả về điểm trung tâm

# Hàm vẽ mũi tên
def draw_arrow(start, end, text=None, color=colors['arrow'], style='arc3,rad=0.0'):
    connection = patches.FancyArrowPatch(
        start, end, arrowstyle='->', color=color, linewidth=1.5,
        connectionstyle=style, shrinkA=0, shrinkB=0, mutation_scale=15
    )
    ax.add_patch(connection)
    
    # Thêm label cho mũi tên nếu có
    if text:
        # Tính vị trí giữa
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        offset_x = 0
        offset_y = -3 if start[1] > end[1] else 3
        
        if 'rad=' in style:
            rad = float(style.split('rad=')[1])
            if abs(rad) > 0:
                if rad > 0:
                    offset_x = -5
                else:
                    offset_x = 5
        
        ax.text(mid_x + offset_x, mid_y + offset_y, text, 
                ha='center', va='center', fontsize=8, bbox=dict(
                    facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'
                ))

# Vẽ sơ đồ luồng phát hiện té ngã

# Kích thước và vị trí chuẩn
box_width = 25
box_height = 6
x_center = 50  # Trung tâm của sơ đồ
margin = 12    # Khoảng cách giữa các box

# 1. Bắt đầu
start = draw_oval(x_center - box_width/2, 95, box_width, box_height, "BẮT ĐẦU", colors['special'], 10)

# 2. Thu thập dữ liệu IMU
data_collection = draw_rect(x_center - box_width/2, 85, box_width, box_height, 
                          "Thu thập dữ liệu IMU\nAccel X,Y,Z | Gyro X,Y,Z", colors['input'], 10)

# 3. Tiền xử lý dữ liệu
preprocessing = draw_rect(x_center - box_width/2, 75, box_width, box_height, 
                         "Tiền xử lý dữ liệu\nLàm mịn | Loại bỏ nhiễu | Chuẩn hóa", colors['process'], 10)

# 4. Phân đoạn dữ liệu
segmentation = draw_rect(x_center - box_width/2, 65, box_width, box_height, 
                        "Phân đoạn dữ liệu\nCửa sổ trượt: 100 mẫu, bước nhảy: 50", colors['process'], 10)

# 5. Trích xuất đặc trưng
feature_extraction = draw_rect(x_center - box_width/2, 55, box_width, box_height, 
                              "Trích xuất đặc trưng\n14 đặc trưng (cơ bản + nâng cao)", colors['process'], 10)

# 6. Dự đoán mô hình CNN-LSTM
prediction = draw_rect(x_center - box_width/2, 45, box_width, box_height, 
                      "Dự đoán mô hình CNN-LSTM\nPhân loại hoạt động + % độ tin cậy", colors['process'], 10)

# 7. Kiểm tra có phải té ngã không
check_fall = draw_diamond(x_center - box_width/2, 35, box_width, box_height, 
                         "Phát hiện\nté ngã?", colors['decision'], 10)

# 8. Kiểm tra độ tin cậy
check_confidence = draw_diamond(x_center + box_width/2 + 5, 35, box_width, box_height, 
                               "Độ tin cậy\n> 90%?", colors['decision'], 10)

# 9. Phân tích mẫu té ngã
analyze_pattern = draw_rect(x_center + box_width/2 + 5, 25, box_width, box_height, 
                           "Phân tích mẫu té ngã\nRơi tự do -> Va chạm -> Bất động", colors['process'], 10)

# 10. Kiểm tra chuỗi thời gian
check_sequence = draw_diamond(x_center + box_width/2 + 5, 15, box_width, box_height, 
                             "Xác nhận\nmẫu té ngã?", colors['decision'], 10)

# 11. Kích hoạt cảnh báo
alert = draw_rect(x_center - box_width/2, 5, box_width, box_height, 
                 "Kích hoạt cảnh báo\nGửi SMS qua Twilio", colors['highlight'], 10)

# 12. Tiếp tục giám sát
continue_monitoring = draw_rect(x_center - box_width/2, 15, box_width, box_height, 
                               "Tiếp tục giám sát\nHoạt động người dùng", colors['output'], 10)

# 13. Kết thúc
end = draw_oval(x_center - box_width/2, 25, box_width, box_height, "Trở về\ngiám sát", colors['special'], 10)

# Vẽ các mũi tên
draw_arrow(start, data_collection, "")
draw_arrow(data_collection, preprocessing, "Dữ liệu thô 50Hz")
draw_arrow(preprocessing, segmentation, "Dữ liệu đã lọc")
draw_arrow(segmentation, feature_extraction, "Cửa sổ dữ liệu")
draw_arrow(feature_extraction, prediction, "Vector đặc trưng")
draw_arrow(prediction, check_fall, "Kết quả phân loại")

# Nhánh trái - không té ngã
draw_arrow(check_fall, continue_monitoring, "Không", style='arc3,rad=-0.3')
draw_arrow(continue_monitoring, end, "")
draw_arrow(end, data_collection, "", style='arc3,rad=-0.5')

# Nhánh phải - nghi ngờ té ngã
draw_arrow(check_fall, check_confidence, "Có", style='arc3,rad=0')
draw_arrow(check_confidence, analyze_pattern, "Có")
draw_arrow(analyze_pattern, check_sequence, "")

# Nhánh phụ từ kiểm tra độ tin cậy
draw_arrow(check_confidence, continue_monitoring, "Không", style='arc3,rad=0.3')

# Nhánh phụ từ kiểm tra mẫu té ngã
draw_arrow(check_sequence, alert, "Có", style='arc3,rad=-0.3')
draw_arrow(check_sequence, continue_monitoring, "Không", style='arc3,rad=0.3')

# Nối từ cảnh báo về giám sát
draw_arrow(alert, continue_monitoring, "Sau khi gửi cảnh báo", style='arc3,rad=-0.3')

# Thêm các chú thích
ax.text(15, 40, "Phát hiện thời gian thực\n(<2 giây)", 
        ha='center', va='center', fontsize=9, fontweight='bold', 
        bbox=dict(facecolor='#ffffcc', alpha=0.9, boxstyle='round,pad=0.5'))

ax.text(85, 40, "Giảm cảnh báo giả\nAccuracy >95%", 
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(facecolor='#ffffcc', alpha=0.9, boxstyle='round,pad=0.5'))

# Thêm tiêu đề
plt.title("Sơ đồ luồng xử lý phát hiện té ngã và gửi cảnh báo", fontsize=14, fontweight='bold', pad=20)

# Ẩn các trục
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Thêm chú thích màu sắc
labels = [
    ('Dữ liệu đầu vào', colors['input']),
    ('Xử lý', colors['process']),
    ('Quyết định', colors['decision']),
    ('Kết quả/Đầu ra', colors['output']),
    ('Cảnh báo', colors['highlight']),
    ('Bắt đầu/Kết thúc', colors['special'])
]

handles = [patches.Patch(color=color, label=label) for label, color in labels]
ax.legend(handles=handles, loc='upper right', fontsize=8)

# Thêm thông tin bổ sung
footnote = """
* Thuật toán phát hiện té ngã sử dụng cả kết quả phân loại từ mô hình CNN-LSTM và phân tích các đặc điểm chuyển động.
* Mẫu té ngã điển hình bao gồm: Giai đoạn rơi tự do (gia tốc thấp) → Va chạm (gia tốc cao đột ngột) → Bất động sau té ngã.
* Hệ thống có thể phân biệt giữa té ngã và các chuyển động thay đổi đột ngột thông thường (như ngồi xuống nhanh).
"""
plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=8, 
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))

# Lưu và hiển thị
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('fall_detection_flowchart.png', dpi=300, bbox_inches='tight')
plt.show() 