import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    # Tạo figure và axis
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Định nghĩa vị trí và kích thước các box
    box_width = 2
    box_height = 1
    spacing = 1
    start_x = 1
    
    # Vẽ các box
    boxes = []
    labels = ['Cảm biến', 'Raspberry Pi', 'Mô hình', 'Web UI/Gmail']
    colors = ['#FFB6C1', '#98FB98', '#87CEEB', '#DDA0DD']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = start_x + i * (box_width + spacing)
        box = patches.Rectangle((x, 1), box_width, box_height, 
                              facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(box)
        boxes.append(box)
        
        # Thêm nhãn
        ax.text(x + box_width/2, 1 + box_height/2, label,
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Vẽ các mũi tên
    arrow_props = dict(arrowstyle='->', color='black', lw=1.5)
    for i in range(len(boxes)-1):
        start = (boxes[i].get_x() + box_width, boxes[i].get_y() + box_height/2)
        end = (boxes[i+1].get_x(), boxes[i+1].get_y() + box_height/2)
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
        
        # Thêm nhãn cho mũi tên
        mid_x = (start[0] + end[0]) / 2
        mid_y = start[1] + 0.2
        labels = ['Dữ liệu', 'Xử lý', 'Kết quả']
        ax.text(mid_x, mid_y, labels[i], ha='center', va='bottom', fontsize=8)
    
    # Cài đặt giới hạn trục và ẩn trục
    ax.set_xlim(0, start_x + len(boxes) * (box_width + spacing))
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Thêm tiêu đề
    plt.title('Kiến trúc của hệ thống', pad=20, fontsize=14, fontweight='bold')
    
    # Lưu và hiển thị
    plt.savefig('system_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    create_architecture_diagram()
    print("Diagram has been generated as 'system_architecture.png'") 