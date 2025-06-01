from graphviz import Digraph

def create_architecture_diagram():
    # Tạo đồ thị có hướng
    dot = Digraph(comment='System Architecture')
    dot.attr(rankdir='LR', fontsize='12')  # Bố trí từ trái sang phải

    # Cảm biến
    dot.node('sensors', 'Cảm biến\n(MPU-6050)', shape='box', style='filled', color='lightyellow')

    # Vi xử lý
    dot.node('raspberry', 'Vi điều khiển\n(Raspberry Pi)', shape='box', style='filled', color='lightblue')

    # Mô hình học sâu
    dot.node('model', 'Mô hình học sâu\n(CNN-LSTM)', shape='box', style='filled', color='lightgreen')

    # Giao diện
    dot.node('ui', 'Giao diện người dùng\n(Web UI / SMS)', shape='box', style='filled', color='lightpink')

    # Kết nối các thành phần
    dot.edge('sensors', 'raspberry', label='Truyền dữ liệu\nthô (I2C)')
    dot.edge('raspberry', 'model', label='Xử lý và\ndự đoán hành động')
    dot.edge('model', 'ui', label='Xuất kết quả\nvà cảnh báo')

    # Xuất sơ đồ
    dot.render('system_architecture', format='png', cleanup=True)
    print("✅ Sơ đồ kiến trúc đã được tạo: 'system_architecture.png'")

if __name__ == '__main__':
    create_architecture_diagram()
