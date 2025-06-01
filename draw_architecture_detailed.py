import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_detailed_architecture_diagram():
    # Tạo figure và axis
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Định nghĩa vị trí và kích thước các khối chính
    main_block_width = 4
    main_block_height = 6
    spacing_x = 1
    spacing_y = 0.5
    start_x = 1
    start_y = 1
    
    # Vẽ khối chính: Thiết bị cảm biến (Edge Device)
    edge_x = start_x
    edge_y = start_y
    rect_edge = patches.Rectangle((edge_x, edge_y), main_block_width, main_block_height, 
                                 facecolor='#E0F7FA', edgecolor='#008000', linewidth=2)
    ax.add_patch(rect_edge)
    ax.text(edge_x + main_block_width/2, edge_y + main_block_height, 'Thiết bị cảm biến\n(Edge Device)', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#008000')
    
    # Thành phần con trong Thiết bị cảm biến
    sensor_height = 1.5
    pi_height = 2
    sensor_y = edge_y + main_block_height - 1 - sensor_height
    pi_y = edge_y + 0.5

    rect_sensor = patches.Rectangle((edge_x + 0.2, sensor_y), main_block_width - 0.4, sensor_height, 
                                  facecolor='white', edgecolor='gray')
    ax.add_patch(rect_sensor)
    ax.text(edge_x + main_block_width/2, sensor_y + sensor_height/2, 'Cảm biến mpu6050', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(edge_x + 0.3, sensor_y + sensor_height/2 - 0.3, '- Gia tốc 3 trục (accelX/Y/Z)\n- Vận tốc góc 3 trục (gyroX/Y/Z)', 
            ha='left', va='top', fontsize=9)

    rect_pi = patches.Rectangle((edge_x + 0.2, pi_y), main_block_width - 0.4, pi_height, 
                              facecolor='white', edgecolor='gray')
    ax.add_patch(rect_pi)
    ax.text(edge_x + main_block_width/2, pi_y + pi_height/2, 'Raspberry Pi', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(edge_x + 0.3, pi_y + pi_height/2 - 0.2, '- Thu thập dữ liệu từ cảm biến\n- Gửi dữ liệu qua WebSocket', 
            ha='left', va='top', fontsize=9)


    # Vẽ khối chính: Máy chủ xử lý
    server_x = edge_x + main_block_width + spacing_x
    server_y = start_y
    rect_server = patches.Rectangle((server_x, server_y), main_block_width + 1, main_block_height, 
                                  facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=2)
    ax.add_patch(rect_server)
    ax.text(server_x + (main_block_width + 1)/2, server_y + main_block_height, 'Máy chủ xử lý', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2196F3')
    
    # Thành phần con trong Máy chủ xử lý
    ws_in_height = 1.2
    buffer_height = 1.2
    model_height = 1.2
    ws_out_height = 1.5

    ws_in_y = server_y + main_block_height - 0.5 - ws_in_height
    buffer_y = ws_in_y - spacing_y - buffer_height
    model_y = buffer_y - spacing_y - model_height
    ws_out_y = model_y - spacing_y - ws_out_height

    rect_ws_in = patches.Rectangle((server_x + 0.2, ws_in_y), main_block_width + 1 - 0.4, ws_in_height, facecolor='white', edgecolor='gray')
    ax.add_patch(rect_ws_in)
    ax.text(server_x + (main_block_width + 1)/2, ws_in_y + ws_in_height/2, 'WebSocket Server (8080)\nNhận dữ liệu từ Raspberry Pi', 
            ha='center', va='center', fontsize=9)

    rect_buffer = patches.Rectangle((server_x + 0.2, buffer_y), main_block_width + 1 - 0.4, buffer_height, facecolor='white', edgecolor='gray')
    ax.add_patch(rect_buffer)
    ax.text(server_x + (main_block_width + 1)/2, buffer_y + buffer_height/2, 'Bộ đệm dữ liệu (window_size=100)\nLưu trữ 100 mẫu dữ liệu gần nhất (~2 giây)', 
            ha='center', va='center', fontsize=9)

    rect_model = patches.Rectangle((server_x + 0.2, model_y), main_block_width + 1 - 0.4, model_height, facecolor='white', edgecolor='gray')
    ax.add_patch(rect_model)
    ax.text(server_x + (main_block_width + 1)/2, model_y + model_height/2, 'Mô hình CNN-LSTM\nDự đoán 6 hành động: falling, jogging, jumping, sitting, standing, walking', 
            ha='center', va='center', fontsize=9)

    rect_ws_out = patches.Rectangle((server_x + 0.2, ws_out_y), main_block_width + 1 - 0.4, ws_out_height, facecolor='white', edgecolor='gray')
    ax.add_patch(rect_ws_out)
    ax.text(server_x + (main_block_width + 1)/2, ws_out_y + ws_out_height/2, 'Giao tiếp kết quả\n\n• WebSocket Server (8081): Gửi kết quả đến giao diện\n• HTTP Server (8000): Phục vụ giao diện web', 
            ha='center', va='center', fontsize=9)
            
    # Vẽ khối chính: Giao diện người dùng
    ui_x = server_x + main_block_width + 1 + spacing_x
    ui_y = start_y
    rect_ui = patches.Rectangle((ui_x, ui_y), main_block_width - 0.5, main_block_height, 
                               facecolor='#F3E5F5', edgecolor='#9C27B0', linewidth=2)
    ax.add_patch(rect_ui)
    ax.text(ui_x + (main_block_width - 0.5)/2, ui_y + main_block_height, 'Giao diện người dùng', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#9C27B0')

    # Thành phần con trong Giao diện người dùng
    browser_height = 1.5
    display_height = 3
    browser_y = ui_y + main_block_height - 0.5 - browser_height
    display_y = browser_y - spacing_y - display_height

    rect_browser = patches.Rectangle((ui_x + 0.2, browser_y), main_block_width - 0.5 - 0.4, browser_height, facecolor='white', edgecolor='gray')
    ax.add_patch(rect_browser)
    ax.text(ui_x + (main_block_width - 0.5)/2, browser_y + browser_height/2, 'Web Browser\nKết nối WebSocket và hiển thị kết quả', 
            ha='center', va='center', fontsize=9)

    rect_display = patches.Rectangle((ui_x + 0.2, display_y), main_block_width - 0.5 - 0.4, display_height, facecolor='white', edgecolor='gray')
    ax.add_patch(rect_display)
    ax.text(ui_x + (main_block_width - 0.5)/2, display_y + display_height/2, 'Hiển thị thời gian thực\n\n• Hành động hiện tại\n• Độ tin cậy của dự đoán\n• Cảnh báo khi phát hiện ngã', 
            ha='center', va='center', fontsize=9)

    # Vẽ khối chính: Hệ thống cảnh báo
    alert_x = server_x + (main_block_width + 1)/2 - (main_block_width - 1)/2 # center below server
    alert_y = start_y - 1.5 - main_block_height/3
    rect_alert = patches.Rectangle((alert_x, alert_y), main_block_width - 1, main_block_height/3, 
                                 facecolor='#FFEBEE', edgecolor='#F44336', linewidth=2)
    ax.add_patch(rect_alert)
    ax.text(alert_x + (main_block_width - 1)/2, alert_y + main_block_height/3, 'Hệ thống cảnh báo', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#F44336')
            
    # Thành phần con trong Hệ thống cảnh báo
    sms_height = main_block_height/3 - 1
    sms_y = alert_y + 0.5
    rect_sms = patches.Rectangle((alert_x + 0.2, sms_y), main_block_width - 1 - 0.4, sms_height, facecolor='white', edgecolor='gray')
    ax.add_patch(rect_sms)
    ax.text(alert_x + (main_block_width - 1)/2, sms_y + sms_height/2, 'SMS (Twilio)\nCảnh báo khẩn cấp đến điện thoại', 
            ha='center', va='center', fontsize=9)

    # Vẽ các mũi tên kết nối
    arrow_props = dict(arrowstyle='->', color='black', lw=1.5)
    
    # Raspberry Pi -> WebSocket Server (8080)
    start = (rect_pi.get_x() + rect_pi.get_width(), rect_pi.get_y() + rect_pi.get_height()/2)
    end = (rect_ws_in.get_x(), rect_ws_in.get_y() + rect_ws_in.get_height()/2)
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    ax.text((start[0] + end[0])/2, start[1] + 0.2, 'Gửi dữ liệu', ha='center', va='bottom', fontsize=8)

    # WebSocket Server (8080) -> Bộ đệm dữ liệu
    start = (rect_ws_in.get_x() + rect_ws_in.get_width()/2, rect_ws_in.get_y())
    end = (rect_buffer.get_x() + rect_buffer.get_width()/2, rect_buffer.get_y() + rect_buffer.get_height())
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    # ax.text((start[0] + end[0])/2, (start[1] + end[1])/2, 'Dữ liệu', ha='center', va='center', fontsize=8)
    
    # Bộ đệm dữ liệu -> Mô hình CNN-LSTM
    start = (rect_buffer.get_x() + rect_buffer.get_width()/2, rect_buffer.get_y())
    end = (rect_model.get_x() + rect_model.get_width()/2, rect_model.get_y() + rect_model.get_height())
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    # ax.text((start[0] + end[0])/2, (start[1] + end[1])/2, 'Dữ liệu', ha='center', va='center', fontsize=8)

    # Mô hình CNN-LSTM -> Giao tiếp kết quả
    start = (rect_model.get_x() + rect_model.get_width()/2, rect_model.get_y())
    end = (rect_ws_out.get_x() + rect_ws_out.get_width()/2, rect_ws_out.get_y() + rect_ws_out.get_height())
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    # ax.text((start[0] + end[0])/2, (start[1] + end[1])/2, 'Kết quả', ha='center', va='center', fontsize=8)

    # Giao tiếp kết quả -> Web Browser
    start = (rect_ws_out.get_x() + rect_ws_out.get_width(), rect_ws_out.get_y() + rect_ws_out.get_height()/2)
    end = (rect_browser.get_x(), rect_browser.get_y() + rect_browser.get_height()/2)
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    ax.text((start[0] + end[0])/2, start[1] + 0.2, 'Gửi kết quả', ha='center', va='bottom', fontsize=8)

    # Giao tiếp kết quả -> SMS (Twilio)
    start = (rect_ws_out.get_x() + rect_ws_out.get_width()/2, rect_ws_out.get_y())
    end = (rect_sms.get_x() + rect_sms.get_width()/2, rect_sms.get_y() + rect_sms.get_height())
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    ax.text((start[0] + end[0])/2, (start[1] + end[1])/2 - 0.3, 'Cảnh báo', ha='center', va='center', fontsize=8)

    # Cài đặt giới hạn trục và ẩn trục
    ax.set_xlim(0, ui_x + main_block_width - 0.5 + 1)
    ax.set_ylim(alert_y - 1, start_y + main_block_height + 1)
    ax.axis('off')
    
    # Thêm tiêu đề
    plt.title('Kiến trúc tổng quan hệ thống phát hiện ngã', pad=20, fontsize=16, fontweight='bold')
    
    # Lưu và hiển thị
    plt.savefig('system_architecture_detailed.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    create_detailed_architecture_diagram()
    print("Detailed diagram has been generated as 'system_architecture_detailed.png'") 