import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_simclr_architecture(save_path=None, figsize=(15, 10)):  # 增加圖形寬度
    """
    Draw SimCLR neural network architecture diagram, showing connections between layers
    
    Args:
        save_path (str, optional): Path to save the image, if not specified the image will be displayed
        figsize (tuple, optional): Size of the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layer names and descriptions
    layers = [
        ("Input", "Input Image (3, 224, 224)"),
        ("MobileNetV3", "Feature Extraction (576)"),
        ("Linear 1", "Fully Connected Layer (576 → 512)"),
        ("BatchNorm1d", "Batch Normalization"),
        ("ReLU", "Activation Function"),
        ("Dropout", "Dropout (0.2)"),
        ("Linear 2", "Fully Connected Layer (512 → 512)"),
        ("BatchNorm1d", "Batch Normalization"),
        ("ReLU", "Activation Function"),
        ("Dropout", "Dropout (0.2)"),
        ("Linear 3", "Fully Connected Layer (512 → 128)"),
        ("Normalize", "L2 Normalization"),
        ("Output", "Feature Vector (128)")
    ]
    
    # Set drawing range
    ax.set_xlim(0, 120)  # 增加畫布寬度
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define visual style for layers
    layer_height = 5
    layer_space = (100 - len(layers) * layer_height) / (len(layers) + 1)
    layer_width = 35  # 稍微縮小方塊寬度
    start_x = 25  # 調整起始位置
    text_offset = 8  # 增加文字與方塊的間距
    
    # Draw each layer
    boxes = []
    for i, (name, desc) in enumerate(layers):
        y_pos = 100 - (i + 1) * (layer_height + layer_space)
        
        # Set different colors for different layer types
        color = '#3498db'  # 默認藍色
        if 'MobileNetV3' in name:
            color = '#e74c3c'  # 紅色
        elif 'Linear' in name:
            color = '#2ecc71'  # 綠色
        elif 'BatchNorm' in name or 'Dropout' in name:
            color = '#f39c12'  # 橙色
        elif 'Input' in name or 'Output' in name:
            color = '#9b59b6'  # 紫色
        
        # Draw layer rectangle
        rect = patches.Rectangle((start_x, y_pos), layer_width, layer_height, 
                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        boxes.append((start_x, y_pos, layer_width, layer_height))
        
        # Add layer name and description with more space between them
        ax.text(start_x + layer_width/2, y_pos + layer_height/2, name,
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(start_x + layer_width + text_offset, y_pos + layer_height/2, desc,
               va='center', fontsize=10)
    
    # 繪製層之間的連接
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2] / 2
        y1 = boxes[i][1]
        x2 = boxes[i+1][0] + boxes[i+1][2] / 2
        y2 = boxes[i+1][1] + boxes[i+1][3]
        ax.arrow(x1, y1, 0, y2-y1-0.5, head_width=1, head_length=0.5, fc='black', ec='black', alpha=0.7)
    
    plt.title('SimCLR Network Architecture', fontsize=16, fontweight='bold')
    
    # 添加圖例
    legend_elements = [
        patches.Patch(facecolor='#9b59b6', alpha=0.7, label='Input/Output Layers'),
        patches.Patch(facecolor='#e74c3c', alpha=0.7, label='Backbone Network (MobileNetV3)'),
        patches.Patch(facecolor='#2ecc71', alpha=0.7, label='Fully Connected Layers'),
        patches.Patch(facecolor='#f39c12', alpha=0.7, label='Normalization/Dropout')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if save_path:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # 設置輸出目錄和文件名
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成架構圖
    plot_simclr_architecture(save_path=os.path.join(output_dir, "model_architecture.png"))
