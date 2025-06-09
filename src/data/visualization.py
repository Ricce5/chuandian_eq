import matplotlib.pyplot as plt

def plot_earthquake_2d(df, title="Earthquake Distribution 2D"):
    """
    绘制地震二维分布图：经度-纬度为坐标，震级控制大小，时间控制颜色。

    参数：
        df: DataFrame，包含 ["Longitude", "Latitude", "Magnitude", "Time"]
        title: 图像标题
    """
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        df["Longitude"],
        df["Latitude"],
        s=df["Magnitude"]**3,     # 点大小随震级增大
        c=df["Time"],             # 点颜色代表时间
        cmap="viridis",           
        alpha=0.7,
        edgecolors='k',
        linewidths=0.3
    )

    plt.colorbar(scatter, label="Time (year)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_earthquake_3d(df, title="3D Earthquake Visualization", color_by="Time"):
    """
    以3D形式绘制地震数据：经度-纬度-深度为坐标，震级控制大小，颜色代表指定特征（默认是时间）

    参数：
        df: DataFrame，包含 ["Longitude", "Latitude", "Depth_m", "Magnitude", color_by]
        title: 图像标题
        color_by: 控制点颜色的列名（默认是 'Time'）
    """

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘图
    sc = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        -df["Depth_m"] / 1000,
        s=df["Magnitude"]**3,
        c=df[color_by],
        cmap='viridis',
        alpha=0.7,
        edgecolors='k',
        linewidths=0.3
    )

    # 颜色条
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label(color_by)

    # 轴标签和标题
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth (km)")
    ax.set_title(title)

    ax.view_init(elev=20, azim=120)
    plt.tight_layout()
    plt.show()

    