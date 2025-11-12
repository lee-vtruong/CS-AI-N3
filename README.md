# Project 01 - Search Algorithms

Dự án triển khai và so sánh **8 thuật toán tối ưu hóa** trên **2 bài toán**: Rastrigin (liên tục) và Knapsack (rời rạc).

## 📋 Mô tả dự án

Dự án này là phần triển khai cho môn **Nhập môn AI**, bao gồm:
- **8 thuật toán tìm kiếm/tối ưu hóa**: PSO, ABC, FA, CS, GA, HC, ACO, A*
- **2 bài toán benchmark**: 
  - Rastrigin Function (continuous optimization)
  - 0/1 Knapsack Problem (discrete optimization)
- **Phân tích hiệu suất**: Convergence analysis, robustness testing, scalability evaluation

## 🗂️ Cấu trúc dự án

```
algorithm/
├── algorithms/          # Code nguồn của 8 thuật toán
│   ├── pso.py          # Particle Swarm Optimization
│   ├── abc.py          # Artificial Bee Colony
│   ├── fa.py           # Firefly Algorithm
│   ├── cs.py           # Cuckoo Search
│   ├── ga.py           # Genetic Algorithm
│   ├── hc.py           # Hill Climbing
│   ├── aco.py          # Ant Colony Optimization
│   └── a_star.py       # A* Search
├── problems/            # Định nghĩa bài toán
│   ├── rastrigin.py    # Rastrigin function
│   └── knapsack.py     # Knapsack problem
├── experiments/         # Scripts chạy thí nghiệm
│   ├── run_rastrigin.py
│   └── run_knapsack.py
├── visualizations/      # Scripts vẽ biểu đồ (Python files only)
│   ├── plot_convergence.py
│   ├── plot_robustness.py
│   └── plot_rastrigin_3d.py
├── results/            # Kết quả thí nghiệm (CSV & PNG files)
│   ├── *.csv          # Dữ liệu thô và tổng hợp
│   └── *.png          # Biểu đồ visualizations
└── README.md
```

## 📊 Thuật toán được triển khai

### Thuật toán đa mục đích (Continuous & Discrete)
6 thuật toán sau có thể giải **CẢ HAI** bài toán Rastrigin (continuous) và Knapsack (discrete):
1. **PSO** - Particle Swarm Optimization
2. **ABC** - Artificial Bee Colony
3. **FA** - Firefly Algorithm
4. **CS** - Cuckoo Search
5. **GA** - Genetic Algorithm
6. **HC** - Hill Climbing

**Lưu ý**: Mỗi thuật toán có 2 phiên bản:
- Phiên bản `_continuous`: Sử dụng cho bài toán Rastrigin (tối ưu hóa liên tục)
- Phiên bản `_discrete`: Sử dụng cho bài toán Knapsack (tối ưu hóa rời rạc)
  - PSO, ABC, FA, CS: Sử dụng phương pháp sigmoid để chuyển đổi lời giải liên tục thành nhị phân
  - GA: Sử dụng crossover/mutation phù hợp với từng loại bài toán
  - HC: Sử dụng chiến lược tìm kiếm hàng xóm phù hợp với từng loại bài toán

### Thuật toán chuyên biệt cho Discrete Optimization (Knapsack)
2 thuật toán sau chỉ được triển khai cho bài toán Knapsack:
1. **ACO** - Ant Colony Optimization
2. **A*** - A* Search

## 🔧 Yêu cầu

- **Python 3.7+**
- **NumPy** (tính toán)
- **Matplotlib** (visualization)

### Cài đặt thư viện

```bash
pip install numpy matplotlib
```

## 🚀 Cách chạy

### Bước 1: Chạy thí nghiệm

**Lưu ý**: Quá trình này sẽ mất vài phút (khoảng 5-15 phút tùy cấu hình máy).

```bash
# Chạy thí nghiệm cho Rastrigin Function
python experiments/run_rastrigin.py

# Chạy thí nghiệm cho Knapsack Problem
python experiments/run_knapsack.py
```

### Bước 2: Tạo các biểu đồ

```bash
# Vẽ biểu đồ hội tụ (convergence curves)
python visualizations/plot_convergence.py

# Vẽ biểu đồ độ ổn định (robustness - box plots)
python visualizations/plot_robustness.py

# Vẽ biểu đồ heatmap & contour của hàm Rastrigin
python visualizations/plot_rastrigin_3d.py
```

### Bước 3: Xem kết quả

- **Tất cả kết quả**: Trong thư mục `results/`
  - Dữ liệu CSV (raw data và summary)
  - Biểu đồ PNG (visualizations)

## 📈 Thí nghiệm được thực hiện

### Rastrigin Function
- **Algorithms tested**: 6 thuật toán (PSO, ABC, FA, CS, GA, HC)
- **Dimensions**: 10, 30
- **Number of runs**: 20 (cho mỗi thuật toán)
- **Population size**: 50
- **Max iterations**: 1000

### Knapsack Problem
- **Algorithms tested**: 8 thuật toán (PSO, ABC, FA, CS, GA, HC, ACO, A*)
- **Problem sizes**: 20 items, 50 items
- **Number of runs**: 20 (cho mỗi thuật toán)
- **Population size**: 50
- **Max iterations**: 1000

## 📊 Kết quả đầu ra

### Trong thư mục `results/`:

**CSV Files:**
- `rastrigin_summary.csv` - Thống kê tổng hợp (mean, std, time)
- `rastrigin_convergence.csv` - Dữ liệu hội tụ theo iteration
- `rastrigin_raw_fitness.csv` - Dữ liệu thô từ 20 lần chạy
- `knapsack_summary.csv` - Thống kê tổng hợp
- `knapsack_convergence.csv` - Dữ liệu hội tụ
- `knapsack_raw_fitness.csv` - Dữ liệu thô

**PNG Files (Visualizations):**
- `convergence_rastrigin_D10.png` - Convergence cho D=10
- `convergence_rastrigin_D30.png` - Convergence cho D=30
- `convergence_knapsack_N20.png` - Convergence cho N=20
- `convergence_knapsack_N50.png` - Convergence cho N=50
- `robustness_rastrigin_D10.png` - Box plot cho D=10
- `robustness_rastrigin_D30.png` - Box plot cho D=30
- `robustness_knapsack_N20.png` - Box plot cho N=20
- `robustness_knapsack_N50.png` - Box plot cho N=50
- `rastrigin_3d_surface.png` - Heatmap và contour plot
- `rastrigin_cross_sections.png` - Cross-section plots

## 🎯 Rastrigin Function

Hàm Rastrigin là một hàm benchmark phổ biến trong tối ưu hóa, có dạng:

```
f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]
```

- **Domain**: x_i ∈ [-5.12, 5.12]
- **Global minimum**: f(0, 0, ..., 0) = 0
- **Đặc điểm**: Highly multimodal (nhiều cực trị địa phương)

## 🎒 Knapsack Problem

Bài toán cái túi 0/1:
- **Input**: n items với weight và value
- **Constraint**: Tổng weight ≤ capacity
- **Objective**: Maximize tổng value

## 🔬 Phân tích

Dự án thực hiện các phân tích sau:

1. **Convergence Analysis**: Đánh giá tốc độ hội tụ của các thuật toán
2. **Robustness Testing**: Kiểm tra độ ổn định qua 20 lần chạy
3. **Scalability Evaluation**: So sánh hiệu suất với các problem size khác nhau
4. **Statistical Comparison**: Mean, Standard Deviation, Min/Max fitness

## 📚 Tài liệu tham khảo

### Thuật toán
- Kennedy & Eberhart (1995) - Particle Swarm Optimization
- Karaboga (2005) - Artificial Bee Colony
- Yang (2008) - Firefly Algorithm
- Yang & Deb (2009) - Cuckoo Search
- Goldberg (1989) - Genetic Algorithms
- Dorigo (1992) - Ant Colony Optimization

### Bài toán
- Rastrigin (1974) - Systems of Extremal Control
- Knapsack Problem - Classic NP-Complete problem

## 👤 Tác giả

Dự án được phát triển cho môn **Nhập môn AI - HCMUS**

## 📝 Ghi chú

- Tất cả code được viết từ đầu chỉ sử dụng **NumPy** (không dùng các thư viện tối ưu hóa có sẵn như scipy, scikit-learn, deap, etc.)
- Mã nguồn tuân thủ chuẩn function signature đã định trong yêu cầu
- Các tham số thuật toán đã được điều chỉnh để phù hợp với từng bài toán

## 🐛 Troubleshooting

**Nếu gặp lỗi import:**
```bash
# Chạy từ thư mục project_01/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Nếu matplotlib không hiển thị được:**
```bash
# Kiểm tra backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

## 📧 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng mở issue hoặc liên hệ qua email.

---

**Good luck with your experiments! 🚀**

