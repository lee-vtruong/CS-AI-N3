# Implementation Summary - Mô tả Bài toán và Thiết lập Thực nghiệm

## 1. Mô tả Bài toán

Dự án này thực hiện so sánh và đánh giá hiệu năng của **8 thuật toán tối ưu hóa** trên **2 bài toán benchmark** khác nhau về bản chất: một bài toán liên tục (continuous) và một bài toán rời rạc (discrete).

### 1.1. Bài toán Rastrigin (Continuous Optimization)

#### 1.1.1. Định nghĩa

Hàm Rastrigin là một hàm benchmark phổ biến trong tối ưu hóa số, được định nghĩa như sau:

```
f(x) = 10n + Σ[i=1 to n][x_i² - 10cos(2πx_i)]
```

Trong đó:
- **n**: Số chiều (dimensions) của không gian tìm kiếm
- **x_i**: Giá trị của biến thứ i, với x_i ∈ [-5.12, 5.12]
- **Domain**: Mỗi biến x_i nằm trong khoảng [-5.12, 5.12]
- **Global minimum**: f(0, 0, ..., 0) = 0

#### 1.1.2. Đặc điểm

- **Highly multimodal**: Hàm có rất nhiều local optima (cực tiểu địa phương), tạo ra một landscape phức tạp
- **Difficult to optimize**: Do có nhiều local optima, các thuật toán dễ bị kẹt tại các điểm tối ưu địa phương
- **Scalable**: Có thể test với số chiều khác nhau để đánh giá khả năng mở rộng (scalability)
- **Standard benchmark**: Được sử dụng rộng rãi trong cộng đồng nghiên cứu để so sánh các thuật toán tối ưu hóa

#### 1.1.3. Mục tiêu

Tìm nghiệm tối ưu toàn cục (global optimum) tại điểm x* = (0, 0, ..., 0) với giá trị hàm mục tiêu f(x*) = 0.

### 1.2. Bài toán 0/1 Knapsack (Discrete Optimization)

#### 1.2.1. Định nghĩa

Bài toán 0/1 Knapsack là một bài toán tối ưu hóa tổ hợp kinh điển, thuộc lớp NP-Complete. Bài toán được mô tả như sau:

**Input:**
- **n items**: Mỗi item i có:
  - `weight_i`: Trọng lượng của item thứ i
  - `value_i`: Giá trị của item thứ i
- **capacity**: Sức chứa tối đa của túi (knapsack)

**Constraint:**
- Tổng trọng lượng của các items được chọn không được vượt quá capacity:
  ```
  Σ[i=1 to n] (x_i × weight_i) ≤ capacity
  ```
  trong đó x_i ∈ {0, 1} (0 = không chọn, 1 = chọn)

**Objective:**
- Tối đa hóa tổng giá trị của các items được chọn:
  ```
  Maximize: Σ[i=1 to n] (x_i × value_i)
  ```

#### 1.2.2. Đặc điểm

- **NP-Complete problem**: Không có thuật toán đa thức thời gian để giải chính xác cho trường hợp tổng quát
- **Binary decision**: Mỗi item chỉ có thể được chọn hoặc không chọn (0 hoặc 1)
- **Constraint satisfaction**: Nghiệm phải thỏa mãn ràng buộc về trọng lượng
- **Real-world applications**: Có nhiều ứng dụng thực tế như resource allocation, portfolio optimization, etc.

#### 1.2.3. Implementation Details

Trong implementation của dự án:

- **Data generation**: 
  - Weights được sinh ngẫu nhiên trong khoảng [1, 20]
  - Values được sinh ngẫu nhiên trong khoảng [10, 100]
  - Capacity được đặt bằng 40% tổng trọng lượng: `capacity = 0.4 × Σ(weights)`
  - Sử dụng seed=42 để đảm bảo reproducibility

- **Fitness function**:
  - Để phù hợp với framework minimization, fitness được định nghĩa là `-total_value`
  - Nghiệm vi phạm constraint (vượt quá capacity) được gán fitness = 0 (penalty)
  - Nghiệm hợp lệ có fitness < 0, càng âm càng tốt (tương ứng với total_value càng cao)

#### 1.2.4. Mục tiêu

Tìm tập hợp items (binary vector) sao cho:
- Tổng trọng lượng ≤ capacity
- Tổng giá trị đạt được là lớn nhất

## 2. Thiết lập Thực nghiệm

### 2.1. Các Thuật toán được Đánh giá

Dự án triển khai và so sánh **8 thuật toán tối ưu hóa**:

#### 2.1.1. Thuật toán Đa mục đích (Multi-purpose Algorithms)

Các thuật toán sau có thể giải được **CẢ HAI** bài toán Rastrigin (continuous) và Knapsack (discrete):

1. **PSO (Particle Swarm Optimization)**
   - Continuous version: `pso_continuous`
   - Discrete version: `pso_discrete` (sử dụng sigmoid để chuyển đổi continuous → binary)
   - Parameters: `w` (inertia weight), `c1` (cognitive coefficient), `c2` (social coefficient)

2. **ABC (Artificial Bee Colony)**
   - Continuous version: `abc_continuous`
   - Discrete version: `abc_discrete` (sử dụng sigmoid để chuyển đổi)
   - Parameters: `limit` (abandonment limit)

3. **FA (Firefly Algorithm)**
   - Continuous version: `firefly_continuous`
   - Discrete version: `firefly_discrete` (sử dụng sigmoid để chuyển đổi)
   - Parameters: `alpha` (randomization parameter), `beta0` (attractiveness at r=0), `gamma` (light absorption coefficient)

4. **CS (Cuckoo Search)**
   - Continuous version: `cuckoo_search_continuous`
   - Discrete version: `cuckoo_search_discrete` (sử dụng sigmoid để chuyển đổi)
   - Parameters: `pa` (probability of discovering alien eggs), `beta` (parameter for Levy flight)

5. **GA (Genetic Algorithm)**
   - Continuous version: `genetic_continuous`
   - Discrete version: `genetic_discrete` (sử dụng crossover/mutation phù hợp)
   - Parameters: `crossover_rate`, `mutation_rate`

6. **HC (Hill Climbing)**
   - Continuous version: `hill_climbing_continuous`
   - Discrete version: `hill_climbing_discrete` (sử dụng neighbor search strategy phù hợp)
   - Parameters: `step_size` (cho continuous)

#### 2.1.2. Thuật toán Chuyên biệt cho Discrete (Knapsack)

Các thuật toán sau chỉ được triển khai cho bài toán Knapsack:

7. **ACO (Ant Colony Optimization)**
   - Discrete version: `aco_discrete`
   - Parameters: `alpha` (pheromone importance), `beta` (heuristic importance), `rho` (evaporation rate), `Q` (pheromone deposit constant)

8. **SA (Simulated Annealing)**
   - Discrete version: `simulated_annealing_discrete`
   - Parameters: `initial_temp`, `cooling_rate`, `min_temp`

### 2.2. Tham số Thực nghiệm Chung

#### 2.2.1. Bài toán Rastrigin

- **Số lần chạy (N_RUNS)**: 10 lần cho mỗi thuật toán
  - Mục đích: Đảm bảo tính thống kê và độ tin cậy của kết quả
  - Tính toán: Mean, Standard Deviation, Min, Max từ 10 lần chạy

- **Số chiều (Dimensions)**: [10, 30]
  - D=10: Bài toán nhỏ, test khả năng cơ bản
  - D=30: Bài toán lớn hơn, đánh giá scalability
  - Mục đích: So sánh hiệu năng khi tăng độ phức tạp của bài toán

- **Kích thước quần thể (POP_SIZE)**: 50
  - Áp dụng cho tất cả thuật toán population-based (PSO, ABC, FA, CS, GA)
  - HC không sử dụng population

- **Số lần lặp tối đa (MAX_ITER)**: 50
  - Giới hạn số lần lặp để đảm bảo thời gian chạy hợp lý
  - Đủ để quan sát quá trình hội tụ của các thuật toán

- **Domain bounds**: [-5.12, 5.12] cho mỗi chiều
  - Theo chuẩn của hàm Rastrigin

#### 2.2.2. Bài toán Knapsack

- **Số lần chạy (N_RUNS)**: 10 lần cho mỗi thuật toán
  - Tương tự như Rastrigin để đảm bảo tính thống kê

- **Kích thước bài toán (N_ITEMS_LIST)**: [20, 50]
  - N=20: Bài toán nhỏ, test khả năng cơ bản
  - N=50: Bài toán lớn hơn, đánh giá scalability
  - Mục đích: So sánh hiệu năng khi tăng số lượng items

- **Kích thước quần thể (POP_SIZE)**: 50
  - Áp dụng cho tất cả thuật toán population-based
  - ACO, HC, SA có cách xử lý riêng

- **Số lần lặp tối đa (MAX_ITER)**: 50
  - Tương tự như Rastrigin

- **Data generation**:
  - Weights: Random integers trong [1, 20]
  - Values: Random integers trong [10, 100]
  - Capacity: 40% tổng trọng lượng
  - Seed: 42 (đảm bảo reproducibility)

### 2.3. Tham số Thuật toán Cụ thể

#### 2.3.1. Rastrigin - Algorithm Parameters

```python
ALGO_PARAMS = {
    'pso': {
        'w': 0.8,      # Inertia weight
        'c1': 2.0,     # Cognitive coefficient
        'c2': 2.0      # Social coefficient
    },
    'abc': {
        'limit': 10    # Abandonment limit
    },
    'fa': {
        'alpha': 0.5,   # Randomization parameter
        'beta0': 1.0,   # Attractiveness at r=0
        'gamma': 0.95   # Light absorption coefficient
    },
    'cs': {
        'pa': 0.25,     # Probability of discovering alien eggs
        'beta': 1.5    # Parameter for Levy flight
    },
    'ga': {
        'crossover_rate': 0.8,
        'mutation_rate': 0.02
    },
    'hc': {
        'step_size': 0.1
    }
}
```

#### 2.3.2. Knapsack - Algorithm Parameters

```python
ALGO_PARAMS = {
    'pso': {
        'w': 0.8,
        'c1': 2.0,
        'c2': 2.0
    },
    'abc': {
        'limit': 10
    },
    'fa': {
        'alpha': 0.5,
        'beta0': 1.0,
        'gamma': 0.95
    },
    'cs': {
        'pa': 0.25,
        'beta': 1.5
    },
    'ga': {
        'crossover_rate': 0.8,
        'mutation_rate': 0.05  # Higher mutation rate for discrete
    },
    'hc': {
        'step_size': 0.1
    },
    'aco': {
        'alpha': 1.0,   # Pheromone importance
        'beta': 2.0,    # Heuristic importance
        'rho': 0.5      # Evaporation rate
    },
    'sa': {
        'initial_temp': 100.0,
        'cooling_rate': 0.95,
        'min_temp': 0.01
    }
}
```

### 2.4. Metrics và Đánh giá Hiệu năng

#### 2.4.1. Metrics được Thu thập

Với mỗi thuật toán và mỗi cấu hình bài toán, các metrics sau được thu thập:

1. **Fitness Values**:
   - Average Fitness: Giá trị fitness trung bình qua N_RUNS lần chạy
   - Standard Deviation: Độ lệch chuẩn, đo độ ổn định
   - Min Fitness: Giá trị tốt nhất trong N_RUNS lần chạy
   - Max Fitness: Giá trị tệ nhất trong N_RUNS lần chạy

2. **Convergence History**:
   - Lịch sử giá trị fitness tốt nhất tại mỗi iteration
   - Được lưu trữ để phân tích quá trình hội tụ
   - Average convergence: Trung bình hóa qua N_RUNS lần chạy

3. **Computational Complexity**:
   - Average Time: Thời gian thực thi trung bình (seconds)
   - Peak Memory: Bộ nhớ đỉnh sử dụng (MB)
   - Được đo bằng `time.time()` và `tracemalloc`

4. **Raw Data**:
   - Fitness values từ tất cả N_RUNS lần chạy
   - Dùng để vẽ box plots và phân tích robustness

#### 2.4.2. Phân tích được Thực hiện

1. **Convergence Analysis**:
   - Vẽ convergence curves cho tất cả thuật toán
   - So sánh tốc độ hội tụ
   - Đánh giá khả năng tìm nghiệm tốt

2. **Robustness Testing**:
   - Box plots để hiển thị distribution của fitness values
   - Đánh giá độ ổn định và consistency
   - So sánh variance giữa các thuật toán

3. **Scalability Evaluation**:
   - So sánh hiệu năng giữa các kích thước bài toán khác nhau
   - Rastrigin: D=10 vs D=30
   - Knapsack: N=20 vs N=50

4. **Complexity Analysis**:
   - So sánh thời gian thực thi (time complexity)
   - So sánh bộ nhớ sử dụng (memory complexity)
   - Trade-off giữa chất lượng nghiệm và chi phí tính toán

5. **Time-Quality Trade-off**:
   - Scatter plots: Time vs Fitness
   - Xác định thuật toán cân bằng tốt nhất giữa tốc độ và chất lượng

### 2.5. Môi trường Thực nghiệm

#### 2.5.1. Công nghệ và Thư viện

- **Programming Language**: Python 3.7+
- **Core Libraries**:
  - NumPy: Tính toán số học và ma trận
  - Matplotlib: Visualization và plotting
- **Development Tools**:
  - Jupyter Notebook: Interactive experimentation và analysis
  - CSV: Lưu trữ dữ liệu kết quả
  - tracemalloc: Đo lường memory usage

#### 2.5.2. Implementation Approach

- **From scratch**: Tất cả thuật toán được implement từ đầu, không sử dụng các thư viện optimization có sẵn (scipy, scikit-learn, deap, etc.)
- **Pure NumPy**: Chỉ sử dụng NumPy cho các phép tính toán
- **Modular design**: Mỗi thuật toán là một module độc lập, dễ bảo trì và mở rộng

#### 2.5.3. Reproducibility

- **Fixed seeds**: Sử dụng seed cố định (seed=42) cho data generation
- **Deterministic**: Các tham số được cố định để đảm bảo reproducibility
- **Version control**: Code được quản lý bằng Git

### 2.6. Cấu trúc Dữ liệu Kết quả

#### 2.6.1. CSV Files

**Rastrigin:**
- `rastrigin_summary.csv`: Summary statistics (Algorithm, Dimension, Avg_Fitness, Std_Dev_Fitness, Min_Fitness, Max_Fitness, Avg_Time, Avg_Peak_Mem_MB)
- `rastrigin_convergence.csv`: Convergence history theo iteration
- `rastrigin_raw_fitness.csv`: Raw fitness values từ tất cả runs

**Knapsack:**
- `knapsack_summary.csv`: Summary statistics (Algorithm, N_Items, Avg_Fitness, Std_Dev_Fitness, Min_Fitness, Max_Fitness, Avg_Time, Avg_Peak_Mem_MB)
- `knapsack_convergence.csv`: Convergence history theo iteration
- `knapsack_raw_fitness.csv`: Raw fitness values từ tất cả runs

#### 2.6.2. Visualization Files (PNG)

- `convergence/`: Convergence curve plots
- `robustness/`: Box plot visualizations
- `complexity/`: Time và memory complexity bar charts
- `tradeoff/`: Time vs fitness scatter plots
- `rastrigin_3d/`: 3D surface và contour plots (chỉ cho Rastrigin)

### 2.7. Quy trình Thực nghiệm

#### 2.7.1. Execution Flow

1. **Initialization**:
   - Setup problem instance (Rastrigin bounds hoặc Knapsack data)
   - Initialize algorithm parameters

2. **For each algorithm**:
   - For each problem size/dimension:
     - For N_RUNS times:
       - Run algorithm với parameters đã định nghĩa
       - Record: fitness, convergence history, execution time, memory usage
     - Calculate statistics: mean, std, min, max
     - Store average convergence history

3. **Data Processing**:
   - Pad convergence histories to same length
   - Calculate average convergence curves
   - Prepare data for visualization

4. **Output**:
   - Save CSV files với summary và raw data
   - Generate visualization plots
   - Display results trong notebook (nếu dùng Jupyter)

#### 2.7.2. Statistical Methodology

- **Multiple runs**: Mỗi cấu hình được chạy N_RUNS=10 lần để đảm bảo tính thống kê
- **Averaging**: Convergence curves được tính trung bình từ N_RUNS runs
- **Padding**: Nếu các runs có số iterations khác nhau, padding với giá trị cuối cùng
- **Error bars**: Standard deviation được tính và hiển thị để đánh giá độ tin cậy

### 2.8. Đặc điểm Kỹ thuật Implementation

#### 2.8.1. Continuous vs Discrete Handling

**Continuous Algorithms (Rastrigin):**
- Làm việc trực tiếp với real-valued vectors
- Search space là continuous domain

**Discrete Algorithms (Knapsack):**
- **PSO, ABC, FA, CS**: Sử dụng sigmoid function để chuyển đổi continuous solutions → binary
  - Sigmoid: `p = 1 / (1 + exp(-x))`
  - Binary: `binary = (p > 0.5).astype(int)`
  - Numerically stable: Clip values để tránh overflow

- **GA**: Sử dụng crossover và mutation operators phù hợp cho binary encoding

- **HC**: Sử dụng neighbor search strategy (flip bits) cho binary space

- **ACO**: Sử dụng pheromone trails và heuristic (value/weight ratio) để construct solutions

- **SA**: Sử dụng neighbor moves (flip bits) với acceptance probability

#### 2.8.2. Function Signatures

**Continuous algorithms:**
```python
solution, fitness, history = algorithm_continuous(
    obj_func, bounds, n_dim, pop_size, max_iter, **params
)
```

**Discrete algorithms (PSO, ABC, FA, CS, GA):**
```python
solution, fitness, history = algorithm_discrete(
    obj_func, context, n_dim, pop_size, max_iter, **params
)
```

**Discrete algorithms (ACO):**
```python
solution, fitness, history = aco_discrete(
    obj_func, context, pop_size, max_iter, **params
)
```

**Discrete algorithms (HC, SA):**
```python
solution, fitness, history = algorithm_discrete(
    obj_func, context, n_dim, max_iter, **params
)
```

### 2.9. Validation và Testing

#### 2.9.1. Problem Validation

- **Rastrigin**: 
  - Kiểm tra global minimum tại (0, 0, ..., 0) với f = 0
  - Verify domain bounds [-5.12, 5.12]

- **Knapsack**:
  - Kiểm tra constraint satisfaction (total_weight ≤ capacity)
  - Verify fitness calculation (negative of total_value)
  - Test penalty mechanism cho invalid solutions

#### 2.9.2. Algorithm Validation

- Kiểm tra convergence: Algorithms phải cải thiện fitness qua iterations
- Kiểm tra bounds: Solutions phải nằm trong domain cho phép
- Kiểm tra constraints: Knapsack solutions phải thỏa mãn capacity constraint

### 2.10. Limitations và Assumptions

#### 2.10.1. Limitations

- **Fixed parameters**: Tham số thuật toán được cố định, không có parameter tuning tự động
- **Limited problem sizes**: Chỉ test với 2 kích thước cho mỗi bài toán
- **Deterministic seeds**: Sử dụng fixed seed có thể không đại diện cho tất cả trường hợp
- **Time constraints**: MAX_ITER=50 có thể chưa đủ để một số thuật toán hội tụ hoàn toàn

#### 2.10.2. Assumptions

- **Problem instances**: 
  - Rastrigin: Standard domain [-5.12, 5.12]
  - Knapsack: Random generation với seed cố định

- **Algorithm behavior**: 
  - Các thuật toán được implement đúng theo lý thuyết
  - Parameters được chọn dựa trên literature và best practices

- **Computational resources**: 
  - Đủ memory và CPU để chạy experiments
  - Không có giới hạn về thời gian thực thi

## 3. Tóm tắt

Dự án này thực hiện một nghiên cứu so sánh toàn diện về hiệu năng của 8 thuật toán tối ưu hóa trên 2 bài toán benchmark khác nhau. Thiết lập thực nghiệm được thiết kế để:

1. **Đảm bảo tính công bằng**: Tất cả thuật toán được test với cùng điều kiện (N_RUNS, POP_SIZE, MAX_ITER)
2. **Đánh giá đa chiều**: Không chỉ về chất lượng nghiệm mà còn về thời gian, bộ nhớ, và độ ổn định
3. **Phân tích sâu**: Từ convergence đến robustness, từ scalability đến complexity
4. **Reproducibility**: Sử dụng fixed seeds và parameters để có thể reproduce kết quả

Kết quả thực nghiệm sẽ cung cấp insights về:
- Thuật toán nào phù hợp nhất cho từng loại bài toán
- Trade-offs giữa chất lượng nghiệm và chi phí tính toán
- Ảnh hưởng của problem size đến hiệu năng thuật toán
- Độ ổn định và reliability của từng phương pháp

