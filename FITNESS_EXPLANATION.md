# Giải thích Best Fitness Values cho các bài toán và thuật toán

## 📊 Tổng quan

### Bài toán Rastrigin (Continuous Optimization)
- **Loại**: **Minimization** (tối thiểu hóa)
- **Fitness function**: `rastrigin(x)` trả về giá trị ≥ 0
- **Global minimum**: f(0, 0, ..., 0) = 0
- **Best Fitness Value**: **Càng nhỏ càng tốt** (giá trị thấp nhất = tốt nhất)

### Bài toán Knapsack (Discrete Optimization)
- **Loại**: **Maximization** (tối đa hóa)
- **Fitness function**: `knapsack_fitness(solution, context)` trả về:
  - `total_value` (số dương) nếu solution hợp lệ
  - `0` nếu solution không hợp lệ (vượt capacity)
- **Best Fitness Value**: **Càng lớn càng tốt** (giá trị cao nhất = tốt nhất)

---

## 🔍 Bảng so sánh chi tiết

### 1. Bài toán Rastrigin (Minimization)

| Thuật toán | Cách xử lý | Fitness bên trong | History lưu | Return value | Ý nghĩa |
|-----------|-----------|-------------------|-------------|--------------|---------|
| **PSO** | Minimization trực tiếp | `obj_func(solution)` ≥ 0 | Giá trị ≥ 0 | Giá trị ≥ 0 | Giá trị càng nhỏ càng tốt |
| **ABC** | Minimization trực tiếp | `obj_func(solution)` ≥ 0 | Giá trị ≥ 0 | Giá trị ≥ 0 | Giá trị càng nhỏ càng tốt |
| **FA** | Minimization trực tiếp | `obj_func(solution)` ≥ 0 | Giá trị ≥ 0 | Giá trị ≥ 0 | Giá trị càng nhỏ càng tốt |
| **CS** | Minimization trực tiếp | `obj_func(solution)` ≥ 0 | Giá trị ≥ 0 | Giá trị ≥ 0 | Giá trị càng nhỏ càng tốt |
| **GA** | Minimization trực tiếp | `obj_func(solution)` ≥ 0 | Giá trị ≥ 0 | Giá trị ≥ 0 | Giá trị càng nhỏ càng tốt |
| **HC** | Minimization trực tiếp | `obj_func(solution)` ≥ 0 | Giá trị ≥ 0 | Giá trị ≥ 0 | Giá trị càng nhỏ càng tốt |

**Kết luận cho Rastrigin**: Tất cả thuật toán đều làm việc trực tiếp với minimization, không cần chuyển đổi.

---

### 2. Bài toán Knapsack (Maximization)

| Thuật toán | Chiến lược | Fitness bên trong | History lưu | History return | Return value | Trạng thái |
|-----------|-----------|------------------|-------------|---------------|--------------|-----------|
| **PSO** | Chuyển Maximization → Minimization | `-obj_func(solution)` ≤ 0 | Số âm (internal) | **Số dương** ✅ | `-best_fitness` (dương) ✅ | **Đã sửa** ✅ |
| **ABC** | Chuyển Maximization → Minimization | `-obj_func(solution)` ≤ 0 | Số âm (internal) | **Số dương** ✅ | `-best_fitness` (dương) ✅ | **Đã sửa** ✅ |
| **FA** | Chuyển Maximization → Minimization | `-obj_func(solution)` ≤ 0 | Số âm (internal) | **Số dương** ✅ | `-best_fitness` (dương) ✅ | **Đã sửa** ✅ |
| **CS** | Chuyển Maximization → Minimization | `-obj_func(solution)` ≤ 0 | Số âm (internal) | **Số dương** ✅ | `-best_fitness` (dương) ✅ | **Đã sửa** ✅ |
| **HC** | Chuyển Maximization → Minimization | `-obj_func(solution)` ≤ 0 | Số âm (internal) | **Số dương** ✅ | `-best_fitness` (dương) ✅ | **Đã sửa** ✅ |
| **SA** | Chuyển Maximization → Minimization | `-obj_func(solution)` ≤ 0 | Số âm (internal) | **Số dương** ✅ | `-best_fitness` (dương) ✅ | **Đã sửa** ✅ |
| **GA** | Maximization trực tiếp | `obj_func(solution)` ≥ 0 | Số dương | **Số dương** ✅ | `best_fitness` (dương) ✅ | **Đúng từ đầu** ✅ |
| **ACO** | Maximization trực tiếp | `obj_func(solution)` ≥ 0 | Số dương | **Số dương** ✅ | `best_fitness` (dương) ✅ | **Đúng từ đầu** ✅ |

---

## ✅ Vấn đề đã được sửa

### Vấn đề ban đầu:

**Trước đây**, chỉ GA và ACO có Best Fitness Values dương trong biểu đồ convergence vì:

1. **PSO, ABC, FA, CS, HC, SA**:
   - Bên trong thuật toán: Chuyển maximization → minimization bằng cách **negate** fitness
   - `fitness_internal = -obj_func(solution, context)` → **Số âm**
   - History lưu: `history.append(best_fitness)` → **Lưu số âm** ❌
   - Khi return: `best_fitness = -best_fitness` → **Chuyển thành dương** ✅
   - **Kết quả cũ**: History (dùng cho convergence plot) chứa số âm, nhưng return value (dùng cho summary) là dương

2. **GA, ACO**:
   - Bên trong thuật toán: Làm việc trực tiếp với maximization
   - `fitness_internal = obj_func(solution, context)` → **Số dương**
   - History lưu: `history.append(best_fitness)` → **Lưu số dương** ✅
   - Khi return: `best_fitness` → **Giữ nguyên dương** ✅
   - **Kết quả**: Cả history và return value đều dương

### Giải pháp đã áp dụng:

**Đã sửa tất cả các thuật toán PSO, ABC, FA, CS, HC, SA** để convert history về maximization trước khi return:

```python
# Trước khi return
best_fitness = -best_fitness  # Convert return value về dương
history = [-h for h in history]  # Convert history về dương ✅
return best_solution, best_fitness, history
```

### Kết quả sau khi sửa:

**Giả sử solution có fitness = 500 (tốt):**

| Thuật toán | Fitness bên trong | History (internal) | History (return) | Return | Convergence Plot |
|-----------|------------------|-------------------|-----------------|--------|-----------------|
| PSO | -500 | -500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |
| ABC | -500 | -500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |
| FA | -500 | -500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |
| CS | -500 | -500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |
| HC | -500 | -500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |
| SA | -500 | -500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |
| GA | 500 | 500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |
| ACO | 500 | 500 | **500** ✅ | 500 ✅ | Hiển thị 500 (đúng) |

---

## 📈 Ý nghĩa Best Fitness Values

### Cho bài toán Rastrigin:
- **Best Fitness Value** = Giá trị hàm Rastrigin tại solution tốt nhất
- **Giá trị tốt nhất**: 0 (hoặc gần 0)
- **Giá trị càng nhỏ càng tốt**
- **Ví dụ**: 
  - Fitness = 0.5 → Rất tốt (gần global minimum)
  - Fitness = 10.0 → Tốt
  - Fitness = 50.0 → Trung bình
  - Fitness = 100.0 → Kém

### Cho bài toán Knapsack:
- **Best Fitness Value** = Tổng giá trị (total value) của items được chọn
- **Giá trị tốt nhất**: Càng cao càng tốt (phụ thuộc vào problem instance)
- **Giá trị càng lớn càng tốt**
- **Ví dụ** (với N=20 items):
  - Fitness = 800 → Rất tốt
  - Fitness = 600 → Tốt
  - Fitness = 400 → Trung bình
  - Fitness = 200 → Kém
  - Fitness = 0 → Solution không hợp lệ hoặc rất kém

---

## 🔧 Cách đọc biểu đồ Convergence

### Biểu đồ Rastrigin:
- **Trục Y**: Best Fitness Value (càng nhỏ càng tốt)
- **Đường đi xuống**: Thuật toán đang cải thiện ✅
- **Đường đi lên**: Thuật toán đang tệ hơn ❌
- **Giá trị gần 0**: Thuật toán hoạt động tốt ✅

### Biểu đồ Knapsack (sau khi sửa):
- **Trục Y**: Best Fitness Value (càng lớn càng tốt)
- **Tất cả thuật toán (PSO, ABC, FA, CS, GA, HC, ACO, SA)**: 
  - Hiển thị số dương ✅ (đã được sửa)
  - **Đường đi lên**: Đang cải thiện ✅ (fitness tăng = tốt hơn)
  - **Đường đi xuống**: Đang tệ hơn ❌ (fitness giảm = kém hơn)
- **So sánh trực tiếp**: Có thể so sánh trực tiếp giữa các thuật toán vì tất cả đều dương

---

## 💡 Kết luận

1. **Best Fitness Value thực sự**:
   - **Rastrigin**: Giá trị hàm số (≥ 0), càng nhỏ càng tốt
   - **Knapsack**: Tổng giá trị items (≥ 0), càng lớn càng tốt

2. **Vấn đề đã được sửa**:
   - ✅ **Đã sửa code**: History của PSO, ABC, FA, CS, HC, SA giờ cũng được convert về dương trước khi return
   - ✅ **Kết quả**: Tất cả thuật toán đều hiển thị số dương trong convergence plot
   - ✅ **So sánh trực tiếp**: Có thể so sánh trực tiếp giữa tất cả các thuật toán

3. **Cách đọc biểu đồ (sau khi sửa)**:
   - **Tất cả thuật toán**: Đọc trực tiếp (số dương, càng lớn càng tốt)
   - **Đường đi lên**: Thuật toán đang cải thiện ✅
   - **Đường đi xuống**: Thuật toán đang tệ hơn ❌
   - **Ví dụ**: 800 tốt hơn 600 (vì 800 > 600)

4. **Lợi ích của việc sửa**:
   - ✅ Dễ so sánh giữa các thuật toán
   - ✅ Trực quan hơn (số dương = tốt, số lớn = tốt hơn)
   - ✅ Nhất quán với return value trong summary

---

## 📝 Ghi chú kỹ thuật

### Code pattern cho PSO, ABC, FA, CS, HC, SA (sau khi sửa):
```python
# Bên trong thuật toán
fitness = -obj_func(solution, context)  # Negate để chuyển max → min
history.append(best_fitness)  # Lưu số âm (internal)

# Khi return
best_fitness = -best_fitness  # ✅ Convert về dương
history = [-h for h in history]  # ✅ Convert history về dương
return best_solution, best_fitness, history  # ✅ Cả hai đều dương
```

### Code pattern cho GA, ACO:
```python
# Bên trong thuật toán
fitness = obj_func(solution, context)  # Làm việc trực tiếp với max
history.append(best_fitness)  # ✅ Lưu số dương

# Khi return
return best_solution, best_fitness, history  # ✅ Cả hai đều dương
```

