import numpy as np

def generate_knapsack_data(n_items, seed=42):
    """Tạo dữ liệu ngẫu nhiên cho bài toán Knapsack."""
    np.random.seed(seed)
    weights = np.random.randint(1, 20, n_items)
    values = np.random.randint(10, 100, n_items)
    # Đặt capacity bằng 40% tổng trọng lượng
    capacity = int(np.sum(weights) * 0.4)
    return weights, values, capacity, n_items

def knapsack_fitness(solution, context):
    """
    Hàm fitness cho Knapsack.
    'solution' là vector nhị phân (n_items,).
    'context' là dict {'weights', 'values', 'capacity'}.
    """
    weights = context['weights']
    values = context['values']
    capacity = context['capacity']
    
    total_weight = np.dot(solution, weights)
    total_value = np.dot(solution, values)
    
    # Áp dụng cơ chế phạt (penalty) nếu vượt quá capacity
    if total_weight > capacity:
        return 0  # Lời giải không hợp lệ, fitness = 0 (vì ta muốn tối đa hóa)
    
    return total_value

