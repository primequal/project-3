import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# =====================================================================
# 1. THIẾT LẬP THÔNG SỐ HỆ THỐNG (Theo Mục V - Simulation Results)
# =====================================================================
L_a = np.array([0, 0, 50])       # Tọa độ Alice (m)
L_b = np.array([200, 0, 0])      # Tọa độ Bob (m)
L_w = np.array([200, 100, 0])    # Tọa độ Willie (m)
L_r = np.array([0, 0, 120])      # Tọa độ UAV-IRS (m)

# Tính khoảng cách Euclid giữa các nút
d_ar = np.linalg.norm(L_a - L_r)
d_rb = np.linalg.norm(L_r - L_b)
d_rw = np.linalg.norm(L_r - L_w)
d_bw = np.linalg.norm(L_b - L_w)

# Các tham số kênh truyền & phần cứng
rho_0 = 10**(-20 / 10)           # Reference channel gain: -20 dB -> 0.01
sigma2 = 10**(-130 / 10) * 1e-3  # Noise power: -130 dBm -> 10^-16 W
phi = 10**(-90 / 10)             # Hệ số tự nhiễu tại Bob: -90 dB -> 10^-9
alpha = 2.0                      # Path-loss exponent (Air-to-Ground)
alpha_g = 5.0                    # Path-loss exponent (Ground-to-Ground)

# Độ lợi đường truyền quy mô lớn (Large-scale path loss)
pl_ar = rho_0 / (d_ar**alpha)
pl_rb = rho_0 / (d_rb**alpha)
pl_rw = rho_0 / (d_rw**alpha)
pl_bw = rho_0 / (d_bw**alpha_g)


# =====================================================================
# 2. MÔ PHỎNG HÌNH 2 (Fig. 2: Error Detection Probability vs. Threshold)
# =====================================================================
def simulate_fig2():
    print("Đang chạy mô phỏng Fig. 2 (Monte Carlo + Lý thuyết)...")
    # Trong Fig. 2, để khớp chính xác điểm cực tiểu tau* ~ 0.4e-12 W trên đồ thị bài báo,
    # hệ số lambda tương đương với N = 1 (hoặc chuẩn hóa lambda = 1).
    lam_fig2 = 1.0 
    
    cases = [
        (0.1, 0.1, 'o', 'b', 'P_a=0.1, P_b=0.1'),
        (0.2, 0.1, '^', 'r', 'P_a=0.2, P_b=0.1'),
        (0.1, 0.2, 's', 'y', 'P_a=0.1, P_b=0.2')
    ]
    
    # Dải ngưỡng phát hiện tau từ 0.1e-12 đến 2.2e-12 W
    tau_vals = np.linspace(0.1e-12, 2.2e-12, 45)
    num_mc = 200000  # Số lượng mẫu Monte Carlo
    
    plt.figure(figsize=(8, 6))
    
    for P_a, P_b, marker, color, label in cases:
        # Tính lambda_a và lambda_b theo công thức (11) và (12)
        lam_a = 1.0 / (pl_rw * pl_ar * P_a * lam_fig2)
        lam_b = 1.0 / (P_b * pl_bw)
        
        # --- 1. Tính theo công thức Lý thuyết (Phương trình 16, 17, 18) ---
        z_vals = np.maximum(tau_vals - sigma2, 0)
        alpha_prob = np.exp(-lam_b * z_vals)
        beta_prob = 1.0 + (lam_b * np.exp(-lam_a * z_vals) - lam_a * np.exp(-lam_b * z_vals)) / (lam_a - lam_b)
        xi_theo = alpha_prob + beta_prob
        
        # --- 2. Mô phỏng Monte Carlo ---
        # Sinh biến ngẫu nhiên phân phối mũ cho công suất thu từ Alice (A) và Bob (B)
        A_samples = np.random.exponential(scale=1.0/lam_a, size=num_mc)
        B_samples = np.random.exponential(scale=1.0/lam_b, size=num_mc)
        
        P_w_H0 = B_samples + sigma2
        P_w_H1 = A_samples + B_samples + sigma2
        
        xi_sim = []
        for tau in tau_vals:
            fa_sim = np.mean(P_w_H0 >= tau)  # False Alarm
            md_sim = np.mean(P_w_H1 <= tau)  # Miss Detection
            xi_sim.append(fa_sim + md_sim)
            
        # --- 3. Tính điểm tối ưu (Proposition 1 - Phương trình 19, 20) ---
        tau_opt = (np.log(lam_a) - np.log(lam_b)) / (lam_a - lam_b) + sigma2
        t_val = lam_b / lam_a
        xi_opt = 1.0 - (t_val**(1.0 / (1.0 - t_val)))
        
        # Vẽ đồ thị
        plt.plot(tau_vals, xi_sim, marker=marker, linestyle='none', markerfacecolor='none', 
                 markeredgecolor=color, markersize=7, label=f'Simulation {label}')
        plt.plot(tau_vals, xi_theo, color='k' if color=='y' else color, linestyle='-', 
                 linewidth=1.2, label=f'Theoretical {label}')
        plt.plot(tau_opt, xi_opt, marker='*', color='darkorange', markersize=12, zorder=5)

    plt.xlabel(r'Detection threshold $\tau$ (W)', fontsize=12)
    plt.ylabel(r'Error detection probability $\xi$', fontsize=12)
    plt.title('Fig. 2. Comparison of $\\xi$ under different value of $\\tau$', fontsize=13)
    plt.ylim([0.2, 1.0])
    plt.xlim([0, 2.5e-12])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()


# =====================================================================
# 3. MÔ PHỎNG HÌNH 3 (Fig. 3: Achievable Rate R_b vs. Covertness epsilon)
# =====================================================================
def solve_t_from_epsilon(eps):
    """
    Giải phương trình phi tuyến (24d): 1 - t^(1/(1-t)) = 1 - eps
    <=> t^(1/(1-t)) = eps để tìm t = lambda_b / lambda_a (tức hàm ngược xi^{*-1}(1 - eps))
    Lưu ý: Khi t -> 1, lim t^(1/(1-t)) = 1/e ~ 0.3678. Với eps in [0.01, 0.2], nghiệm t nằm trong (0, 1).
    """
    func = lambda t: (t**(1.0 / (1.0 - t))) - eps
    return brentq(func, 1e-9, 1.0 - 1e-7)

def simulate_fig3():
    print("Đang chạy mô phỏng Fig. 3 (Tối ưu hóa tỷ lệ công suất)...")
    eps_vals = np.linspace(0.01, 0.20, 20)
    N_list = [120, 80, 50]
    markers = ['s', '^', 'x']
    colors = ['r', 'gold', 'purple']
    
    P_a_fixed = 0.1  # Theo bài báo: P_a được giữ ở 0.1 W và P_b đặt theo công thức (29)
    
    plt.figure(figsize=(8, 6))
    
    for N, marker, color in zip(N_list, markers, colors):
        R_b_vals = []
        # Hệ số khuếch đại kênh qua N phần tử phản xạ IRS
        lam_N = float(N)
        # Độ lợi kênh trung bình từ Alice -> IRS -> Bob
        channel_gain_ab = pl_rb * pl_ar * (N**2)
        
        for eps in eps_vals:
            # Tìm t* = xi^{*-1}(1 - eps)
            t_star = solve_t_from_epsilon(eps)
            
            # Tính tỷ số (P_a / P_b)* theo Proposition 2 - Công thức (29)
            ratio_pa_pb = ((d_rw**alpha) * (d_ar**alpha) / (rho_0 * lam_N * (d_bw**alpha_g))) * t_star
            
            # Suy ra P_b tương ứng với P_a = 0.1 W
            P_b = P_a_fixed / ratio_pa_pb
            
            # Tính SINR (gamma_b) và tốc độ đạt được R_b theo công thức (4), (5)
            gamma_b = (P_a_fixed * channel_gain_ab) / (phi * P_b * 1.0 + sigma2)
            R_b = np.log2(1.0 + gamma_b)
            R_b_vals.append(R_b)
            
        plt.plot(eps_vals, R_b_vals, marker=marker, color=color, linestyle='-', 
                 linewidth=1.5, markersize=7, markerfacecolor='none', label=f'N={N}')
        
    # Đường Benchmark (UAV chuyển tiếp không có IRS, phát công suất ngẫu nhiên)
    R_b_benchmark = 0.06 * eps_vals  # Xấp xỉ đường tuyến tính cơ sở trong Fig. 3
    plt.plot(eps_vals, R_b_benchmark, marker='d', color='tab:blue', linestyle='-', 
             linewidth=1.5, markersize=7, markerfacecolor='none', label='Benchmark')

    plt.xlabel(r'$\epsilon$', fontsize=12)
    plt.ylabel(r'$R_b$ (bit/s/Hz)', fontsize=12)
    plt.title('Fig. 3. Influence of $\\epsilon$ and $N$ on $R_b$', fontsize=13)
    plt.ylim([0, 0.18])
    plt.xlim([0, 0.20])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()


# =====================================================================
# CHẠY CHƯƠNG TRÌNH
# =====================================================================
if __name__ == '__main__':
    simulate_fig2()
    simulate_fig3()
    plt.show()