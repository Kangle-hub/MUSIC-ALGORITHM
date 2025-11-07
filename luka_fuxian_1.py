import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

"""
初始化参数
K 用户数量
I IRS反射单元数量
wavelength 波长
d_I 反射单元间距
gamma IRS->BS的AOD
"""
K = 3
I = 128
wavelength = 1 # 归一化处理
d_I = wavelength / 2
gamma = 45

def steering_vector_irs(theta_deg):
    """IRS的导向矢量（ULA模型）"""
    theta_rad = theta_deg * np.pi / 180
    n = np.arange(I)
    phase = -2 * np.pi * n * d_I * np.cos(theta_rad) / wavelength

    return np.exp(1j * phase).reshape(-1, 1)

def generate_irs_patterns(L):
    """生成IRS反射系数模式"""
    phi_bar = np.zeros((L, I), dtype=complex)
    for l in range(L):
        phases = np.random.uniform(0, 2 * np.pi, I)
        phi_bar[l] = np.exp(1j * phases)

    return phi_bar

def create_virtual_steering_vector(theta_deg, phi_bar, L):
    """创建虚拟导向矢量 a_bar(theta)"""
    b_gamma = steering_vector_irs(gamma).flatten()  # 改为一维数组
    b_theta = steering_vector_irs(theta_deg).flatten()  # 改为一维数组

    a_bar = np.zeros(L, dtype=complex)
    for l in range(L):
        a_bar[l] = np.dot(b_gamma, phi_bar[l] * b_theta)

    return a_bar

def generate_temporal_signals(true_aoas, L, Q, phi_bar, SNR_dB):
    """生成时域多维接收信号"""
    SNR = 10 ** (SNR_dB / 10)
    signal_power = 1
    sigma2 = signal_power / SNR

    # 一开始仿真结果低信噪比下失效原因 我给的增益均为0.1 过小
    beta = np.ones(K) * 0.35
    delta = 0.35

    y_tilde = np.zeros((L, Q), dtype=complex)

    A_bar = np.zeros((L, K), dtype=complex)
    for k in range(K):
        A_bar[:,k] = create_virtual_steering_vector(true_aoas[k], phi_bar, L)

    for q in range(Q):
        s_tilde = np.random.randn(K) + 1j * np.random.randn(K)
        s_tilde = s_tilde / np.sqrt(2)

        x_tilde = delta * beta * np.sqrt(signal_power) * s_tilde

        noise = np.sqrt(sigma2 / 2) * (np.random.randn(L) + 1j * np.random.randn(L))

        y_tilde[:,q] = A_bar @ x_tilde + noise

    return y_tilde

def music_spectrum(y_tilde, phi_bar, L, Q, search_angles):
    """计算MUSIC谱"""
    S = np.zeros((L, L), dtype=complex)
    for q in range(Q):
        S += np.outer(y_tilde[:, q], np.conj(y_tilde[:, q]))
    S = S / Q

    eigenvalues, eigenvectors = eigh(S)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    U_noise = eigenvectors[:, K:]

    spectrum = np.zeros(len(search_angles))
    for i, theta in enumerate(search_angles):
        a_bar = create_virtual_steering_vector(theta, phi_bar, L)
        numerator = np.abs(np.dot(np.conj(a_bar), a_bar))
        denominator = np.abs(np.dot(np.conj(a_bar), U_noise @ np.conj(U_noise.T) @ a_bar))
        if denominator > 1e-10:
            spectrum[i] = numerator / denominator
        else:
            spectrum[i] = 1e10

    return spectrum

def find_peaks(spectrum, search_angles, num_peaks):
    """找到谱的峰值"""
    peaks = []

    for i in range(1, len(spectrum) - 1):
        if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
            peaks.append((spectrum[i], search_angles[i]))

    peaks.sort(reverse=True)
    estimated_aoas = [peak[1] for peak in peaks[:num_peaks]]

    return np.sort(estimated_aoas)

def main():
    """绘制Fig.3 - MUSIC谱"""
    np.random.seed(23)

    true_aoas = np.array([72.9078, 34.0409, 19.3314])
    search_angles = np.linspace(0, 90, 900001)

    L = 6
    Q = 12
    SNR_dB = 10

    phi_bar = generate_irs_patterns(L)
    y_tilde = generate_temporal_signals(true_aoas, L, Q, phi_bar, SNR_dB)
    spectrum = music_spectrum(y_tilde, phi_bar, L, Q, search_angles)
    normalized_spectrum = spectrum / np.max(spectrum)

    plt.figure(figsize=(10, 6))
    plt.plot(search_angles, normalized_spectrum, 'b-', linewidth=2, label='Normalized Spectrum')

    for aoa in true_aoas:
        plt.axvline(x=aoa, color='r', linestyle='--', linewidth=2, alpha=0.7)

    plt.plot([], [], 'r--', linewidth=2, label='True AOA')

    plt.xlabel('AOA (Degree)', fontsize=12)
    plt.ylabel('Normalized Spectrum', fontsize=12)
    plt.title('Normalized Spectrum of the MUSIC Algorithm', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim([0, 90])
    plt.ylim([0, 1.1])

    estimated_aoas = find_peaks(normalized_spectrum, search_angles, 3)
    print(f"True AOAs: {true_aoas}")
    print(f"Estimated AOAs: {[f'{aoa:.4f}' for aoa in estimated_aoas]}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== IRS Assisted MUSIC Algorithm for AOA Estimation ===")
    print("Generating Figure 3...")
    main()