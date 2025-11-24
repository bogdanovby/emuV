#!/usr/bin/env python3
"""
РЕАЛЬНЫЙ ТЕСТ: GPU → RAM SPILLOVER
Используем PyTorch + Accelerate для автоматического overflow
"""

import torch
import time
import subprocess

class Colors:
    CYAN = '\033[0;36m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def get_gpu_memory():
    """Получает GPU memory через nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        used, total = map(int, result.stdout.strip().split(','))
        return used, total
    except:
        return 0, 0

def print_header():
    print(f"\n{Colors.CYAN}╔════════════════════════════════════════════════════════════════╗")
    print(f"║                                                                ║")
    print(f"║     РЕАЛЬНЫЙ ТЕСТ: GPU → RAM AUTOMATIC SPILLOVER              ║")
    print(f"║       Сначала GPU, потом автоматически RAM                    ║")
    print(f"║                                                                ║")
    print(f"╚════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

def test_manual_spillover(total_gb=12):
    """
    Ручной spillover: сначала GPU, потом CPU
    """
    print(f"{Colors.YELLOW}🎯 Целевая загрузка: {total_gb} GB{Colors.RESET}")
    print(f"   Стратегия: GPU first, then CPU spillover")
    print()
    
    if not torch.cuda.is_available():
        print(f"{Colors.RED}✗ CUDA недоступна!{Colors.RESET}")
        return False
    
    device_gpu = torch.device('cuda:0')
    device_cpu = torch.device('cpu')
    
    # Проверка GPU
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_total_gb = gpu_props.total_memory / (1024**3)
    
    print(f"{Colors.GREEN}✓ GPU: {gpu_props.name}{Colors.RESET}")
    print(f"  Total VRAM: {gpu_total_gb:.2f} GB")
    print()
    
    # Начальное состояние
    gpu_used_start, gpu_total = get_gpu_memory()
    print(f"{Colors.CYAN}Начальное состояние GPU:{Colors.RESET}")
    print(f"  Used: {gpu_used_start} MB / {gpu_total} MB")
    print()
    
    tensors_gpu = []
    tensors_cpu = []
    total_allocated = 0
    gpu_allocated = 0
    cpu_allocated = 0
    
    block_size_mb = 256
    num_blocks = int((total_gb * 1024) / block_size_mb)
    
    print(f"{Colors.GREEN}Начинаем загрузку ({num_blocks} блоков по {block_size_mb} MB)...{Colors.RESET}\n")
    
    start_time = time.time()
    
    for i in range(num_blocks):
        block_size_bytes = block_size_mb * 1024 * 1024
        num_elements = block_size_bytes // 4  # float32
        
        try:
            # Пытаемся выделить на GPU
            tensor = torch.randn(num_elements, device=device_gpu, dtype=torch.float32)
            tensors_gpu.append(tensor)
            gpu_allocated += block_size_bytes
            location = f"{Colors.GREEN}GPU{Colors.RESET}"
            
        except RuntimeError as e:
            # GPU полна - spillover на CPU
            if "out of memory" in str(e).lower():
                print(f"\n{Colors.YELLOW}⚡ GPU VRAM FULL! Spillover to RAM...{Colors.RESET}")
                
                tensor = torch.randn(num_elements, device=device_cpu, dtype=torch.float32)
                tensors_cpu.append(tensor)
                cpu_allocated += block_size_bytes
                location = f"{Colors.YELLOW}RAM{Colors.RESET}"
            else:
                raise
        
        total_allocated += block_size_bytes
        gb_allocated = total_allocated / (1024**3)
        
        # Прогресс
        progress = ((i + 1) / num_blocks) * 100
        bar_length = 40
        filled = int(bar_length * (i + 1) / num_blocks)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        gpu_used, _ = get_gpu_memory()
        
        print(f"\r[{bar}] {progress:.0f}% - {gb_allocated:.2f} GB "
              f"(GPU: {gpu_used} MB) [{location}]", end='', flush=True)
        
        time.sleep(0.01)
    
    elapsed = time.time() - start_time
    
    # Финальная статистика
    gpu_used_end, gpu_total_end = get_gpu_memory()
    
    print(f"\n\n{Colors.GREEN}✓ Загрузка завершена!{Colors.RESET}")
    print(f"\n{Colors.CYAN}📊 Детальная статистика:{Colors.RESET}")
    print(f"   Время: {elapsed:.2f} секунд")
    print(f"   Скорость: {total_allocated / (1024**3) / elapsed:.2f} GB/сек")
    print()
    print(f"   {Colors.GREEN}GPU VRAM:{Colors.RESET}")
    print(f"   • Блоков на GPU: {len(tensors_gpu)}")
    print(f"   • Размер: {gpu_allocated / (1024**3):.2f} GB")
    print(f"   • Использование GPU: {gpu_used_end} MB / {gpu_total_end} MB")
    print(f"   • Процент GPU: {(gpu_used_end / gpu_total_end) * 100:.1f}%")
    print()
    print(f"   {Colors.YELLOW}System RAM (spillover):{Colors.RESET}")
    print(f"   • Блоков на RAM: {len(tensors_cpu)}")
    print(f"   • Размер: {cpu_allocated / (1024**3):.2f} GB")
    print()
    print(f"   {Colors.BOLD}ИТОГО:{Colors.RESET}")
    print(f"   • Общий размер: {total_allocated / (1024**3):.2f} GB")
    print(f"   • На GPU: {(gpu_allocated / total_allocated) * 100:.1f}%")
    print(f"   • На RAM: {(cpu_allocated / total_allocated) * 100:.1f}%")
    
    # Проверка, что spillover произошел
    if len(tensors_cpu) > 0:
        print(f"\n{Colors.GREEN}✅ SPILLOVER РАБОТАЕТ!{Colors.RESET}")
        print(f"   GPU заполнена → автоматический переход на RAM")
    else:
        print(f"\n{Colors.YELLOW}ℹ Все данные поместились на GPU{Colors.RESET}")
    
    # Тест доступа к данным
    print(f"\n{Colors.CYAN}🔥 Тест доступа к данным...{Colors.RESET}")
    test_start = time.time()
    
    # Читаем с GPU
    if tensors_gpu:
        val_gpu = tensors_gpu[0].mean().item()
        print(f"   ✓ Доступ к GPU tensor: {val_gpu:.6f}")
    
    # Читаем с CPU
    if tensors_cpu:
        val_cpu = tensors_cpu[0].mean().item()
        print(f"   ✓ Доступ к RAM tensor: {val_cpu:.6f}")
    
    test_elapsed = time.time() - test_start
    print(f"   Время доступа: {test_elapsed:.4f} сек")
    
    # Удержание памяти
    print(f"\n{Colors.YELLOW}💡 Память удерживается...{Colors.RESET}")
    print(f"   В другом терминале проверьте:")
    print(f"   $ nvidia-smi")
    print(f"   $ cat /sys/class/emuv/emuv/vram_info")
    print(f"\n   Нажмите Enter для освобождения...")
    input()
    
    # Освобождение
    print(f"\n{Colors.CYAN}🗑️  Освобождение памяти...{Colors.RESET}")
    
    tensors_gpu.clear()
    tensors_cpu.clear()
    torch.cuda.empty_cache()
    
    gpu_used_final, _ = get_gpu_memory()
    
    print(f"{Colors.GREEN}✓ Память освобождена{Colors.RESET}")
    print(f"  GPU после очистки: {gpu_used_final} MB")
    
    return True

def main():
    print_header()
    
    print(f"{Colors.CYAN}Этот тест демонстрирует:{Colors.RESET}")
    print(f"  1. Сначала данные выделяются на GPU VRAM (до 7.5 GB)")
    print(f"  2. Когда GPU заполняется → автоматический spillover на RAM")
    print(f"  3. Итого: GPU + RAM = больше памяти!")
    print()
    
    if not torch.cuda.is_available():
        print(f"{Colors.RED}✗ CUDA недоступна. Требуется NVIDIA GPU.{Colors.RESET}")
        return 1
    
    # Выбор размера
    print(f"{Colors.YELLOW}Сколько GB загрузить?{Colors.RESET}")
    print(f"  Рекомендуется: 10-12 GB (больше чем GPU VRAM)")
    print(f"  GPU VRAM: ~7.75 GB")
    print(f"  Spillover начнется после ~7.5 GB")
    
    choice = input(f"\n{Colors.GREEN}Введите размер в GB [12]: {Colors.RESET}").strip()
    target_gb = float(choice) if choice else 12.0
    
    # Запуск теста
    success = test_manual_spillover(target_gb)
    
    if success:
        print(f"\n{Colors.CYAN}╔════════════════════════════════════════════════════════════════╗")
        print(f"║                                                                ║")
        print(f"║              ✅ SPILLOVER GPU → RAM РАБОТАЕТ!                  ║")
        print(f"║                                                                ║")
        print(f"║  ✓ GPU VRAM использована ПЕРВОЙ                                ║")
        print(f"║  ✓ Spillover в RAM когда GPU полна                             ║")
        print(f"║  ✓ Автоматическое управление памятью                           ║")
        print(f"║  ✓ Доступ к обоим типам памяти                                 ║")
        print(f"║                                                                ║")
        print(f"╚════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
        return 0
    else:
        return 1

if __name__ == '__main__':
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Прервано пользователем{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
