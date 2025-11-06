#!/usr/bin/env python3
"""
Python скрипт для тестирования виртуальной видеокарты
Использование: python3 test_emuv.py
"""

import os
import sys

def test_sysfs():
    """Тест sysfs интерфейса"""
    print("=== Тест sysfs интерфейса ===")
    
    vram_info_path = "/sys/class/emuv/emuv/vram_info"
    
    if not os.path.exists(vram_info_path):
        print(f"✗ Файл {vram_info_path} не найден")
        return False
    
    try:
        with open(vram_info_path, 'r') as f:
            content = f.read()
            print("✓ Файл найден и прочитан")
            print("\nСодержимое:")
            print(content)
            return True
    except PermissionError:
        print(f"✗ Нет прав на чтение {vram_info_path}")
        return False
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False

def test_device():
    """Тест символьного устройства"""
    print("\n=== Тест символьного устройства ===")
    
    device_path = "/dev/emuv"
    
    if not os.path.exists(device_path):
        print(f"✗ Устройство {device_path} не найдено")
        return False
    
    if os.geteuid() != 0:
        print(f"⚠ Устройство найдено, но требуется root для чтения")
        print("  Запустите: sudo python3 test_emuv.py")
        return False
    
    try:
        with open(device_path, 'r') as f:
            content = f.read(1024)
            print("✓ Устройство прочитано")
            print("\nСодержимое:")
            print(content)
            return True
    except PermissionError:
        print(f"✗ Нет прав на чтение {device_path}")
        return False
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False

def test_module():
    """Проверка загрузки модуля"""
    print("\n=== Проверка модуля ===")
    
    try:
        with open('/proc/modules', 'r') as f:
            modules = f.read()
            if 'emuv' in modules:
                print("✓ Модуль emuv загружен")
                # Извлекаем информацию о модуле
                for line in modules.split('\n'):
                    if 'emuv' in line:
                        parts = line.split()
                        print(f"  Размер: {parts[1]} байт")
                        print(f"  Использование: {parts[2]} ссылок")
                        return True
            else:
                print("✗ Модуль emuv не загружен")
                return False
    except Exception as e:
        print(f"✗ Ошибка при проверке модуля: {e}")
        return False

def test_sysfs_dir():
    """Проверка sysfs директории"""
    print("\n=== Проверка sysfs директории ===")
    
    sysfs_dir = "/sys/class/emuv"
    
    if not os.path.exists(sysfs_dir):
        print(f"✗ Директория {sysfs_dir} не найдена")
        return False
    
    if not os.path.isdir(sysfs_dir):
        print(f"✗ {sysfs_dir} не является директорией")
        return False
    
    print(f"✓ Директория найдена")
    
    try:
        entries = os.listdir(sysfs_dir)
        print(f"  Содержимое ({len(entries)} элементов):")
        for entry in entries:
            entry_path = os.path.join(sysfs_dir, entry)
            if os.path.isdir(entry_path):
                print(f"    [DIR]  {entry}/")
            elif os.path.isfile(entry_path):
                print(f"    [FILE] {entry}")
        return True
    except Exception as e:
        print(f"✗ Ошибка при чтении директории: {e}")
        return False

def main():
    print("=" * 50)
    print("Тест виртуальной видеокарты emuv")
    print("=" * 50)
    print()
    
    results = []
    
    results.append(("Модуль", test_module()))
    results.append(("Sysfs директория", test_sysfs_dir()))
    results.append(("Sysfs файл", test_sysfs()))
    results.append(("Устройство", test_device()))
    
    print("\n" + "=" * 50)
    print("Результаты:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print(f"\nПройдено: {passed}/{total}")
    
    if passed == total:
        print("\n✓ Все тесты пройдены успешно!")
        return 0
    else:
        print(f"\n✗ Провалено тестов: {total - passed}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

