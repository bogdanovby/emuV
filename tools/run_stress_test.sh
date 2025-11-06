#!/bin/bash
# Скрипт для запуска нагрузочного теста виртуальной видеокарты

set -e

echo "=========================================="
echo "Нагрузочный тест виртуальной видеокарты"
echo "=========================================="
echo ""

# Проверка наличия модуля
if ! lsmod | grep -q "^emuv "; then
    echo "ОШИБКА: Модуль emuv не загружен!"
    echo "Загрузите модуль: sudo insmod emuv.ko"
    exit 1
fi

# Компиляция теста
if [ ! -f "stress_test_emuv" ] || [ "stress_test_emuv.c" -nt "stress_test_emuv" ]; then
    echo "Компиляция stress_test_emuv..."
    gcc -o stress_test_emuv stress_test_emuv.c -lpthread
    if [ $? -ne 0 ]; then
        echo "ОШИБКА: Не удалось скомпилировать тест"
        exit 1
    fi
    echo "✓ Компиляция завершена"
fi

echo ""
echo "Запуск нагрузочного теста..."
echo "Тест будет использовать всю доступную VRAM (10 GB)"
echo "Для остановки нажмите Ctrl+C"
echo ""

# Запуск теста
# Можно указать размер в ГБ: sudo ./run_stress_test.sh [размер_в_ГБ]
SIZE=${1:-10}
sudo ./stress_test_emuv $SIZE

echo ""
echo "Тест завершен"

