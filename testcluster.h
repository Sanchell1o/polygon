
#ifndef POLYGON_TESTCLUSTER_H
#define POLYGON_TESTCLUSTER_H
#include <vector>

// Функция для вычисления метрики
double calculateMetric(const std::vector<double>& cluster);

// Рекурсивная функция для нахождения всех разбиений
void findClusters(const std::vector<double>& array, int index, int K,
                  std::vector<std::vector<std::vector<double>>>& allClusters,
                  std::vector<std::vector<double>>& currentClusters);

// Основная функция для нахождения кластеров
void findOptimalClusters(const std::vector<double>& array, int K);

// Функция для запуска тестов
void runTests();
#endif //POLYGON_TESTCLUSTER_H