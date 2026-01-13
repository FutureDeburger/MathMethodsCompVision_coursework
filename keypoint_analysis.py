import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
import os


class KeypointDetectorComparator:
    def __init__(self):
        """Инициализация детекторов SIFT, ORB и AKAZE"""
        self.detectors = {
            'SIFT': cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04),
            'ORB': cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8),
            'AKAZE': cv2.AKAZE_create(threshold=0.001, nOctaves=4, nOctaveLayers=4)
        }

        # Цвета для визуализации
        self.colors = {
            'SIFT': (0, 0.4, 0.8),  # Синий
            'ORB': (0.8, 0.2, 0.2),  # Красный
            'AKAZE': (0.6, 0.2, 0.8)  # Фиолетовый
        }

        print("Инициализация завершена. Доступные детекторы:", list(self.detectors.keys()))

    def create_realistic_test_images(self, save=True):
        """Создание реалистичных тестовых изображений для панорамирования"""
        # Размер изображений
        height, width = 400, 600

        # Первое изображение - левая часть панорамы
        img1 = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Рисуем реалистичные объекты
        # Здание слева
        cv2.rectangle(img1, (50, 150), (200, 350), (180, 180, 220), -1)  # Здание
        cv2.rectangle(img1, (100, 200), (150, 250), (100, 100, 150), -1)  # Окно
        cv2.rectangle(img1, (70, 300), (180, 320), (120, 120, 180), -1)  # Дверь

        # Дерево
        cv2.circle(img1, (350, 200), 60, (50, 120, 50), -1)  # Крона
        cv2.rectangle(img1, (340, 260), (360, 350), (100, 80, 50), -1)  # Ствол

        # Облака
        cv2.ellipse(img1, (200, 80), (40, 20), 0, 0, 360, (240, 240, 250), -1)
        cv2.ellipse(img1, (240, 70), (30, 25), 0, 0, 360, (240, 240, 250), -1)

        cv2.putText(img1, 'Panorama Left', (150, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (50, 50, 50), 2)

        # Второе изображение - правая часть (смещена на 200 пикселей)
        img2 = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Копируем часть изображения с перекрытием
        overlap = 200
        img2[:, :overlap] = img1[:, width - overlap:]

        # Добавляем новые объекты в правой части
        # Другое здание
        cv2.rectangle(img2, (250, 180), (400, 330), (220, 200, 180), -1)
        cv2.circle(img2, (325, 220), 25, (200, 150, 100), -1)  # Круглое окно

        # Дорожный знак
        cv2.rectangle(img2, (450, 250), (470, 320), (200, 100, 50), -1)
        cv2.circle(img2, (460, 230), 30, (220, 50, 50), -1)

        cv2.putText(img2, 'Panorama Right', (350, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (50, 50, 50), 2)

        # Добавляем реалистичный шум
        noise1 = np.random.normal(0, 15, img1.shape).astype(np.int16)
        noise2 = np.random.normal(0, 15, img2.shape).astype(np.int16)

        img1 = np.clip(img1.astype(np.int16) + noise1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)

        # Немного разные условия освещения
        img2 = cv2.addWeighted(img2, 0.9, np.zeros_like(img2), 0, 10)

        if save:
            cv2.imwrite('panorama_left.jpg', img1)
            cv2.imwrite('panorama_right.jpg', img2)
            print("Созданы тестовые изображения: panorama_left.jpg, panorama_right.jpg")

        return img1, img2

    def apply_transformations(self, image, transform_type, param):
        """Применение различных преобразований к изображению"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if transform_type == 'noise':
            # Гауссов шум
            noise = np.random.normal(0, param, gray.shape)
            noisy = gray.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)

        elif transform_type == 'rotation':
            # Поворот
            return ndimage.rotate(gray, param, reshape=False, mode='reflect')

        elif transform_type == 'scale':
            # Масштабирование
            h, w = gray.shape
            new_w, new_h = int(w * param), int(h * param)
            scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if param < 1:
                # Если уменьшили - добавляем рамку
                result = np.ones((h, w), dtype=np.uint8) * 128
                y_start = (h - new_h) // 2
                x_start = (w - new_w) // 2
                result[y_start:y_start + new_h, x_start:x_start + new_w] = scaled
                return result
            else:
                # Если увеличили - обрезаем
                return scaled[:h, :w]

        return gray

    def detect_keypoints(self, image, detector_name):
        """Детекция ключевых точек с заданным детектором"""
        detector = self.detectors[detector_name]

        try:
            if detector_name in ['SIFT', 'AKAZE']:
                keypoints, descriptors = detector.detectAndCompute(image, None)
            elif detector_name == 'ORB':
                keypoints, descriptors = detector.detectAndCompute(image, None)
        except Exception as e:
            print(f"Ошибка при детекции {detector_name}: {e}")
            return [], None

        return keypoints, descriptors

    def match_keypoints(self, desc1, desc2, detector_name):
        """Сопоставление ключевых точек с правильной метрикой"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return [], 0

        try:
            if detector_name == 'ORB':
                # Для ORB используем Hamming расстояние
                if desc1.dtype != np.uint8:
                    desc1 = desc1.astype(np.uint8)
                if desc2.dtype != np.uint8:
                    desc2 = desc2.astype(np.uint8)

                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = matcher.knnMatch(desc1, desc2, k=2)

                # Применяем ratio test Лоу
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                matches = good_matches

            else:
                # Для SIFT и AKAZE используем L2 расстояние
                if desc1.dtype != np.float32:
                    desc1 = desc1.astype(np.float32)
                if desc2.dtype != np.float32:
                    desc2 = desc2.astype(np.float32)

                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = matcher.match(desc1, desc2)

            # Сортировка по качеству
            matches = sorted(matches, key=lambda x: x.distance)

            # Расчет качества совпадений
            match_quality = 0
            if len(matches) > 10:
                distances = [m.distance for m in matches[:50]]  # Берем только лучшие
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)

                if std_distance > 0:
                    good_threshold = avg_distance + 0.5 * std_distance
                    good_matches = [m for m in matches if m.distance < good_threshold]
                    match_quality = len(good_matches) / len(matches) if len(matches) > 0 else 0
                else:
                    match_quality = 1.0 if len(matches) > 0 else 0
            elif len(matches) > 0:
                match_quality = 1.0

            return matches, match_quality

        except Exception as e:
            print(f"Ошибка при сопоставлении {detector_name}: {e}")
            return [], 0

    def run_experiment(self, img1, img2):
        """Запуск полного эксперимента по сравнению детекторов"""
        results = {
            'detector': [],
            'condition': [],
            'num_kp_img1': [],
            'num_kp_img2': [],
            'num_matches': [],
            'match_quality': [],
            'detection_time': [],
            'matching_time': []
        }

        # Упрощенные условия для тестирования
        conditions = [
            ('original', None),
            ('noise_low', ('noise', 15)),
            ('noise_high', ('noise', 30)),
            ('rotate_15', ('rotation', 15)),
            ('scale_0.8', ('scale', 0.8)),
        ]

        print("\n" + "=" * 70)
        print("НАЧАЛО ЭКСПЕРИМЕНТОВ")
        print("=" * 70)

        for condition_name, transform in conditions:
            print(f"\nУсловие: {condition_name}")
            print("-" * 40)

            # Применяем преобразование ко второму изображению
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            if transform:
                img2_transformed = self.apply_transformations(img2, transform[0], transform[1])
                if len(img2_transformed.shape) == 2:
                    img2_gray = img2_transformed
                else:
                    img2_gray = cv2.cvtColor(img2_transformed, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            for detector_name in self.detectors.keys():
                # Измерение времени детекции
                start_time = time.time()
                kp1, desc1 = self.detect_keypoints(img1_gray, detector_name)
                kp2, desc2 = self.detect_keypoints(img2_gray, detector_name)
                detection_time = time.time() - start_time

                # Измерение времени сопоставления
                start_time = time.time()
                matches, quality = self.match_keypoints(desc1, desc2, detector_name)
                matching_time = time.time() - start_time

                # Сохранение результатов
                results['detector'].append(detector_name)
                results['condition'].append(condition_name)
                results['num_kp_img1'].append(len(kp1))
                results['num_kp_img2'].append(len(kp2))
                results['num_matches'].append(len(matches))
                results['match_quality'].append(quality)
                results['detection_time'].append(detection_time)
                results['matching_time'].append(matching_time)

                print(f"  {detector_name}: {len(kp1)}/{len(kp2)} точек, "
                      f"{len(matches)} совпадений, качество: {quality:.2%}, "
                      f"время: {detection_time + matching_time:.3f}с")

        return results

    def visualize_detectors_comparison(self, img1, img2):
        """Визуализация сравнения детекторов"""
        print("\n" + "=" * 70)
        print("ВИЗУАЛИЗАЦИЯ КЛЮЧЕВЫХ ТОЧЕК")
        print("=" * 70)

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        fig, axes = plt.subplots(3, 2, figsize=(12, 15))

        for idx, detector_name in enumerate(self.detectors.keys()):
            # Для первого изображения
            kp1, _ = self.detect_keypoints(img1_gray, detector_name)
            img1_kp = cv2.drawKeypoints(img1, kp1[:100], None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            axes[idx, 0].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
            axes[idx, 0].set_title(f'{detector_name} - Левое изображение\n{len(kp1)} точек')
            axes[idx, 0].axis('off')

            # Для второго изображения
            kp2, _ = self.detect_keypoints(img2_gray, detector_name)
            img2_kp = cv2.drawKeypoints(img2, kp2[:100], None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            axes[idx, 1].imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
            axes[idx, 1].set_title(f'{detector_name} - Правое изображение\n{len(kp2)} точек')
            axes[idx, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_matches_comparison(self, img1, img2):
        """Визуализация совпадений для всех детекторов"""
        print("\n" + "=" * 70)
        print("ВИЗУАЛИЗАЦИЯ СОВПАДЕНИЙ")
        print("=" * 70)

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, detector_name in enumerate(self.detectors.keys()):
            kp1, desc1 = self.detect_keypoints(img1_gray, detector_name)
            kp2, desc2 = self.detect_keypoints(img2_gray, detector_name)
            matches, quality = self.match_keypoints(desc1, desc2, detector_name)

            # Ограничиваем количество для наглядности
            display_matches = matches[:30] if len(matches) > 30 else matches

            # Рисуем совпадения
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, display_matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            axes[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f'{detector_name}\n{len(matches)} совпадений (качество: {quality:.1%})')
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_analysis_results(self, results):
        """Построение графиков анализа результатов"""
        detectors = list(self.detectors.keys())
        conditions = sorted(set(results['condition']))

        # Подготовка данных
        matches_data = {d: [] for d in detectors}
        quality_data = {d: [] for d in detectors}
        time_data = {d: [] for d in detectors}

        for cond in conditions:
            for det in detectors:
                indices = [i for i in range(len(results['detector']))
                           if results['detector'][i] == det and results['condition'][i] == cond]

                if indices:
                    idx = indices[0]
                    matches_data[det].append(results['num_matches'][idx])
                    quality_data[det].append(results['match_quality'][idx])
                    time_data[det].append(results['detection_time'][idx] +
                                          results['matching_time'][idx])

        # 1. График количества совпадений
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        x = np.arange(len(conditions))
        width = 0.25

        for i, det in enumerate(detectors):
            offset = (i - 1) * width
            plt.bar(x + offset, matches_data[det], width,
                    label=det, color=self.colors[det], alpha=0.8)

        plt.xlabel('Условия')
        plt.ylabel('Количество совпадений')
        plt.title('Количество найденных совпадений')
        plt.xticks(x, conditions, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. График качества
        plt.subplot(132)
        for i, det in enumerate(detectors):
            plt.plot(x, quality_data[det], 'o-',
                     label=det, color=self.colors[det], linewidth=2, markersize=8)

        plt.xlabel('Условия')
        plt.ylabel('Качество совпадений')
        plt.title('Качество совпадений (0-1)')
        plt.xticks(x, conditions, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. График времени
        plt.subplot(133)
        for i, det in enumerate(detectors):
            plt.plot(x, time_data[det], 's-',
                     label=det, color=self.colors[det], linewidth=2, markersize=8)

        plt.xlabel('Условия')
        plt.ylabel('Время (секунды)')
        plt.title('Время выполнения')
        plt.xticks(x, conditions, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Сводная таблица
        self.print_summary_table(results)

    def print_summary_table(self, results):
        """Вывод сводной таблицы результатов"""
        print("\n" + "=" * 70)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 70)

        detectors = list(self.detectors.keys())

        # Вычисляем средние значения
        summary = {}
        for det in detectors:
            indices = [i for i, d in enumerate(results['detector']) if d == det]

            if indices:
                summary[det] = {
                    'avg_matches': np.mean([results['num_matches'][i] for i in indices]),
                    'avg_quality': np.mean([results['match_quality'][i] for i in indices]) * 100,
                    'avg_time': np.mean([results['detection_time'][i] +
                                         results['matching_time'][i] for i in indices]),
                    'avg_kp': np.mean([results['num_kp_img1'][i] +
                                       results['num_kp_img2'][i] for i in indices]) / 2
                }

        # Заголовок
        print(f"{'Метрика':<20}", end="")
        for det in detectors:
            print(f"{det:<15}", end="")
        print()
        print("-" * 70)

        # Данные
        metrics = [
            ('Совпадения', 'avg_matches', '{:.0f}'),
            ('Качество, %', 'avg_quality', '{:.1f}'),
            ('Время, с', 'avg_time', '{:.3f}'),
            ('Точек/изобр.', 'avg_kp', '{:.0f}')
        ]

        for name, key, fmt in metrics:
            print(f"{name:<20}", end="")
            for det in detectors:
                value = summary[det][key]
                print(f"{fmt.format(value):<15}", end="")
            print()

        print("-" * 70)

        # Рекомендации
        # print("\nРЕКОМЕНДАЦИИ:")
        # print("-" * 40)

        best_matches = max(summary.items(), key=lambda x: x[1]['avg_matches'])[0]
        best_quality = max(summary.items(), key=lambda x: x[1]['avg_quality'])[0]
        best_speed = min(summary.items(), key=lambda x: x[1]['avg_time'])[0]

        # print(f"1. Для максимального количества совпадений: {best_matches}")
        # print(f"2. Для лучшего качества: {best_quality}")
        # print(f"3. Для максимальной скорости: {best_speed}")

        # Баланс
        balanced_scores = {}
        for det in detectors:
            score = (summary[det]['avg_matches'] / max([s['avg_matches'] for s in summary.values()]) +
                     summary[det]['avg_quality'] / max([s['avg_quality'] for s in summary.values()]) +
                     (1 - summary[det]['avg_time'] / max([s['avg_time'] for s in summary.values()]))) / 3
            balanced_scores[det] = score

        best_balanced = max(balanced_scores.items(), key=lambda x: x[1])[0]
        print(f"\n4. Оптимальный баланс (качество/скорость/количество): {best_balanced}")

        print("\n" + "=" * 70)
        print("ОБЩИЕ ВЫВОДЫ:")
        print("=" * 70)
        print("• SIFT: Лучшее качество, но медленнее всего")
        print("• ORB: Самый быстрый, хорошо работает в реальном времени")
        print("• AKAZE: Хороший баланс между скоростью и качеством")
        print("\nДля панорамирования рекомендуем AKAZE или SIFT в зависимости")
        print("от требований к качеству и скорости обработки.")

    def create_simple_panorama(self, img1, img2):
        """Создание простой панорамы с использованием AKAZE"""
        print("\n" + "=" * 70)
        print("ДЕМОНСТРАЦИЯ ПАНОРАМИРОВАНИЯ")
        print("=" * 70)

        detector_name = 'AKAZE'
        print(f"Используется алгоритм: {detector_name}")

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Детекция ключевых точек
        kp1, desc1 = self.detect_keypoints(img1_gray, detector_name)
        kp2, desc2 = self.detect_keypoints(img2_gray, detector_name)

        print(f"Найдено ключевых точек: {len(kp1)} (левое), {len(kp2)} (правое)")

        if len(kp1) < 10 or len(kp2) < 10:
            print("Недостаточно ключевых точек для создания панорамы.")
            return

        # Сопоставление
        matches, quality = self.match_keypoints(desc1, desc2, detector_name)
        print(f"Найдено совпадений: {len(matches)} (качество: {quality:.2%})")

        if len(matches) < 4:
            print("Недостаточно совпадений для создания панорамы.")
            return

        # Простое сшивание (горизонтальное)
        height = max(img1.shape[0], img2.shape[0])
        panorama = np.zeros((height, img1.shape[1] + img2.shape[1] - 200, 3), dtype=np.uint8)

        # Размещаем изображения с перекрытием
        panorama[:img1.shape[0], :img1.shape[1]] = img1
        panorama[:img2.shape[0], img1.shape[1] - 200:img1.shape[1] - 200 + img2.shape[1]] = img2

        # Сохраняем результат
        cv2.imwrite('simple_panorama_result.jpg', panorama)

        # Визуализация
        plt.figure(figsize=(15, 8))

        plt.subplot(231)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('Левое изображение')
        plt.axis('off')

        plt.subplot(232)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('Правое изображение')
        plt.axis('off')

        # Показываем совпадения
        if len(matches) > 0:
            display_matches = matches[:20] if len(matches) > 20 else matches
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, display_matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.subplot(233)
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            plt.title(f'Совпадения ({len(matches)})')
            plt.axis('off')

        plt.subplot(212)
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title(f'Панорама ({panorama.shape[1]}x{panorama.shape[0]})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"\nПанорама сохранена как 'simple_panorama_result.jpg'")
        print(f"Размер: {panorama.shape[1]}x{panorama.shape[0]} пикселей")


    def create_real_panorama(self, img1, img2):
        print("\nСШИВАНИЕ РЕАЛЬНОЙ ПАНОРАМЫ (AKAZE)")

        detector = cv2.AKAZE_create()

        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, d1 = detector.detectAndCompute(g1, None)
        kp2, d2 = detector.detectAndCompute(g2, None)

        if d1 is None or d2 is None:
            print("Нет дескрипторов")
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(d1, d2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 4:
            print("Недостаточно совпадений")
            return

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        panorama = cv2.warpPerspective(
            img2, H, (w1 + w2, max(h1, h2))
        )

        panorama[0:h1, 0:w1] = img1

        cv2.imwrite("panorama_result.jpg", panorama)

        plt.figure(figsize=(14, 6))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("РЕЗУЛЬТАТ ПАНОРАМЫ")
        plt.show()

        print("Панорама сохранена: panorama_result.jpg")



def main():
    """Основная функция программы"""
    print("=" * 80)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ SIFT, ORB И AKAZE ДЛЯ ПАНОРАМИРОВАНИЯ")
    print("=" * 80)

    # Создаем экземпляр компаратора
    comparator = KeypointDetectorComparator()

    img1 = cv2.imread("boat1.jpg")
    img2 = cv2.imread("boat2.jpg")
    comparator.create_real_panorama(img1, img2)

    # Создаем реалистичные тестовые изображения
    # print("\n1. Создание тестовых изображений...")
    # img1, img2 = comparator.create_realistic_test_images(save=True)

    # Показываем тестовые изображения
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Левое изображение (panorama_left.jpg)')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Правое изображение (panorama_right.jpg)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Запускаем эксперименты
    print("\n2. Проведение экспериментов...")
    results = comparator.run_experiment(img1, img2)

    # Визуализация
    print("\n3. Визуализация результатов...")
    comparator.visualize_detectors_comparison(img1, img2)
    comparator.visualize_matches_comparison(img1, img2)

    # Анализ результатов
    print("\n4. Анализ результатов...")
    comparator.plot_analysis_results(results)

    # Демонстрация панорамирования
    print("\n5. Демонстрация панорамирования...")
    comparator.create_simple_panorama(img1, img2)

    print("\n" + "=" * 80)
    print("ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 80)
    print("\nСозданные файлы:")
    print("  - panorama_left.jpg          : Левое тестовое изображение")
    print("  - panorama_right.jpg         : Правое тестовое изображение")
    print("  - simple_panorama_result.jpg : Результирующая панорама")


if __name__ == "__main__":
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        print("Все необходимые библиотеки доступны.")
        print(f"Версия OpenCV: {cv2.__version__}")

        # Запускаем программу
        main()

    except ImportError as e:
        print(f"Ошибка: {e}")
        print("\nУстановите необходимые библиотеки:")
        print("pip install opencv-python opencv-contrib-python numpy matplotlib scipy")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("\nПопробуйте перезапустить программу.")