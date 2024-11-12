# Работа с данными
import numpy as np
import pandas as pd

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Улучшение модели
from sklearn.model_selection import train_test_split

# Метрики
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Torch
import torch
from torch import nn

# Остальное
from IPython.display import clear_output
from tqdm.notebook import tqdm
from PIL import Image
import os, random

# Config
from src.config import *


def set_all_seeds(seed=42):
    # python's seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # torch's seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset, test_size=0.2):
    """
    Функция для разделения датасета на тренировочную и тестовую выборки с сохранением распределения классов.
    Датасет состоит из кортежей, где первый элемент — изображение (PIL), а второй элемент — метка класса.
    
    Параметры:
    ----------
    dataset : list of tuples
        Список кортежей вида (PIL.Image, class), где `PIL.Image` — объект изображения,
        а `class` — метка класса, представленная любым типом данных, пригодным для использования в качестве ключа.
        
    test_size : float, optional, default=0.2
        Доля данных, которая будет выделена для тестовой выборки. По умолчанию, 20% данных будет использовано для тестирования.
        
    Возвращает:
    -----------
    trainset : list of tuples
        Список кортежей для тренировочной выборки, где каждый кортеж состоит из изображения (PIL.Image) и метки класса.
        
    testset : list of tuples
        Список кортежей для тестовой выборки, где каждый кортеж состоит из изображения (PIL.Image) и метки класса.
    
    Примечания:
    -----------
    - Датасет перемешивается перед разделением.
    - Данные каждого класса разделяются пропорционально на тренировочную и тестовую выборки.
    """
    
    trainset, testset = train_test_split(dataset, test_size=test_size, random_state=42, shuffle=True, stratify=list(map(lambda x: x[1], dataset)))
    
    return trainset, testset


def denormalize(image):
    """
    Денормализует тензор изображения и преобразует его в формат uint8 с диапазоном [0, 255].

    :param tensor: Нормализованный тензор изображения (C, H, W).
    :return: Денормализованное изображение в формате uint8 с диапазоном [0, 255].
    """

    # Преобразуем mean и std в тензоры и переносим их на то же устройство, что и image
    tensor_mean = torch.tensor(mean).view(-1, 1, 1).to(image.device)
    tensor_std = torch.tensor(std).view(-1, 1, 1).to(image.device)

    # Денормализация: (тензор * std) + mean
    denormalize_image = image * tensor_std + tensor_mean

    # Преобразуем в диапазон [0, 255] и к типу uint8
    return (denormalize_image * 255).clamp(0, 255).byte()


def show_image(dataset, classes, amount=None, unique=True, grid_size=(3, 3), figsize=(10, 10)):
    """
    Функция для отображения изображений из датасета в виде сетки, с возможностью выбирать уникальные классы.
    Изображения могут быть как черно-белыми, так и цветными (RGB). Функция выполняет денормализацию в диапазон [0, 1].
    """
    # Установка стиля
    sns.set(style="white", palette="muted", color_codes=True)

    # Установка количества изображений для отображения
    if amount is None:
        amount = len(classes) if unique else len(dataset)
    amount = min(amount, len(dataset))

    rows, cols = grid_size
    max_cells = rows * cols  # Максимальное количество ячеек на сетке
    amount = min(amount, max_cells)  # Ограничиваем количество отображаемых изображений

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    idx = 0
    shown = 0

    while shown < amount and idx < len(dataset):
        image_tensor, label = dataset[idx]
        idx += 1

        # Пропускаем дубликаты классов, если нужны уникальные изображения
        if unique and any(label == dataset[i][1] for i in range(idx - 1)):
            continue

        # Денормализация для отображения (диапазон [0, 1])
        image_tensor = denormalize(image_tensor).float() / 255.0
        ax = axes[shown]
        shown += 1

        # Определяем тип изображения (RGB или ч/б)
        if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
            image = image_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
            ax.imshow(image)
        elif image_tensor.dim() == 3 and image_tensor.shape[0] == 1 or image_tensor.dim() == 2:
            if image_tensor.dim() == 3:
                image_tensor = image_tensor[0]  # Убираем лишний канал
            ax.imshow(image_tensor.numpy(), cmap='gray')

        ax.set_title(f'Class: {classes[label]}', fontsize=12, color='blue')
        ax.axis('off')

    # Отключаем лишние оси, если изображений меньше, чем ячеек
    for i in range(shown, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def reading_dataset_from_file(path, series, transform=None):
    """
    Функция для чтения изображений из указанной директории и создания датасета в формате (изображение, индекс класса).
    Изображения могут быть преобразованы в нужный формат (RGB или черно-белый), изменены по размеру и, при необходимости,
    к ним могут быть применены дополнительные трансформации.

    Параметры:
    ----------
    path : str
        Путь к директории, содержащей изображения.
    
    series : pandas.Series
        Серия, содержащая информацию о классах изображений. Значения должны быть метками классов, а индексы — именами файлов изображений.
    
    transform : torchvision.transforms, optional, default=None
        Трансформация, которая будет применена к каждому изображению после его загрузки (например, нормализация, аугментация).

    Возвращает:
    -----------
    data : list of tuples
        Список кортежей, где каждый кортеж состоит из изображения (PIL.Image) и индекса класса.
    
    Примечания:
    -----------
    - Функция автоматически преобразует изображения в нужный формат (RGB или черно-белый) и изменяет их размер.
    - При возникновении ошибки при чтении или обработке файла выводится сообщение с указанием файла и ошибки.
    - Серия должна содержать метки классов, соответствующие каждому изображению.
    """
    data = []
    
    for image_name in tqdm(series.index):  # Проходим только по индексам, которые есть в серии
        try:
            # Полный путь к изображению
            image_path = os.path.join(path, image_name)
            
            # Проверяем, существует ли файл
            if not os.path.isfile(image_path):
                print(f"Файл {image_name} не найден.")
                continue
            
            # Загружаем изображение
            image = Image.open(image_path)

            # Преобразуем изображение в RGB или черно-белое в зависимости от параметра image_size
            if image_size[0] == 3:
                image = image.convert("RGB")
            elif image_size[0] == 1:
                image = image.convert("L")

            # Изменяем размер изображения, если он не соответствует заданному
            if image.size != image_size[1:]:
                image = image.resize(image_size[1:])

            # Применяем трансформации, если они заданы
            if transform is not None:
                image = transform(image)

            # Получаем индекс класса для изображения из серии
            class_index = series.loc[image_name]

            # Добавляем кортеж (изображение, индекс класса) в список
            data.append((image, class_index))
        
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_name}: {e}")

    return data


def reading_dataset_from_folders(root_dir, transform=None):
    """
    Функция для чтения изображений из директорий, где каждая папка представляет отдельный класс.
    Изображения могут быть преобразованы в нужный формат (RGB или черно-белый), изменены по размеру и,
    при необходимости, к ним могут быть применены дополнительные трансформации.

    Параметры:
    ----------
    root_dir : str
        Путь к корневой директории, содержащей папки с изображениями. Каждая папка представляет собой отдельный класс.
    
    transform : torchvision.transforms, optional, default=None
        Трансформация, которая будет применена к каждому изображению после его загрузки (например, нормализация, аугментация).

    Возвращает:
    -----------
    data : list of tuples
        Список кортежей, где каждый кортеж состоит из изображения (PIL.Image) и индекса класса.
    
    classes : list
        Список классов (имена папок), где каждый элемент соответствует метке класса.

    Примечания:
    -----------
    - Функция автоматически преобразует изображения в нужный формат (RGB или черно-белое) и изменяет их размер.
    - При возникновении ошибки при чтении или обработке файла выводится сообщение с указанием файла и ошибки.
    - Все классы должны быть представлены в виде папок, где имена папок соответствуют классам изображений.
    """

    data = []
    # Получаем список классов (названия папок)
    classes = sorted(os.listdir(root_dir))
    # Создаем маппинг классов на индексы
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}  
    
    # Прогресс-бар для классов
    for class_name in tqdm(classes, desc="Загрузка классов", unit="класс"):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Пропускаем файлы, если это не папка

        # Получаем список изображений текущего класса
        image_names = os.listdir(class_dir)

        # Вложенный прогресс-бар для изображений
        for image_name in tqdm(image_names, desc=f"Обработка {class_name}", leave=False):
            try:
                # Формируем путь к изображению и открываем его
                image_path = os.path.join(class_dir, image_name)
                image = Image.open(image_path)

                # Преобразуем изображение в RGB или черно-белое в зависимости от параметра image_size
                if image_size[0] == 3:
                    image = image.convert("RGB")
                elif image_size[0] == 1:
                    image = image.convert("L")

                # Изменяем размер изображения, если он не соответствует заданному
                if image.size != image_size[1:]:
                    image = image.resize(image_size[1:])

                # Применяем трансформации, если они заданы
                if transform is not None:
                    image = transform(image)

                # Добавляем кортеж (изображение, индекс класса) в список данных
                data.append((image, class_to_idx[class_name]))  
            except Exception as e:
                # Выводим сообщение об ошибке при обработке изображения
                print(f"Ошибка при обработке изображения {image_name} в классе {class_name}: {e}")

    return data, classes


def reading_testset(path, transform=None):
    """
    Функция для чтения изображений из указанной директории и создания датасета в формате (изображение).
    Изображения могут быть преобразованы в нужный формат (RGB или черно-белый), изменены по размеру и, при необходимости,
    к ним могут быть применены дополнительные трансформации.

    Параметры:
    ----------
    path : str
        Путь к директории, содержащей изображения.
    
    transform : torchvision.transforms, optional, default=None
        Трансформация, которая будет применена к каждому изображению после его загрузки (например, нормализация, аугментация).

    Возвращает:
    -----------
    data : list of PIL.Image
        Список изображений в формате PIL.Image.
    
    Примечания:
    -----------
    - Функция автоматически преобразует изображения в нужный формат (RGB или черно-белый) и изменяет их размер.
    - При возникновении ошибки при чтении или обработке файла выводится сообщение с указанием файла и ошибки.
    """
    data = []
    
    image_names = os.listdir(path)
    for image_name in tqdm(image_names):  # Проходим по всем файлам в директории
        try:
            # Полный путь к изображению
            image_path = os.path.join(path, image_name)

            # Проверяем, существует ли файл и является ли он изображением
            if not os.path.isfile(image_path):
                print(f"Файл {image_name} не найден или не является файлом.")
                continue
            
            # Загружаем изображение
            image = Image.open(image_path)

            # Преобразуем изображение в RGB или черно-белое в зависимости от параметра image_size
            if image_size[0] == 3:
                image = image.convert("RGB")
            elif image_size[0] == 1:
                image = image.convert("L")

            # Изменяем размер изображения, если он не соответствует заданному
            if image.size != image_size[1:]:
                image = image.resize(image_size[1:])

            # Применяем трансформации, если они заданы
            if transform is not None:
                image = transform(image)

            # Добавляем изображение в список
            data.append(image)
        
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_name}: {e}")

    return data, image_names


def metrics(y_true, y_pred, count_round=4):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    rounded_scores = np.round([accuracy, precision, recall, f1], count_round)
    index_labels = ['accuracy', 'precision', 'recall', 'f1']
    
    return pd.Series(rounded_scores, index=index_labels)


class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0, act_funс=nn.ReLU):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(input_size, output_size)
        self.act_funс = act_funс()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.linear(x)
        x = self.act_funс(x)
        x = self.dropout(x)
        return x
    

class Conv2dBlock(nn.Module):
    def __init__(self, input_size, output_size, maxpool2d=True, act_funс=nn.ReLU):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(input_size)
        self.conv2d = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=3, stride=1, padding=1)
        self.act_funс = act_funс()
        if maxpool2d:
            self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        else:
            self.maxpool2d = None

    def forward(self, x):
        x = self.batchnorm2d(x)
        x = self.conv2d(x)
        x = self.act_funс(x)
        if self.maxpool2d is not None:
            x = self.maxpool2d(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, *submodule):
        super().__init__()
        self.submodule = nn.Sequential(*submodule)

    def forward(self, x):
        return x + self.submodule(x)  # Простое добавление входа к выходу подмодуля


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Переносим модель на устройство (CPU или GPU)
        self.__model = model.to(self.device)

        # Инициализируем лучшую метрику
        self.best_score = None

        # Флаг для остановки обучения
        self.stop_fiting = False

    def forward(self, x):
        # Прямой проход модели
        return self.__model(x)

    def train(self, dataloader, optimizer, loss_fn, metric):
        # Устанавливаем модель в режим обучения
        self.__model.train()
        total_loss = 0    
        y_true, y_pred = [], []

        # Используем tqdm для отображения прогресса, включая loss и метрику
        progress_bar = tqdm(dataloader, desc='Training')
        try:
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                output = self.__model(x)
                loss = loss_fn(output, y)

                # Обратное распространение ошибки
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(output.argmax(dim=1).cpu().numpy())

                # Обновляем описание tqdm с текущими значениями
                current_loss = total_loss / len(y_true)
                current_metric = metric(y_true, y_pred)
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", metric=f"{current_metric:.4f}")
        except KeyboardInterrupt:
            # Устанавливаем флаг для остановки обучения
            self.stop_fiting = True
            print("\nОбучение прервано пользователем. Завершаем текущую эпоху...")

        # Возвращаем средний loss и метрику за эпоху
        return total_loss / len(dataloader), metric(y_true, y_pred)

    @torch.inference_mode()
    def evaluate(self, dataloader, loss_fn, metric):
        # Устанавливаем модель в режим оценки
        self.__model.eval()
        total_loss = 0
        y_true, y_pred = [], []

        progress_bar = tqdm(dataloader, desc='Evaluating')
        try:
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)

                output = self.__model(x)
                loss = loss_fn(output, y)

                total_loss += loss.item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(output.argmax(dim=1).cpu().numpy())

                current_loss = total_loss / len(y_true)
                try:
                    current_metric = metric(y_true, y_pred)
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", metric=f"{current_metric:.4f}")
                except:
                    current_metric = accuracy_score(y_true, y_pred)
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", metric=f"{current_metric:.4f}")

        except KeyboardInterrupt:
            # Устанавливаем флаг для остановки обучения
            self.stop_fiting = True
            print("\nОценка прервана пользователем. Завершаем текущую эпоху...")

        # Возвращаем средний loss и метрику за эпоху
        return total_loss / len(dataloader), metric(y_true, y_pred)

    @staticmethod
    def plot_stats(train_loss, valid_loss, train_score, valid_score, loss_name, metric_name, title):
        # Настраиваем график
        plt.figure(figsize=(16, 8))
        epochs = range(1, len(train_loss) + 1)

        # Визуализация потерь
        plt.subplot(1, 2, 1)

        sns.lineplot(x=epochs, y=train_loss, label='Train Loss', linestyle='--', marker='o', color='#1f77b4', linewidth=3)
        sns.lineplot(x=epochs, y=valid_loss, label='Valid Loss', linestyle='-', marker='o', color='#bc4b51', linewidth=3)
        plt.plot(epochs, valid_loss, 'o', markerfacecolor='none', markeredgecolor='#bc4b51', markersize=7, linewidth=2)

        plt.title(f'{title} - {loss_name}')
        plt.xlabel('Epochs')
        plt.legend()
        plt.gca().set_ylabel('')
        plt.xticks(epochs)  # Устанавливаем натуральные значения по оси x
        plt.xlim(1, len(train_loss))  # Ограничиваем ось x от 1 до максимального значения

        # Визуализация кастомной метрики
        plt.subplot(1, 2, 2)

        sns.lineplot(x=epochs, y=train_score, label=f'Train {metric_name}', linestyle='--', marker='o', linewidth=3)
        sns.lineplot(x=epochs, y=valid_score, label=f'Valid {metric_name}', linestyle='-', marker='o', linewidth=3)
        plt.plot(epochs, valid_score, 'o', markerfacecolor='none', markeredgecolor='#DD8452', markersize=7, linewidth=2)

        plt.title(f'{title} - {metric_name}')
        plt.xlabel('Epochs')
        plt.legend()
        plt.gca().set_ylabel('')
        plt.xticks(epochs)  # Устанавливаем натуральные значения по оси x
        plt.xlim(1, len(train_score))  # Ограничиваем ось x от 1 до максимального значения

        plt.tight_layout()
        plt.show()

    def fit(self, trainloader, validloader, optimizer, loss_fn, num_epochs, metric=None, model_name='Model', eps=0.001, min_loss=False, visualize=True):
        # Настраиваем стиль графиков
        sns.set_style('whitegrid')
        sns.set_palette('Set2')
        
        if metric is None:
            metric = accuracy_score

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = f"../models/{model_name}.pth"

        train_loss_history, valid_loss_history = [], []
        train_score_history, valid_score_history = [], []
        best_epoch, best_score, best_loss = 0, None, float('inf')

        def save_model():
            # Сохраняем веса модели
            torch.save(self.__model.state_dict(), save_path)

        for epoch_num in range(1, num_epochs + 1):
            # Обучение на тренировочных данных
            train_loss, train_score = self.train(trainloader, optimizer, loss_fn, metric)
            # Оценка на валидационных данных
            valid_loss, valid_score = self.evaluate(validloader, loss_fn, metric)

            # Сохраняем историю потерь и метрик
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            train_score_history.append(train_score)
            valid_score_history.append(valid_score)

            # Очищаем вывод для обновления информации
            clear_output(wait=True)

            print(f"Epoch: {epoch_num}/{num_epochs}\n")
            print(f'Loss: {loss_fn._get_name()}')
            print(f" - Train: {train_loss:.4f}\n - Valid: {valid_loss:.4f}\n")

            print(f"Score: {metric.__name__}")
            print(f" - Train: {train_score:.4f}\n - Valid: {valid_score:.4f}\n")

            # Сохраняем лучшую модель на основе улучшения метрики
            if (best_score is None or (valid_score - best_score > eps)) and (not min_loss or valid_loss < best_loss) and not self.stop_fiting:
                print("(Model saved)")
                best_epoch, best_score, best_loss = epoch_num, valid_score, valid_loss
                save_model()

            # Визуализация результатов после второй эпохи
            if epoch_num > 1:
                if visualize:
                    self.plot_stats(train_loss_history, valid_loss_history, train_score_history, valid_score_history, loss_fn._get_name(), metric.__name__, model_name)

                print(f"Best valid score: {best_score:.4f} ({best_epoch} epoch)\n")

            # Проверяем флаг остановки обучения
            if self.stop_fiting:
                print("Обучение остановлено пользователем после текущей эпохи.")
                break

        # Загружаем лучшие веса модели
        self.load(save_path)
        self.best_score = best_score

    @torch.inference_mode()
    def predict(self, inputs, batch_size=50):
        # Предсказание на новых данных
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Извлекаем тензоры из входных данных
        input_tensors = [item[0] if isinstance(item, tuple) else item for item in inputs]
        predictions = []

        # Разбиваем данные на батчи и делаем предсказания
        for i in range(0, len(input_tensors), batch_size):
            batch = torch.stack(input_tensors[i:i + batch_size])
            preds = self.__model(batch.to(self.device))
            predictions.append(preds)

        return torch.cat(predictions, dim=0).argmax(axis=1)
    
    def load(self, path):
        # Загружаем веса и применяем их к модели
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.__model.load_state_dict(state_dict)
