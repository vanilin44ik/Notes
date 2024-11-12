# Данные
import numpy as np
import pandas as pd

# Место
import h3.api.numpy_int as h3

# Время
import holidays
import datetime


def calculate_azimuth(lat1, lat2, lon_diff):
    # Переводим широты и разницу долгот в радианы один раз
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    delta_lon_rad = np.radians(lon_diff)
    
    # Предварительные вычисления для синусов и косинусов широт
    cos_lat2, sin_lat2 = np.cos(lat2_rad), np.sin(lat2_rad)
    cos_lat1, sin_lat1 = np.cos(lat1_rad), np.sin(lat1_rad)
    
    # Вычисление x и y
    x = np.sin(delta_lon_rad) * cos_lat2
    y = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * np.cos(delta_lon_rad)
    
    # Вычисление азимута и приведение результата к диапазону 0-360
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def calculate_distances(lat1, lon1, lat2, lon2):
    # Преобразуем входные данные в радианы для дальнейших расчетов
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    
    # Радиус Земли в километрах
    R = 6371.0
    
    # Евклидово расстояние в км
    euclidean = R * np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    
    # Манхэттенское расстояние в км
    lat_diff = R * np.abs(lat2 - lat1)
    lon_diff = R * np.abs(lon2 - lon1)
    manhattan = lat_diff + lon_diff
    
    # Геодезическое расстояние (гаверсинус) в км
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    geodesic = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Косинусное расстояние (безразмерное)
    dot_product = lat1 * lat2 + lon1 * lon2
    norm_vec1 = np.sqrt(lat1**2 + lon1**2)
    norm_vec2 = np.sqrt(lat2**2 + lon2**2)
    cosine = 1 - (dot_product / (norm_vec1 * norm_vec2))
    
    # Чебышёвское расстояние в км
    chebyshev = R * np.maximum(lon_diff, lat_diff)
    
    # Формируем словарь с результатами
    distances = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'geodesic': geodesic,
        'cosine': cosine,
        'chebyshev': chebyshev
    }
    
    return pd.DataFrame(distances)



def add_1geo_features(dataset, prefix, points, h3_resolution=10):
    # Корректировка широты в пределах [-90, 90]
    dataset[f'{prefix}_latitude'] = np.clip(dataset[f'{prefix}_latitude'], -90, 90)

    latitudes = dataset[f'{prefix}_latitude'].to_numpy()
    longitudes = dataset[f'{prefix}_longitude'].to_numpy()

    for name, (coords, radius) in points.items():

        lat_diff = latitudes - coords[0]
        lon_diff = longitudes - coords[1]

        # Расстояние
        # dataset = pd.concat([dataset, calculate_distances(latitudes, longitudes, coords[0], coords[1]).add_prefix(f"{prefix}_distance_{name}_")], axis=1)
        
        # Радиус Земли в километрах
        R = 6371.0
        
        # Евклидово расстояние в км
        dataset[f"{prefix}_distance_{name}_euclidean"] = R * np.sqrt((np.radians(latitudes) - np.radians(coords[0])) ** 2 + (np.radians(longitudes) - np.radians(coords[1])) ** 2)

        # Азимут
        dataset[f'{prefix}_azimuth_{name}'] = calculate_azimuth(latitudes, coords[0], lon_diff)

        # Индикатор попадания в радиус
        dataset[f'{prefix}_indicator_{name}'] = (dataset[f'{prefix}_distance_{name}_euclidean'] <= radius).astype(int)

        # Индикатор попадания в квадрат вокруг точки
        dataset[f'{prefix}_in_square_{name}'] = (
            (np.abs(lat_diff) <= radius) & 
            (np.abs(lon_diff) <= radius)
        ).astype(int)

    # Присвоение H3 ячейки с использованием np.vectorize
    vectorized_h3 = np.vectorize(lambda lat, lon: h3.latlng_to_cell(lat, lon, h3_resolution))
    dataset[f'{prefix}_h3_cell'] = vectorized_h3(latitudes, longitudes)

    # Тригонометрические преобразования для широты и долготы
    radians_lat = np.radians(latitudes)
    radians_lon = np.radians(longitudes)

    dataset[f'{prefix}_latitude_sin'] = np.sin(radians_lat)
    dataset[f'{prefix}_latitude_cos'] = np.cos(radians_lat)
    dataset[f'{prefix}_longitude_sin'] = np.sin(radians_lon)
    dataset[f'{prefix}_longitude_cos'] = np.cos(radians_lon)

    # Плотность на основе H3 ячеек
    density = dataset[f'{prefix}_h3_cell'].value_counts()
    dataset[f'{prefix}_density'] = dataset[f'{prefix}_h3_cell'].map(density)

    # Добавление полярных координат
    dataset[f'{prefix}_radius'] = np.sqrt(latitudes ** 2 + longitudes ** 2)
    dataset[f'{prefix}_theta'] = np.arctan2(latitudes, longitudes)

    return dataset


def add_2geo_features(dataset, prefix1, prefix2):
    # Извлечение необходимых массивов для оптимизации
    lat1 = dataset[f'{prefix1}_latitude'].to_numpy()
    lon1 = dataset[f'{prefix1}_longitude'].to_numpy()
    lat2 = dataset[f'{prefix2}_latitude'].to_numpy()
    lon2 = dataset[f'{prefix2}_longitude'].to_numpy()

    # Вычисление разности широты и долготы
    lat_diff = lat2 - lat1
    lon_diff = lon2 - lon1

    # Расстояния
    dataset = pd.concat([dataset, calculate_distances(lat1, lon1, lat2, lon2).add_prefix(f"{prefix1}_distance_{prefix2}_")], axis=1)

    # Азимут (с использованием векторизированной функции calculate_azimuth)
    dataset[f'{prefix1}_{prefix2}_azimuth'] = calculate_azimuth(lat1, lat2, lon_diff)

    # Отношение расстояния к азимуту
    dataset[f'{prefix1}_{prefix2}_distance_to_azimuth_ratio'] = (
        dataset[f"{prefix1}_distance_{prefix2}_euclidean"] / (dataset[f'{prefix1}_{prefix2}_azimuth'] + 1e-5)
    )

    # Совпадение ячеек H3
    dataset[f'{prefix1}_{prefix2}_same_h3_cell'] = (dataset[f'{prefix1}_h3_cell'] == dataset[f'{prefix2}_h3_cell']).astype(int)

    # Тригонометрические разложения разности широты и долготы
    dataset[f'{prefix1}_{prefix2}_latitude_diff_sin'] = np.sin(np.radians(lat_diff))
    dataset[f'{prefix1}_{prefix2}_latitude_diff_cos'] = np.cos(np.radians(lat_diff))
    dataset[f'{prefix1}_{prefix2}_longitude_diff_sin'] = np.sin(np.radians(lon_diff))
    dataset[f'{prefix1}_{prefix2}_longitude_diff_cos'] = np.cos(np.radians(lon_diff))

    # Произведение координат для выявления парных взаимодействий
    dataset[f'{prefix1}_{prefix2}_latitude_product'] = lat1 * lat2
    dataset[f'{prefix1}_{prefix2}_longitude_product'] = lon1 * lon2

    # Разница плотностей H3 ячеек
    dataset[f'{prefix1}_{prefix2}_density_diff'] = np.abs(
        dataset[f'{prefix1}_density'].to_numpy() - dataset[f'{prefix2}_density'].to_numpy()
    )

    # Взаимодействия тригонометрических функций
    dataset[f'{prefix1}_{prefix2}_sin_latitude_product'] = dataset[f'{prefix1}_latitude_sin'] * dataset[f'{prefix2}_latitude_sin']
    dataset[f'{prefix1}_{prefix2}_cos_latitude_product'] = dataset[f'{prefix1}_latitude_cos'] * dataset[f'{prefix2}_latitude_cos']
    dataset[f'{prefix1}_{prefix2}_sin_longitude_product'] = dataset[f'{prefix1}_longitude_sin'] * dataset[f'{prefix2}_longitude_sin']
    dataset[f'{prefix1}_{prefix2}_cos_longitude_product'] = dataset[f'{prefix1}_longitude_cos'] * dataset[f'{prefix2}_longitude_cos']

    return dataset


def add_1time_features(dataset, timestamp_col, country_code='US'):
    prefix = f"{timestamp_col}_"
    
    # Основные временные признаки
    dataset[f'{prefix}year'] = dataset[timestamp_col].dt.year
    dataset[f'{prefix}month'] = dataset[timestamp_col].dt.month
    dataset[f'{prefix}day'] = dataset[timestamp_col].dt.day
    dataset[f'{prefix}day_of_week'] = dataset[timestamp_col].dt.dayofweek
    dataset[f'{prefix}hour'] = dataset[timestamp_col].dt.hour
    dataset[f'{prefix}minute'] = dataset[timestamp_col].dt.minute
    dataset[f'{prefix}second'] = dataset[timestamp_col].dt.second
    dataset[f'{prefix}day_of_year'] = dataset[timestamp_col].dt.dayofyear
    dataset[f'{prefix}quarter'] = dataset[timestamp_col].dt.quarter

    # Праздники
    years = dataset[f'{prefix}year'].unique().tolist()
    country_holidays = holidays.CountryHoliday(country_code, years=years)
    holiday_dates = set(pd.to_datetime(list(country_holidays.keys())).date)

    pre_holiday_dates = {date - datetime.timedelta(days=1) for date in holiday_dates}
    post_holiday_dates = {date + datetime.timedelta(days=1) for date in holiday_dates}

    dataset_dates = dataset[timestamp_col].dt.date

    dataset[f'{prefix}is_holiday'] = dataset_dates.isin(holiday_dates).astype(int)
    dataset[f'{prefix}is_pre_holiday'] = dataset_dates.isin(pre_holiday_dates).astype(int)
    dataset[f'{prefix}is_post_holiday'] = dataset_dates.isin(post_holiday_dates).astype(int)

    # Расчет дней до ближайшего праздника
    holiday_dates_array = np.array(sorted(holiday_dates), dtype='datetime64[D]')
    dataset_dates_array = dataset_dates.values.astype('datetime64[D]')
    diff_matrix = np.abs(dataset_dates_array[:, np.newaxis] - holiday_dates_array[np.newaxis, :]).astype('timedelta64[D]').astype(int)
    min_diffs = diff_matrix.min(axis=1)
    dataset[f'{prefix}days_until_holiday'] = min_diffs

    # Индикаторы выходных, час пик и рабочих дней
    dataset[f'{prefix}is_weekend'] = dataset[f'{prefix}day_of_week'].isin([5, 6]).astype(int)
    dataset[f'{prefix}is_rush_hour'] = dataset[f'{prefix}hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    dataset[f'{prefix}is_working_day'] = (~dataset[f'{prefix}is_weekend'] & ~dataset[f'{prefix}is_holiday']).astype(int)

    # Сезон на основе месяца
    season_mapping = {12: 'winter', 1: 'winter', 2: 'winter',
                      3: 'spring', 4: 'spring', 5: 'spring',
                      6: 'summer', 7: 'summer', 8: 'summer',
                      9: 'autumn', 10: 'autumn', 11: 'autumn'}
    dataset[f'{prefix}season'] = dataset[f'{prefix}month'].map(season_mapping)

    # Время суток на основе часа
    conditions = [
        (dataset[f'{prefix}hour'] >= 5) & (dataset[f'{prefix}hour'] < 12),
        (dataset[f'{prefix}hour'] >= 12) & (dataset[f'{prefix}hour'] < 17),
        (dataset[f'{prefix}hour'] >= 17) & (dataset[f'{prefix}hour'] < 21),
    ]
    choices = ['morning', 'afternoon', 'evening']
    dataset[f'{prefix}time_of_day'] = np.select(conditions, choices, default='night')

    # Циклические признаки для периодических данных
    dataset[f'{prefix}month_sin'] = np.sin(2 * np.pi * dataset[f'{prefix}month'] / 12)
    dataset[f'{prefix}month_cos'] = np.cos(2 * np.pi * dataset[f'{prefix}month'] / 12)
    dataset[f'{prefix}hour_sin'] = np.sin(2 * np.pi * dataset[f'{prefix}hour'] / 24)
    dataset[f'{prefix}hour_cos'] = np.cos(2 * np.pi * dataset[f'{prefix}hour'] / 24)
    dataset[f'{prefix}day_of_week_sin'] = np.sin(2 * np.pi * dataset[f'{prefix}day_of_week'] / 7)
    dataset[f'{prefix}day_of_week_cos'] = np.cos(2 * np.pi * dataset[f'{prefix}day_of_week'] / 7)
    dataset[f'{prefix}day_of_year_sin'] = np.sin(2 * np.pi * dataset[f'{prefix}day_of_year'] / 365)
    dataset[f'{prefix}day_of_year_cos'] = np.cos(2 * np.pi * dataset[f'{prefix}day_of_year'] / 365)

    # Количество дней с начала наблюдений
    dataset[f'{prefix}days_since_start'] = (dataset[timestamp_col] - dataset[timestamp_col].min()).dt.days

    # Новый год и Рождество
    dataset[f'{prefix}is_new_year'] = ((dataset[timestamp_col].dt.month == 1) & (dataset[timestamp_col].dt.day == 1)).astype(int)
    dataset[f'{prefix}is_christmas'] = ((dataset[timestamp_col].dt.month == 12) & (dataset[timestamp_col].dt.day == 25)).astype(int)
    
    # Признаки окончания месяца и квартала
    dataset[f'{prefix}is_month_end'] = dataset[timestamp_col].dt.is_month_end.astype(int)
    dataset[f'{prefix}is_quarter_end'] = dataset[timestamp_col].dt.is_quarter_end.astype(int)

    # Половина месяца (первая или вторая)
    dataset[f'{prefix}is_first_half_month'] = (dataset[f'{prefix}day'] <= 15).astype(int)

    # Месяц в сезоне
    dataset[f'{prefix}month_in_season'] = (dataset[f'{prefix}month'] % 3) + 1

    # Праздничные выходные (длинные выходные)
    dataset[f'{prefix}is_long_weekend'] = (dataset[f'{prefix}is_holiday'] & (dataset[f'{prefix}day_of_week'] == 0)).astype(int)

    # Количество дней до конца сезона
    dataset[f'{prefix}days_to_season_end'] = 90 - (dataset[f'{prefix}day_of_year'] % 90)

    return dataset


def add_2time_features(dataset, timestamp_col1, timestamp_col2):
    # Префикс для новых признаков
    prefix = f'{timestamp_col1}_{timestamp_col2}_'
    
    # Извлечение временных меток
    timestamp1 = dataset[timestamp_col1]
    timestamp2 = dataset[timestamp_col2]
    
    # Вычисление разницы времени в секундах и днях
    time_diff = (timestamp2 - timestamp1).dt.total_seconds().values
    time_diff_days = (timestamp2 - timestamp1).dt.days.values
    
    dataset[f'{prefix}time_diff_seconds'] = time_diff
    dataset[f'{prefix}time_diff_days'] = time_diff_days
    
    # Абсолютные значения разницы времени
    dataset[f'{prefix}abs_time_diff_seconds'] = np.abs(time_diff)
    dataset[f'{prefix}abs_time_diff_days'] = np.abs(time_diff_days)
    
    # Разница дней недели по модулю 7
    day_of_week1 = timestamp1.dt.dayofweek.values
    day_of_week2 = timestamp2.dt.dayofweek.values
    day_of_week_diff = (day_of_week2 - day_of_week1) % 7
    dataset[f'{prefix}day_of_week_diff'] = day_of_week_diff
    
    # Разница часов по модулю 24
    hour1 = timestamp1.dt.hour.values
    hour2 = timestamp2.dt.hour.values
    hour_diff = (hour2 - hour1) % 24
    dataset[f'{prefix}hour_diff'] = hour_diff
    
    # Индикаторы совпадения дня, часа и минуты
    date1 = timestamp1.dt.date.values
    date2 = timestamp2.dt.date.values
    is_same_day = (date1 == date2).astype(int)
    dataset[f'{prefix}is_same_day'] = is_same_day
    
    is_same_hour = ((date1 == date2) & (hour1 == hour2)).astype(int)
    dataset[f'{prefix}is_same_hour'] = is_same_hour
    
    minute1 = timestamp1.dt.minute.values
    minute2 = timestamp2.dt.minute.values
    is_same_minute = ((date1 == date2) & (hour1 == hour2) & (minute1 == minute2)).astype(int)
    dataset[f'{prefix}is_same_minute'] = is_same_minute
    
    # Взаимодействия циклических признаков (предполагается, что add_1time_features уже применена)
    # Извлечение значений признаков для оптимизации
    hour_sin1 = dataset[f'{timestamp_col1}_hour_sin'].values
    hour_sin2 = dataset[f'{timestamp_col2}_hour_sin'].values
    hour_cos1 = dataset[f'{timestamp_col1}_hour_cos'].values
    hour_cos2 = dataset[f'{timestamp_col2}_hour_cos'].values
    
    day_of_week_sin1 = dataset[f'{timestamp_col1}_day_of_week_sin'].values
    day_of_week_sin2 = dataset[f'{timestamp_col2}_day_of_week_sin'].values
    day_of_week_cos1 = dataset[f'{timestamp_col1}_day_of_week_cos'].values
    day_of_week_cos2 = dataset[f'{timestamp_col2}_day_of_week_cos'].values
    
    month_sin1 = dataset[f'{timestamp_col1}_month_sin'].values
    month_sin2 = dataset[f'{timestamp_col2}_month_sin'].values
    month_cos1 = dataset[f'{timestamp_col1}_month_cos'].values
    month_cos2 = dataset[f'{timestamp_col2}_month_cos'].values
    
    day_of_year_sin1 = dataset[f'{timestamp_col1}_day_of_year_sin'].values
    day_of_year_sin2 = dataset[f'{timestamp_col2}_day_of_year_sin'].values
    day_of_year_cos1 = dataset[f'{timestamp_col1}_day_of_year_cos'].values
    day_of_year_cos2 = dataset[f'{timestamp_col2}_day_of_year_cos'].values
    
    # Произведение циклических признаков
    dataset[f'{prefix}hour_sin_product'] = hour_sin1 * hour_sin2
    dataset[f'{prefix}hour_cos_product'] = hour_cos1 * hour_cos2
    dataset[f'{prefix}day_of_week_sin_product'] = day_of_week_sin1 * day_of_week_sin2
    dataset[f'{prefix}day_of_week_cos_product'] = day_of_week_cos1 * day_of_week_cos2
    dataset[f'{prefix}month_sin_product'] = month_sin1 * month_sin2
    dataset[f'{prefix}month_cos_product'] = month_cos1 * month_cos2
    dataset[f'{prefix}day_of_year_sin_product'] = day_of_year_sin1 * day_of_year_sin2
    dataset[f'{prefix}day_of_year_cos_product'] = day_of_year_cos1 * day_of_year_cos2
    
    # Разница циклических признаков
    dataset[f'{prefix}hour_sin_diff'] = hour_sin1 - hour_sin2
    dataset[f'{prefix}hour_cos_diff'] = hour_cos1 - hour_cos2
    dataset[f'{prefix}day_of_week_sin_diff'] = day_of_week_sin1 - day_of_week_sin2
    dataset[f'{prefix}day_of_week_cos_diff'] = day_of_week_cos1 - day_of_week_cos2
    
    # Индикаторы праздников
    is_holiday1 = dataset[f'{timestamp_col1}_is_holiday'].values
    is_holiday2 = dataset[f'{timestamp_col2}_is_holiday'].values
    dataset[f'{prefix}both_holidays'] = (is_holiday1 & is_holiday2).astype(int)
    dataset[f'{prefix}holiday_difference'] = is_holiday1 - is_holiday2
    
    # Разница индикаторов час пик
    is_rush_hour1 = dataset[f'{timestamp_col1}_is_rush_hour'].values
    is_rush_hour2 = dataset[f'{timestamp_col2}_is_rush_hour'].values
    dataset[f'{prefix}rush_hour_difference'] = is_rush_hour1 - is_rush_hour2
    
    # Нормализованная разница времени по модулю 24 часов (в секундах)
    time_diff_mod_24h = time_diff % 86400  # 86400 секунд в сутках
    dataset[f'{prefix}time_diff_mod_24h_seconds'] = time_diff_mod_24h
    
    # Тригонометрические функции от нормализованной разницы времени
    time_diff_day_frac = time_diff_mod_24h / 86400
    dataset[f'{prefix}time_diff_day_frac_sin'] = np.sin(2 * np.pi * time_diff_day_frac)
    dataset[f'{prefix}time_diff_day_frac_cos'] = np.cos(2 * np.pi * time_diff_day_frac)
    
    # Взаимодействие количества дней с начала наблюдений
    days_since_start1 = dataset[f'{timestamp_col1}_days_since_start'].values
    days_since_start2 = dataset[f'{timestamp_col2}_days_since_start'].values
    dataset[f'{prefix}days_since_start_product'] = days_since_start1 * days_since_start2
    
    # Разница количества дней с начала наблюдений
    dataset[f'{prefix}days_since_start_diff'] = days_since_start1 - days_since_start2
    
    # Разница сезонов (предполагая, что сезон закодирован строками)
    # Создаем маппинг сезонов в числовые значения
    season_mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    season1 = dataset[f'{timestamp_col1}_season'].map(season_mapping).values
    season2 = dataset[f'{timestamp_col2}_season'].map(season_mapping).values
    season_diff = (season2 - season1) % 4
    dataset[f'{prefix}season_diff'] = season_diff
    
    # Разница времени суток (предполагая, что время суток закодировано строками)
    time_of_day_mapping = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
    time_of_day1 = dataset[f'{timestamp_col1}_time_of_day'].map(time_of_day_mapping).values
    time_of_day2 = dataset[f'{timestamp_col2}_time_of_day'].map(time_of_day_mapping).values
    time_of_day_diff = (time_of_day2 - time_of_day1) % 4
    dataset[f'{prefix}time_of_day_diff'] = time_of_day_diff
    
    return dataset


def add_2geo_2time_features(dataset, prefix_geo1, prefix_geo2, timestamp_col1, timestamp_col2):
    # Префикс для новых признаков
    prefix = f'{prefix_geo1}_{prefix_geo2}_{timestamp_col1}_{timestamp_col2}_'
    
    # Извлечение необходимых данных для оптимизации
    # Географические координаты
    lat1 = dataset[f'{prefix_geo1}_latitude'].to_numpy()
    lon1 = dataset[f'{prefix_geo1}_longitude'].to_numpy()
    lat2 = dataset[f'{prefix_geo2}_latitude'].to_numpy()
    lon2 = dataset[f'{prefix_geo2}_longitude'].to_numpy()
    
    # Временные метки
    timestamp1 = dataset[timestamp_col1]
    timestamp2 = dataset[timestamp_col2]
    
    # Вычисление временной разницы в секундах
    time_diff_seconds = (timestamp2 - timestamp1).dt.total_seconds().values
    # Обработка случаев с нулевой или отрицательной временной разницей
    time_diff_seconds_safe = np.where(time_diff_seconds <= 0, np.nan, time_diff_seconds)
    
    # Расстояния между точками (предполагается, что они уже вычислены в add_2geo_features)
    distance_col = f'{prefix_geo1}_distance_{prefix_geo2}_euclidean'
    if distance_col in dataset.columns:
        distance_euclidean = dataset[distance_col].values
    else:
        # Если расстояние не вычислено, рассчитываем евклидово расстояние
        # Преобразуем широту и долготу в радианы
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        # Радиус Земли в километрах
        R = 6371.0
        # Евклидово расстояние в км
        distance_euclidean = R * np.sqrt((lat2_rad - lat1_rad)**2 + (lon2_rad - lon1_rad)**2)
        dataset[distance_col] = distance_euclidean
    
    # Вычисление скорости (км/ч)
    # Скорость = расстояние (км) / время (ч)
    speed = distance_euclidean / (time_diff_seconds_safe / 3600)  # преобразуем секунды в часы
    dataset[f'{prefix}speed_euclidean'] = speed
    
    # Вычисление азимута между точками (если не вычислен ранее)
    azimuth_col = f'{prefix_geo1}_{prefix_geo2}_azimuth'
    if azimuth_col in dataset.columns:
        azimuth = dataset[azimuth_col].values
    else:
        lon_diff = lon2 - lon1
        azimuth = calculate_azimuth(lat1, lat2, lon_diff)
        dataset[azimuth_col] = azimuth
    
    # Взаимодействие азимута и временной разницы
    dataset[f'{prefix}azimuth_time_diff_product'] = azimuth * time_diff_seconds
    
    # Взаимодействие расстояния и циклических временных признаков
    hour_sin1 = dataset[f'{timestamp_col1}_hour_sin'].values
    hour_sin2 = dataset[f'{timestamp_col2}_hour_sin'].values
    dataset[f'{prefix}distance_hour_sin_product'] = distance_euclidean * (hour_sin1 + hour_sin2)
    
    # Взаимодействие скорости и циклических временных признаков
    hour_cos1 = dataset[f'{timestamp_col1}_hour_cos'].values
    hour_cos2 = dataset[f'{timestamp_col2}_hour_cos'].values
    dataset[f'{prefix}speed_hour_cos_product'] = speed * (hour_cos1 + hour_cos2)
    
    # Взаимодействие плотности ячеек H3 и разницы дней недели
    if f'{prefix_geo1}_density' in dataset.columns and f'{prefix_geo2}_density' in dataset.columns:
        density_diff = dataset[f'{prefix_geo1}_density'].values - dataset[f'{prefix_geo2}_density'].values
        day_of_week_diff = dataset[f'{timestamp_col1}_day_of_week'].values - dataset[f'{timestamp_col2}_day_of_week'].values
        dataset[f'{prefix}density_day_of_week_diff_product'] = density_diff * day_of_week_diff
    
    # Косинусное взаимодействие разницы часов и широты
    hour_diff = dataset[f'{timestamp_col1}_hour'].values - dataset[f'{timestamp_col2}_hour'].values
    lat_diff = lat2 - lat1
    dataset[f'{prefix}cos_hour_lat_diff'] = np.cos(np.radians(hour_diff)) * lat_diff
    
    # Индикатор движения в час пик
    is_rush_hour1 = dataset[f'{timestamp_col1}_is_rush_hour'].values
    is_rush_hour2 = dataset[f'{timestamp_col2}_is_rush_hour'].values
    dataset[f'{prefix}movement_in_rush_hour'] = (is_rush_hour1 | is_rush_hour2).astype(int)
    
    # Логарифм скорости для стабилизации распределения
    dataset[f'{prefix}log_speed'] = np.log1p(speed)
    
    # Определение направления движения по широте и долготе
    lat_direction_numeric = np.where(lat2 > lat1, 1, -1)
    lon_direction_numeric = np.where(lon2 > lon1, 1, -1)
    dataset[f'{prefix}lat_direction_numeric'] = lat_direction_numeric
    dataset[f'{prefix}lon_direction_numeric'] = lon_direction_numeric
    
    # Взаимодействие направления движения и временной разницы
    dataset[f'{prefix}lat_direction_time_diff'] = lat_direction_numeric * time_diff_seconds
    dataset[f'{prefix}lon_direction_time_diff'] = lon_direction_numeric * time_diff_seconds
    
    # Разница тригонометрических преобразований широты и долготы, умноженная на временную разницу
    lat_sin_diff = dataset[f'{prefix_geo2}_latitude_sin'].values - dataset[f'{prefix_geo1}_latitude_sin'].values
    lon_cos_diff = dataset[f'{prefix_geo2}_longitude_cos'].values - dataset[f'{prefix_geo1}_longitude_cos'].values
    dataset[f'{prefix}lat_sin_diff_time_diff_product'] = lat_sin_diff * time_diff_seconds
    dataset[f'{prefix}lon_cos_diff_time_diff_product'] = lon_cos_diff * time_diff_seconds
    
    # Разница времени суток
    time_of_day_mapping = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
    time_of_day1 = dataset[f'{timestamp_col1}_time_of_day'].map(time_of_day_mapping).values
    time_of_day2 = dataset[f'{timestamp_col2}_time_of_day'].map(time_of_day_mapping).values
    time_of_day_diff = (time_of_day2 - time_of_day1) % 4
    dataset[f'{prefix}time_of_day_movement'] = time_of_day_diff
    
    # Взаимодействие между сезоном и направлением движения
    season_mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    season1 = dataset[f'{timestamp_col1}_season'].map(season_mapping).values
    season2 = dataset[f'{timestamp_col2}_season'].map(season_mapping).values
    season_diff = (season2 - season1) % 4
    direction_sum = lat_direction_numeric + lon_direction_numeric
    dataset[f'{prefix}season_direction_product'] = season_diff * direction_sum
    
    # Обработка значений скорости и логарифма скорости для случаев с некорректной временной разницей
    dataset[f'{prefix}speed_euclidean'] = np.where(np.isnan(time_diff_seconds_safe), np.nan, speed)
    dataset[f'{prefix}log_speed'] = np.where(np.isnan(time_diff_seconds_safe), np.nan, dataset[f'{prefix}log_speed'])
    
    return dataset
