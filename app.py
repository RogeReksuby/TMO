import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ЗАГОЛОВОК ПРИЛОЖЕНИЯ
# ============================================================
st.set_page_config(page_title="Выгорание разработчиков", page_icon="🔥", layout="wide")
st.title('🔥 Предсказание профессионального выгорания разработчиков')
st.markdown('---')

# ============================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================

@st.cache_data
def load_and_prepare_data():
    """Загрузка и предобработка данных (кэшируется)"""
    
    # Загружаем данные (замените путь на ваш)
    data = pd.read_csv('data/developer_burnout_dataset_7000.csv')
    data = data.drop_duplicates()
    
    st.write(f"Исходный размер: {data.shape}")
    
    # Очистка burnout_level
    data['burnout_level'] = data['burnout_level'].replace('', np.nan)
    data = data.dropna(subset=['burnout_level'])
    st.write(f"После удаления пустого burnout_level: {data.shape}")
    
    # Список всех числовых колонок
    numeric_cols = ['age', 'experience_years', 'daily_work_hours', 'sleep_hours',
                    'caffeine_intake', 'bugs_per_day', 'commits_per_day',
                    'meetings_per_day', 'screen_time', 'exercise_hours', 'stress_level']
    
    # Сначала заменяем все inf и -inf на NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Заполняем пропуски медианой во ВСЕХ числовых колонках
    for col in numeric_cols:
        median_val = data[col].median()
        if pd.isna(median_val):
            median_val = 0  # Если все значения NaN, заполняем нулём
        data[col].fillna(median_val, inplace=True)
    
    # Проверяем, что в исходных колонках нет NaN
    for col in numeric_cols:
        if data[col].isna().any():
            #st.warning(f"Колонка {col} всё ещё содержит NaN! Заполняем нулями.")
            data[col].fillna(0, inplace=True)
    
    # Создание признаков с защитой от деления на ноль
    denom_wlb = data['sleep_hours'] + data['exercise_hours']
    denom_wlb = denom_wlb.replace(0, 1)  # Заменяем 0 на 1, чтобы избежать деления на 0
    data['work_life_balance'] = data['daily_work_hours'] / denom_wlb
    
    data['bugs_per_commit'] = data['bugs_per_day'] / (data['commits_per_day'] + 1)
    
    # Кодирование целевой переменной
    le = LabelEncoder()
    data['burnout_encoded'] = le.fit_transform(data['burnout_level'])
    
    # Признаки для модели
    feature_cols = ['age', 'experience_years', 'daily_work_hours', 'sleep_hours',
                    'caffeine_intake', 'bugs_per_day', 'commits_per_day',
                    'meetings_per_day', 'screen_time', 'exercise_hours',
                    'work_life_balance', 'bugs_per_commit']
    
    X = data[feature_cols].copy()
    y = data['burnout_encoded'].copy()
    
    # Жёсткая очистка: заменяем ВСЁ проблемное на медиану или 0
    for col in X.columns:
        # Заменяем inf
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        # Если есть NaN — заполняем медианой, если медиана NaN — заполняем 0
        if X[col].isna().any():
            fill_val = X[col].median()
            if pd.isna(fill_val):
                fill_val = 0
            X[col] = X[col].fillna(fill_val)
    
    # Финальная проверка перед масштабированием
    total_nan = X.isna().sum().sum()
    if total_nan > 0:
        st.error(f"После очистки осталось {total_nan} NaN в колонках: {X.columns[X.isna().any()].tolist()}")
        # Заполняем нулями всё, что осталось
        X = X.fillna(0)
    
    # Масштабирование
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Финальная проверка
    total_nan_after = X_scaled.isna().sum().sum()
    if total_nan_after > 0:
        st.error(f"После масштабирования NaN в колонках: {X_scaled.columns[X_scaled.isna().any()].tolist()}")
        st.error(f"Значения: {X_scaled[X_scaled.isna().any(axis=1)]}")
        # Заполняем нулями
        X_scaled = X_scaled.fillna(0)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, le, feature_cols, data
   
# Загружаем данные
with st.spinner('Загрузка данных...'):
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols, full_data = load_and_prepare_data()

st.success(f'✅ Данные загружены! Обучающая выборка: {X_train.shape[0]} записей, Тестовая: {X_test.shape[0]} записей')

# ============================================================
# БОКОВАЯ ПАНЕЛЬ — ГИПЕРПАРАМЕТР МОДЕЛИ
# ============================================================
st.sidebar.header('⚙️ Настройка модели')

st.sidebar.markdown("""
**Logistic Regression** — простая и интерпретируемая модель.
Параметр **C** отвечает за силу регуляризации:
- **C → 0** (маленькое) — сильная регуляризация, модель проще
- **C → ∞** (большое) — слабая регуляризация, модель сложнее
""")

# Слайдер для гиперпараметра C
c_value = st.sidebar.slider(
    'Параметр C (обратный коэффициент регуляризации)',
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.01,
    help='Меняйте этот параметр и нажимайте "Переобучить модель"'
)

# Кнопка переобучения
retrain_button = st.sidebar.button('🔄 Переобучить модель', type='primary')

# ============================================================
# ОБУЧЕНИЕ МОДЕЛИ
# ============================================================
@st.cache_resource
def train_model(c_val):
    """Обучает модель с заданным C"""
    model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def get_predictions(_model, _X_test):
    """Получает предсказания модели"""
    y_pred = _model.predict(_X_test)
    y_pred_proba = _model.predict_proba(_X_test)
    return y_pred, y_pred_proba

# Обучаем или загружаем модель
if retrain_button or 'current_model' not in st.session_state:
    st.session_state.current_model = train_model(c_value)
    st.session_state.current_c = c_value

model = st.session_state.current_model
y_pred, y_pred_proba = get_predictions(model, X_test)

# ============================================================
# ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================
st.markdown(f'### 📊 Результаты модели (C = {st.session_state.current_c:.2f})')

# Метрики в ряд
col1, col2, col3, col4, col5 = st.columns(5)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

with col1:
    st.metric('Accuracy', f'{accuracy:.3f}')
with col2:
    st.metric('Precision', f'{precision:.3f}')
with col3:
    st.metric('Recall', f'{recall:.3f}')
with col4:
    st.metric('F1-score', f'{f1:.3f}')
with col5:
    st.metric('ROC-AUC', f'{roc_auc:.3f}')

# ============================================================
# ВИЗУАЛИЗАЦИИ
# ============================================================
st.markdown('---')
col_left, col_right = st.columns(2)

with col_left:
    st.subheader('Матрица ошибок')
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=label_encoder.classes_
    )
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(f'Матрица ошибок (C={st.session_state.current_c:.2f})')
    st.pyplot(fig)

with col_right:
    st.subheader('Важность признаков')
    # Важность признаков — абсолютные значения коэффициентов
    feature_importance = pd.DataFrame({
        'Признак': feature_cols,
        'Важность': np.abs(model.coef_).sum(axis=0)
    }).sort_values('Важность', ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(feature_importance['Признак'], feature_importance['Важность'], color='steelblue')
    ax.set_title('Топ-10 важных признаков')
    ax.set_xlabel('Сумма абсолютных коэффициентов')
    st.pyplot(fig)

# ============================================================
# ГРАФИК ВЛИЯНИЯ ГИПЕРПАРАМЕТРА C
# ============================================================
st.markdown('---')
st.subheader('📈 Влияние гиперпараметра C на качество модели')

if st.button('Построить график зависимости от C (может занять время)'):
    c_values = np.logspace(-2, 1, 15)  # 15 значений от 0.01 до 10
    metrics_data = {'C': [], 'Accuracy': [], 'F1': [], 'ROC_AUC': []}
    
    progress_bar = st.progress(0)
    
    for i, c_val in enumerate(c_values):
        temp_model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        temp_model.fit(X_train, y_train)
        temp_pred = temp_model.predict(X_test)
        temp_proba = temp_model.predict_proba(X_test)
        
        metrics_data['C'].append(c_val)
        metrics_data['Accuracy'].append(accuracy_score(y_test, temp_pred))
        metrics_data['F1'].append(f1_score(y_test, temp_pred, average='weighted'))
        metrics_data['ROC_AUC'].append(roc_auc_score(y_test, temp_proba, multi_class='ovr', average='weighted'))
        
        progress_bar.progress((i + 1) / len(c_values))
    
    progress_bar.empty()
    
    # Строим график
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_data['C'], metrics_data['Accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(metrics_data['C'], metrics_data['F1'], 's-', label='F1-score', linewidth=2)
    ax.plot(metrics_data['C'], metrics_data['ROC_AUC'], '^-', label='ROC-AUC', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('C (логарифмическая шкала)')
    ax.set_ylabel('Значение метрики')
    ax.set_title('Зависимость метрик от гиперпараметра C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.info('Оптимальное значение C обычно находится в диапазоне 0.1–1.0. При C > 1 регуляризация ослабевает, и модель может начать переобучаться.')

# ============================================================
# ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ ДЛЯ ОДНОГО РАЗРАБОТЧИКА
# ============================================================
st.markdown('---')
st.subheader('🧑‍💻 Предсказание для конкретного разработчика')

st.markdown('Введите данные разработчика, чтобы узнать предсказанный уровень выгорания:')

# Разбиваем на 3 колонки для компактности
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Возраст', min_value=20, max_value=50, value=30)
    experience = st.number_input('Опыт (лет)', min_value=0, max_value=30, value=5)
    work_hours = st.slider('Рабочие часы в день', 4.0, 14.0, 8.0, 0.5)
    sleep_hours = st.slider('Часы сна', 4.0, 9.0, 7.0, 0.5)

with col2:
    caffeine = st.slider('Кофеин (чашек/день)', 0, 10, 3)
    bugs = st.slider('Багов в день', 0, 20, 10)
    commits = st.slider('Коммитов в день', 0, 30, 10)
    meetings = st.slider('Встреч в день', 0, 10, 3)

with col3:
    screen_time = st.slider('Экранное время (часов)', 5.0, 18.0, 12.0, 0.5)
    exercise = st.slider('Упражнения (часов/день)', 0.0, 2.0, 0.5, 0.1)

# Кнопка предсказания
if st.button('🔮 Предсказать уровень выгорания', type='primary'):
    # Создаём признаки
    wlb = work_hours / (sleep_hours + exercise) if (sleep_hours + exercise) > 0 else work_hours
    bpc = bugs / (commits + 1)
    
    input_data = pd.DataFrame([[
        age, experience, work_hours, sleep_hours, caffeine,
        bugs, commits, meetings, screen_time, exercise, wlb, bpc
    ]], columns=feature_cols)
    
    # Масштабируем
    input_scaled = scaler.transform(input_data)
    
    # Предсказываем
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    
    # Выводим результат
    st.markdown('---')
    
    # Цветовая индикация
    if predicted_class == 'Low':
        st.success(f'### 🟢 Предсказанный уровень выгорания: **{predicted_class}** (низкий)')
    elif predicted_class == 'Medium':
        st.warning(f'### 🟡 Предсказанный уровень выгорания: **{predicted_class}** (средний)')
    else:
        st.error(f'### 🔴 Предсказанный уровень выгорания: **{predicted_class}** (высокий)')
    
    # Вероятности по классам
    st.markdown('**Вероятности по классам:**')
    prob_cols = st.columns(3)
    class_names = label_encoder.classes_
    emojis = ['🟢', '🟡', '🔴']
    
    for i, (name, prob, emoji) in enumerate(zip(class_names, prediction_proba, emojis)):
        with prob_cols[i]:
            st.metric(f'{emoji} {name}', f'{prob:.1%}')
    
    # Дополнительные рекомендации
    st.markdown('**Анализ факторов риска:**')
    risks = []
    if work_hours > 10:
        risks.append('⚠️ Высокая рабочая нагрузка (>10 часов/день)')
    if sleep_hours < 6:
        risks.append('⚠️ Недостаток сна (<6 часов)')
    if bugs > 15:
        risks.append('⚠️ Большое количество багов')
    if meetings > 6:
        risks.append('⚠️ Много встреч')
    if wlb > 1.5:
        risks.append('⚠️ Нарушен work-life balance')
    
    if risks:
        for risk in risks:
            st.write(risk)
    else:
        st.write('✅ Критических факторов риска не выявлено')

# ============================================================
# ИНФОРМАЦИЯ О МОДЕЛИ
# ============================================================
st.sidebar.markdown('---')
st.sidebar.markdown("""
### ℹ️ О модели
- **Тип**: LogisticRegression (мультиклассовая)
- **Признаков**: 12
- **Классов**: Low / Medium / High
- **Обучена на**: 5488 записях
""")

st.sidebar.markdown("""
### 📝 Как использовать
1. Измените параметр **C** в слайдере
2. Нажмите **«Переобучить модель»**
3. Смотрите, как изменились метрики
4. Введите данные разработчика и получите предсказание
""")
