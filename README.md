# Deep Dive ML — Phishing Detection

## Overview

**Deep Dive ML** — веб-система для классификации фишинговых и безопасных сообщений (email, текст со ссылками, HTML). Шесть обученных моделей машинного обучения дают отдельные предсказания; итоговый вердикт формирует **взвешенный ансамбль**. Результаты сохраняются в SQLite, пользователь может оставить обратную связь, которая постепенно корректирует веса моделей.

Стек: Flask API, vanilla JavaScript UI, scikit-learn / XGBoost, TF-IDF-признаки текста и бинарный признак наличия URL.

## Problem

Фишинг и спам в письмах остаются одной из главных угроз: злоумышленники маскируют ссылки, срочные «уведомления» и поддельные сервисы под легитимную переписку. Ручная проверка каждого письма не масштабируется.

Задача проекта — **бинарная классификация текста**:

| Метка | Значение |
|-------|----------|
| `0`   | безопасное (ham) |
| `1`   | фишинг / спам |

Система должна:

- работать на произвольном вставленном тексте (и опционально URL);
- показывать прозрачный разбор по каждой модели;
- накапливать историю анализов и учиться на пользовательском feedback.

## Dataset

Исходные данные: **`email_text.csv`** в корне репозитория.

| Поле   | Описание |
|--------|----------|
| `label` | `0` — safe, `1` — phishing/spam |
| `text`  | тело письма или фрагмент текста (нижний регистр, без лишней разметки) |

Объём: **~53 670** строк (заголовок + записи). Примеры класса `1` — рекламный спам, «фарма», подозрительные ссылки; класс `0` — техническая переписка, обсуждения на mailing-list и т.п.

**Предобработка при inference (сервер):**

- TF-IDF-векторизация текста (сохранённый `preprocessing/tfidf_vectorizer.pkl`);
- извлечение URL regex из текста, если URL не передан отдельно;
- для большинства моделей: **TF-IDF + признак `has_url`** (1.0 / 0.0);
- **Naive Bayes** использует только текстовые TF-IDF-признаки.

> Для обучения моделей используйте тот же CSV и артефакты в `models/` и `preprocessing/` (файлы `.pkl` могут не входить в репозиторий из‑за размера).

## Models

Обучены и подключаются через pickle:

| Файл | Алгоритм | Входные признаки |
|------|----------|------------------|
| `logistic_regression.pkl` | Logistic Regression | TF-IDF + `has_url` |
| `naive_bayes.pkl` | Naive Bayes | TF-IDF |
| `random_forest.pkl` | Random Forest | TF-IDF + `has_url` |
| `svm.pkl` | SVM | TF-IDF + `has_url` |
| `svm_tuned.pkl` | SVM (подбор гиперпараметров) | TF-IDF + `has_url` |
| `xgboost.pkl` | XGBoost | TF-IDF + `has_url` |

**Ансамбль:** weighted voting — каждая модель голосует с весом, пропорциональным качеству (из `models/metrics_table.csv`: 60% Val F1 + 40% Val ROC-AUC, нормализация и ограничение весов в диапазоне **0.90–1.10**). После feedback веса обновляются и сохраняются в `feedback_data/model_weights.json`.

## Results

Метрики на validation (файл `models/metrics_table.csv`):

| Model | CV F1 (mean) | Val Accuracy | Val F1 | Val ROC-AUC |
|-------|----------------|--------------|--------|-------------|
| Logistic Regression | 0.955 | 0.975 | 0.957 | 0.994 |
| Naive Bayes | 0.949 | 0.976 | 0.960 | 0.992 |
| SVM | 0.966 | 0.978 | 0.963 | 0.995 |
| Random Forest | 0.955 | 0.975 | 0.958 | 0.992 |
| XGBoost | 0.958 | 0.977 | 0.960 | 0.994 |
| **SVM (Tuned)** | **0.971** | **0.984** | **0.972** | **0.997** |

Лучший одиночный результат по Val F1 и ROC-AUC — **SVM (Tuned)**. Ансамбль согласует модели и снижает риск опоры на одну ошибочную модель за счёт взвешенного голосования и пользовательской обратной связи.

## How to Run

### Требования

- Python 3.9+ (рекомендуется)
- Установленные артефакты: `models/*.pkl`, `preprocessing/tfidf_vectorizer.pkl`, при необходимости `preprocessing/url_scaler.pkl`

### Установка

```bash
cd Deep_Dive_ML
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
```

### Запуск сервера

```bash
python backend/main.py
```

При успешной загрузке моделей откройте в браузере: **http://localhost:5000**

API (кратко):

| Метод | Endpoint | Назначение |
|-------|----------|------------|
| GET | `/api/health` | статус и число загруженных моделей |
| POST | `/api/predict` | анализ текста (`text`, опционально `url`) |
| POST | `/api/feedback` | feedback: `phishing` / `safe` / `not_sure` |
| GET | `/api/analyses` | список сохранённых анализов |
| GET | `/api/statistics` | агрегированная статистика |

### Просмотр базы данных

```bash
python view_database.py --analyses
python view_database.py --detail <id>
```

Подробнее о схеме SQLite: [DATABASE.md](DATABASE.md).

## Project Structure

```
Deep_Dive_ML/
├── backend/
│   ├── main.py              # Flask: predict, feedback, API
│   └── database.py          # SQLite (analyses, model_results)
├── frontend/
│   ├── index.html           # UI анализа
│   ├── script.js            # вызовы API, отображение результатов
│   └── style.css
├── models/
│   ├── *.pkl                # обученные модели (локально)
│   └── metrics_table.csv    # метрики обучения / начальные веса
├── preprocessing/
│   ├── tfidf_vectorizer.pkl
│   └── url_scaler.pkl
├── data/
│   └── analyses.db          # создаётся при первом анализе
├── feedback_data/
│   ├── model_weights.json   # текущие веса ансамбля
│   └── model_stats.json     # статистика по feedback
├── logs/
│   └── server.log
├── email_text.csv           # датасет для обучения
├── requirements.txt
├── view_database.py
├── DATABASE.md
└── README.md
```

## Tech Stack

| Слой | Технологии |
|------|------------|
| ML | scikit-learn, XGBoost, NumPy, SciPy, pandas |
| Признаки | TF-IDF (`TfidfVectorizer`), бинарный признак URL |
| Backend | Flask, Flask-CORS, pickle |
| Хранение | SQLite (`data/analyses.db`) |
| Frontend | HTML5, CSS3, JavaScript (Fetch API) |
| Визуализация (обучение) | matplotlib, seaborn (зависимости в requirements) |



**Deep Dive ML** — учебный/демонстрационный проект по ML для детекции фишинга с веб-интерфейсом и ансамблевым inference.
