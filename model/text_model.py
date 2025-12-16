
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
from transformers import pipeline
from datasets import Dataset
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import kagglehub

# Download latest version
# path = kagglehub.dataset_download("mar1mba/russian-sentiment-dataset")
# path = "C:\\Users\\Admin\\.cache\\kagglehub\\datasets\\mar1mba\\russian-sentiment-dataset\\versions\\2"
# path2 = "C:\\Users\\Admin\\.cache\\kagglehub\\datasets\\mar1mba\\russian-sentiment-dataset\\versions\\2"
# df = pd.read_csv(path)

# url = "hf://datasets/seara/ru_go_emotions/raw/train-00000-of-00001-86de8ef1d0ae28df.parquet"
# save_path = "Документы"
# df = pd.read_parquet(url)
# df.to_csv(save_path, index=False, encoding='utf-8')
# print(df.head(100))

# ==================== 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ====================

# Пример данных для обучения (замените на свои)
data = {
    'text': [
        "Отличный товар, всем рекомендую!",
        "Ужасное качество, не покупайте",
        "Нормально, но могло быть лучше",
        "Просто супер, я в восторге!",
        "Разочарован, не оправдало ожиданий",
        "Хороший продукт за свои деньги",
        "Кошелёк просто ужасен",
        "Лучшая покупка за последнее время",
        "Не советую, полный развод",
        "Качество на высоте, доволен"
    ],
    'label': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = позитив, 0 = негатив
}

# Создаем DataFrame и Dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Разделяем на train и validation
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

print(f"Размер обучающей выборки: {len(train_dataset)}")
print(f"Размер валидационной выборки: {len(val_dataset)}")

# ==================== 2. ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА ====================

model_name = "cointegrated/rubert-tiny2"  # Русская модель
print(f"Загрузка модели: {model_name}")

# Загружаем токенизатор с указанием pad_token
# По сути разные предложения имеют разную длину. Дополняем короткие с помощбю "PAD", чтобы сделать одинаковые
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'


# Функция для токенизации
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding=True,
        truncation=True, # Обрезает длинные сообщения до 128 токенов, если такие есть
        max_length=128
    )


# Токенизируем данные
print("Токенизация данных...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Загружаем модель для классификации
print("Загрузка модели...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # бинарная классификация
    ignore_mismatched_sizes=True
)


# ==================== 3. НАСТРОЙКА ОБУЧЕНИЯ ====================

# Функция для вычисления метрик
def compute_metrics(pred):
    labels = pred.label_ids
    print("!!!!!!!!!!!!!!!!!!!!!!!!", labels)
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': acc,
        'f1': f1
    }


# ИСПРАВЛЕНО: Используем правильные имена параметров для TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',  # папка для результатов
    eval_strategy="epoch",  # ИСПРАВЛЕНО: было evaluation_strategy
    learning_rate=2e-5,  # скорость обучения
    per_device_train_batch_size=4,  # уменьшено для экономии памяти
    per_device_eval_batch_size=4,
    num_train_epochs=300,  # количество эпох
    weight_decay=0.01,  # регуляризация
    save_strategy="epoch",  # сохранять каждую эпоху
    load_best_model_at_end=True,  # загружать лучшую модель
    logging_dir='./logs',  # логи для tensorboard
    logging_steps=10,
    report_to="none",  # не отправлять в wandb/tensorboard
    fp16=False,  # отключить смешанную точность если нет GPU
    gradient_accumulation_steps=1,
    warmup_steps=100,
    save_total_limit=2,
    metric_for_best_model='accuracy',
    greater_is_better=True
)

# Создаем тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ==================== 4. ОБУЧЕНИЕ МОДЕЛИ ====================

print("\n" + "=" * 50)
print("Начинаю обучение...")
print("=" * 50)

try:
    trainer.train()

    # Оценка модели
    print("\n" + "=" * 50)
    print("Оценка модели на валидации:")
    print("=" * 50)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

except Exception as e:
    print(f"Ошибка при обучении: {e}")
    print("Пробую альтернативный подход...")

    # Альтернативный простой подход
    from torch.utils.data import DataLoader

    # Создаем DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Настраиваем оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Простой цикл обучения
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            # Переносим данные на устройство
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['label']
            }

            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Эпоха {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

# ==================== 5. СОХРАНЕНИЕ МОДЕЛИ ====================

try:
    # Сохраняем модель и токенизатор
    model.save_pretrained("./my_sentiment_model")
    tokenizer.save_pretrained("./my_sentiment_model")
    print("\n✅ Модель сохранена в папку './my_sentiment_model'")
except Exception as e:
    print(f"Ошибка при сохранении модели: {e}")

# ==================== 6. СОЗДАНИЕ ПАЙПЛАЙНА ДЛЯ ПРЕДСКАЗАНИЙ ====================

try:
    # Создаем пайплайн для классификации
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

    # ==================== 7. ТЕСТИРОВАНИЕ ====================

    test_texts = [
        "Это просто великолепно!",
        "Ужасный сервис, больше не вернусь",
        "Нормально, но есть недостатки"
    ]

    print("\n" + "=" * 50)
    print("Тестирование обученной модели:")
    print("=" * 50)

    for text in test_texts:
        try:
            result = classifier(text)[0]
            # Определяем метку в читаемом формате
            if 'LABEL' in str(result['label']):
                label_num = int(result['label'].replace('LABEL_', ''))
                label = "ПОЗИТИВ" if label_num == 1 else "НЕГАТИВ"
            else:
                label = result['label']

            print(f"Текст: '{text}'")
            print(f"  → {label} (уверенность: {result['score']:.2%})")
            print()
        except Exception as e:
            print(f"Ошибка при предсказании для текста '{text}': {e}")

except Exception as e:
    print(f"Ошибка при создании пайплайна: {e}")

    # Альтернативный способ предсказаний
    print("\nАльтернативный способ предсказаний:")
    model.eval()

    test_texts = ["Это тестовый текст для проверки"]
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        print(f"Текст: '{text}'")
        print(f"Вероятности: {predictions[0].tolist()}")
        print(f"Предсказанный класс: {predictions[0].argmax().item()}")

print("\n" + "=" * 50)
print("Скрипт завершен!")
print("=" * 50)


# # Пример использования
# new_texts = [
#     "Я в полном восторге от этого продукта!",
#     "Качество оставляет желать лучшего",
#     "Стоит своих денег, рекомендую"
# ]
#
# print("Предсказания для новых текстов:")
# predictions = predict_sentiment(new_texts)
# for pred in predictions:
#     print(f"'{pred['text'][:30]}...' → {pred['sentiment']} ({pred['confidence']:.1%})")
