import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
import pandas as pd
import warnings
import os
import json
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings('ignore')

need_to_train = False


# ==================== 1. КОНФИГУРАЦИЯ ====================

class Config:
    # Пути
    DATA_PATH = "Датасет"
    SAVE_DIR = "./multi_label_model"
    BASE_MODEL = "cointegrated/rubert-tiny2"

    # Параметры
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    EPOCHS = 20
    MAX_LENGTH = 128
    TEST_SIZE = 0.2
    SEED = 34

    # Эмоции (28 + neutral)
    EMOTIONS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]


# ==================== 2. МОДЕЛЬ ====================

class EmotionClassifier(nn.Module):
    """Модель для multi-label классификации эмоций"""

    def __init__(self, model_name, num_labels):
        super().__init__()
        # Загружаем BERT
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Классификатор
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels),
            nn.Sigmoid()  # Для multi-label
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # Получаем эмбеддинги от BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Используем [CLS] токен
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Предсказания
        logits = self.classifier(pooled_output)

        # Loss если есть labels
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(logits, labels.float())

        return {'loss': loss, 'logits': logits}


# ==================== 3. ЗАГРУЗКА ДАННЫХ ====================

def load_and_prepare_data():
    """Загружает и подготавливает данные"""
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    # Загрузка
    df = pd.read_csv(Config.DATA_PATH, encoding='utf-8')
    print(f"Загружено строк: {len(df)}")

    # Проверяем доступные эмоции
    available_emotions = [e for e in Config.EMOTIONS if e in df.columns]
    print(f"Найдено эмоций: {len(available_emotions)}")
    print(f"Эмоции: {available_emotions[:5]}..." if len(available_emotions) > 5 else available_emotions)

    # Подготовка
    df['text'] = df['ru_text']
    df['labels'] = df[available_emotions].values.tolist()

    # Оставляем только нужные колонки
    df = df[['text', 'labels']]

    # Уменьшаем для теста
    df = df.head(10000)  # 500 примеров
    print(f"Используем {len(df)} примеров")

    # Создаем mapping
    id2label = {i: emotion for i, emotion in enumerate(available_emotions)}
    label2id = {emotion: i for i, emotion in enumerate(available_emotions)}

    # Создаем Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(
        test_size=Config.TEST_SIZE,
        seed=Config.SEED
    )

    return dataset, available_emotions, id2label, label2id


# ==================== 4. ТОКЕНИЗАЦИЯ ====================

def tokenize_data(dataset, tokenizer):
    """Токенизирует данные"""
    print("\n" + "=" * 60)
    print("ТОКЕНИЗАЦИЯ")
    print("=" * 60)

    def tokenize_function(examples):
        # Токенизация без token_type_ids для rubert
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors=None,
            return_token_type_ids=False,
        )
        tokenized['labels'] = examples['labels']
        return tokenized

    # Применяем токенизацию
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Форматируем для PyTorch
    tokenized_dataset.set_format(type='torch',
                                 columns=['input_ids', 'attention_mask', 'labels'])

    print(f"Токенизация завершена")
    print(f"Пример input_ids: {tokenized_dataset['train'][0]['input_ids'].shape}")

    return tokenized_dataset


# ==================== 5. ОБУЧЕНИЕ ====================

def train_model(model, train_loader, val_loader, epochs=3):
    """Обучает модель"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        print(f"\nЭпоха {epoch + 1}/{epochs}")
        print("-" * 40)

        # ===== ОБУЧЕНИЕ =====
        model.train()
        train_loss = 0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Перемещаем на устройство
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            # Forward
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx:3d}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        print(f"Средний train loss: {avg_train_loss:.4f}")

        # ===== ВАЛИДАЦИЯ =====
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).float()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs['loss'].item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        print(f"Средний val loss: {avg_val_loss:.4f}")

        # ===== МЕТРИКИ =====
        if (epoch + 1) % 2 == 0:
            evaluate_model(model, val_loader, device)

    print("\n✅ Обучение завершено")
    return history


# ==================== 6. ОЦЕНКА ====================

def evaluate_model(model, data_loader, device):
    """Оценивает модель"""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = (outputs['logits'].cpu().numpy() > 0.5).astype(int)

            all_labels.append(labels)
            all_preds.append(preds)

    # Объединяем батчи
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Вычисляем метрики
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )

    # Exact match
    exact_match = np.mean(np.all(all_preds == all_labels, axis=1))

    print(f"  Метрики: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Exact Match={exact_match:.3f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match
    }


# ==================== 7. СОХРАНЕНИЕ ====================

def save_model(model, tokenizer, id2label, label2id, save_dir):
    """Сохраняет модель и конфигурацию"""
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ МОДЕЛИ")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 1. Сохраняем веса модели
    torch.save(model.state_dict(), f"{save_dir}/model_weights.pth")
    print(f"✅ Веса модели сохранены в {save_dir}/model_weights.pth")

    # 2. Сохраняем BERT часть отдельно
    model.bert.save_pretrained(save_dir)

    # 3. Сохраняем токенизатор
    tokenizer.save_pretrained(save_dir)

    # 4. Сохраняем конфигурацию
    config = {
        "num_labels": len(id2label),
        "id2label": id2label,
        "label2id": label2id,
        "model_type": "bert",
        "hidden_size": model.bert.config.hidden_size,
        "problem_type": "multi_label_classification",
        "classifier_architecture": str(model.classifier)
    }

    with open(f"{save_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"✅ Конфигурация сохранена")
    print(f"✅ Токенизатор сохранен")
    print(f"   Сохранено эмоций: {len(id2label)}")


# ==================== 8. ЗАГРУЗКА ====================

def load_model(save_dir, num_labels=None):
    """Загружает сохраненную модель"""
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА МОДЕЛИ")
    print("=" * 60)

    if not os.path.exists(save_dir):
        print(f"❌ Папка {save_dir} не найдена")
        return None, None, None, None

    try:
        # 1. Загружаем конфигурацию
        config_path = f"{save_dir}/config.json"
        if not os.path.exists(config_path):
            print(f"❌ Конфиг не найден: {config_path}")
            return None, None, None, None

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        id2label = config.get("id2label", {})
        label2id = config.get("label2id", {})
        saved_num_labels = config.get("num_labels", num_labels)

        if num_labels is None:
            num_labels = saved_num_labels

        print(f"✅ Конфигурация загружена: {num_labels} эмоций")

        # 2. Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

        print("✅ Токенизатор загружен")

        # 3. Создаем модель
        model = EmotionClassifier(Config.BASE_MODEL, num_labels)

        # 4. Загружаем веса
        weights_path = f"{save_dir}/model_weights.pth"
        if os.path.exists(weights_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            print(f"✅ Веса модели загружены на {device}")
        else:
            print("⚠️ Веса не найдены, модель создана с нуля")

        return model, tokenizer, id2label, label2id

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# ==================== 9. ПРЕДСКАЗАНИЯ ====================

def predict_emotions(model, tokenizer, texts, id2label, threshold=0.3):
    EMOTIONS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    EMOTIONS_RU = [
        'восхищение',  # admiration
        'веселье',  # amusement
        'гнев',  # anger
        'раздражение',  # annoyance
        'одобрение',  # approval
        'забота',  # caring
        'замешательство',  # confusion
        'любопытство',  # curiosity
        'желание',  # desire
        'разочарование',  # disappointment
        'неодобрение',  # disapproval
        'отвращение',  # disgust
        'смущение',  # embarrassment
        'возбуждение',  # excitement
        'страх',  # fear
        'благодарность',  # gratitude
        'горе',  # grief
        'радость',  # joy
        'любовь',  # love
        'нервозность',  # nervousness
        'оптимизм',  # optimism
        'гордость',  # pride
        'осознание',  # realization
        'облегчение',  # relief
        'раскаяние',  # remorse
        'печаль',  # sadness
        'удивление',  # surprise
        'нейтральность'  # neutral
    ]


    """Предсказывает эмоции для списка текстов с диагностикой"""
    model.eval()
    device = next(model.parameters()).device

    print(f"\n{'=' * 60}")
    print(f"ПРЕДСКАЗАНИЯ (порог: {threshold})")
    print(f"id2label: {len(id2label)} эмоций")
    print('=' * 60)

    results = []

    for text_idx, text in enumerate(texts, 1):
        print(f"\n{'=' * 40}")
        print(f"Текст #{text_idx}: '{text}'")
        print('=' * 40)

        # Токенизация
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors='pt',
            return_token_type_ids=False,
        )

        # Переносим на устройство
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Предсказание
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = outputs['logits'][0].cpu().numpy()

        # 1. Показываем топ-10 эмоций
        print("\nТоп-10 наиболее вероятных эмоций:")
        sorted_indices = np.argsort(probabilities)[::-1][:10]
        for rank, idx in enumerate(sorted_indices, 1):
            prob = probabilities[idx]
            emotion_name = id2label.get(idx, f"emotion_{idx}")
            print(f"  {rank:2d}. [{idx:2d}] {EMOTIONS_RU[idx]:20s}: {prob:.3f}")

        # 2. Находим эмоции выше порога
        emotions = []
        for idx, prob in enumerate(probabilities):
            if prob >= threshold:
                emotion_name = id2label.get(idx, f"emotion_{idx}")
                emotions.append({
                    'emotion': EMOTIONS_RU[idx],
                    'probability': float(prob),
                    'idx': idx
                })

        # 3. Сортируем по уверенности
        emotions.sort(key=lambda x: x['probability'], reverse=True)

        # 4. Сохраняем результат
        results.append({
            'text': text,
            'emotions': emotions,
            'probabilities': probabilities.tolist(),
            'max_probability': float(np.max(probabilities))
        })

        # 5. Выводим результат
        if emotions:
            print(f"\n🎯 ЭМОЦИИ ВЫШЕ ПОРОГА ({threshold}):")
            for emotion in emotions[:5]:  # Показываем первые 5
                print(f"  • {emotion['emotion']}: {emotion['probability']:.3f}")
            if len(emotions) > 5:
                print(f"    ... и еще {len(emotions) - 5} эмоций")
        else:
            max_idx = np.argmax(probabilities)
            max_prob = probabilities[max_idx]
            max_emotion = id2label.get(max_idx, f"emotion_{max_idx}")
            print(f"\n⚠️ Нет эмоций выше порога {threshold}")
            print(f"   Наиболее вероятная: {max_emotion} ({max_prob:.3f})")

    return results


# ==================== 10. ОСНОВНАЯ ПРОГРАММА ====================

def main():
    """Основная функция"""
    print("=" * 60)
    print("МОДЕЛЬ КЛАССИФИКАЦИИ ЭМОЦИЙ")
    print("=" * 60)

    # 1. Загружаем данные
    dataset, available_emotions, id2label, label2id = load_and_prepare_data()

    # 2. Загружаем или создаем модель
    model, tokenizer, loaded_id2label, loaded_label2id = load_model(Config.SAVE_DIR,
                                                                    len(available_emotions))

    if model is None:
        print("\n🆕 Создаем новую модель...")

        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

        # Создаем модель
        num_labels = len(available_emotions)
        model = EmotionClassifier(Config.BASE_MODEL, num_labels)

        print(f"✅ Новая модель создана: {num_labels} эмоций")
    else:
        print("\n✅ Модель загружена из сохранения")
        # Используем загруженные mapping
        if loaded_id2label:
            id2label = loaded_id2label
            label2id = loaded_label2id

    # 3. Токенизация данных
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    # 4. Создаем DataLoader
    train_loader = DataLoader(
        tokenized_dataset['train'],
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        tokenized_dataset['test'],
        batch_size=Config.BATCH_SIZE
    )

    print(f"\nРазмер train loader: {len(train_loader)} батчей")
    print(f"Размер val loader: {len(val_loader)} батчей")

    if need_to_train:
        # 5. Обучаем модель
        history = train_model(model, train_loader, val_loader, epochs=Config.EPOCHS)

        # 6. Сохраняем модель
        save_model(model, tokenizer, id2label, label2id, Config.SAVE_DIR)

    # 7. Тестируем
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ")
    print("=" * 60)

    # test_texts = [
    #     "Я очень рад этой новости!",
    #     "Мне страшно и тревожно",
    #     "Это злит и раздражает меня",
    #     "Чувствую благодарность и любовь",
    #     "Нейтральное сообщение"
    # ]

    test_texts = ["Я получил 2 по экзамену",
                  "Я получил 5 по экзамену"]

    predictions = predict_emotions(model, tokenizer, test_texts, id2label, threshold=0.4)

    for pred in predictions:
        print(f"\nТекст: '{pred['text']}'")
        print(f"Макс. вероятность: {pred['max_probability']:.3f}")

        if pred['emotions']:
            print("Эмоции:")
            for emotion in pred['emotions']:
                print(f"  • {emotion['emotion']}: {emotion['probability']:.3f}")
        else:
            print("Нет эмоций выше порога")

    # 8. Вывод истории обучения
    # print("\n" + "=" * 60)
    # print("ИСТОРИЯ ОБУЧЕНИЯ")
    # print("=" * 60)
    #
    # for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
    #     print(f"Эпоха {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    print("\n" + "=" * 60)
    print("ПРОГРАММА ЗАВЕРШЕНА!")
    print("=" * 60)


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    main()