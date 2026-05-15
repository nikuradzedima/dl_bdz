# SoundStream Neural Audio Codec



Проект представляет собой реализацию нейросетевого аудиокодека SoundStream для сжатия и восстановления речевого аудио. Основная задача модели это преобразовать исходный waveform в дискретное представление, а затем восстановить аудио как можно ближе к оригиналу.

Кодек обучается и оценивается на LibriSpeech, для обучения используется `train-clean-100`, для оценки `test-clean`.


## Архитектура

Архитектура выглядит следующим образом

```text
waveform [B, 1, T]
  -> Encoder
  -> RVQ, 8 codebooks x 1024 entries
  -> Decoder
  -> reconstructed waveform [B, 1, T]
```


## Структура Кода

Репозиторий имеет следующую структура (старался делать как у Петра)

```text
train.py
inference.py
evaluate.py
src/
  configs/        конфиги
  datasets/       LibriSpeech dataset
  logger/         Comet ML writer
  loss/           spectral, adversarial и feature matching losses
  metrics/        обертки для STOI и NISQA
  model/          encoder, decoder, RVQ, discriminators
  trainer/        GAN training loop
  analysis/       функции для графиков в отчете
scripts/
  download_checkpoint.py
  download_librispeech.py
notebooks/
  demo.ipynb
reports/
  REPORT_TEMPLATE.md
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для логирования в Comet ML создать `.env` в корне репозитория и заполнить

```text
COMET_API_KEY=...
COMET_WORKSPACE=...
COMET_PROJECT_NAME=soundstream-codec
CHECKPOINT_URL=...
```

## Данные

Ожидаемая структура LibriSpeech

```text
data/LibriSpeech/
  train-clean-100/
  test-clean/
```

Скачать можно данные как

```bash
python scripts/download_librispeech.py --root data
```

Если используешь Kaggle, можно загрузить данные в самом Kaggle

## Обучение

Запуск обучения 

```bash
python train.py -cn=soundstream
```

Checkpoints сохраняются сюда

```bash
saved/soundstream_6kbps/
```

## Инференс

Скачать финальный checkpoint

```bash
python scripts/download_checkpoint.py
```

Восстановить локальный аудиофайл

```bash
python inference.py input_path=/path/to/audio.wav output_path=outputs/reconstructed.wav
```

Или восстановить аудио по URL

```bash
python inference.py input_url=https://example.com/audio.wav output_path=outputs/reconstructed.wav
```

## Логирование

В Comet ML логируются следующие метрики и лоссы

- `generator_loss`
- `discriminator_loss`
- `spectral_loss`
- `adversarial_loss`
- `feature_matching_loss`
- `commitment_loss`
- `codebook_perplexity`
- `STOI`
- `NISQA`
