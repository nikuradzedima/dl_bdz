# SoundStream Neural Audio Codec


Мною был реализован нейросетевой аудиокодек SoundStream по статье.
Модель сжимает звуковой сигнал в дискретное представление через Encoder и RVQ, а
затем восстанавливает через Decoder.

## Архитектура

```text
waveform -> Encoder -> RVQ -> Decoder -> reconstructed waveform
```

- sample rate: `16 kHz`;
- strides encoder-а: `[2, 4, 5, 5]`;
- общий stride: `200`;
- latent frames: `80` в секунду;
- RVQ: `8` quantizer-ов, codebook size `1024`;
- bitrate: `80 * 8 * log2(1024) = 6400 bps`.

Для обучения также использовались waveform discriminator и STFT discriminator.

## Обучение

- train: `LibriSpeech train-clean-100`;
- test: `LibriSpeech test-clean`;
- crop length: `8000` samples, то есть `0.5` секунды;
- batch size: `12`;
- optimizer: Adam, learning rate `2e-4`;
- всего: `45000` шагов;
- устройство: Kaggle GPU.

Во время обучения валидация считалась на `100` batches, чтобы не тратить много
времени на каждую эпоху. Финальные метрики ниже посчитаны отдельно на полном
`test-clean`.


## Лоссы

В generator loss входят

- multi-scale spectral loss
- adversarial loss
- feature matching loss
- commitment loss

Для discriminator используется hinge loss.

## Графики

Метрики графиков выглядят следующим образом, то есть это сами метрики, а также лоссы

![STOI и NISQA](assets/quality_metrics.png)

![Основные лоссы](assets/reconstruction_losses.png)

![Codebook perplexity](assets/codebook_perplexity.png)

## Результаты

Финальная оценка модели

| Модель | STOI | NISQA |
| --- | ---: | ---: |
| SoundStream | 0.9070 | 3.6354 |

## Ссылки

- Comet ML report https://www.comet.com/dimadmitrij734/dl-bdz-1/reports/arGTBJQ3zAIN9HNG1FixbwI2Y
- Checkpoint https://huggingface.co/dimadmitrij734/bdz_first/resolve/main/model_best.pth