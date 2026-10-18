# Восстановление стабильного ForteVision

Проверенное окружение: Ubuntu, Python 3.12, NVIDIA L4 24 GB, PyTorch 2.8.0
с CUDA 12.8. Для восстановления архива оставьте не менее 27 GiB свободного
места. Установка зависимостей требует сети; обработка видео работает локально.

## Скачать и проверить

Скачайте **все файлы** [релиза v0.1.0-hackathon](https://github.com/song-ev/Datathon-2026/releases/tag/v0.1.0-hackathon)
в один каталог. Девять частей `production-weights.tar.part001`–`part009`
образуют один архив. Не распаковывайте части по отдельности.

```bash
export FORTE_HOME=/home/ubuntu/forte-datathon
export RELEASE_DIR="$HOME/fortevision-release"
mkdir -p "$RELEASE_DIR"
gh release download v0.1.0-hackathon --repo song-ev/Datathon-2026 --dir "$RELEASE_DIR"
python3.12 "$RELEASE_DIR/restore_production_weights.py" "$RELEASE_DIR" "$FORTE_HOME"
mkdir -p "$FORTE_HOME/src"
python3.12 -m zipfile -e "$RELEASE_DIR/ForteVision-source.zip" "$FORTE_HOME/src"
```

Используйте пустой каталог назначения: восстановление не перезаписывает
существующие модели. Скрипт проверяет SHA-256 файлов релиза и каждой модели.
Архив исходников фиксирует стабильный коммит `42d9e3a` независимо от изменений
ветки main. GitHub CLI нужен только для команды скачивания; можно скачать
файлы через браузер.

## Установить и запустить

```bash
python3.12 -m venv "$FORTE_HOME/.venv"
"$FORTE_HOME/.venv/bin/pip" install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$FORTE_HOME/.venv/bin/pip" install -r "$FORTE_HOME/src/requirements.txt"
export MODEL_PATH="$FORTE_HOME/models/Qwen3-VL-4B-Instruct"
export DETECTOR_MODE=fusion
bash "$FORTE_HOME/src/scripts/run.sh"
```

Для systemd сначала создайте настройку режима, затем установите сервис
(скрипт рассчитан на пользователя ubuntu и указанный выше FORTE_HOME):

```bash
sudo mkdir -p /etc/systemd/system/fortevision.service.d
printf '[Service]\nEnvironment=DETECTOR_MODE=fusion\n' | sudo tee /etc/systemd/system/fortevision.service.d/stable.conf
bash "$FORTE_HOME/src/scripts/install_service.sh"
```

Сервис слушает порт 8000. API-ключ создаётся в `private/api-key.txt` при первом
запуске. Передайте его жюри отдельно. Для внешнего доступа настройте HTTPS
reverse proxy; адрес прежнего Brev не переносится автоматически.

Проверяйте готовность через `/api/v1/health`, а анализ через `POST /api/v1/analyze`
и `GET /api/v1/jobs/{job_id}` с заголовком `X-API-Key`. Контракт и очередь
описаны в README. Не публикуйте private/, runtime/, ключи или исходные видео.

## Состав и результаты

Замороженная Qwen3-VL-4B-Instruct, upstream revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, Apache-2.0; финальный
классификатор пространственного движения и временной локализатор.
Нормативная база находится в `forte/rules.json`. LoRA, 8B и другие
экспериментальные веса не требуются и не поставляются.

Оценка разработки на 40 уникальных открытых видео: accuracy 62,5%, recall
нарушений 71,4%, precision 62,5%; TP=15, FP=9, TN=10, FN=6.
Модель и порог выбраны по этим же out-of-fold предсказаниям: это не независимая
оценка на скрытых тестах жюри. Локализация ±2 секунды: 8/21 положительных видео.
Прототип требует проверки специалистом.
