## Установка зависимостей (UNIX-системы)

Используйте python версии ^3.10. Установите poetry в эту версию python.

```
pip install poetry
poetry config virtualenvs.create false
```

Склонируйте репозитория
```
git clone ...
cd hse_asml_course
```

Создайте виртуальную среду из этой версии python. Активируйте её.
```
cd hse_asml_course
python3.10 -m venv venv
source venv/bin/activate
```

Установите зависимости.
```
poetry shell
poetry install
```
