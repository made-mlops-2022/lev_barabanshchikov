## Барабанщиков Лев, ML-21. ДЗ #3

### Суть этого ДЗ -- познакомиться с `airflow`.

### Легенда:

1. Откуда-то берутся данные... Мы их используем для обучения МЛ модельки для задачи классификации.
2. Еженедельно, мы переобучаем модель на новых данных, ручками смотрим на метрики и если класс, то выкатываем ее на
   прод.
3. Ежедневно, текущая выбранная нами модель скорит данные и записывает предсказания куда-то.
4. Эти предсказания используют -- все счастливы =)

### Запуск

```bash
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
chmod +x ./docker-create-databases/create-databases.sh
cd images/airflow-ml-base
docker build -t airflow-ml-base:latest .
cd -
USER=<email> PASS=<pw> docker-compose up --build
```
