# Работа над хакатоном MORE.Tech 4.0 от ВТБ

* [ML](#ml)
* [Back-end](#backend)
  * [1. POST /add_user](#add_user)
  * [2. GET /get_news/$id](#get_news)
  * [3. GET /get_trends/$id](#get_trends)
* [Парсинг](#pars)
* [Инструкции](#instruction)
  * [Запуск веб-приложения](#start_app)

## <a name="ml"></a> ML

В качестве ролей в решении хакатона были приняты бухгалтер и генеральный директор.

Модель можно разбить на несколько этапов
1. С помощью предобученной языковой модели rubert-tiny мы получили эмбеддинги датасета с парсингами сайтов - массив действительных чисел с 312 столбцами. Далее весь наш рабочий процесс строился на полученных эмбеддингах. В выборе модели мы исходили из того, насколько быстро мы будем получать наши эмбеддинги, поэтому и остановились на быстрой, легковесной модели rubert-tiny
2. Массив эмбеддингов использовался в качестве данных для кластеризации методом k-средних. На выходе предполагалось получить k групп текстов, близких по смыслу и несущих информацию о схожих событиях. `KMeans` из пакета `sk-learn`, гиперпараметр числа кластеров `k` подбирался по методу локтя функцией `KElbowVisualizer`.

Пример выхода KElbowVisualizer: 
\
![Пример выхода KElbowVisualizer](https://user-images.githubusercontent.com/98041378/194737110-283f5594-b961-47f5-95c0-4c3610fac70d.png)

3. Визуализация результатов кластеризации в tSNE\
tSNE - Бухгалтеры:
\
![tSNE - Бухгалтеры](https://user-images.githubusercontent.com/98041378/194737052-e8350a86-23cd-400a-a500-cc3cd96234b9.png)
\
tSNE - Гендиры:
\
![tSNE - Гендиры](https://user-images.githubusercontent.com/98041378/194737075-e083f6a3-3322-48e3-a6fd-c14be64f4ba5.png)
\
Стохатическая проекция результатов кластеризации позволяет увидеть результаты работы алгоритма `KMeans`

Гистограмма кластеров - Бухгалтеры:
\
![image](https://user-images.githubusercontent.com/98041378/194737322-bc19c561-1f4b-446f-b732-91880d9e502a.png)


4. Выделение трендов с помощью анализа временных рядов кластеров новостей
\Временной ряд - Бухгалтеры:
\
![image](https://user-images.githubusercontent.com/98041378/194737420-a7dc3c16-9f31-4691-81ec-1fd998ca289c.png)
\

Временной ряд - Гендиры: 
\
![image](https://user-images.githubusercontent.com/98041378/194737433-5a4d3176-70a4-4527-8033-c200f9deccaf.png)
\

На выходе алгоритма для определения существования тренда используется критерий, основанный на скользящей средней и скользящем СКО. 

5. Cуммаризация трендов

Выделенные кластеры не обладают интерпретацией, имеют только численное представление, а потому не могут в полной мере восприниматься как тренды - некоторые семантические события. Суммаризация на основе TF-IDF позволяет выделить список наиболее реевантных н-грам (слов/словосочетаний), формирующий общее представление о контенте кластера. Далее вручную каждому кластеру было дано название с наиболее близкой ассоциацией. Именно суммаризированные тренды выводятся при запросе "Тренды"

6. Дайджесты 

Дайджесты строятся исходя из трендов на текущий момент времени, или же на определенную дату. Мы решили реализовать этот функционал для более удобной откладки кода. Всего у нас существует две роли: бухгалтер и ген. дир. Каждой из этих ролей сопоставляется список ключевых слов, который был отобран вручную. Далее, модель, в нашем случае rubert-tiny, превращает этот список слов в эмбеддинг (вектор длины 312). Аналогичная ситуация происходит и с новостями, мы их векторизуем, т.е. получаем эмбеддинги длины 312. Пул новостей, также проходит отбор по "трендовости", из нескольких новостных кластеров берутся кластеры, которые алгоритм, описаннный в пунктах выше, определяет, как "трендовые". На данном этапе мы работаем уже со сформированном списком "трендовых" новостей. Для того, чтобы оценить актуальность новости для каждой из ролей мы смотрим на косинусную схожесть между эмбеддингами новостей и эмбеддингом роли. Исходя из этого и строится топ-3 нашего новостного дайджеста!
Принцип выдачи дайджестов осно

## <a name="backend"></a> Back-end
Разработка веб-приложения велась с использованием веб-фреймворка Flask (Python)
### <a name="add_user"></a> 1. POST /add_user
Принимает на вход данные о новом пользователе в формате `json`.

При успешной обработке возвращает `id` пользователя.

Пример входных данных: `{"name": "John", "role": "ceo"}`

Пример ответа: `{"id": 1}`

При невалидных данных возвращает `400`: `{"code": 400,"message": "Validation Failed"}`

### <a name="get_news"></a> 2. GET /get_news/$id
Принимает на вход `id` пользователя и возвращает новости в формате `json`.

Отвечает за получение новостей пользователями.

Пример ответа: `{"title": "Заголовок", "text": "Текст"}`

При невалидном `id` возвращает: `{"code": 404,"message": "User not found"}`

### <a name="get_trends"></a> 3. GET /get_trends/$id
Принимает на вход `id` пользователя и возвращает тренды в формате `json`.

Отвечает за получение трендов пользователями.

Пример ответа: `{"trends": ["trend1", "trend2"]}`

При невалидном `id` возвращает: `{"code": 404,"message": "User not found"}`

## <a name="pars"></a> Парсинг
Парсинг данных для обучения проводился на Python с использованием BeautifulSoup, requests, feedparser.

Были проанализированны обычные новостные сайты, а также тематические телеграм-каналы.

## <a name="instruction"></a> Инструкции
### <a name="start_app"></a> Запуск веб-приложения
#### Настраиваем виртуальное окружение
##### Устанавливаем `python3-venv`
      sudo apt install python3-venv
##### Создаем виртуальное окружение
      python3 -m venv venv
##### Переходим в созданное окружение
      source venv/bin/activate
##### Устанавливаем необходимые пакеты
      pip install -r requirements.txt
##### Запускаем веб-приложение
      gunicorn --workers=4 --bind=127.0.0.1:5000 main:app
