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
1. С помощью предобученной языковой модели `rubert-tiny` мы получили эмбеддинги датасета с парсингами сайтов - массив действительных чисел с 312 столбцами
2. Массив эмбеддингов использовался в качестве данных для кластеризации методом k-средних. На выходе предполагалось получить k групп текстов, близких по смыслу и несущих информацию о схожих событиях. `KMeans` из пакета `sk-learn`, гиперпараметр числа кластеров `k` подбирался по методу локтя функцией `KElbowVisualizer`.

Пример выхода KElbowVisualizer: 
\
![Пример выхода KElbowVisualizer](https://user-images.githubusercontent.com/98041378/194737110-283f5594-b961-47f5-95c0-4c3610fac70d.png)

3. Визуализация результатов кластеризации в tSNE
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
Временной ряд - Бухгалтеры:
\
![image](https://user-images.githubusercontent.com/98041378/194737420-a7dc3c16-9f31-4691-81ec-1fd998ca289c.png)
\

Временной ряд - Гендиры: 
\
![image](https://user-images.githubusercontent.com/98041378/194737433-5a4d3176-70a4-4527-8033-c200f9deccaf.png)\
\


## <a name="backend"></a> Back-end
### <a name="add_user"></a> 1. POST /add_user
### <a name="get_news"></a> 2. GET /get_news/$id
### <a name="get_trends"></a> 3. GET /get_trends/$id
## <a name="pars"></a> Парсинг
## <a name="instruction"></a> Инструкции
### <a name="start_app"></a> Запуск веб-приложения
