**Как запускать?**
<br>
Можно изменить файл [train.sh](train.sh) и передать свои параметры для тренировки.
<br><br>
**Структура датасета**

Датасет должен быть представлен в формате COCO
<br>
[COCO Tutorial](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) – здесь представлена
структура ожидаемых данных и вкратце приведено описание каждого из полей.

Пример одной аннотации:

```

{
  // [x0,y0, x1, y1, ...] – вершины многоугольника, описывающего область сегментации.
  "segmentation": [
    [
      510.66,
      423.01,
      511.72,
      420.03,
      ...,
      510.45,
      423.01
    ]
  ],
  // класс к которому относится аннотированая область 
  "image_id": 289343,
  "id": 1768,
  "category_id": 18,
  "area": 702.1057499999998,
  "iscrowd": 0,
  "bbox": [
    473.07,
    395.93,
    38.65,
    28.67
  ]
}

```

<br>
Для одной картинки может быть несколько областей, тогда в файлике с аннотациями должно быть несколько словарей для каждой из областей:

```
"annotations": [
    {
      "segmentation": [...],
      "category_id": 1,
      "image_id": IMAGE_ID
    },
    {
      "segmentation": [...],
      "category_id": 2
      "image_id": IMAGE_ID
    },
    ...
]
```
