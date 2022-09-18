# <p align="center"> ✂️ CarveKit ✂️  </p>

<p align="center"> <img src="/docs/imgs/logo.png"> </p>

<p align="center">
<img src="https://github.githubassets.com/favicons/favicon-success.svg"> <a src="https://github.com/OPHoperHPO/image-background-remove-tool/actions">
<img src="https://github.com/OPHoperHPO/image-background-remove-tool/workflows/Test%20release%20version/badge.svg?branch=master"> <a src="https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/carvekit_try.ipynb">
<img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>
</p>

**********************************************************************
<p align="center"> <img align="center" width="512" height="288" src="/docs/imgs/compare/readme.jpg"> </p>

> Изображения с более высоким разрешением с примером выше можно увидеть в директории docs/imgs/compare/ и docs/imgs/input folders.

#### 📙 README Language
[Russian](/docs/readme/ru.md)
[English](/README.md)

## 📄 О проекте:  
Автоматизированное высококачественное удаление фона с изображения с использованием нейронных сетей


## 🎆 Особенности:  
- Высокое качество выходного изображения
- Пакетная обработка изображений
- Поддержка NVIDIA CUDA и процессорной обработки
- Легкое взаимодействие и запуск
- 100% совместимое с remove.bg API FastAPI HTTP API
- Удаляет фон с волос
- Простая интеграция с вашим кодом

## ⛱ Попробуйте сами на [Google Colab](https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/carvekit_try.ipynb) 
## ⛓️ Как это работает?

1. Пользователь выбирает картинку или папку с картинками для обработки
2. Происходит предобработка фотографии для обеспечения лучшего качества выходного изображения
3. С помощью технологии машинного обучения убирается фон у изображения
4. Происходит постобработка изображения для улучшения качества обработанного изображения
## 🎓 Интегрированные нейронные сети:
* [U^2-net](https://github.com/NathanUA/U-2-Net)
* [BASNet](https://github.com/NathanUA/BASNet)
* [DeepLabV3](https://github.com/tensorflow/models/tree/master/research/deeplab)


## 🖼️ Методы предварительной обработки и постобработки изображений:
### 🔍 Методы предобработки:
* `none` - методы предобработки не используются.
> Они будут добавлены в будущем.
### ✂ Методы постобработки:
* `none` - методы постобработки не используются
* `fba` (по умолчанию) - Этот алгоритм улучшает границы изображения при удалении фона с изображений с волосами и т.д. с помощью нейронной сети FBA Matting. Этот метод дает наилучший результат в сочетании с u2net без каких-либо методов предварительной обработки.

## 🏷 Настройка для обработки на CPU:
1. `pip install carvekit --extra-index-url https://download.pytorch.org/whl/cpu`
> Проект поддерживает версии Python от 3.8 до 3.10.4.

## 🏷 Настройка для обработки на GPU:  
1. Убедитесь, что у вас есть графический процессор NVIDIA с 8 ГБ видеопамяти.
2. Установите `CUDA Toolkit и Видео дравер для вашей видеокарты.`
3. `pip install carvekit --extra-index-url https://download.pytorch.org/whl/cu113`
> Проект поддерживает версии Python от 3.8 до 3.10.4.

## 🧰 Интеграция в код:  
### Если вы хотите быстрее приступить к работе без дополнительной настройки
``` python
import torch
from carvekit.api.high import HiInterface

interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                               device='cuda' if torch.cuda.is_available() else 'cpu',
                               seg_mask_size=320, matting_mask_size=2048)
images_without_background = interface(['./tests/data/cat.jpg'])                               
cat_wo_bg = images_without_background[0]
cat_wo_bg.save('2.png')
                   
```

### Если вы хотите провести детальную настройку
``` python
import PIL.Image

from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.u2net import U2NET
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

u2net = U2NET(device='cpu',
              batch_size=1)

fba = FBAMatting(device='cpu',
                 input_tensor_size=2048,
                 batch_size=1)

trimap = TrimapGenerator()

preprocessing = PreprocessingStub()

postprocessing = MattingMethod(matting_module=fba,
                               trimap_generator=trimap,
                               device='cpu')

interface = Interface(pre_pipe=preprocessing,
                      post_pipe=postprocessing,
                      seg_pipe=u2net)

image = PIL.Image.open('tests/data/cat.jpg')
cat_wo_bg = interface([image])[0]
cat_wo_bg.save('2.png')
                   
```


## 🧰 Запустить через консоль:  
 * ```python3 -m carvekit  -i <input_path> -o <output_path> --device <device>```  
 
### Все доступные аргументы:  
````
Usage: carvekit [OPTIONS]

  Performs background removal on specified photos using console interface.

Options:
  -i ./2.jpg                   Путь до входного файла или директории  [обязателен]
  -o ./2.png                   Путь для сохранения результата обработки
  --pre none                   Метод предобработки
  --post fba                   Метод постобработки
  --net u2net                  Нейронная сеть для сегментации
  --recursive                  Включение рекурсивного поиска изображений в папке
  --batch_size 10              Размер пакета изображений, загруженных в ОЗУ 

  --batch_size_seg 5           Размер пакета изображений для обработки с помощью
                               сегментации

  --batch_size_mat 1           Размер пакета изображений для обработки с помощью
                               матирования

  --seg_mask_size 320          Размер исходного изображения для сегментирующей
                               нейронной сети

  --matting_mask_size 2048     Размер исходного изображения для матирующей
                               нейронной сети

  --device cpu                 Обрабатывающий девайс
  --help                       Показывает это сообщение

````
## 📦 Запустить фреймворк / FastAPI HTTP API сервер с помощью Docker:

Использование API через Docker — это **быстрый** и эффективный способ получить работающий API.\
**Этот HTTP API на 100% совместим с API клиентами сайта remove.bg** 
<p align="center"> 
<img src="/docs/imgs/screenshot/frontend.png"> 
<img src="/docs/imgs/screenshot/docs_fastapi.png"> 
</p>

>### Важная информация:
>1. Образ Docker имеет фронтенд по умолчанию по адресу `/` и документацию к API по адресу `/docs`.
>2. Аутентификация **включена** по умолчанию. \
> **Ключи доступа сбрасываются** при каждом перезапуске контейнера, если не установлены специальные переменные окружения. \
См. `docker-compose.<device>.yml` для более подробной информации. \
> **Вы можете посмотреть свои ключи доступа в логах докера.**
> 
>3. Примеры работы с API.\
> См. `docs/code_examples/python` для уточнения деталей
### 🔨 Создать и запустить контейнер:
1. Установите `docker-compose`
2. Запустите `docker-compose -f docker-compose.cpu.yml up -d`  # для обработки на ЦП
3. Запустите `docker-compose -f docker-compose.cuda.yml up -d`  # для обработки на GPU
> Также вы можете монтировать папки с вашего компьютера в docker container
> и использовать интерфейс командной строки внутри контейнера докера для обработки
> файлов в этой папке.

## ☑️ Тестирование

### ☑️ Тестирование с локальным окружением
1. `pip install -r requirements_test.txt`
2. `pytest`

### ☑️ Тестирование с Docker
1. Запустите `docker-compose -f docker-compose.cpu.yml run carvekit_api pytest`  # для тестирования на ЦП
2. Run `docker-compose -f docker-compose.cuda.yml run carvekit_api pytest`  # для тестирования на GPU


## 👪 Credits: [Больше информации](/docs/CREDITS.md)

## 💵 Поддержать развитие проекта
  Вы можете поблагодарить меня за разработку этого проекта и угостить меня чашечкой кофе ☕

| Blockchain |            Cryptocurrency           |          Network          |                                              Wallet                                             |
|:----------:|:-----------------------------------:|:-------------------------:|:-----------------------------------------------------------------------------------------------:|
|  Ethereum  | ETH / USDT / USDC / BNB / Dogecoin  |          Mainnet          |                            0x7Ab1B8015020242D2a9bC48F09b2F34b994bc2F8                           |
|  Ethereum  |  ETH / USDT / USDC / BNB / Dogecoin | BSC (Binance Smart Chain) |                            0x7Ab1B8015020242D2a9bC48F09b2F34b994bc2F8                           |
|   Bitcoin  |                 BTC                 |             -             |                            bc1qmf4qedujhhvcsg8kxpg5zzc2s3jvqssmu7mmhq                           |
|    ZCash   |                 ZEC                 |             -             |                               t1d7b9WxdboGFrcVVHG2ZuwWBgWEKhNUbtm                               |
|    Tron    |                 TRX                 |             -             |                                TH12CADSqSTcNZPvG77GVmYKAe4nrrJB5X                               |
|   Monero   |                 XMR                 |          Mainnet          | 48w2pDYgPtPenwqgnNneEUC9Qt1EE6eD5MucLvU3FGpY3SABudDa4ce5bT1t32oBwchysRCUimCkZVsD1HQRBbxVLF9GTh3 |
|     TON    |                 TON                 |             -             |                         EQCznqTdfOKI3L06QX-3Q802tBL0ecSWIKfkSjU-qsoy0CWE                        |

## 📧 __Обратная связь__
Буду рад отзывам о проекте и предложениям об интеграции.

По всем вопросам писать: [farvard34@gmail.com](mailto://farvard34@gmail.com)
