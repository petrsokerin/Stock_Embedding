import sys
from urllib.parse   import urlencode
from urllib.request import urlopen, Request
from datetime       import timedelta, datetime
from time           import sleep
import yaml

def get_assets_timestemps_codes():
    with open('configs/downloads/ticker_codes.yaml', 'r') as f:
        assets = yaml.load(f, Loader=yaml.FullLoader)

    with open('configs/downloads/timeframes.yaml', 'r') as f:
        timeframes = yaml.load(f, Loader=yaml.FullLoader)
    
    return assets, timeframes

def get_piece(asset, timeframe, first, last, path) :

    assets, timeframes = get_assets_timestemps_codes()

    try :

        domain = "http://export.finam.ru/"
        code = assets[asset]

        properties = urlencode([
		    ("market",    0),                       # Тип рынка
		    ("em",        code),           # Код актива
	            ("code",      asset),                   # Имя актива
                    ("apply",     0),                       # Избранное
		    ("df",        first.day),               # Начальная дата, номер дня (1-31)
		    ("mf",        first.month - 1),         # Начальная дата, номер месяца (0-11)
		    ("yf",        first.year),              # Начальная дата, год
		    ("from",      first),                   # Начальная дата
		    ("dt",        last.day),                # Конечная дата, номер дня (1-31)
		    ("mt",        last.month - 1),          # Конечная дата, номер месяца (0-11)
		    ("yt",        last.year),               # Конечная дата, год
		    ("to",        last),                    # Конечная дата
		    ("p",         timeframes[timeframe]),   # Таймфрейм
		    ("f",         asset + "_" + timeframe), # Имя сформированного файла
		    ("e",         ".txt"),                  # Расширение сформированного файла
		    ("cn",        asset),                   # Имя актива	
		    ("dtf",       1),                       # Формат даты
		    ("tmf",       1),                       # Формат времени
		    ("MSOR",      0),                       # Время свечи
		    ("mstime",    "on"),                    # Московское время	
		    ("mstimever", 1),                       # Коррекция часового пояса	
		    ("sep",       1),                       # Разделитель полей
		    ("sep2",      1),                       # Разделитель разрядов
		    ("datf",      5),                       # Формат записи в файл
		    ("at",        0)                        # Заголовки столбцов
        ])

        url = domain + asset + "_" + timeframe + ".txt?" + properties
        
        request = Request(url, headers = {"User-Agent": "Mozilla/5.0"}) # ?

        text = urlopen(request).readlines()
        
        lines = []
        
        for line in text:
            lines.append(line.strip().decode("utf-8") + "\n")

        if len(lines) != 0:
            with open(path, "a") as f:
                for line in reversed(lines):
                    f.write(line)	

    except:
        print("Exception: ", sys.exc_info()[0])
        raise

def get(asset, timeframe, path) :

    try:
        total = 365 * 1
        batch = 365 * 1

        first = datetime.now().date() - timedelta(days = 1)
        last  = first                 - timedelta(days = batch)
        
        for i in range(0, total, batch):
            
            get_piece(asset, timeframe, last, first, path)

            first = last  - timedelta(days = 2)
            last  = first - timedelta(days = batch)

            sleep(1)

    except:
        print("Exception: ", sys.exc_info()[0])
        raise
    
def get_for_levels(asset, timeframe, path) :

    try:
        
        fout = open(path, "w")
        
        fout.close()

        total = 365 * 10

        first = datetime.now().date() - timedelta(days = 1)
        last  = first                 - timedelta(days = total)
            
        get_piece(asset, timeframe, last, first, path)

        sleep(1)

    except:
        print("Exception: ", sys.exc_info()[0])
        raise
