"""Calendário de cada dia e distância entre dois dias.

Os dias sazonais são os que mudam o comportamento de quem faz transações: fim de semana, feriado, véspera, dia seguinte, ponte,
fim de semana prolongado, início, meio e fim de mês, Carnaval, Black Friday, datas comerciais, fim de ano e férias escolares.
Cada um vira uma marca (`FLAGS`) que a regressão do nível, a escolha dos dias parecidos e o LightGBM usam.
O modelo só aprende o efeito de uma marca se ela aparecer no histórico, por isso o ideal são 14 meses de dados ou mais.

Os feriados vêm da biblioteca `holidays`, para o país e a região de `COUNTRY` e `SUBDIVISION`, nos anos que os dados cobrem.
Para outro país, troque essas duas constantes. As demais datas (Black Friday, datas comerciais, fim de ano) seguem regras do calendário brasileiro.
"""
from functools import lru_cache

import holidays as holidays_lib
import numpy as np
import pandas as pd

COUNTRY = "BR"  # código do país na biblioteca holidays
SUBDIVISION = None  # estado ou região, por exemplo "SP"; None usa só os feriados nacionais
# Feriados que a biblioteca classifica como opcionais, mas que na prática paralisam o dia
DE_FACTO_HOLIDAYS = ("Carnaval", "Corpus Christi")
MONTH_START_DAYS = range(5, 11)  # dias de pagamento
MID_MONTH_DAYS = range(15, 21)  # adiantamento e vales

# Marcas por dia, na ordem usada pela regressão. Peso: quanto a diferença da marca afasta dois dias na escolha de dias parecidos.
FLAG_WEIGHTS = {
    "holiday": 3.0, "eve": 2.0, "post": 1.5, "bridge": 2.0, "long_weekend": 1.5, "carnival": 3.0,
    "black_friday": 3.0, "cyber_monday": 3.0, "commercial": 1.0, "new_year": 2.0, "year_end": 0.5,
    "school_break": 0.4, "month_end": 0.5, "mid_month": 0.3,
}
FLAGS = tuple(FLAG_WEIGHTS)
_WEIGHTS = np.array(list(FLAG_WEIGHTS.values()))
ONE_DAY = pd.Timedelta(days=1)


def _nth_weekday(year, month, weekday, n):
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.dayofweek) % 7 + 7 * (n - 1))


class Calendar:
    """Datas especiais dos anos pedidos e as marcas de cada dia."""

    def __init__(self, years):
        years = list(years)
        public = holidays_lib.country_holidays(COUNTRY, subdiv=SUBDIVISION, years=years)
        optional = holidays_lib.country_holidays(COUNTRY, subdiv=SUBDIVISION, years=years, categories=("optional",))
        self.holidays = {pd.Timestamp(d) for d in public} | {
            pd.Timestamp(d) for d, name in optional.items() if any(key in name for key in DE_FACTO_HOLIDAYS)}
        carnival = {}  # primeiro dia de Carnaval de cada ano (segunda-feira)
        for d, name in optional.items():
            if "Carnaval" in name:
                carnival[d.year] = min(carnival.get(d.year, d), d)
        # sábado a quarta-feira de cinzas
        self.carnival = {pd.Timestamp(first) + pd.Timedelta(days=k) for first in carnival.values() for k in range(-2, 3)}
        fridays = [_nth_weekday(y, 11, 4, 4) for y in years]  # Black Friday: 4ª sexta-feira de novembro
        self.black_friday = set(fridays)
        self.cyber_monday = {f + pd.Timedelta(days=3) for f in fridays}
        # Dia do Consumidor (15/3), Dia das Mães (2º domingo de maio), Namorados (12/6) e Dia dos Pais (2º domingo de agosto)
        self.commercial = {d for y in years for d in (pd.Timestamp(y, 3, 15), _nth_weekday(y, 5, 6, 2), pd.Timestamp(y, 6, 12),
                                                     _nth_weekday(y, 8, 6, 2))}

    def is_bridge(self, d):
        """Segunda antes de feriado de terça ou sexta depois de feriado de quinta."""
        return d not in self.holidays and ((d.dayofweek == 0 and d + ONE_DAY in self.holidays)
                                           or (d.dayofweek == 4 and d - ONE_DAY in self.holidays))

    def _is_off(self, d):
        return d.dayofweek >= 5 or d in self.holidays or self.is_bridge(d)

    def is_long_weekend(self, d):
        """Sábado ou domingo em uma sequência de 3 ou mais dias sem expediente (com feriado ou ponte colado)."""
        if d.dayofweek < 5:
            return False
        run = 1
        for step in (-ONE_DAY, ONE_DAY):
            t = d + step
            while self._is_off(t):
                run += 1
                t += step
        return run >= 3

    def flags(self, d):
        bridge = self.is_bridge(d)
        workday = d.dayofweek < 5 and not bridge  # véspera e dia seguinte só valem em dia útil que não seja ponte (senão contariam o efeito duas vezes)
        return {
            "holiday": d in self.holidays, "eve": workday and d + ONE_DAY in self.holidays,
            "post": workday and d - ONE_DAY in self.holidays, "bridge": bridge, "long_weekend": self.is_long_weekend(d),
            "carnival": d in self.carnival, "black_friday": d in self.black_friday, "cyber_monday": d in self.cyber_monday,
            "commercial": d in self.commercial, "new_year": (d.month == 12 and d.day >= 27) or (d.month == 1 and d.day <= 2),
            "year_end": d.month == 12 and d.day <= 23, "school_break": d.month == 7 or (d.month == 1 and d.day >= 3),
            "month_end": d.day >= d.days_in_month - 1, "mid_month": d.day in MID_MONTH_DAYS,
        }


@lru_cache(maxsize=8)
def _calendar(first_year, last_year):
    return Calendar(range(first_year - 1, last_year + 2))  # um ano de folga em cada ponta, para a véspera e o dia seguinte


def build_day_features(days):
    """Um dicionário por dia: data, dia da semana, início de mês, as marcas de `FLAGS` e o vetor `vec` delas."""
    calendar = _calendar(min(days).year, max(days).year)
    out = []
    for d in days:
        flags = calendar.flags(d)
        out.append(dict(day=d, dow=d.dayofweek, month_start=d.day in MONTH_START_DAYS, **flags,
                        vec=np.array([flags[name] for name in FLAGS])))
    return out


def day_distance(a, b):
    """Distância entre dois dias. Menor é mais parecido.

    Soma: o peso de cada marca que difere (`FLAG_WEIGHTS`), dia da semana (0 igual, 0,8 ambos úteis ou ambos de fim de semana,
    2 caso contrário), início de mês (0,2 por dia de diferença no dia do mês, até 1; 0,5 se só um dos dois está no início do mês)
    e 0,001 por dia de distância entre as datas.
    """
    if a["dow"] == b["dow"]:
        dow = 0.0
    elif (a["dow"] >= 5) == (b["dow"] >= 5):
        dow = 0.8
    else:
        dow = 2.0
    if a["month_start"] and b["month_start"]:
        month = min(1.0, 0.2 * abs(a["day"].day - b["day"].day))
    else:
        month = 0.5 * (a["month_start"] != b["month_start"])
    recency = 0.001 * abs((a["day"] - b["day"]).days)
    return float(_WEIGHTS @ (a["vec"] != b["vec"])) + dow + month + recency
