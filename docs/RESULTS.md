# Resultados e revisão estatística

Os números abaixo mostram o que a demonstração alcança com os canais simulados. Todos vêm de `uv run cli.py evaluate` sobre os dados simulados com
os dados de referência (`data/reference/`). O período de teste cobre os 21 dias finais do histórico e o modelo é treinado só com dias anteriores. Com a data de hoje
o calendário muda e os números variam um pouco. Os valores são WAPE em %, e quanto menor, melhor. As tabelas completas ficam em
`out/metrics.csv` e `out/coverage.csv`.

O histórico simulado tem 400 dias, mais de um ano, para cada dia sazonal aparecer ao menos uma vez. Tem também 26 eventos de 20 minutos a
4 horas: 18 quedas com retenção e 8 picos anormais. O modelo não presume duração nenhuma. Ele marca como sujo tudo o que os dias parecidos
não explicam, pelo tempo que durar.

Nomes que aparecem nas tabelas:

| Nome | Significado |
|---|---|
| `normal` | dia comum, sem evento e sem dia sazonal forte |
| `month_start` | início de mês (dias 5 a 10) |
| `holiday` | feriado |
| `special` | dia sazonal forte que não é feriado nem início de mês: véspera, dia seguinte, ponte, fim de semana prolongado, Carnaval, Black Friday, Cyber Monday, data comercial e virada do ano |
| `event` | intervalos afetados por queda, retenção ou pico, medidos contra o volume que teria acontecido |
| `noise_floor` | piso de ruído: erro de uma previsão perfeita |
| `model` | o modelo deste projeto (perfil, ritmo do dia e estados) |
| `model_lgbm_p50` | o centro da previsão trocado pela mediana do LightGBM |
| `no_cleaning` | o mesmo modelo, sem limpeza do histórico e sem tratar eventos |
| `last_week` | o valor do mesmo intervalo, 7 dias antes |
| `profile_day_ahead` | o perfil do dia, calculado antes do dia começar |
| `statsforecast_seasonal_naive`, `statsforecast_mstl` | modelos de séries temporais da biblioteca StatsForecast |

## Como ler o erro

Transações são contagens. Mesmo conhecendo a taxa verdadeira, o observado varia ao acaso, então uma previsão perfeita erra
cerca de `0,8 / √taxa` por intervalo. Esse é o piso de ruído, que nenhum modelo vence. Há dois erros:

- contra o observado: o que um painel mostraria. Compare com o piso;
- contra a taxa verdadeira: só existe na simulação e mede o erro do modelo sem o ruído.

## 1. Erro próximo do piso de ruído

Erro de 5 minutos à frente, contra o observado. O simulador faz o nível oscilar devagar ao longo do dia (4%), o que o torna
imprevisível. Por isso sobra de 0,4 a 0,8 ponto percentual acima do piso em dias comuns (app 2,75% contra 2,24%, site 5,33% contra 4,97%,
whatsapp 14,96% contra 14,13%). Nos dias sazonais a distância fica entre 0,1 e 0,9 ponto. O feriado é um dia só na avaliação, então o
erro dele contra o observado depende do ruído sorteado nesse dia.

| canal | cenário | noise_floor | model | model_lgbm_p50 | no_cleaning | last_week |
|---|---|---|---|---|---|---|
| app | normal | 2.24 | 2.75 | 2.63 | 2.77 | 16.88 |
| app | month_start | 2.01 | 2.51 | 2.44 | 3.00 | 19.09 |
| app | holiday | 2.75 | 3.37 | 3.30 | 3.37 | 103.33 |
| app | special | 2.49 | 3.11 | 3.06 | 3.11 | 33.33 |
| site | normal | 4.97 | 5.33 | 5.25 | 5.41 | 17.80 |
| site | month_start | 4.36 | 4.88 | 4.79 | 4.88 | 24.34 |
| site | holiday | 6.10 | 6.46 | 6.25 | 6.71 | 99.27 |
| site | special | 5.44 | 5.63 | 5.55 | 5.63 | 35.46 |
| whatsapp | normal | 14.13 | 14.96 | 14.90 | 15.14 | 25.73 |
| whatsapp | month_start | 12.47 | 12.75 | 12.65 | 12.82 | 27.25 |
| whatsapp | holiday | 16.93 | 17.02 | 16.94 | 17.02 | 103.15 |
| whatsapp | special | 15.70 | 16.55 | 16.51 | 16.56 | 50.52 |

`last_week` é o que um painel simples usaria. O LightGBM (`model_lgbm_p50`) melhora um pouco em quase todos os cenários. A limpeza
(`model` contra `no_cleaning`) pesa mais no início de mês do app: 2,51% contra 3,00%.

## 2. Erro contra a taxa verdadeira

5 minutos à frente:

| canal | cenário | model | model_lgbm_p50 | no_cleaning | last_week |
|---|---|---|---|---|---|
| app | normal | 1.60 | 1.44 | 1.64 | 16.75 |
| app | month_start | 1.54 | 1.48 | 2.08 | 18.97 |
| app | holiday | 1.82 | 1.83 | 1.82 | 103.54 |
| app | special | 1.76 | 1.60 | 1.76 | 33.16 |
| app | event | 3.49 | 3.27 | 39.88 | 10.26 |
| site | normal | 2.13 | 1.91 | 2.24 | 16.86 |
| site | month_start | 2.18 | 1.98 | 2.20 | 23.90 |
| site | holiday | 3.26 | 2.81 | 3.39 | 100.60 |
| site | special | 2.59 | 2.31 | 2.60 | 34.26 |
| site | event | 6.54 | 6.51 | 37.38 | 19.44 |
| whatsapp | normal | 3.81 | 3.39 | 4.14 | 20.78 |
| whatsapp | month_start | 3.82 | 3.31 | 4.06 | 24.24 |
| whatsapp | holiday | 5.04 | 3.81 | 5.03 | 101.87 |
| whatsapp | special | 3.84 | 3.50 | 3.84 | 48.27 |
| whatsapp | event | 6.31 | 6.37 | 21.72 | 14.20 |

1 hora à frente:

| canal | cenário | model | model_lgbm_p50 | no_cleaning | last_week |
|---|---|---|---|---|---|
| app | normal | 2.84 | 2.72 | 3.06 | 16.75 |
| app | month_start | 3.09 | 2.85 | 3.80 | 18.97 |
| app | holiday | 4.22 | 3.41 | 4.22 | 103.54 |
| app | special | 3.14 | 2.85 | 3.14 | 33.16 |
| app | event | 2.77 | 2.65 | 20.42 | 10.26 |
| site | normal | 2.74 | 2.78 | 2.91 | 16.86 |
| site | month_start | 2.97 | 2.85 | 2.99 | 23.90 |
| site | holiday | 3.87 | 3.89 | 4.28 | 100.60 |
| site | special | 3.11 | 3.10 | 3.11 | 34.26 |
| site | event | 6.79 | 6.65 | 24.77 | 19.44 |
| whatsapp | normal | 3.86 | 3.85 | 4.00 | 20.78 |
| whatsapp | month_start | 4.10 | 3.78 | 4.36 | 24.24 |
| whatsapp | holiday | 5.02 | 4.51 | 5.01 | 101.87 |
| whatsapp | special | 3.84 | 3.93 | 3.84 | 48.27 |
| whatsapp | event | 5.00 | 5.24 | 13.83 | 14.20 |

A linha `event` mede os intervalos de queda, retenção e pico contra o volume que teria acontecido sem o evento. Os eventos duram de 20 minutos
a 4 horas. O tratamento decide o resultado nos três canais: em 5 minutos o app erra 3.49% com tratamento e 39.88% sem, o site 6.54% contra 37.38%
e o whatsapp 6.31% contra 21.72%.

A saída de RECOVERY e SURGE passou a depender do ruído de contagem do canal (METHOD.md, 4.3). Antes, 3 intervalos entre 0,80 e 1,25 bastavam para voltar a NORMAL,
e no app a cauda da retenção (+10% a +25%) entrava no alfa. O erro em `event` a 5 minutos caiu de 5,64% para 3.49% no app e de 7,49% para 6.54% no site.
Numa segunda janela de avaliação (2026-09-10 a 2026-09-30) o app foi de 8,55% para 5,13%. O whatsapp, de volume baixo e ruído alto, mantém a faixa de antes e não mudou.

Numa versão anterior, com eventos de até 30 minutos, o whatsapp não ganhava com o tratamento em 5 minutos (3,98% contra 3,29%): a janela do
alfa cresce quando o volume é baixo (até 72 intervalos), então um evento de 6 intervalos pesava pouco nela. Com incidentes de 1 a 4 horas ele
pesa e o tratamento passa a decidir. O erro não foi separado por duração do evento.

## 3. Dia seguinte: perfil contra StatsForecast

Previsão do dia inteiro, feita antes do dia, contra a taxa verdadeira. O StatsForecast usa o mesmo histórico limpo.

| canal | cenário | profile_day_ahead | statsforecast_seasonal_naive | statsforecast_mstl |
|---|---|---|---|---|
| app | normal | 4.08 | 8.26 | 9.37 |
| app | month_start | 5.52 | 18.58 | 13.47 |
| app | holiday | 9.45 | 52.75 | 66.36 |
| app | special | 5.09 | 18.43 | 27.55 |
| app | event | 3.51 | 12.50 | 8.07 |
| site | normal | 3.55 | 8.50 | 6.02 |
| site | month_start | 4.20 | 22.58 | 22.29 |
| site | holiday | 3.39 | 35.43 | 43.23 |
| site | special | 3.47 | 12.58 | 12.28 |
| site | event | 6.77 | 10.43 | 8.22 |
| whatsapp | normal | 5.09 | 15.50 | 8.72 |
| whatsapp | month_start | 4.94 | 23.35 | 21.84 |
| whatsapp | holiday | 4.29 | 41.70 | 45.84 |
| whatsapp | special | 5.29 | 22.70 | 17.55 |
| whatsapp | event | 4.33 | 11.32 | 7.66 |

Em dia comum o perfil erra de 3,6% a 5,1% e o StatsForecast de 6% a 16%. Em início de mês, feriado e dia sazonal forte o StatsForecast erra de
12% a 66%, contra 3% a 9% do perfil, porque modela só a sazonalidade diária e semanal e não conhece o calendário. O pior caso do perfil é o
feriado do app (9,45%), e o feriado é um dia só na avaliação.

## 4. Faixa de 80%

Fração do alvo dentro de P10 a P90, para um alvo de 80%, com a calibração conformal (METHOD.md, 5.1.2) e os horizontes de 5 min a 8 h medidos
pelo LightGBM. Em `event` o alvo é a taxa verdadeira, não o observado. Referência, 21 dias de avaliação, uma seed:

| canal | cenário | 5 min | 1 h | 2 h | 4 h | 8 h |
|---|---|---|---|---|---|---|
| app | normal | 81 | 79 | 77 | 77 | 80 |
| app | month_start | 80 | 76 | 75 | 73 | 73 |
| app | holiday | 76 | 74 | 68 | 62 | 69 |
| app | special | 78 | 76 | 74 | 75 | 78 |
| app | event | 72 | 81 | 88 | 93 | 84 |
| site | normal | 81 | 81 | 80 | 79 | 79 |
| site | month_start | 80 | 80 | 79 | 77 | 77 |
| site | holiday | 80 | 79 | 78 | 84 | 86 |
| site | special | 83 | 82 | 81 | 81 | 82 |
| site | event | 81 | 94 | 99 | 89 | 100 |
| whatsapp | normal | 79 | 79 | 79 | 79 | 78 |
| whatsapp | month_start | 81 | 81 | 81 | 80 | 80 |
| whatsapp | holiday | 84 | 84 | 85 | 84 | 82 |
| whatsapp | special | 79 | 80 | 80 | 80 | 81 |
| whatsapp | event | 100 | 100 | 100 | 100 | 100 |

Em dias comuns (`normal`) a cobertura fica entre 77% e 81% em todos os horizontes, em vez de depender de uma extrapolação depois de 1 hora.
Com `uv run cli.py backtest --seeds 1,2 --origins 2` (4 execuções por célula), a cobertura média ficou entre 0,73 e 0,84 em todas as células
que não são `event`, e em `normal` entre 0,78 e 0,81. O feriado do app em 4 h (62% nesta tabela) mostra o limite: é um dia só, e a calibração
vem de dias comuns.

**Faixa em `event`.** O app cobre 72% em 5 minutos (43% antes das correções abaixo, 50% antes da calibração) e o site 81% (65%). O whatsapp cobre 100% porque a faixa de
ruído de Poisson é comparada à taxa sem ruído. O que foi investigado:

- Em `event` o alvo é a taxa verdadeira, que não existe em dado real. A calibração conformal usa só linhas NORMAL e limpas e não vê o evento.
- Separando as origens por estado (app, 5 minutos, 2026-06-01 a 10-21): a pior cobertura não era a dos estados de evento, e sim a de NORMAL **logo depois** do evento
  (15% em 48 casos, com a projeção 10% acima da taxa verdadeira). A máquina voltava a NORMAL cedo e o alfa aprendia com a cauda da retenção.
- Duas correções: (1) a saída de RECOVERY e SURGE passou a ter faixa estreita em canal de volume alto, o que corrigiu o centro (acima);
  (2) a faixa em estado de evento e nos 12 intervalos seguintes usa a largura do horizonte `idade do alfa + h`, no lugar de `h` (METHOD.md, 5.1.3).
- A correção (2) sozinha foi pequena (app 43% para 55% em 5 minutos) e piorou o interval score de `event` (app 0,184 para 0,199, whatsapp 0,378 para 0,442),
  porque a largura cresce sem corrigir o viés do centro. O ganho principal vem da correção (1).
- Via rápida de detecção (METHOD.md, 4.3): quedas e picos extremos (razão < 0,35 ou > 1,8) são confirmados em 1 intervalo, desde que estejam a 6 desvios de contagem do esperado.
  Em 4 janelas (out/2026, set/2026, ago/2026, jun/2026), sem nenhum intervalo falso a mais: WAPE de `event` em 5 minutos de 3,49% para 2,32% (app, janela A), 5,13% para 4,90% (app, B),
  5,79% para 1,04% (app, C), 4,87% para 3,75% (site, B), 3,69% para 1,93% (site, D); o whatsapp ficou igual. Sem a condição de ruído, o whatsapp teve de 710 a 2.280 intervalos falsos.
  As quedas moderadas (0,35 a 0,55) continuam a esperar os 3 intervalos.
- Falta: o app ainda cobre só de 28% a 75% em 5 minutos conforme a janela (uma janela de 21 dias tem poucos eventos). Os limiares foram ajustados nos dados simulados, onde as quedas são abruptas.

**Ridge escolhido por validação temporal.** Testei `ridge_grid = (1e-4 … 1)` com 4 janelas de 14 dias. A escolha foi sempre 0,1. O erro contra a taxa verdadeira
ficou igual em dias comuns e piorou um pouco em feriado (site, 1 h, 3,87% para 4,09%; app, 5 min com LightGBM, 1,82% para 2,04%).
As janelas de 14 dias quase não têm feriado, então a validação favorece o encolhimento dos efeitos raros. A opção continua disponível e fica desligada por padrão.
Para usá-la com proveito, as dobras precisam incluir dias sazonais.

**Nível do dia por GLM Poisson/NegBin (testado, descartado).** O ruído de contagem num total diário (menos de 1,5%, mesmo no whatsapp) é bem menor que a variação de nível entre dias (cerca de 3%),
então os pesos de um NegBin ficam quase iguais entre os dias e o ajuste se aproxima do mínimo quadrado em log que já existe. Medi a única diferença que sobra, estimar o nível do dia pela
razão das somas em vez da mediana, em 4 janelas (60 combinações de canal e cenário): o erro do perfil do dia seguinte mudou no máximo ±0,4 ponto percentual, sem tendência. Nada foi mantido no código.

**Centro da projeção (P50 do LightGBM).** Erro contra a taxa verdadeira (WAPE, %), projeção base e com o P50, referência de 21 dias:

| canal | cenário | 5 min base | 5 min P50 | 1 h base | 1 h P50 |
|---|---|---|---|---|---|
| app | normal | 1,60 | 1,44 | 2,84 | 2,72 |
| app | holiday | 1,82 | 1,83 | 4,22 | 3,41 |
| site | normal | 2,13 | 1,91 | 2,74 | 2,78 |
| site | holiday | 3,26 | 2,81 | 3,87 | 3,89 |
| whatsapp | normal | 3,81 | 3,39 | 3,86 | 3,85 |
| whatsapp | holiday | 5,04 | 3,81 | 5,02 | 4,51 |

O ganho vai de 0 a 25% em 5 minutos e some perto de 1 hora, porque o efeito do alfa e o do P50 se misturam. O erro mínimo teórico é o piso de ruído
(≈0,8/√λ) e os valores do app já estão perto dele. No backtest (2 seeds × 2 períodos, 5 minutos) o P50 teve erro médio menor que o centro base nas 12 combinações de canal e cenário
fora de `event`, mas em vários casos a diferença é menor que o desvio entre execuções. A diferença no feriado vem de poucos dias.

Para dias à frente a faixa vem dos erros dos últimos 28 dias e cobriu de 77% a 80%, conforme o canal, nos últimos 14 dias do
histórico. O painel mostra os 30 dias seguintes com feriados e fins de semana prolongados. No app, o simulado ficou 3,4% abaixo do projetado em
um feriado de segunda, 5,8% abaixo em um feriado de sexta e de 3% a 4% acima nos dois dias de um fim de semana prolongado.
Cada caso é um único dia e não constitui uma medida estatística.

## 5. Limpeza do histórico

Contra o gabarito, só nos dias de treino (anteriores ao período de teste):

| canal | intervalos sujos achados | marcações corretas | alarme falso em dia sazonal |
|---|---|---|---|
| app | 86,6% | 91,7% | 0,0% |
| site | 89,8% | 91,3% | 0,0% |
| whatsapp | 85,5% | 86,4% | 0,0% |

Nenhum dia sazonal foi marcado como sujo. As marcações sem evento real somam de 0,015% a 0,023% do histórico. Os intervalos que escapam são,
quase todos, a cauda final de retenções longas: em um incidente de 3 horas nos três canais e em um de 4 horas no whatsapp, a limpeza achou
de 74% a 79% do que o gabarito marca. A cauda já está a menos de 10% do normal, dentro da variação do dia.

A retenção depois de uma queda não tem duração presumida. A limpeza continua marcando enquanto a razão entre o observado e o esperado fica acima de
1,10 e para quando 3 intervalos seguidos voltam abaixo disso. Esse limiar foi escolhido nos dados simulados. Numa medida anterior, com 1,25, a limpeza
deixava a cauda da retenção no treino e achava só 78% a 84% dos intervalos sujos, contra 93% a 99% com 1,10.

## 6. Efeito de cada dia sazonal estimado pelo modelo

A regressão do nível estima o efeito de cada dia sazonal pelo histórico. A tabela compara o efeito estimado (app, site e whatsapp) com o que o
simulador aplicou, e mostra quantos dias de cada tipo existem no histórico de 400 dias. Mesmo os vistos uma única vez (Black Friday e Cyber Monday)
ficam a menos de 10% do valor simulado, e os vistos várias vezes ficam a 6% ou menos. O teste
`test_level_model_learns_the_effect_of_each_seasonal_day_that_shows_up_in_the_history` exige menos de 8% de diferença nos efeitos vistos 5 vezes ou mais.

| dia sazonal | dias no histórico | efeito simulado | app | site | whatsapp |
|---|---|---|---|---|---|
| feriado | 14 | 0.70 | 0.70 | 0.71 | 0.69 |
| véspera de feriado | 8 | 1.15 | 1.15 | 1.14 | 1.16 |
| dia seguinte ao feriado | 7 | 0.92 | 0.92 | 0.94 | 0.92 |
| ponte | 5 | 0.82 | 0.82 | 0.82 | 0.83 |
| fim de semana prolongado | 20 | 0.88 | 0.88 | 0.89 | 0.89 |
| Carnaval | 5 | 0.75 | 0.74 | 0.74 | 0.71 |
| Black Friday | 1 | 1.70 | 1.64 | 1.75 | 1.62 |
| Cyber Monday | 1 | 1.25 | 1.28 | 1.19 | 1.14 |
| data comercial | 4 | 1.10 | 1.09 | 1.11 | 1.14 |
| virada do ano | 7 | 0.85 | 0.85 | 0.86 | 0.85 |
| fim de ano (1 a 23/12) | 23 | 1.10 | 1.09 | 1.10 | 1.10 |
| férias escolares | 60 | 0.93 | 0.93 | 0.93 | 0.94 |
| fim de mês | 26 | 1.08 | 1.08 | 1.07 | 1.08 |
| meio de mês (dias 15 a 20) | 82 | 1.05 | 1.05 | 1.05 | 1.05 |

Com dados reais o efeito verdadeiro é desconhecido, então essa comparação só vale na simulação. O que vale para dados reais é o limite: um dia
sazonal que não aparece no histórico não tem efeito aprendido, e o modelo o trata como dia comum.

## 7. Evolução do erro durante a revisão

A medida é o erro contra a taxa verdadeira, na média de canais e horizontes (5 minutos, 1 hora e perfil do dia), nos intervalos sem evento.
O ajuste foi feito na janela de validação e confirmado na janela de teste. Estas quatro linhas foram medidas com o simulador
anterior, sem a oscilação lenta do nível, e por isso não se comparam com os valores das seções 1 a 3.

| etapa | validação | teste |
|---|---|---|
| 0. mediana de 5 dias parecidos | 5,32 | 5,55 |
| 1. média, correção de tendência e suavização adaptativa | 4,29 | 5,40 |
| 2. distância por dia do mês e margem de similaridade | 3,34 | 3,97 |
| 3. nível por regressão vezes forma dos dias parecidos | 2,25 | 2,31 |

O que cada etapa corrigiu:

1. **Média no lugar da mediana.** Com o histórico limpo, a média aproveita melhor os dados.
2. **Distância por dia do mês e margem.** O início de mês tem níveis diferentes (×1,4 nos dias 5 a 7 e ×1,2 nos dias 8 a 10) e o perfil
   misturava os dois. A margem impede que um feriado seja comparado com dias comuns.
3. **Nível por regressão.** Um feriado que também caía no início do mês (×0,7 e ×1,4 ao mesmo tempo) virava o "dia mais parecido" de
   outro feriado, que ficou 41% acima do real. A regressão em escala log soma os efeitos e combina o que nunca apareceu junto.
   O erro do perfil para o dia seguinte caiu de 5,5% para 2,9%.
4. **Janela do alfa medida em transações.** Com a oscilação do nível no simulador, o app preferia janelas curtas e o whatsapp, longas.
   Medir a janela em transações (cerca de 3.000) serve aos dois. Em 5 minutos, contra a taxa verdadeira, o app ficou em 1,55% e o whatsapp em 3,48%.

O LightGBM tinha erro de 25% no feriado antes da etapa 3. A causa estava no perfil, que ele apenas reproduzia.

## 8. Validação com o simulador difícil (`simulate_hard.py`)

Sem dados reais, o simulador de referência favorece o modelo: sazonalidade multiplicativa, mesma forma de curva em todo tipo de dia, tendência linear e eventos todos abruptos. O simulador difícil
quebra essas premissas: sazonalidade aditiva com intensidade por canal, formas próprias em fim de semana e feriado, ciclo anual de ±6%, degrau de +12% e mudança de inclinação, ruído de nível AR(1),
12 dias anômalos sem explicação no calendário, deriva dentro do dia, contagens superdispersas e eventos sorteados por seed (queda, longo, parcial, pico com rampa, micro). Foi escrito por mim,
então os defeitos que ele revela são os que consegui imaginar; ele não substitui dados reais.

Os problemas achados e corrigidos:

- **Tendência.** No dia seguinte, o erro do perfil no app foi de 9,8%, pior que MSTL (6,0%) e que o sazonal ingênuo (8,0%), com viés de +6%: a reta extrapolava mal sob degrau, ciclo anual e troca de inclinação. As dobras
  ancoradas no fim do histórico (`level_knots=(90, 30)`, penalidade 1000x) levaram o erro a 4,7% (app), 6,1% (site) e 5,6% (whatsapp) (medido antes do realismo extra descrito abaixo). No simulador de referência o resultado ficou igual ou melhor (app 4,08% para 3,64%; feriado 9,45% para 8,38%).
- **Falsa limpeza.** Dias com curva própria (ponte, Carnaval, fim de ano) afundavam a razão por horas e eram marcados como sujos, então o efeito nunca era aprendido. A exigência de início abrupto (`edge_step=1,4`) elevou a precisão da limpeza de 64% para 94% (app), 50% para 94% (site) e 47% para 76% (whatsapp), e a falsa marcação em dia sazonal caiu de 0,5 a 1% para menos de 0,16%. A cobertura da limpeza caiu pouco (app 72%, site 73%, whatsapp 56%); no simulador de referência nada mudou.

O simulador difícil também recebeu elementos de realismo que o calendário do modelo desconhece: choque de nível comum aos três canais (desvio de 2%), efeito de quinzena (dias 19 a 21, +8%),
4 a 6 campanhas sem aviso de 2 a 5 dias (+10% a +35%, em um canal ou em todos) e degradações lentas (15% a 40%, rampa de 30 a 60 minutos). Os números abaixo já incluem tudo isso.

Resultado em 3 seeds x 3 janelas de 21 dias (9 execuções, `uv run cli.py backtest --hard --seeds 1,2,3`), WAPE contra a taxa verdadeira, média ±desvio:

| | app 5 min | site 5 min | whatsapp 5 min | app 1 h | site 1 h | whatsapp 1 h |
|---|---|---|---|---|---|---|
| `normal`, difícil | 2,50 ±0,08 | 2,93 ±0,07 | 4,14 ±0,17 | 4,24 ±0,62 | 4,64 ±0,50 | 4,77 ±0,36 |
| `normal`, referência | 1,56 ±0,06 | 2,25 ±0,05 | 3,76 ±0,11 | 2,69 ±0,14 | 2,96 ±0,12 | 3,86 ±0,15 |
| `event`, difícil | 5,72 ±2,31 | 5,91 ±2,26 | 6,00 ±1,93 | 4,79 ±1,62 | 5,34 ±1,51 | 4,74 ±1,81 |

- Os erros em dia comum ficam de 0,4 a 0,9 ponto percentual (5 min) acima dos da referência, e o erro em `event` é maior e mais instável, esperado com degradações lentas e micro no sorteio. Sem a via lenta (seção 4.3 de METHOD.md) o `event` a 5 min era 10,42 ±7,19 (app), 10,59 ±6,33 (site) e 6,30 ±2,03 (whatsapp), e o desvio caía junto com o erro quando ela entrou.
- No dia seguinte (janela de 21 dias, seed 42), o perfil erra 5,6% (app), 5,3% (site) e 7,9% (whatsapp) em dia comum, contra 7,4% a 16,0% do melhor baseline StatsForecast. Feriado do site chega a 17%: é um dia só, e uma campanha ou choque comum pode cair nele.
- A faixa de 80% mantém a cobertura de 0,79 a 0,83 em dia comum, início de mês e especial, em todos os horizontes. No feriado a cobertura cai a 0,74 a 0,78 em 8 h (app, site, whatsapp).
- **Em `event`, a via lenta resolveu o que o alfa absorvia.** Sem ela, a cobertura a 5 min era de 0,74 (app), 0,70 (site) e 0,97 (whatsapp), porque a degradação em rampa deixava o estado em NORMAL e a faixa estreita. Com ela (janela de 21 dias, seed 42, dias de evento), o site vai a 0,91 e o app e o whatsapp ficam em 1,00 e 0,98. O custo em dia comum é nulo (tabela acima), e no simulador de referência o resultado é idêntico, sem nenhum falso alarme a mais.
  A confirmação leva cerca de 1 h, então o ganho vem do restante da degradação, que dura de 2 h a 5 h. Em casos isolados a via lenta ainda erra: numa janela do whatsapp o `event` piorou de 9,2 para 11,8 (5 min), provavelmente porque a saída pela faixa de retorno encerra o evento antes do fim e o alfa volta a absorver o resto (hipótese, não medida). Nos 9 sorteios a média é a da tabela.
- A limpeza do histórico acha 45% a 54% dos intervalos afetados (app, site) e 45% (whatsapp), por causa dessas mesmas degradações, com 93% (app, site) e 72% (whatsapp) das marcações corretas.
- **Marcar degradações lentas na limpeza não ajudou, e foi descartado.** Testei uma regra que marcava patamares de 1 h ou mais fora da faixa 0,88 a 1,14 (com ruído e degrau de entrada). O que se achou subiu
  de 50% para 85% dos intervalos afetados (app), mas a precisão caiu de 93% para 28%: a oscilação intradiária do simulador (desvio de 5%, memória de 3 h) cria patamares iguais a uma degradação de 15%.
  Com limiares estritos (0,80 a 1,20, 24 intervalos, 10 desvios) a precisão sobe a 40% a 55%, e o erro ficou igual ao sem a regra (diferenças de até 0,05 ponto percentual, a 5 min e a 1 h, em dia comum e `event`).
  Com limiares frouxos o erro piorou (`event` do app a 5 min, de 2,42 para 3,14). A mediana dos dias parecidos já resiste a degradações lentas ocasionais, e a via lenta ao vivo cobre o que importa.
- O LightGBM P50 vence o centro do alfa a 1 h em dia comum no app (3,77 contra 4,26) e no site (4,11 contra 4,75), e empata no whatsapp (4,69 contra 4,79). O painel já usa o P50 como centro; a coluna `model` da avaliação é o centro do alfa.

## 9. Limites

- Os dados são simulados. No simulador de referência a forma da curva é igual em todos os tipos de dia, o que favorece a estimativa da forma; o simulador difícil (seção 8) corrige isso, mas foi escrito pelo mesmo autor do modelo. Os limiares (`edge_step`, dobras) foram ajustados nos dois. Só dados reais dizem se o modelo é bom o bastante.
- Os limiares (0,55, 1,45, 1,10, 3 intervalos e os da via lenta: 0,88, 1,14, 12 intervalos, 6 desvios) e os pesos de cada dia sazonal na escolha de dias parecidos foram escolhidos para os dados simulados.
- Os efeitos de cada dia sazonal só são aprendidos se o dia aparece no histórico. Com menos de um ano, Carnaval, Black Friday e fim de ano ficam sem exemplo.
- Confirmar um evento leva 15 minutos, e nesse tempo a previsão ainda sofre o efeito dele. Só quedas e picos extremos têm via rápida (1 intervalo); degradações em rampa levam cerca de 1 h pela via lenta, e rampas bem mais lentas que isso (deriva de horas) seguem absorvidas pelo alfa.
- O LightGBM é medido de 5 minutos a 8 horas (7 horizontes). Depois de 8 horas a faixa é extrapolada.
- A calibração conformal separa os últimos 28 dias do treino e só vale para o estado NORMAL. Em `event` a faixa do app ainda cobre menos de 80% (seção 4).
- Os eventos simulados vão de 20 minutos a 4 horas. Um incidente que ocupe mais da metade do trecho diurno (9 horas) distorce o nível do dia na
  limpeza, e um dia com mais de 30% dos intervalos sujos sai da base de dias parecidos.
- Uma mudança permanente de patamar (campanha, nova integração) é indistinguível de um incidente longo. Na limpeza ela deixa de ser
  sujeira quando os dias parecidos passam a mostrar o novo patamar, o que leva alguns dias. No ajuste ao vivo cada dia recomeça em NORMAL.
- `tune` e as colunas "contra a taxa verdadeira" só funcionam com dados simulados, que têm gabarito.
