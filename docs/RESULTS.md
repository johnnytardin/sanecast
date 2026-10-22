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
| app | normal | 1.62 | 1.42 | 1.64 | 16.75 |
| app | month_start | 1.57 | 1.43 | 2.08 | 18.97 |
| app | holiday | 1.82 | 1.78 | 1.82 | 103.54 |
| app | special | 1.76 | 1.58 | 1.76 | 33.16 |
| app | event | 5.64 | 5.12 | 39.88 | 10.26 |
| site | normal | 2.14 | 1.90 | 2.24 | 16.86 |
| site | month_start | 2.18 | 1.99 | 2.20 | 23.90 |
| site | holiday | 3.26 | 2.62 | 3.39 | 100.60 |
| site | special | 2.59 | 2.32 | 2.60 | 34.26 |
| site | event | 7.49 | 7.50 | 37.38 | 19.44 |
| whatsapp | normal | 3.81 | 3.34 | 4.14 | 20.78 |
| whatsapp | month_start | 3.82 | 3.22 | 4.06 | 24.24 |
| whatsapp | holiday | 5.04 | 3.81 | 5.03 | 101.87 |
| whatsapp | special | 3.84 | 3.38 | 3.84 | 48.27 |
| whatsapp | event | 6.31 | 6.43 | 21.72 | 14.20 |

1 hora à frente:

| canal | cenário | model | model_lgbm_p50 | no_cleaning | last_week |
|---|---|---|---|---|---|
| app | normal | 2.86 | 2.74 | 3.06 | 16.75 |
| app | month_start | 3.07 | 2.76 | 3.80 | 18.97 |
| app | holiday | 4.22 | 3.45 | 4.22 | 103.54 |
| app | special | 3.14 | 2.83 | 3.14 | 33.16 |
| app | event | 3.33 | 3.27 | 20.42 | 10.26 |
| site | normal | 2.75 | 2.78 | 2.91 | 16.86 |
| site | month_start | 2.97 | 2.88 | 2.99 | 23.90 |
| site | holiday | 3.87 | 3.90 | 4.28 | 100.60 |
| site | special | 3.11 | 3.15 | 3.11 | 34.26 |
| site | event | 7.13 | 7.22 | 24.77 | 19.44 |
| whatsapp | normal | 3.86 | 3.81 | 4.00 | 20.78 |
| whatsapp | month_start | 4.10 | 3.70 | 4.36 | 24.24 |
| whatsapp | holiday | 5.02 | 4.56 | 5.01 | 101.87 |
| whatsapp | special | 3.84 | 3.81 | 3.84 | 48.27 |
| whatsapp | event | 5.00 | 5.25 | 13.83 | 14.20 |

A linha `event` mede os intervalos de queda, retenção e pico contra o volume que teria acontecido sem o evento. Os eventos duram de 20 minutos
a 4 horas. O tratamento decide o resultado nos três canais: no app o erro é 5,64% com tratamento e 39,88% sem, no site 7,49% contra 37,38%
e no whatsapp 6,31% contra 21,72%.

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

Fração do alvo dentro de P10 a P90, para um alvo de 80%. Em `event` o alvo é a taxa verdadeira, não o observado.

| canal | cenário | 5 min | 1 h |
|---|---|---|---|
| app | normal | 81 | 79 |
| app | month_start | 83 | 78 |
| app | holiday | 77 | 79 |
| app | special | 77 | 76 |
| app | event | 50 | 78 |
| site | normal | 81 | 81 |
| site | month_start | 81 | 80 |
| site | holiday | 80 | 78 |
| site | special | 84 | 82 |
| site | event | 76 | 76 |
| whatsapp | normal | 79 | 79 |
| whatsapp | month_start | 81 | 81 |
| whatsapp | holiday | 84 | 84 |
| whatsapp | special | 79 | 80 |
| whatsapp | event | 100 | 100 |

Depois de 1 hora a largura é extrapolada (`band_growth_exponent = 0,2`). Essa cobertura foi medida em vários horários numa versão
anterior do modelo e ficou entre 74% e 91% conforme o canal. Não foi medida de novo depois das últimas mudanças.

Em `event` o app cobre só 50% em 5 minutos, abaixo do alvo de 80%: a faixa fica estreita demais nesses intervalos. A causa não foi investigada.

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

## 8. Limites

- Os dados são simulados. A forma da curva é igual em todos os tipos de dia, o que favorece a estimativa da forma. Em dados reais, feriados costumam ter forma própria.
- Os limiares (0,55, 1,45, 1,10 e 3 intervalos) e os pesos de cada dia sazonal na escolha de dias parecidos foram escolhidos para os dados simulados.
- Os efeitos de cada dia sazonal só são aprendidos se o dia aparece no histórico. Com menos de um ano, Carnaval, Black Friday e fim de ano ficam sem exemplo.
- Confirmar um evento leva 15 minutos, e nesse tempo a previsão ainda sofre o efeito dele.
- O LightGBM é medido em 5 minutos e 1 hora. Depois de 1 hora a faixa é extrapolada.
- Os eventos simulados vão de 20 minutos a 4 horas. Um incidente que ocupe mais da metade do trecho diurno (9 horas) distorce o nível do dia na
  limpeza, e um dia com mais de 30% dos intervalos sujos sai da base de dias parecidos.
- Uma mudança permanente de patamar (campanha, nova integração) é indistinguível de um incidente longo. Na limpeza ela deixa de ser
  sujeira quando os dias parecidos passam a mostrar o novo patamar, o que leva alguns dias. No ajuste ao vivo cada dia recomeça em NORMAL.
- `tune` e as colunas "contra a taxa verdadeira" só funcionam com dados simulados, que têm gabarito.
