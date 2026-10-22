---
name: sanecast
description: Mapa e regras do projeto sanecast, uma demonstração de como projetar dados futuros a partir de dados históricos (contagem de transações por canal fictício, em intervalos de 5 minutos, com limpeza automática de dado sujo, ritmo ao vivo, faixa P10 a P90, simulador e painel HTML). Use sempre que o pedido envolver este repositório, mesmo sem citar o nome. Exemplos: executar o demo, simular dados, medir erro, mudar limiar de queda ou pico, mexer no perfil do dia, no alfa, no LightGBM, no calendário de feriados, no painel (dashboard.html), no Docker, ou explicar por que a previsão errou num dia.
---

# sanecast

Demonstração de como projetar dados futuros a partir de dados históricos. O exemplo são as transações por minuto de três canais fictícios
(app, site, whatsapp). O modelo descobre sozinho quando o dado está sujo (queda, retenção, pico anormal), treina só com histórico limpo,
projeta o resto do dia e os próximos 30 dias. Os dados vêm de um simulador. Ao mexer no projeto, leia primeiro
`README.md` (como executar) e `docs/METHOD.md` (o método). Este arquivo guarda o que não está óbvio no código.

## Mapa

| Quero | Arquivo |
|---|---|
| ler o CSV e agrupar em intervalos de 5 min (288 por dia) | `sanecast/data.py` |
| achar queda, retenção e pico anormal no histórico | `sanecast/cleaning.py` |
| montar o dia esperado (regressão do nível vezes forma dos dias parecidos) | `sanecast/profile.py`, `day_features.py` |
| ritmo do dia (alfa) e estados NORMAL, OUTAGE, SURGE, RECOVERY | `sanecast/forecasting.py` |
| faixa P10 a P90 (LightGBM) e faixa dos dias futuros | `sanecast/quantile_model.py`, `snapshots.py` |
| orquestrar tudo | `sanecast/pipeline.py` (`Forecaster`) |
| modelos de comparação (StatsForecast `SeasonalNaive` e `MSTL`, semana passada) | `sanecast/baselines.py`, usados só em `evaluate` |
| simular dados e eventos | `sanecast/simulate.py` |
| qualquer número ajustável | `sanecast/config.py` (`Config`) |
| painel e gráfico | `sanecast/dashboard.py`, `templates/dashboard.html`, `report.py` |

## Comandos

```bash
uv sync                                   # cria o .venv a partir do uv.lock (ou: docker compose run --rm forecast demo)
uv run cli.py demo              # simula, avalia, gráfico e painel (~2 min)
uv run cli.py evaluate          # erro e cobertura, últimos 21 dias
uv run cli.py dashboard         # painel com 7 dias passados, hoje e 30 à frente
uv run cli.py tune              # variações de hiperparâmetros
uv run tests/test_pipeline.py   # 16 testes, ~3 min
```

Bibliotecas: LightGBM só na faixa P10 a P90 (`quantile_model.py`). StatsForecast só como comparação na avaliação, nunca na projeção
entregue. Todo o resto (limpeza, regressão do nível, alfa, estados) é pandas e numpy, com código próprio.

Arquivos: `transactions_1min.csv` é a única entrada real. `truth_1min.csv` (gabarito) e `future_1min.csv` (30 dias à frente, só para
comparar no painel) existem só na simulação. O painel não lê CSV ao abrir: o Python grava os dados dentro do HTML. Nada do modelo é salvo em
disco, tudo é refeito a cada execução. A tabela completa está no README, seção "Visão geral".

## Regras que o projeto segue

- Dependências e ambiente pelo uv: `uv add <pacote>` para acrescentar, `uv lock --upgrade` para atualizar, `uv sync` para criar o `.venv`. O `pyproject.toml` e o `uv.lock` são a fonte das versões. Não crie `requirements.txt`.

- O modelo recebe só `ts,channel,count`. Nunca crie coluna de incidente na entrada. O que é sujo o modelo descobre pelos dias parecidos.
- Código em inglês (arquivos, funções, variáveis). Comentários em português, diretos, explicando o que o código faz. A justificativa de uma
  decisão vai para `docs/`, não para comentário.
- Textos de documentação, painel e imagens seguem o estilo do no-ai-slop: frases curtas, sem travessão, sem palavras de efeito.
  Evite tradução literal do inglês: use "intervalo" para `bin`, "painel" para `dashboard`, "modelos de comparação" para `baselines`.
- Os canais são fictícios e genéricos (app, site, whatsapp). Não use nomes de produto ou de negócio real, nem "mvp", em nomes.
- O modelo não presume duração de incidente nem de pico anormal: detecta pelo dado, de 15 minutos a horas. Os eventos simulados vão de 20 minutos a 4 horas,
  e a retenção termina quando o volume volta a menos de 110% do esperado (`retention_exit_ratio`).

## Cuidados antes de mudar algo

1. **Os números da documentação valem para os dados de referência** (`data/reference/`, gerados com `REFERENCE_DAY` em `simulate.py`). Os testes usam `data/reference/`, que não muda
   com o relógio. Se mudar o modelo, execute `evaluate` sobre esses dados e atualize README e `docs/RESULTS.md`. Comparar um dia com outro
   sem fixar a data mistura efeito do calendário com efeito da mudança.
2. **O histórico simulado não pode mudar sem querer.** O futuro (`future_1min.csv`) usa um sorteio próprio (`seed + 1`) justamente para isso.
   Depois de mexer em `simulate.py`, confira que a simulação com `REFERENCE_DAY` gera o mesmo `transactions_1min.csv` de antes (md5).
3. **O modelo nunca lê `truth_1min.csv` nem `future_1min.csv`.** O primeiro mede o acerto, o segundo só alimenta a linha "simulado" do painel.
4. **Hiperparâmetro novo:** acrescente em `Config` com comentário, uma linha em `GRID` (`tuning.py`) e escolha pela validação, confirmando no teste.
   Não ajuste olhando só o período de teste.
5. **O painel depende do formato do JSON** que `dashboard.build_dashboard` grava no lugar de `__DATA__`. Se mudar um, mude o outro e rode
   `test_dashboard_payload_has_history_with_snapshots_and_future_with_projection_only`. Confira o resultado abrindo o HTML no Chrome
   (`google-chrome --headless --screenshot=... "file://.../out/dashboard.html?now=AAAA-MM-DDTHH:MM"`). Também aceita `?view=multi&range=all`.
6. **Calendário:** os feriados vêm da biblioteca `holidays` (`COUNTRY` e `SUBDIVISION` em `day_features.py`, qualquer ano). O resto dos dias sazonais (ponte, fim de semana prolongado, Black Friday, datas comerciais, fim de ano, férias) segue `FLAGS`. Para outro país,
   troque `COUNTRY` e revise as regras. Sem isso o feriado vira um "dia estranho".

## Ao explicar um erro de previsão

Compare sempre com o piso de ruído (`noise_floor`): transações são contagens, e uma previsão perfeita ainda erra cerca de `0,8 / √λ` por
intervalo (app 2,2%, site 5,0%, whatsapp 14,3%). Um erro de 2,7% no app é quase o piso. Um erro de 12% no whatsapp também. Depois olhe
o estado do momento (evento detectado há menos de 15 minutos ainda não foi confirmado) e a cobertura da faixa, que deve ficar perto de 80%.

## Limites conhecidos (não os esconda)

- Os dados são simulados e a forma da curva é a mesma em todos os tipos de dia. Em dados reais, feriados têm forma própria.
- Confirmar um evento leva 3 intervalos (15 minutos).
- A faixa depois de 1 hora é extrapolada. Nos dias futuros ela vem dos erros dos últimos 28 dias (cobertura medida de 77% a 79%).
- A limpeza marca de 0,03% a 0,07% do histórico sem evento real (whatsapp: 38 intervalos em 14 trechos). Incidente com mais de 9 horas no dia distorce o nível do dia.
- Feriados distantes no futuro podem errar mais (um feriado a 30 dias no app: 7% abaixo do projetado, um único dia).
