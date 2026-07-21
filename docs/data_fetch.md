# Fetch de dados por traffic source

O módulo `src.data_fetch` usa o MySQL apenas para resolver o id e o nome de cada
traffic source. Em seguida, busca no ClickHouse todas as requisições daquela
fonte, independentemente da campanha. O `campaign_id` continua presente em cada
registro exportado.

Copie `.env.example` para `.env` e preencha os valores localmente.

## Exemplos

Listar as fontes ativas sem consultar o ClickHouse:

```powershell
python -m src.data_fetch --source-names outbrain taboola --list-only
```

Exportar duas fontes em um intervalo UTC:

```powershell
python -m src.data_fetch `
  --source-names outbrain taboola `
  --start "2026-07-01 00:00:00" `
  --end "2026-07-08 00:00:00" `
  --limit-per-group 10000
```

Também é possível filtrar por ids:

```powershell
python -m src.data_fetch --source-ids 3 7
```

## Saída

```text
data/raw/<traffic-source>/
  <traffic-source>-bot.json
  <traffic-source>-unsafe.json
  <traffic-source>-safe.json
```

- `bot` contém a decisão ClickHouse `bot`.
- `unsafe` contém `offer`, chamada de `unsafeClicks` nos relatórios do core.
- `safe` permanece separado, sem presumir que seja a classe humana/unsafe.
- Todas as campanhas da fonte são agregadas nos mesmos arquivos.
- `campaign_id`, decisão original, data, headers, parâmetros e body são
  preservados em cada registro.
- Linhas com `duplicated = true` são excluídas por padrão.
- O intervalo é semiaberto: início inclusivo e fim exclusivo.
- `manifest.json` registra os arquivos e suas contagens.
