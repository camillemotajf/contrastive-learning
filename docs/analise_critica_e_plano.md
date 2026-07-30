# Contrastive learning neste projeto: o que está implementado, o que funciona, e como atacar falsos positivos/negativos

Documento de análise crítica. Escrito a partir da leitura de `src/`, `configs/`,
`results/` e dos dumps em `data/raw/`. A Parte 2 e a Parte 3 contêm conclusões
que **contradizem a hipótese central do trabalho** — elas estão aqui porque é
melhor descobrir isso agora do que num parecer de revisor.

---

## Parte 1 — Mapa do algoritmo: o que exatamente está implementado

### 1.0 A cadeia completa

```
JSON bruto (headers, request)
   ↓ split_raw()                      split estratificado no TEXTO, antes de qualquer fit
   ↓ Preprocessor.fit(train only)     TF-IDF char_wb (2,4), max_features=5000
   ↓                                  → TruncatedSVD(64) → hstack(4 features manuais) → z-score
   ↓ = X ∈ R^68
   ↓ TrafficEncoder                   68 → 128 → 64 → embed_dim, L2-norm (hiperesfera)
   ↓ h ∈ R^64
   ├→ instance_head → z ∈ R^16 (L2-norm)     → InstanceLoss
   └→ cluster_head  → p ∈ Δ^k   (softmax)    → ClusterLoss
```

Dois pontos técnicos que estão **corretos** e valem ser defendidos no artigo:

1. **Ausência de vazamento.** `split_raw` divide o texto cru; `Preprocessor.fit`
   só vê o treino; toda view aumentada passa por `transform`, nunca por um novo
   `fit`. Isso é raro de ver feito certo e é um ponto forte.
2. **L2-norm na saída do encoder.** É o que faz o produto interno virar cosseno
   e torna as losses NT-Xent bem-comportadas (temperatura com significado,
   gradientes limitados).

Um ponto técnico que está **errado por omissão**: o `TfidfVectorizer` recebe
apenas `headers` (`Preprocessor._text` retorna `list(headers)` e ignora
`requests`). O `request` inteiro entra no modelo como **dois números**:
`len(r)` e `r.count(":")`. Toda a informação de `utm_*`, `placement`, `adset`,
templates `{{...}}` — que é onde vive parte grande do sinal de fraude — é
descartada. Volto nisso na Parte 4 (P2).

### 1.1 Os quatro algoritmos contrastivos, e para que cada um serve

Você tem **quatro** coisas diferentes no repositório, frequentemente confundidas
como "o algoritmo de contrastive learning". Elas têm papéis distintos:

| # | Algoritmo | Arquivo | Usa rótulo? | Papel no artigo |
|---|---|---|---|---|
| A | SimCLR / NT-Xent puro (SSL) | `src/models/ssl.py` | Não | Referência de quanto se aprende **sem rótulo nenhum** + curva de label-efficiency |
| B | Triplet (supervisionado) | `src/models/triplet.py` | Sim | Baseline contrastivo **supervisionado** — o teto do que o contraste dá com rótulo perfeito |
| C | Contrastive Clustering (CC) | `src/models/contrastive_clustering.py` | Não | O método central: agrupa e produz embedding + distribuição de cluster |
| D | Augmentações HTTP | `src/data/http_augmentations.py` | Não | Define a **invariância** que A e C aprendem — é aqui que está a contribuição de domínio |

#### A. SSL / NT-Xent (`ssl_pretrain`)

Duas views por *dropout + ruído gaussiano no vetor SVD* (`_augment`: máscara
15%, σ=0.05). `InstanceLoss` puxa as duas views da mesma amostra e empurra todas
as outras. Depois: backbone congelado + regressão logística (protocolo de
avaliação linear).

**Para que serve:** medir se existe estrutura aprendível sem rótulo, e dar a
curva de eficiência de rótulo. **Não** é um detector de ruído.

**Nota honesta:** a augmentação aqui é no espaço SVD, o que não tem significado
em HTTP. Você mesmo documentou isso. É por isso que o CC usa D em vez disso — e
é a razão do `ssl_probe` ser o pior classificador de todos (F1 0.761).

#### B. Triplet

Âncora, positivo da mesma classe, negativo de outra classe, `TripletMarginLoss`
(margem 1.0, L2). Usa rótulos de treino.

**Para que serve:** é o **limite superior** do que contraste consegue quando o
rótulo é tratado como verdade. Se o triplet supervisionado não bate o
RandomForest, contraste não-supervisionado dificilmente vai. (E não bate:
0.872 vs 0.886.)

**Nota importante para o seu objetivo:** o triplet é *exatamente* o algoritmo
mais sensível a rótulo errado — um positivo mal rotulado força o encoder a
aproximar um bot de um humano. Isso é uma faca de dois lados que você pode
explorar: a *perda por amostra* do triplet ao longdo treino é um sinal de ruído
(ver AUM na Parte 4, P3).

#### C. Contrastive Clustering (Li et al., 2021) — o núcleo

A ideia original, em uma frase: **a linha da matriz de probabilidades é a
representação da amostra; a coluna é a representação do cluster.** Então você
aplica contraste duas vezes na mesma matriz, em dois eixos diferentes.

Dado um batch de tamanho `B` e `k` clusters, o forward produz para cada view:
- `Z ∈ R^{B×16}` (instance head, L2-norm por linha)
- `P ∈ R^{B×k}` (cluster head, softmax por linha)

**`InstanceLoss` — contraste nas LINHAS (instance-level).**

```python
z = cat(z_i, z_j)                  # (2B, 16)
sim = z @ z.T / τ_i                # τ_i = 0.5
sim.fill_diagonal_(-1e9)           # remove auto-similaridade
positives = [B..2B-1, 0..B-1]      # o positivo de i é i+B, e vice-versa
loss = CrossEntropy(sim, positives)
```

É NT-Xent/InfoNCE: para cada amostra, classificação (2B-1)-classes onde a
classe correta é a outra view dela mesma. **O que isso serve:** produz um
embedding onde as duas serializações HTTP da mesma request caem no mesmo lugar,
e requests diferentes se separam. É o que dá o `embeddings_train` usado depois
por KNN e por distância a centroide.

**`ClusterLoss` — contraste nas COLUNAS (cluster-level) + entropia.**

```python
pi = normalize(P_i.T, dim=1)       # (k, B) — cada linha é um "vetor de cluster"
p  = cat(pi, pj)                   # (2k, B)
sim = p @ p.T / τ_c                # τ_c = 1.0
sim.fill_diagonal_(-1e9)
positives = [k..2k-1, 0..k-1]      # cluster c da view 1 ↔ cluster c da view 2
contrast = CrossEntropy(sim, positives)

# regularizador de entropia
ne = Σ p̄_i log p̄_i + Σ p̄_j log p̄_j        # p̄ = média das probs no batch
loss = contrast + λ_H · ne                   # ne = −(H(P̄_i)+H(P̄_j))
```

Duas coisas acontecendo:

- **Contraste de colunas:** força o cluster `c` a significar a mesma coisa nas
  duas views (consistência) e a ser diferente dos outros clusters
  (não-redundância). É isso que faz `argmax(p)` ser um agrupamento e não uma
  softmax arbitrária.
- **Entropia marginal:** `ne` é o negativo da entropia da distribuição marginal
  de clusters. Minimizar `+λ_H·ne` = **maximizar** `H(P̄)` = empurrar para uso
  balanceado dos clusters. Sem esse termo o modelo colapsa em "tudo no
  cluster 0", que minimiza trivialmente o contraste de colunas. `λ_entropy=1.0`
  no seu config. Diagnóstico do colapso: `cluster_distribution.entropy` em
  `results/.../metrics.json` (seu k=4 dá 1.279, próximo de ln4=1.386 → **não
  houve colapso**; o problema é outro, ver Parte 2).

Loss total: `λ_ins · InstanceLoss + λ_clu · ClusterLoss`, tudo 1.0.

#### D. Augmentações HTTP — a contribuição de domínio

Aqui está a parte original e genuinamente boa do trabalho. Em vez de perturbar o
vetor SVD (sem significado), você gera **duas serializações plausíveis da mesma
request**:

| Operação | Parâmetro | Invariância que você está ensinando |
|---|---|---|
| Reordenar chaves | `reorder_keys` | Ordem de header não muda a request (verdade em HTTP/2+) |
| Dropar headers instáveis | `drop_unstable_prob` | `date`, `cf-ray`, `x-request-id`, `cookie` são ruído por-request |
| Dropar headers opcionais | `drop_optional_prob` | Um cliente real pode ou não mandar `dnt`, `pragma`, `te` |
| Mascarar valores | `mask_value_prob` | O valor específico importa menos que a presença da chave |
| Variar caixa da chave | `change_case_prob` | Chaves HTTP são case-insensitive |
| Jitter de whitespace | `whitespace_jitter` | Serialização compacta vs espaçada |
| Mascarar versão do UA | `ua_mask_prob` | `Chrome/120.0` ≈ `Chrome/XXX.X` — a marca importa, a versão menos |

`SEMANTIC_HEADERS = {user-agent, accept, from}` nunca são dropados sob
`preserve_ua=True`. Os configs `*_masking_aug` **destroem** deliberadamente o
sinal (UA → `<MASK>`, template → `<MASK>`) e existem para o teste de atalho:
*o CC continua funcionando quando você tira a pista mais óbvia?* Isso é a
ablação certa e está bem desenhada.

**Crítica:** três augmentações supõem invariâncias que **não valem para detecção
de bot**. Dropar `sec-fetch-*` com prob. 0.15 e mascarar valores com prob. 0.15
apaga exatamente a coerência `Sec-Fetch-Dest/Mode/Site` que é um dos sinais mais
fortes de tráfego sintético. Você está pedindo ao encoder para ser *invariante ao
sinal de classe*. Sob `medium_http_aug` isso acontece em ~15% dos casos por view
— o suficiente para degradar. Isso é uma hipótese concreta e testável para o
fracasso do CC (ver P2).

### 1.2 Os oito scores de suspeição e o que cada um mede

Convenção do repositório: **maior = mais suspeito**. Nenhum é probabilidade.

| Score | Espaço | O que realmente mede |
|---|---|---|
| `knn_raw` | X (SVD 68d) | fração dos k=20 vizinhos com rótulo diferente |
| `knn_cc` | embedding CC (64d) | idem, no espaço aprendido |
| `confident_learning` | X, via RF out-of-fold | `P(outra classe) − P(rótulo dado)` |
| `cc_cluster_entropy` | `P` | `−Σ p log p` — incerteza do modelo sobre o cluster |
| `cc_view_instability` | `P` sob 5 views | variância média de `p` entre views aumentadas |
| `cc_cluster_label_mismatch` | `argmax(P)` + y | `1 − fração do cluster que compartilha o rótulo da amostra` |
| `centroid_own_distance` | embedding CC | distância ao centroide da própria classe |
| `centroid_relative_distance` | embedding CC | `d(própria) − d(oposta)`; **>0 = mais perto da classe errada** |
| `ensemble` | — | média dos min-max de 6 deles |

**Confident Learning** (`src/noise/confident_learning.py`) é o mais sofisticado
e merece o detalhe:

1. `oof_probabilities`: RF(150) em `StratifiedKFold(4)` com
   `cross_val_predict(method="predict_proba")`. Cada amostra é pontuada por um
   modelo que **nunca a viu** — sem auto-confirmação do erro.
2. Limiar por classe: `t[c] = mean(P(c) | y=c)`. Barra calibrada por classe, não
   0.5 chapado.
3. Confident joint: flag quando o modelo **confiantemente** (acima de `t`)
   atribui uma classe **diferente** do rótulo dado.

O comentário no código sobre `n_jobs` (RF paraleliza, CV sequencial; paralelizar
os dois oversubscreve e no `spawn` do Windows trava) está certo e é uma
observação prática valiosa — mantenha.

### 1.3 Ruído sintético: por que existe

`src/noise/synthetic_noise.py` viola rótulos conhecidos e devolve
`(y_noisy, noise_mask)`. **Isso existe porque no dado real você não tem ground
truth**, então não há como medir se um detector funciona. Com o `noise_mask`, a
detecção vira um problema supervisionado: ROC-AUC, AUPRC, precision@k, recall@k.

Modos: `symmetric`, `bot_to_human`, `human_to_bot`, `class_conditional`.

**Esta é a peça que precisa mudar.** Ver Parte 3.

---

## Parte 2 — Leitura honesta dos resultados que você já tem

Fonte: `results/synthetic_noise/summary_aggregate.csv`,
`results/cross_source/*/summary.json`, `results/baseline/metrics.json`,
`results/contrastive_clustering/k4_medium_http_aug_s42/metrics.json`,
`results/real_label_audit/summary_unsafe_audit.json`.

### 2.1 Detecção de ruído sintético — ROC-AUC média

| score | rate 5% | rate 20% | veredicto |
|---|---|---|---|
| `ensemble` | **0.9306** | **0.9121** | melhor AUC global |
| `knn_cc` | 0.9207 | 0.8892 | forte |
| `knn_raw` | 0.9194 | 0.9036 | forte |
| `confident_learning` | 0.9108 | 0.8975 | forte |
| `centroid_relative_distance` | 0.7964 | 0.7901 | moderado |
| `centroid_own_distance` | 0.7388 | 0.6714 | fraco |
| `cc_cluster_label_mismatch` | 0.6134 | 0.5844 | muito fraco |
| `cc_cluster_entropy` | **0.5033** | **0.4939** | **aleatório** |
| `cc_view_instability` | **0.4521** | **0.5004** | **aleatório / pior que aleatório** |

**Conclusão desconfortável nº 1: os três sinais que você derivou do
Contrastive Clustering não funcionam.** `cluster_entropy` = 0.50 é literalmente
uma moeda. `view_instability` = 0.45 em 5% é *pior* que aleatório de forma
consistente (std 0.003). Esses são os dois sinais mais "originais" do trabalho, e
eles não carregam informação sobre ruído de rótulo. Faz sentido em retrospecto:
ambos são calculados **sem olhar o rótulo** — medem dificuldade/ambiguidade da
amostra, não discordância com o rótulo. Um bot óbvio com rótulo trocado tem
entropia de cluster *baixa* (o modelo está seguro do cluster) e o rótulo trocado
não afeta `p` de forma alguma.

**Conclusão desconfortável nº 2: o ensemble está sendo puxado para baixo por
esses dois scores.** Compare no rate 5%:

| | ROC-AUC | AUPRC | precision@100 |
|---|---|---|---|
| `confident_learning` | 0.9108 | **0.7090** | **0.605** |
| `ensemble` | **0.9306** | 0.6453 | 0.530 |

O ensemble ganha em AUC (ordenação global) e **perde em AUPRC e em
precision@100** — e o topo da lista é exatamente o que um analista humano vai
revisar. Colocar dois scores de AUC≈0.5 numa média aritmética de min-max dilui o
sinal justamente onde importa.

**Conclusão desconfortável nº 3: a hipótese "embedding CC melhora a detecção de
ruído" está refutada pelos seus próprios dados.**

| fonte | `knn_raw` AUC | `knn_cc` AUC |
|---|---|---|
| outbrain | 0.9045 | 0.9070 |
| facebook | 0.9935 | 0.9936 |
| taboola | **0.9369** | 0.8991 |
| tiktok | **0.9354** | 0.9015 |

Empate em duas fontes, **perda clara em duas** (−4 e −3.4 pontos). Não há
suporte para a pergunta 4 do seu plano ("KNN-CC > KNN-raw?"). A resposta é não.

### 2.2 O CC como clusterizador e como representação

| fonte | CC ARI | CC probe F1 | RF F1 |
|---|---|---|---|
| outbrain | 0.079 | 0.785 | 0.886 |
| facebook | 0.400 | 0.972 | **0.997** |
| taboola | 0.092 | 0.793 | 0.898 |
| tiktok | 0.007 | 0.809 | 0.977 |

O CC **não recupera** a estrutura humano/bot (ARI 0.007–0.09 em três das quatro
fontes) e o probe sobre o embedding CC é sempre pior que o RF sobre as features
cruas. Não houve colapso de cluster (entropia 1.28 ≈ ln4), então o problema não é
o regularizador — é que a estrutura dominante nos headers **não é** humano/bot.
Provavelmente é host/campanha/dispositivo. Isso é um resultado legítimo e
reportável, mas não é o resultado que o trabalho estava buscando.

O ARI 0.40 do facebook não é uma vitória: **73% dos "bots" do facebook têm
literalmente `facebookexternalhit/1.1` no User-Agent**. O RF acerta F1 0.9967
porque a tarefa é uma regex. Qualquer método parece bom nessa fonte; ela não
deve ser usada para sustentar conclusões sobre dificuldade.

### 2.3 Um artefato concreto que precisa de correção

`results/real_label_audit/facebook/all_noise_scores.csv`: **100% das amostras com
`observed_label=0` têm `heuristic_bot_flag=True`**, todas via `has_template`. A
causa: todo request do facebook contém `"domain": "{{domain}}"` — uma macro do
Meta que não é substituída, e que aparece em tráfego perfeitamente humano.

Consequência: `has_template`, descrito no código como heurística de **alta
precisão**, tem precisão ≈ 0 nessa fonte, e o `heuristic_bot_flag` composto fica
inútil. Como esse flag é usado como "âncora" para interpretar os rankings, toda
a leitura qualitativa do audit do facebook está comprometida.

Correção: calibrar as heurísticas **por fonte** e medir a precisão de cada regra
contra o gold set antes de chamá-la de alta precisão. Regra prática: se um flag
dispara em >20% de uma classe, ele não é uma âncora, é uma feature.

### 2.4 Custo computacional

`results/cross_source/outbrain/summary.json`: `runtime_seconds: 129769` = **36
horas** para uma fonte. O gargalo não é o RF (68 dimensões, segundos). São:

1. `train_cc`: `pre.transform()` (TF-IDF + SVD) é chamado **duas vezes por
   batch, dentro do loop de treino**, em CPU. Para 20 épocas × ~10 batches ×
   2 views isso domina tudo.
2. `view_instability`: 5 × `pre.transform` no corpus inteiro.

Correção barata: pré-computar um **pool de V=8 views aumentadas por amostra**
(uma vez, vetorizado) e amostrar 2 do pool por batch. Perde-se um pouco de
diversidade, ganha-se ~50× de velocidade. Isso é o que viabiliza rodar as
5 taxas × 5 seeds que o config pede (o agregado salvo tem só 5% e 20% — a grade
completa nunca terminou).

---

## Parte 3 — O problema conceitual central

Este é o item mais importante do documento.

### 3.1 Seu rótulo não é ground truth — é a saída de um filtro em produção

Nos dumps brutos existe o campo `decision`:

```
outbrain-unsafe.json : decision = "offer"  (10.000)   → seu label 0
outbrain-bot.json    : decision = "bot"    (10.000)   → seu label 1
outbrain-safe.json   : decision = "safe"      (100)   → ignorado pelo código
```

O rótulo é a **decisão do sistema de filtragem que já está rodando**. Portanto:

- "Falso positivo / falso negativo na base" = **erro do filtro de produção**.
- Qualquer modelo treinado nesses rótulos aprende a **imitar o filtro**,
  incluindo os erros dele.

### 3.2 A consequência que invalida a validação atual

Erro de um filtro determinístico/aprendido **não é aleatório**. Ele é
sistemático: correlacionado com o modo de falha da regra. Exemplos concretos no
seu domínio:

- Bot residencial com UA de iPhone real e `Sec-Fetch-*` coerente → passa como
  humano. **Sempre**. Todos eles.
- Humano legítimo atrás de VPN/proxy corporativo, `x-forwarded-for` de
  datacenter → bloqueado como bot. **Sempre**.

Agora coloque isso ao lado do que `inject_synthetic_label_noise(mode="symmetric")`
faz: escolhe amostras **uniformemente ao acaso** e inverte o rótulo.

| | Ruído simétrico (o que você valida) | Ruído sistemático (o que você tem) |
|---|---|---|
| Localização no espaço de features | aleatória | concentrada numa região |
| Vizinhança da amostra ruidosa | rótulos majoritariamente **corretos** | vizinhos com o **mesmo rótulo errado** |
| `knn_disagreement` | dispara alto | **≈ 0** — há concordância local |
| Modelo OOF (Confident Learning) | discorda, pois aprendeu a regra certa | **concorda**, pois aprendeu a mesma regra errada |

**Ruído simétrico é o caso mais fácil que existe, e é o único em que você mediu
seus detectores.** É por isso que KNN e CL dão AUC 0.92 no sintético e o
Confident Learning sinaliza apenas **0.2%–2.6%** no dado real
(`confident_learning_flag_rate`: outbrain 0.0096, facebook 0.0021,
taboola 0.0264, tiktok 0.0032). Não é que exista pouco ruído. É que o detector é
estruturalmente cego ao tipo de ruído que existe.

O `top100_ensemble_AND_heuristic.intersection = 13` do
`suspect_overlap_report.json` conta a mesma história: os detectores e as
heurísticas externas estão praticamente sem correlação.

Nada disso é fatal para o artigo. Medido explicitamente, **vira a contribuição
principal** — ver Parte 5.

---

## Parte 4 — O que fazer, em ordem de prioridade

### P0 — Trocar o modelo de ruído sintético (2–3 dias, alto impacto)

Sem isso, nenhuma afirmação sobre detecção de ruído no seu domínio se sustenta.
Adicione dois modos a `src/noise/synthetic_noise.py`:

```python
def inject_systematic_noise(y, X, rate, rule=None, random_state=42):
    """Ruído CORRELACIONADO com features — imita o modo de falha de um filtro.

    Em vez de escolher amostras ao acaso, escolhe uma REGIÃO do espaço e
    inverte a maioria dos rótulos ali. Os rótulos errados ficam mutuamente
    consistentes — exatamente o que quebra KNN e Confident Learning.
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(random_state)
    y_noisy, mask = y.copy(), np.zeros(len(y), bool)

    # 1) particiona o espaço de features em regiões
    km = KMeans(n_clusters=20, random_state=random_state, n_init=10).fit(X)
    reg = km.labels_

    # 2) sorteia regiões até cobrir `rate` do dataset e vira ~85% delas
    order = rng.permutation(20)
    budget = int(rate * len(y))
    for c in order:
        idx = np.where(reg == c)[0]
        if mask.sum() >= budget:
            break
        chosen = rng.choice(idx, size=int(0.85 * len(idx)), replace=False)
        y_noisy[chosen] = 1 - y_noisy[chosen]
        mask[chosen] = True
    return y_noisy, mask


def inject_rule_based_noise(y, headers, requests, rate, random_state=42):
    """Ruído semanticamente motivado: inverte só bots que "parecem humanos".

    Prioriza bots com UA mobile plausível + Sec-Fetch-* completo + sem template.
    É o falso negativo REAL do seu filtro, reproduzido de forma controlada.
    """
    # score de "humanidade aparente" e flip dos bots com score mais alto
    ...
```

Depois rode a grade completa (`noise_rates × seeds × {symmetric, systematic,
rule_based}`) e **reporte a tabela de degradação**. Minha previsão: CL e KNN caem
de ~0.91 para 0.55–0.70 no modo `systematic`. Essa tabela é o resultado mais
publicável do trabalho.

### P1 — Construir um gold set humano (1–2 semanas, obrigatório)

Sem gold set, o audit no dado real é ranking não validado — você não pode
escrever nenhuma frase sobre falsos positivos reais. 300–500 amostras bastam.

**Regra de ouro:** rotule usando **sinais que não são features do modelo**, senão
o gold set só confirma o modelo.

| Evidência independente | Como usar |
|---|---|
| ASN / tipo de IP (residencial vs datacenter vs mobile) | consulta de ASN no `x-forwarded-for` |
| Repetição de IP / fingerprint na janela | agregação temporal, fora do escopo por-request |
| Comportamento pós-clique | tempo na página, scroll, conversão |
| Coerência `UA` ↔ `Sec-CH-UA` ↔ `accept-language` ↔ TLS JA3/JA4 | inconsistência = sintético |
| Cadência temporal | intervalos inter-request com variância anormalmente baixa |

**Amostragem** — não pegue só o top-k, isso enviesa tudo:

1. Estratifique por decil de `score_ensemble` (~30 por decil).
2. Adicione ~100 aleatórias uniformes, para estimar a taxa-base.
3. Guarde a probabilidade de inclusão de cada amostra e use o estimador
   **Horvitz–Thompson** para extrapolar a prevalência de ruído na base inteira
   com intervalo de confiança. Isso transforma "achamos 26 suspeitos" em
   "estimamos 4.1% ± 1.2% de rótulos incorretos" — muito mais forte.

`experiments/gold_set_suspects.csv` (99 linhas) é um começo, mas é uma lista de
suspeitos, não um gold set: não tem rótulo revisado nem evidência anexada.

### P2 — Consertar as features antes de culpar o CC (3–5 dias, alto impacto)

O CC pode estar falhando porque a **entrada** é ruim, não porque a ideia é ruim.
Quatro correções, em ordem de retorno:

1. **Incluir o `request` no TF-IDF.** Hoje ele é ignorado (`_text` retorna só
   headers). Use dois vetorizadores (headers, request) e concatene os dois SVDs.
   Barato, e provavelmente o maior ganho isolado do projeto.

2. **Teste de vazamento por grupo — faça isso antes de qualquer outra coisa.**
   O split é aleatório e `host`/`campaign_id` estão no texto vetorizado. Refaça
   com `GroupKFold` agrupando por `host` (e depois por `campaign_id`). Se a AUC
   do RF cair de 0.96 para ~0.80, parte da performance atual é **memorização de
   campanha**, não detecção de bot. Isso muda a interpretação de tudo. É um teste
   de meia hora.

3. **Features estruturadas explícitas**, além do n-grama: conjunto e ordem das
   chaves de header; coerência `Sec-Fetch-Dest/Mode/Site/User`; presença e
   consistência de `Sec-CH-UA` vs `User-Agent`; `accept-language` vs
   geolocalização do IP; contagem de hops no `x-forwarded-for`; hora do dia.
   Ablação: RF só com estruturadas vs RF só com TF-IDF. Se empatarem, o TF-IDF
   está memorizando identificadores.

4. **Ablação das augmentações que apagam o sinal.** Rode `medium_http_aug` com
   `drop_optional_prob=0` e `mask_value_prob=0` (preservando `sec-fetch-*`). Se
   o ARI do CC subir, a hipótese "as augmentações estavam ensinando invariância
   ao sinal de classe" se confirma — e isso é um achado metodológico bonito sobre
   desenho de augmentação em dados HTTP.

### P3 — Reformular a abordagem: semi-supervisionado de verdade (1–2 semanas)

Este é o ponto que você pediu diretamente. O CC é **não-supervisionado** — ele
joga fora seus rótulos. Mas você tem rótulo (ruidoso) para 100% das amostras.
O nome da literatura que resolve exatamente o seu problema é **Learning with
Noisy Labels (LNL)**, e ela é semi-supervisionada por construção.

**Por que LNL e não CC:** LNL usa o rótulo ruidoso *e* modela o fato de ele estar
errado, em vez de ignorá-lo. Os três abaixo dão de graça um ranking de ruído.

**P3.a — AUM / small-loss (2 dias, comece por aqui).**
Redes memorizam ruído *depois* de aprender o padrão. Grave a margem por amostra
a cada época e integre:

```python
# durante o treino do encoder+cabeça linear (ou do triplet), por época:
margin[i] = logit[i, y_i] - max_{c != y_i} logit[i, c]
AUM[i] = margin[i].mean(over epochs)   # AUM baixo/negativo = suspeito
```

Pleiss et al. (2020) acrescentam *threshold samples* (um conjunto com rótulo
deliberadamente aleatório) para calibrar o corte sem precisar de gold set.
Esperado: bem acima de `cluster_entropy` (0.50) e competitivo com CL — **e, ao
contrário do CL, não depende de um segundo modelo que compartilha o mesmo viés**.

**P3.b — Co-teaching (3 dias).**
Duas redes com inicializações diferentes. Em cada batch, cada rede seleciona as
`R(t)%` amostras de **menor perda** e as passa para a *outra* rede treinar. Os
vieses não se reforçam porque as redes discordam. Amostras persistentemente
descartadas pelas duas = candidatas a ruído. Barato, robusto, e o ranking sai
como subproduto.

**P3.c — DivideMix (1 semana, o mais forte).**
Ajusta uma **mistura de 2 gaussianas** sobre a distribuição de perdas por
amostra → separa `clean set` (rotulado) de `noisy set`. Trata o noisy set como
**não rotulado** e aplica MixMatch (co-refinement + co-guessing + mixup). Isto é
literalmente "usar semi-supervisionado para encontrar e corrigir rótulos
errados". A probabilidade posterior do GMM é o seu score de suspeição — e é
calibrada, ao contrário de todos os scores atuais. Se quiser menos código,
**ELR (Early-Learning Regularization)** captura ~80% do ganho com ~30 linhas.

**O que fazer com o CC então:** mantenha-o no artigo, mas reposicionado. Ele
responde uma pergunta legítima — *existe estrutura não-supervisionada nos headers
que corresponda a humano/bot?* — e a resposta (ARI 0.007–0.09) é **não**, o que é
um resultado. Não o venda como detector de ruído, porque os números não sustentam.

### P4 — Consertar o ensemble (1 dia)

1. **Remova** `cc_cluster_entropy` e `cc_view_instability` (AUC ≈ 0.5). Reporte a
   ablação leave-one-score-out para justificar por dados, não por escolha.
2. **Troque min-max por média de ranks.** Min-max é dominado por outliers; um
   único valor extremo comprime todo o resto. `scipy.stats.rankdata` normalizado
   é mais robusto e é o padrão em fusão de rankings.
3. **Aprenda os pesos**, com regressão logística treinada no ruído sintético
   (agora `systematic`) ou no gold set, com validação aninhada. Reporte os pesos
   — eles são interpretáveis e dizem quais sinais importam.
4. Reporte **AUPRC e precision@k como métricas primárias**, não ROC-AUC. Com
   prevalência de 1–5%, ROC-AUC é otimista e enganosa; precision@100 é o que
   descreve o trabalho real de revisão.

### P5 — Parar de tratar `unsafe` como "humano" (meio dia, conceitual)

Seu código faz `unsafe → 0` e chama isso de humano, ignorando `safe` (100
amostras em outbrain). Se `unsafe`/`offer` significa "suspeito mas liberado",
então a classe 0 é uma **mistura de humano-limpo com suspeito-liberado**, e parte
do seu "falso negativo" está embutido na *definição da classe*, não no rótulo.

Faça uma de três coisas, e diga explicitamente qual no artigo:

- modele 3 classes (`safe` / `unsafe` / `bot`);
- trate como ordinal (grau de suspeição);
- ou mantenha binário, mas documente que a classe 0 = "não bloqueado pelo filtro"
  e não "humano verificado".

Não escolher é a única opção ruim.

### P6 — O sinal que você já tem e não está usando: concordância cross-source

Se o gold set (P1) atrasar, existe uma segunda opinião com **independência
real** já no repositório: `results/cross_source/`.

Treine em outbrain e prediga em taboola/tiktok, e vice-versa. Amostras que a
produção chamou de `bot` mas que **todos os modelos treinados em outras fontes**
chamam de humano com alta confiança são candidatas a ruído com evidência que não
vem do mesmo filtro. Isso quebra a circularidade que trava o Confident Learning:
o modelo auditor não compartilha o viés do filtro que gerou o rótulo, porque foi
treinado em outra distribuição.

Complemento: estime a **matriz de ruído** com o confident joint de Northcutt
(`cleanlab.count.estimate_joint`). Reportar "estimamos taxa de ruído
bot→humano de X% e humano→bot de Y%, por fonte" é uma contribuição publicável
que não exige gold set — só exige honestidade sobre as premissas
(class-conditional noise, que já é violada aqui, então declare isso).

---

## Parte 5 — Como escrever isso honestamente

### Afirmações que os dados de hoje NÃO sustentam

Se alguma destas estiver no rascunho, tire ou requalifique:

- ❌ "Contrastive Clustering aprende agrupamentos que correspondem a
  humano/bot." → ARI 0.007–0.09 em 3 de 4 fontes.
- ❌ "Embeddings CC melhoram a detecção de ruído sobre features cruas." →
  `knn_cc` é pior que `knn_raw` em taboola e tiktok.
- ❌ "Instabilidade entre views é um sinal de rótulo suspeito." → AUC 0.45.
- ❌ "Entropia de cluster indica ruído." → AUC 0.50.
- ❌ "O ensemble é o melhor detector." → melhor ROC-AUC, **pior** AUPRC e
  precision@100 que o Confident Learning sozinho.
- ❌ "O audit real identificou rótulos incorretos." → sem gold set, são
  candidatos não validados.
- ❌ "`has_template` é uma heurística de alta precisão." → dispara em 100% do
  tráfego label-0 do facebook.

### O enquadramento que os dados sustentam — e que é mais interessante

> Rótulos de tráfego pago não são ground truth: são decisões de um filtro em
> produção, e portanto seu ruído é **sistemático**, não aleatório. Mostramos que
> detectores estado-da-arte de ruído de rótulo (Confident Learning, KNN de
> inconsistência local) atingem ROC-AUC ≈ 0.92 sob ruído simétrico sintético —
> o regime padrão de avaliação da literatura — mas caem para ≈ X sob ruído
> correlacionado com features, que é o regime real deste domínio. Avaliamos
> também sinais derivados de clustering contrastivo (entropia de cluster,
> instabilidade entre views HTTP aumentadas) e mostramos que **não carregam
> informação sobre ruído de rótulo** (AUC 0.45–0.61), apesar de serem
> intuitivamente atraentes. Propomos augmentações no nível HTTP como
> contribuição de domínio e demonstramos, por ablação, que augmentações que
> apagam headers de coerência (`Sec-Fetch-*`) ensinam invariância ao próprio
> sinal de classe. Finalmente, mostramos que concordância **cross-source** é o
> único sinal de segunda opinião com independência genuína do filtro que gerou
> os rótulos.

Esse artigo é mais forte que o original. Um resultado negativo bem medido, com
diagnóstico da causa, é publicável; um resultado positivo em benchmark
inadequado não sobrevive à revisão.

---

## Parte 6 — Checklist de execução

| # | Tarefa | Esforço | Bloqueia? |
|---|---|---|---|
| 1 | Teste de vazamento com `GroupKFold` por `host` e `campaign_id` | 0.5 dia | **sim, tudo** |
| 2 | Incluir `request` no TF-IDF; re-rodar baselines | 1 dia | não |
| 3 | Implementar `systematic` + `rule_based` noise; rodar grade completa | 3 dias | **sim, Parte 5** |
| 4 | Pré-computar pool de views (fim das 36h de runtime) | 1 dia | viabiliza 3 |
| 5 | Remover scores mortos do ensemble; rank-average; ablação LOO | 1 dia | não |
| 6 | Implementar AUM (small-loss) e comparar com CL | 2 dias | não |
| 7 | Co-teaching como detector + baseline | 3 dias | não |
| 8 | Recalibrar heurísticas por fonte; medir precisão de cada regra | 1 dia | interpretação |
| 9 | Decidir e documentar a semântica `safe`/`unsafe`/`bot` | 0.5 dia | **sim, framing** |
| 10 | Gold set 300–500 amostras, estratificado, com Horvitz–Thompson | 1–2 sem | **sim, claims reais** |
| 11 | Estudo de concordância cross-source + matriz de ruído estimada | 3 dias | não |
| 12 | DivideMix ou ELR (opcional, se houver tempo) | 1 sem | não |

Comece pelo item 1. Se houver vazamento de campanha, vários números da Parte 2
mudam e a ordem do resto muda com eles.

---

## Referências

- Li, Y. et al. *Contrastive Clustering.* AAAI 2021. — o método em `src/models/contrastive_clustering.py`
- Chen, T. et al. *SimCLR.* ICML 2020. — NT-Xent em `src/losses/contrastive_losses.py`
- Khosla, P. et al. *Supervised Contrastive Learning.* NeurIPS 2020. — alternativa para P3
- Northcutt, C. et al. *Confident Learning.* JAIR 2021. — `src/noise/confident_learning.py`
- Han, B. et al. *Co-teaching.* NeurIPS 2018. — P3.b
- Li, J. et al. *DivideMix.* ICLR 2020. — P3.c
- Liu, S. et al. *Early-Learning Regularization.* NeurIPS 2020. — P3.c simplificado
- Pleiss, G. et al. *Identifying Mislabeled Data using the Area Under the Margin Ranking.* NeurIPS 2020. — P3.a
