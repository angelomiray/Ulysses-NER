# Projeto de Reconhecimento de Entidades Nomeadas (NER) em Documentos Legislativos

Este projeto evidencia a análise comparativa de desempenho entre modelos para Reconhecimento de Entidades Nomeadas (NER). 
O objetivo principal é avaliar e comparar a eficácia de diferentes abordagens de modelagem, especificamente o modelo baseado 
na biblioteca spaCy e o modelo baseado em Transformers através da abordagem Universal Language Model Fine-tuning (ULMFit).

Os dados utilizados são oriundos do corpus [UlyssesNER-Br](https://github.com/Convenio-Camara-dos-Deputados/ulyssesner-br-propor), sendo este composto por projetos de lei 
e consultas legislativas da Câmara dos Deputados do Brasil.

O projeto envolve três abordagens principais para o treinamento e avaliação dos modelos:

- [**Modelo spaCy NER**](https://colab.research.google.com/drive/1tF2QNm4AJtycoX-zTP5S8E0UVbtbXh4h): Pipeline baseado na biblioteca spaCy, treinado a partir do zero com os dados anotados do corpus UlyssesNER-Br, utilizando o formato DocBin.

- [**Transformers com Fine-Tuning (Masked Language Model)**](https://colab.research.google.com/drive/1YB63a8j64JEFCv5bFN0LsOhf3RGCX2Hn): Utilização do modelo BERT pré-treinado (`neuralmind/bert-base-portuguese-cased`), adaptado por fine-tuning na tarefa de MLM com os dados concatenados, preparando a base para posterior fine-tuning na tarefa de NER.

- [**Transformers para NER (Token Classification)**](https://colab.research.google.com/drive/1AfKMz_ScKzPNKWSVKeGvpJ5JyV5FLvpb): Modelo BERT fine-tuned especificamente para classificação token a token com etiquetas BIO, treinado diretamente para o reconhecimento das entidades nomeadas do corpus.

A comparação entre esses modelos permite analisar métricas tradicionais, como precisão, recall e F1-score, além de produzir insights com base nas diferentes abordagens.

## Análise dos Resultados

A seguir, são mostrados os resultados finais de cada modelo, bem como as análises individuais e comparativas. 
Este repositório possui os notebooks com o pipeline comentado para cada modelo, mas caso haja algum problema de renderização por parte do GitHub, os links do Google Colab estão acima.

### Abordagem com spaCy

**Token accuracy (`token_acc`)**: O valor **1.0** mostra que o modelo segmentou os tokens exatamente como esperado.  
Isso é típico quando se utiliza a tokenização padrão do **spaCy** em textos já limpos e bem formatados.

**Precision (`ents_p`) = 75.9%**: Dos *spans* de entidades previstos pelo modelo, aproximadamente **76%** foram corretos — ou seja, a maior parte dos spans detectados realmente corresponde a entidades anotadas.

**Recall (`ents_r`) = 65.1%**: O modelo conseguiu recuperar cerca de **65%** de todas as entidades reais presentes no conjunto de teste. Isso mostra que ainda há entidades que não foram reconhecidas (*falsos negativos*).

**F1-score (`ents_f`) = 70.1%**: A pontuação **F1** combina *Precision* e *Recall*, fornecendo uma visão equilibrada da qualidade do modelo. Um valor acima de **70%** é um bom ponto de partida, considerando a complexidade de textos legislativos.

```
{'token_acc': 1.0,
 'token_p': 1.0,
 'token_r': 1.0,
 'token_f': 1.0,
 'ents_p': 0.7594108019639935,
 'ents_r': 0.651685393258427,
 'ents_f': 0.7014361300075586,
 'ents_per_type': {'PESSOA': {'p': 0.8256880733944955,
   'r': 0.6666666666666666,
   'f': 0.7377049180327869},
  'PRODUTODELEI': {'p': 0.6595744680851063,
   'r': 0.49206349206349204,
   'f': 0.5636363636363635},
  'ORGANIZACAO': {'p': 0.6972477064220184,
   'r': 0.6129032258064516,
   'f': 0.6523605150214592},
  'FUNDAMENTO': {'p': 0.6987951807228916,
   'r': 0.7945205479452054,
   'f': 0.7435897435897436},
  'DATA': {'p': 0.8860759493670886,
   'r': 0.8333333333333334,
   'f': 0.8588957055214723},
  'LOCAL': {'p': 0.801980198019802,
   'r': 0.5192307692307693,
   'f': 0.6303501945525292},
  'EVENTO': {'p': 0.0, 'r': 0.0, 'f': 0.0}},
 'speed': 12131.672033139972}
```

```

| Entidade      | Precision (p) | Recall (r) | F1-score (f) |
|---------------|----------------|------------|--------------|
| **PESSOA**        | 0.825688       | 0.666667   | 0.737705     |
| **PRODUTODELEI**  | 0.659574       | 0.492063   | 0.563636     |
| **ORGANIZACAO**   | 0.697248       | 0.612903   | 0.652361     |
| **FUNDAMENTO**    | 0.698795       | 0.794521   | 0.743590     |
| **DATA**          | 0.886076       | 0.833333   | 0.858896     |
| **LOCAL**         | 0.801980       | 0.519231   | 0.630350     |
| **EVENTO**        | 0.000000       | 0.000000   | 0.000000     |

```


### Abordagem Transformers ULMFiT

A tabela abaixo mostra as principais métricas de desempenho obtidas durante o treinamento do modelo 
BERT fine-tuned para NER no corpus UlyssesNER-Br:

```
| Época | Training Loss | Validation Loss | F1-score | Precision | Recall   | Accuracy |
| ----- | ------------- | --------------- | -------- | --------- | -------- | -------- |
| **1** | 0.169100      | 0.097358        | 0.822034 | 0.822034  | 0.822034 | 0.976423 |
| **2** | 0.001600      | 0.122304        | 0.807692 | 0.814655  | 0.800847 | 0.973376 |
| **3** | 0.000400      | 0.111022        | 0.830898 | 0.818930  | 0.843220 | 0.978348 |
| **4** | 0.000500      | 0.127228        | 0.863732 | 0.854772  | 0.872881 | 0.979310 |
| **5** | 0.000200      | 0.130753        | 0.866808 | 0.864979  | 0.868644 | 0.980273 |

```

- **Evolução do F1-score**: Observa-se um ganho progressivo de desempenho, com o F1-score aumentando de ~82% na primeira época para ~86% na última.
  Isso demonstra que o modelo conseguiu ajustar-se bem à tarefa de NER no domínio legislativo.

- **Precision e Recall**: Tanto Precision quanto Recall se mantêm equilibrados ao longo das épocas, indicando que o modelo consegue não só prever entidades corretas (alta precision)
  como também recuperá-las de forma ampla (bom recall).

- **Estabilidade**: A Accuracy se mantém acima de 97%, o que reforça a consistência do treinamento.

### Análise Comparativa

Os resultados obtidos demonstram diferenças claras entre as abordagens avaliadas:

- **Modelo spaCy**: Treinado do zero, o pipeline do spaCy depende fortemente da qualidade e quantidade do corpus anotado.
  Apesar de ter atingido métricas aceitáveis (F1-score ~70%), ele tende a exigir mais iterações, fine-tuning manual de regras e maior esforço para lidar com variações linguísticas,
  especialmente em um domínio complexo como textos legislativos.

- **Transformers com ULMFiT**: O fine-tuning do BERT aplicado como Masked Language Model (uma forma prática de ULMFiT) foi decisivo para adaptar o modelo ao estilo textual
  específico do corpus UlyssesNER-Br antes de realizar a etapa de Token Classification para NER.

  - Essa estratégia traz vantagens importantes:

    - **Transferência de Conhecimento**: O BERT já carrega uma base robusta de conhecimento linguístico.
      O fine-tuning com o corpus legislativo reorienta esse conhecimento para o domínio-alvo, sem começar do zero.

    - **Contexto mais rico**: O BERT utiliza atenção bidirecional, capturando relações contextuais complexas entre tokens, o que é especialmente útil para
      evitar ambiguidade de entidades em frases longas.

    - **Desempenho consistente**: O modelo fine-tuned superou o spaCy em F1-score, Precision e Recall, com métricas finais acima de 85% em todas as frentes,
      enquanto o spaCy ficou em torno de 70% de F1-score.

    - **Robustez**: A abordagem Transformers foi mais resiliente em capturar entidades menos frequentes, mantendo bom equilíbrio entre Recall e Precision.





