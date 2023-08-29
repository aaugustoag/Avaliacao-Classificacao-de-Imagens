# Avaliação - Classificação de Imagens
<div align="left"> 
  <img align="center" src="https://img.shields.io/badge/Python-FF8C00?style=for-the-badge&logo=python&logoColor=white"><br>
  <img src="https://img.shields.io/badge/Ciência%20de%20Dados-red">
  <img src="https://img.shields.io/badge/Inteligência%20Artificial-orange">
  <img src="https://img.shields.io/badge/Aprendizado%20de%20Máquina-blue">
</div>

Prática de Classificação de Imagens no contexto da disciplina Aprendizado de Máquina no CEFET-MG.
## Avaliação - Classificação Automática de Segmentos de Imagens

Nesta prática você irá avaliar um dataset de 1.500 segmentos de imagens. Nesse projeto, cada instancia representa um segmento de 3x3 pixels de uma imagem de algum dos seguintes elementos:

<center>
<img src="https://drive.google.com/uc?id=186mCM0DkT3fNN6_kNMyk04u6zU70A7iG" alt="Imagens que foram seguementadas">
</center>
  
Assim, esta tarefa consiste em classificar tais segmentos de 3x3 pixels em um dos tipos de imagens externas (cimento, janela, grama, etc.). Cada instancia é representada da seguinte forma: 

<ol>
    <li>region-centroid-col:  coluna do pixel central da região </li>
    <li>region-centroid-row:  linha do pixel central da região </li>
    <li>region-pixel-count:  o número de pixels em uma região(3x3 = 9 neste caso) </li>
    <li>short-line-density-5: resultados de uma linha extraída no algoritmo que conta quantas linhas de comprimento 5 (qualquer orientação) com baixo contraste, menor ou igual a 5, passam pela região. </li>
    <li>short-line-density-2:  igual a densidade de linha curta-5, mas conta linhas de alto contraste, maiores que 2 </li>
    <li>vedge-mean: mede o contraste de pixels adjacentes horizontalmente na região. Existem 6, a média e o desvio padrão são dados. Este atributo é usado como um detector de borda vertical.</li>
    <li>vegde-sd: desvio padrão do contraste de pixels adjacentes horizontalmente </li>
    <li>hedge-mean: mede o contraste de pixels adjacentes verticalmente. Usado para detecção de linha horizontal. </li>
    <li>hedge-sd: desvio padrão do contraste de pixels adjacentes verticalmente.</li>
    <li>intensity-mean:  a média na região de (R + G + B) / 3 </li>
    <li>rawred-mean: a média sobre a região do valor R (cor vermelha) </li>
    <li>rawblue-mean: a média sobre a região do valor B (cor azul) </li>
    <li>rawgreen-mean: a média sobre a região do valor G (cor verde) </li>
    <li>exred-mean: mede o excesso de vermelho: (2R - (G + B)) </li>
    <li>exblue-mean: mede o excesso de azul: (2B - (G + R)) </li>
    <li>exgreen-mean: mede o excesso de verde:  (2G - (R + B)) </li>
    <li>value-mean: transformação não-linear 3-d de RGB </li>
    <li>saturatoin-mean: média de saturação do RGB</li>
    <li>hue-mean: média de tonalidade do RGB </li>
    <b><li style="color: red">y-i: classe a ser inferida (ver figura acima)</li></b>
</ol>

<a href="https://storm.cis.fordham.edu/~gweiss/data-mining/weka-data/segment-challenge.arff">**Referência**</a>

1. **Implementação do código de avaliação:** Primeiramente você deverá implementar as métricas de avaliação (da classe Resultado). O arquivo `resultado_tests.py` possui os testes unitários.  Veja abaixo como cada métrica, que é uma propriedade da classe (i.e. atributo calculado):
    - **mat_confusão**: Retorna a matriz de confusão correpondente. Será uma matriz em que o número de linhas e coluna é o valor numérico da maior classe na amostra.
    - **acurácia**: A partir da matriz de confusão, calcule a acurácia 
    - **precisao**: A partir da matriz de confusão, calcule a precisão por classe. Caso, a quantidade de instancias preditas para uma determinda classe for zero, então `precisao[c] = 0`. Nesses casos, você deverá [lançar um warning](https://docs.python.org/3.7/library/warnings.html) da classe `UndefinedMetricWarning` com uma mensagem que não havia instancias previstas para essa classe.
    - **revocacao**: De forma similar à `precisao`, calcula a revocação por meio da matriz de confusão. Caso o número de elementos dessa classe seja igual a zero, então a revocação para esta classe também é zero e também deverá ser retornado um warning `UndefinedMetricWarning` com essa informação
    - **f1_por_classe**: Retorna, para cada classe, o seu valor F1. Caso a soma da precisão e revocação dessa classe seja zero, deverá ser retornado zero.
    - **macro_f1**: Calcula a média do f1 por classe. O método `np.average` pode ajudar.

2. **Método eval da classe fold**: O método `eval` passará como parametro um método de aprendizado de máquina (por ex, uma instancia de [Árvore de Decisão](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)). Assim, usando esse parametro, você deverá criar o modelo de treino (usando as features de `df_treino` e a sua classe) e classificar os elementos de `df_data_to_predict`.

3. **Obtenção e divisão do Dataset:** Nesta prática, iremos trabalhar com 80% de treino e 20% de teste. Leia o dataset [`segment.csv`](segment.csv) e divida-o apropriadamente. Não esqueça que a amostra deve ser aleatória. Coloque como `random_state=1`
 
4. **Criação do modelo e avaliação** dos modelos: agora, você deverá avaliar 4 modelos de aprendizado de máquina nessa tarefa. Use os métodos SVM com kernel linear e RBF, KNN e árvore de decisão. Deixe os parametros padrão de cada algoritmo. Por meio do método eval da classe Fold, avalie o método. Apresente a matriz de confusão resultante de cada um desses métodos além das métricas macro f1, precisão e revocação por classe. Também verifique  quão sensível são os resultados se mudarmos o parametro `random_seed`. Você pode criar mais de um bloco de código/texto para isso organizando da forma que julgar melhor. 

    <ul>
        <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">SVM</a></li>
        <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">KNN</a></li>
    <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">Árvore de decisão</a></li>
    </ul>

## Matriz de Confusão:
### Árvore de Decisão
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/06cb3364-ed8d-4aec-960d-b16cd17dd3fa)

### SVM
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/c56d66ae-fc72-4751-a32b-15367bd4de62)

### KNN
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/52bab186-2bc8-4c35-a06b-59d35f9f6271)

## Acurácia
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/36c74a57-7cac-4a47-9a3f-8c65def18631)

![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/0605ca93-8afb-42a7-8f7b-5b8fc8085dcd)

## Precisão
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/6a0c6c8e-327f-44ea-8a8d-feb315eae3d8)

## Revocação
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/b18f2b94-93f7-4016-a461-4620f4112b3e)

## F1 Score
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/a70ab5b4-178e-46a8-a46d-af53c8c20583)

## Macro F1
![image](https://github.com/aaugustoag/Avaliacao-Classificacao-de-Imagens/assets/49174397/5ba6a41a-dbf0-4a1d-8657-91b76f873db1)

5. **Conclusões:** Escreva um texto com uma análise e conclusão dos resultados, por exemplo: quais são as classes mais dificieis/fácieis de prever? Quais se confundem mais? Qual é o melhor método de classificação?

Foram usados 3 métodos para predição de valores do grupo de dados estudado, Árvore (Tree), SVM e Vizinhança (KNN).<br>
À todos foram aplicadas os mesmos testes para comparar qual seria o melhor método em geral e quais seriam os melhores por classe.<br>
No geral, o KNN e o Tree apresentaram performance bastante aceitáveis, com uma pequena vantagem para a Tree, tanto na comparação de Acuracia quanto na Macro F1 (ambos os testes 95 e 93% de acertividade respectivamente).<br>
A performance do SVM em ambos os testes foi muito ruim (52 e 57% de acertividade respectivamente).<br>
Ao se analisar as tabelas de confusão e confirmando nos testes de Precisão, Revocação e F1, dos três métodos utilizados, pode-se verificar uma dificuldade em predizer as classes 2, 3 e 4 e esta dificultade em prever os resultados nestas 3 classes faz com que o SVM se apresente como uma boa alternativa na predição das classes 2 e 4, conforme verificado no teste de Precisão dos métodos por classe.<br>
Como nas demais classes deste teste o SVM performou parecido com os demais métodos, ele torna-se uma solução interessante quando o foco é analisar um maior conteúdo das classes 2 e 4.<br>
Desta maneira, seria interressante concluir que o melhor método, em uma análise geral, onde a intenção é apenas classificar a imagem entre os grupos, é o Tree, por apresentar melhor performance de Acuracia e Macro F1, no entanto, se a intenção é verificar se a imagem pertence ao grupo 2 ou 4, seria interessante considerar o SVN como melhor método, por ele ter apresentado melhor performance no teste de Precisão para estas duas classes.
