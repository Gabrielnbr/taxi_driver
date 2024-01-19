# Taxi Drive: Projeto de Previsão de mobilidade urbana
# 0.0 Orientações
Este projeto é um case técnico solicitado por um empresa como parte do processo seletivo dela. Tive exatamente 1 semana para desenvolver todo o projeto. Os avaliadores esperavam que eu entregasse o projeto até a análise do modelo de Machine Learning e suas métricas, mas fui além. Coloquei em produção para que eles pudessem testar uma versão simples do projeto.

Todo o proejto que está em produção foi dezenvolvido em 1 semana, só não a página "Home" que fiz posterior a apresentação visando disponibilizar este trabalho no meu portfólio.

Ressalto que estou disponibilizando o projeto online devido ao fato dos dados serem extraídos de uma fonte pública.

Este é um projeto com dados extraídos e trabalhados do [Kaggle: Rossmann Store Sales](https://www.kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i).

O Projeto em Produção está na propria nuvem do [Streamlit](https://taxi-driver.streamlit.app/). A API que realiza as predições foi disponibilizada no Render por meio da url https://taxi-4ui9.onrender.com

Caso queira entender mais minha experiência com o projeto, eu disponibilizei um [Artigo no Medium](https://medium.com/@gabrielnobregalvao/taxi-drive-projeto-de-previs%C3%A3o-de-mobilidade-urbana-d921f895f9af).
# 1.0 Problema de Negócio
Devido aos grandes avanços na área da locomoção, os sistemas de táxi estão precisando se modernizar e também modernizar suas estruturas. Para isso fomos contratados para resolver o problema na capacidade de prever o destino final de um táxi enquanto está em serviço, visando a melhor alocação de novos serviços ao mesmo táxi assim que ficar vago.

Para tal problema precisamos contrui uma estrutura preditiva que seja capaz de prever o destino final das corridas com base na sua trajetória inicial.

Para melhorar a eficiência dos sistemas electrónicos de despacho de táxis é importante ser capaz de prever o destino final de um táxi enquanto este está em serviço.

Se um despachante soubesse aproximadamente onde seus motoristas de táxi terminariam suas viagens atuais, ele seria capaz de identificar qual táxi atribuir a cada solicitação de coleta.

Neste desafio, pedimos-lhe que construa uma estrutura preditiva que seja capaz de inferir o destino final das corridas de táxi no Porto, Portugal, com base nas suas trajetórias parciais (iniciais).

# 2.0 Premissas de negócio
Foi realizado o levantamento de algumas premissas que o projeto precisa atender:

1. Os operadores precisam ter em sua visão o ponto inicial de partida e o possível ponto final.
2. Os operadores poderão selecionar o Taxi para verificar sua disponibilidade.
   1. Em futura implementação eles poderão escolher a latitude e longetude, assim será mostrado os táxis mais proximo a este ponto.
3. Os operadores precisam ter noção do diametro da distância do táxi para o cliente.
   1. Deve ser colocado raio de 300m ao redor do táxi para facilitar a visualização.
4. O operador deve ser capaz de identificar o taxi e qual a corrida ele está fazendo.
5. A primeira versão será uma versão de teste para validação do produto final, assim ela será uma entrega em ambiente de homologação.
## 2.1. Descrição dos Dados

| Atributo | Descrição |
| -------- | --------- |
| TRIP_ID | (String) Contém um identificador único para cada viagem |
| CALL_TYPE | (char) Identifica a forma utilizada para demandar este serviço. Pode conter um dos três valores possíveis: 1. ‘A’ se esta viagem foi despachada da central. 2. ‘B’ se esta viagem foi solicitada diretamente a um taxista em ponto específico. 3. ‘C’ caso contrário (ou seja, uma viagem exigida numa rua aleatória). |
| ORIGIN_CALL | (inteiro) Contém um identificador único para cada número de telefone que foi utilizado para demandar, pelo menos, um serviço. Identifica o cliente da viagem se CALL_TYPE=’A’. Caso contrário, assume um valor NULL |
| ORIGIN_STAND | (inteiro): Contém um identificador único do ponto de táxi. Identifica o ponto inicial da viagem se CALL_TYPE=’B’. Caso contrário, assume um valor NULL |
| TAXI_ID | (inteiro) | Contém um identificador único do taxista que realizou cada viagem |
| TIMESTAMP | (inteiro) Unix Timestamp (em segundos). Identifica o início da viagem |
| DAYTYPE | (char) Identifica o dia de início da viagem. Ele assume um dos três valores possíveis: 1. ‘B’ se esta viagem começou num feriado ou em qualquer outro dia especial (ou seja, feriados prolongados, feriados flutuantes, etc.). 2. ‘C’ se a viagem teve início num dia anterior a um dia tipo B. 3. ‘A’ caso contrário (ou seja, um dia normal, dia útil ou fim de semana).
| MISSING_DATA | (Booleano) É FALSE quando o fluxo de dados GPS está completo e TRUE sempre que um (ou mais) locais estão faltando |
| POLYLINE | (String): Contém uma lista de coordenadas GPS (ou seja, formato WGS84) mapeadas como uma string. O início e o fim da string são identificados entre colchetes (ou seja, [ e ], respectivamente). Cada par de coordenadas também é identificado pelos mesmos colchetes que [LONGITUDE, LATITUDE]. Esta lista contém um par de coordenadas para cada 15 segundos de viagem. O último item da lista corresponde ao destino da viagem enquanto o primeiro representa o seu início |
# 3.0 Estratégia da Solução
Para garantir uma boa entrega foi adotado a metodologia CRISP-DM com algumas modificações, na qual consiste em 9 passos. Esses passos são realizados de forma cíclica, ao qual o desenvolvedor passará por cada etapa algumas vezes, mas sempre aprofundando o conhecimento e o detalhismo na quela etapa.

## 3.1. Metodologia CRISP-DM
1. Problema de Negócio: Consiste em receber a demanda do negócio, definir quem é o real Product Owner e os Stakeholders deste projeto.
2. Entendimento de Negócio: Neste momento nós buscamos entender a real dor do negócio por meio de reuniões com o Product Owner e os Stakeholders e como podemos construir as primeiras prototipações da solução.
3. Coleta de Dados: Esta etapa realizamos a busca e a agregação das informações relevantes ao projeto, seja conectando-se em um Datalake, em um Banco de Dados, por meio de tabelas em .cvs ou até realizando um webscrapping.
4. Limpeza dos Dados: Tem como objetivo realizar todas as limpezas necessária afim de poder trabalhar com os dados sem problemas. É aqui que tratamos as strings, retimos os NAN's e cuidamos dos dados para que ele não perca as características do fenômeno.
5. Exploração dos Dados: Neste ponto buscamos entender as relações entre os dados e a variável ao qual buscamos prever. Neste ponto também desenvolvemos novas features cujo possam agregar ao fenômeno.
6. Modelagem dos Dados: Aqui iremos aplicar técnicas matemáticas e estatísticas afim de estruturar os dados da melhor forma possível para que o modelo ao ser treinado possa performar da melhor forma.
7. Aplicação de Algoritmos de Machine Learning: Esta etapa consiste em selecionar e treinar os Algorítmos de Machine Learning afim de começarmos a entender as primeiras previsões. Também aplicamos técnincas como: Validation Holdout, Cross-Validation, Fine Tunning. **Neste caso Não utilizei técnicas de Cross-Validation nem Fine Tunning. Explico mais no ponto 6.0.**
8. Avaliação de Performance: Neste momento selecionamos o modelo que melhor generaliza, por meio de métricas a serem escolhidas conforme problema, ao ponto de conseguir predizer o que procuramos. Aqui há uma bifurcação, no qual se o modelo for bem avaliado ele irá para o passo 9 "Publicação da Solução", caso não, devemos retornar ao início do clíclo afim de obter melhores resultados
9. Publicação da Solução: Após a aprovação do modelo o Product Owner precisará utiliza-lo, desta forma deveremos publicar e disponibilizar de tal forma que ele possa ser acessado sem problemas. **Neste caso a entrega final será somente um modelo teste.** 
## 3.2 Produto Final
A equipe da empresa de táxi receberá para testes um site com as seguintes estruturas:
1. Uma página com a opção de selecionar multiplos Táxis.
2. Uma visualização da base de dados original daquele Táxi.
3. Um botão de Predição, ao qual o operador ao clicar ele retornará o possível destino final do táxi.
4. Uma tabela com os dados da predição e um mapa que possa interagir para visualizao melhor o ponto final do táxi. 
## 3.3 Ferramentas Utilizadas
Para criar a solução, foram utilizadas as seguintes ferramentas:

1. Python 3.10.4
2. Git e GitHub
3. Jupyter Notebook no VSCode
4. Técnicas de manipulação de dados.
5. Técnicas de redução de dimensionalidade e seleção de atributos.
6. Algoritmos de Machine Learning da biblioteca scikit-learn
7. Serviço de Hospedagem em Nuvem do Render. (Back-End)
8. Serviço Streamlit Cloud para aplicação web. (Front-End)
9. Metodologias Ágeis: Scrum e Kambam
10. Software de Gestão: Notion

# 4.0 Análise dos Dados
Após a limpeza e o tratamento inicial dos dados. Foi realizado uma análise estatística da qual houve o desenvolvimento de novas features e levantamento de hipóteses de negócio.
## 4.1 Insights de Negócio
Foram levantadas, respondidas e aplicada um grau de relevância para o negócio em 6 hipóteses conforme quadro a baixo:

| Hipótese | Pergunta | Resultado |
| -------- | -------- | --------- |
| H1 | Ocorrem mais vigens de táxi nos horários de entrada e saída do trabalho. Das 06h as 09h e das 16h as 19h. | FALSA |
| H2 | Ocorrem mais viagens de táxi nos primeiros 15 dias dos meses. | VERDADEIRA |
| H3 | Há mais viagens com o fluxo de dados incompletos do que completos. (chip gsm de monitoramento). | FALSA |
| H4 | As viagens despachadas da central não atingem 30% das viagens totais. | VERDADEIRA |
| H5 | Pelo menos 50% das viagens são pegues em pontos específicos. | FALSA |
| H6 | Ocorrem mais viagens nos meses de alta estação do que no resto do ano. Meses de alta estação: Janeiro (1), Junho (6), Julho(7) e Dezembro(12). | FALSA |
| H7 | Pelo menos 70% das viagens são menores que 10km. | VERDADEIRA |

Irei destacar somente as 5 hipóteses ao qual percebi relevância para o negócio.

### 3. Há mais viagens com o fluxo de dados incompletos do que completos.
Hipotese falsa.
Só há 10 viagens que estão com o fluxo incompleto. Isso é muito bom, daqui podemos criar outras fetures em um momento futuro. Por enquanto penso em duas novas features:
1. "tempo_da_viagem" = Calcular o tempo total da viagem em minutos. Já que cada ping demora 15 segundos, podemos verificar quantos pings de lat e long ocorreram, multiplicar por 15 e teremos o tempo da viagem em segundos.
2. "dt_hora_viagem_final" = Calcular a data e hora final da viagem.
![H3](/reports/figures/h3.png)
### 4. As viagens despachadas da central não atingem 30% das viagens totais. & 5. Pelo menos 50% das viagens são pegues em pontos específicos.
1. Hipótese Verdadeira.
2. Hipótese Falsa, com resalvas.

A central só é responsável por 21,32% das viagens. Mas os pontos de táxis parecem ser bem colocados na cidade pois representa 47,81% das viagens. Se for contabilizado na estrutura atual quase 70% das viagens são iniciadas dentro da estrutura da empresa.

Possuo 2 sugestões:
1. Um novo estudo das viagens do tipo "C" com a estrutura de lat e long e os pontos de parada dos táxis. Com o objetivo de maximizar e redistribuir esses pontos visando aumentar a categoria do tipo B.
2. Um estudo de viabilidade da construção de um aplicativo para que o cliente possa pedir seu taxi sem precisar nescessariamente passar pela central, mas passando por dentro da estrutura da empresa.
![H4_H5](/reports/figures/h4_h5.png)
### 6. Ocorrem mais viagens nos meses de alta estação do que no resto do ano. Meses de alta estação: Janeiro (1), Junho (6), Julho(7) e Dezembro(12).
Hipotese falsa.

Aparentemente o ano todo é estável com baixa variação.

Afim de aumentar a lucratividade, a empresa pode fazer parcerias com hoteis e agências de turismo tanto em Portugal quanto em Porto para buscar e deixar turistas. Qual a complexidade para tal?
![H6](/reports/figures/h6.png)
### 7. Pelo menos 70% das viagens são menores que 10km.
Hipótese verdadeira.

O total de 96,55% das viagens são menores do que 10km. Isso pode indicar também que o sistema de captura de lat e long pode está com algum problema. Devemos levar ao negócio os seguintes questionamentos:
1. O sistema de táxi aceita corrida até quantos quilometros além da cidade de Porto?
2. Caso possa deixar pessoas além da cidade de Porto, como funciona a métrica CALL_TYPE "C", para passageiros pegues em qualquer local fora da cidade de Porto? Os taxistas tem autorização para fazer essas viagens? ou eles só podem pegar as viagens que irão em direção a Porto?
3. Pode ser adicionado aos taxistas de um localizador GPS com chip GSM, para melhorar a captura de lat e long?
![H7](/reports/figures/h7.png)
## 4.2 Problema da Latitude e Longitude
Volto a ressaltar a possibilidade de colocar um rastreador GPS com chip GSM.
A medição Lat e Long possuem erros comprometedores, sendo necessários retirar dos dados. Conforme mapa abaixo:
![Mapa_Lat_Long](/reports/figures/Problema_latlong.PNG)
Foram realizados 2 recortes distintos para facilitar o entendimento de lat e long.
1. O primeiro foi pelo valor absoluto de lat e long inicial e final considerando a dimensionalidade fornecida pelo google maps.
2. O segundo foi pela variação (delta) de lat e long inicial e final. Considerando os valores do calculo de haversine e a distribuição normal dos dados, os cortes estão adequados, mas no próximo ciclo do CRISP deve-se reavaliar os cálculos e decisões perante o lat e long conforme modelo de negócio.

Limitação de latitude e longitude conforme pesquisas e google.
| Latitude | Longitude |
| -------- | --------- |
| 42.2 | -9.5 |
| 38.4 | -6.0 |

Limitação da variação (delta) da latitude e longitude.
| Latitude | Longitude |
| -------- | --------- |
| 0.2 | -0.2 |
| 0.2 | -0.2 |

![Problema_lat_long2](/reports/figures/probl_lat_long2.png)
Com isso pretendo ter uma melhor acertividade inicial do modelo. Em um segundo cíclo do CRISP-DM, posso voltar a retestar esses valores.
# 5. Preparação e seleção dos atributos
Para selecionar as Features, inicialmente tentei com o Boruta, mas devido a limitação computacional não funcionou, então utilizei o método com Random Forest Regressor. Cheguei a este resultado:
![Resultado_features](reports/figures/Caracteristicas%20modelo%201.png)
# 6. Modelos de Machine Learning Utilizados
Fora selecionados 5 algoritmos de regrssão linear para serem treinados, testados e avaliados. Os algotitmos selecionados foram:

Decision Tree Regressor
Gradient Boosting Regressor
KNeighbors Regressor
Linear Regression
Random Forest Regressor
XGBRegressor
# 7. Seleção do Modelo de Machine Learning
Como a base de dados possuia mais de 1.7 milhões de linhas, utilizei a metodologia de Validação Houldout, que consiste em separa os dados em vez de 2 treino e teste, em dataset 3 treino, validação e teste. Essa metodologia tem como objetivo garanti a qualidade do modelo ao realizar mais treinamentos evitando overfetting ou underfetting quando estiver em produção. Foi separado o treino com 60% dos dados, a validação com 20% dos dados e o teste com 20%.

Como era necessário prever 2 valores, lat e long final, foi utilizado o método MultiOuputRegressor vindo da bibliote Sklearn no qual me permite fazer essa predição dupla, no caso lat e long.

Resolvi não utilizar técnicas de cross-validation, nem de finetunning neste primeiro momento. Optei por causa da simplicidade dos dados e no próximo ciclo do CRISP-DM implementar essas técnicas.

Por fim Irei levar em consideração 2 métricas:

R2   = Mais próximo a 1 representa o quão bem minhas features conseguem explicar o modelo.

RMSE = Consigo visualizar a variação da métrica para mais ou para menos na mesma escala. Quanto mais próximo do 0 melhor.
## 7.1. Métricas dos Algoritmos
Com esse método de validação, foram obtidas as seguintes performances finais:

|index | Modelo | R2  | RMSE |
|--- | ----- |  -- | ---- |
|0 | Random Forest Regressor_train | 0.999741  | 0.000477 |
|1 | LinearRegression_test | 0.933698  | 0.007747 |
|2 | LinearRegression_train | 0.550825  | 0.019846 |
|3 | Random Forest Regressor_test | 0.374134  | 0.023722 |

O modelo de Arvore possuiu o melhor R2 no treino, mas um pessimo valor no test, isso pode ser devido a underfitting, então resolvi não utiliza-lo.

O modelo LinearRegression também teve um baixo desempenho, quanto ao R2, com os dados de treino, mas um desempenho muito grande com os dados de teste. Com esses valores muito distintos posso garantir que o modelo linear está com overfitting.

Logo, esse período de 1 semana não foi satisfatório para que eu conseguisse construir um bom modelo.
## 7.2 Escolha do Modelo
Não utilizei, o modelo Random Forest Regressor devido ao peso dele, não foi possível subir ele na máquina virtual do render para coloca-lo em produção.

Decidi treinar o modelo Linear Regression e disponibiliza-lo em produção devido ele ter o melhor desempenho geral. Mantenho minha resalva de que este modelo em produção é somente para teste e não deve ser levado em consideração para utilização do negócio.
# 8.0 Aplicação em Produção
O serviço de predição foi disponibilizado na nuvem do render através do link https://taxi-4ui9.onrender.com com um serviço de API desenvolvido com FLASK.

A aplicação visual para a equipe testar está rodando atualmente no Streamlit. [Neste link](https://taxi-driver.streamlit.app/)

COLOCAR UM PRINT DA APLICAÇÃO EM PRODUÇÃO
# 9.0 Conclusão
Esse projeto foi bem desafiador pois até então não tinha desenvolvido nada parecido em um tempo tão curto. Montar este projeto desde a coleta de dados até coloca-lo em produção requereu um bom planejamento com escolhas claras do que eu poderia ou não fazer no curto período de tempo, logo tive que escolher o que aplicar, quando aplicar e como aplicar.
# 10.0 Próximos Passos de Melhoria
Os pontos de melhoria são:
1. Análise mais aprofundada dos dados de latitude e longitude afim de extrair novas features.
2. Realizar o Fine Tunning e o Cros-Validation duate os treinamentos dos modelos.
3. Melhorar o layout e a interação do usuário final com a plataforma web.
# 11.0 Aprendizado do projeto
O aprendizado do projeto foi descrito no Medium para facilitar a leitura.
# 12.0 Contato Geral
<ul class="actions">
    <table>
        <tr>
            <th><i class="fa-solid fa-folder-tree"></i><a href="https://bit.ly/portfolio-gabriel-nobre"> Portfólio de Projetos</a></th>
            <th><i class="fa-brands fa-linkedin"></i><a href="https://www.linkedin.com/in/gabriel-nobre-galvao/"> Linkedin</a></th>
            <th><i class="fa-brands fa-medium"></i><a href="https://medium.com/@gabrielnobregalvao"> Medium</a></th>
            <th><i class="fa-brands fa-github"></i><a href="https://github.com/Gabrielnbr"> Git Hub</a></th>
            <th><i class="fa-solid fa-envelope"></i><a href="mailto:gabrielnobregalvao@gmail.com"> E-mail</a></th>
        </tr>
    </table>
</ul>
