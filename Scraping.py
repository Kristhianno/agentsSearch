import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool 
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import streamlit as st

# Carregando os ambientes de Rede
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# Criando as instâncias LLMs
llama3 = ChatGroq(api_key=groq_api_key, model="groq/llama3-8b-8192")


# Configurando o Streamlit 
st.title('Agentes AI Scraping')  # Título

# Entrada de dados pelo usuário 
website_url = st.text_input('Insira a URL do site para pesquisa', placeholder ="Ex: https://google.com")
topic = st.text_input('Insira o assunto relacionado à URL para pesquisa', placeholder = 'Ex: Principais concorrentes ')

if st.button('Iniciar Análise'):
    if topic and website_url:
        
        # Criando Ferramentas específicas para os agentes utilizarem
               
        tool = ScrapeWebsiteTool(website_url=website_url)
        
        
        # Criando os Agentes
        
        
        Analista = Agent(
            role='Analista de Dados',
            goal='Analisar os dados que vieram da pesquisa feita da web sobre o {topic} e repassar ao Diretor para tomada de decisão, retorne as mensagens em Português do Brasil',
            tools=[tool],
            backstory="O Analista é um profissional altamente competente e especialista no {topic} com Doutorado em Harvard e mais de 20 anos de experiência,"
                      " capaz de analisar e resolver problemas complexos de maneira simples e objetiva. Retorne as mensagens em Português do Brasil.",
            verbose=True,
            allow_delegation=False,
            llm=llama3
        )
        
        
        Diretor = Agent(
            role='Diretor Comercial',
            goal='Coordenar e tomar decisões precisas sobre o {topic}, retorne as mensagens em Português do Brasil',
            backstory="O Diretor é responsável e assertivo em suas decisões. Retorne as mensagens em Português do Brasil.",
            verbose=True,
            allow_delegation=False,
            llm=llama3
        )

        
        
        tarefa_analista = Task(
            description='Realizar uma análise detalhada e resumir as informações.',
            expected_output='Uma análise avançada do que foi encontrado na pesquisa do {topic}, destacando os principais pontos abordados e as principais hashtags '
                            'desse assunto no momento, e repassar ao Designer. Todas as saídas devem ser em português do Brasil.',
            agent= Analista,
            async_execution=False
        )

        tarefa_diretor = Task(
            description=(
                'Leia toda análise feita pelo Analista e busque informações relevantes na internet que reforcem as análises feitas sobre o assunto do {topic} '
                'e depois pontue em tópicos os principais conceitos abordados do assunto {topic} com sugestões de melhoria. Todas as saídas devem ser em português do Brasil.'
            ),
            expected_output='Uma solução final consolidada para o problema. Todas as saídas devem ser em português do Brasil.',
            agent=Diretor,
            async_execution=False,
        )

        # Construindo a Tripulação
        crew_analise = Crew(
            agents=[Analista, Diretor],
            tasks=[tarefa_analista, tarefa_diretor],
            process=Process.sequential  # Sequential execution
        )

        # Iniciando a análise com Spinner
        with st.spinner('Processando a análise...'):
            resultado_final = crew_analise.kickoff(inputs={'topic': topic})

        # Mostrando o resultado final para o usuário
        st.text_area('Resultado da Pesquisa', value=resultado_final, height=300)

    else:
        st.warning('Por favor, insira uma URL para iniciar a pesquisa e um assunto relacionado.')
