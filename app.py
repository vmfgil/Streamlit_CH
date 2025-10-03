import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
import io
import base64
import time
import random
from datetime import timedelta

# Imports de Process Mining (PM4PY)
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_miner

# Imports para Análise Exploratória (EDA)
import missingno as msno

# --- 1. CONFIGURAÇÃO DA PÁGINA E ESTILO ---
st.set_page_config(
    page_title="Transformação Analítica - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. GESTÃO DE ESTADO E AUTENTICAÇÃO ---

# Usuários fictícios
USERS = {"admin": "admin", "analista": "analise"}

def authenticate(username, password):
    return USERS.get(username) == password

def login_page():
    st.image("logo.png", width=250)
    st.title("Acesso ao Sistema de Análise")
    username = st.text_input("Nome de Usuário")
    password = st.text_input("Senha", type="password")

    if st.button("Entrar", use_container_width=True):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.user_name = username
            st.session_state.current_page = "Dashboard"
            st.rerun()
        else:
            st.error("Nome de usuário ou senha incorretos.")

# --- 3. FUNÇÕES DE CARREGAMENTO E PRÉ-PROCESSAMENTO ---

@st.cache_data
def load_data():
    try:
        # Carregar os 5 CSVs
        df_projects = pd.read_csv('projects.csv')
        df_tasks = pd.read_csv('tasks.csv')
        df_resources = pd.read_csv('resources.csv')
        df_allocations = pd.read_csv('resource_allocations.csv')
        df_dependencies = pd.read_csv('dependencies.csv')
        
        # Conversão de Datas
        df_projects['start_date'] = pd.to_datetime(df_projects['start_date'])
        df_projects['end_date'] = pd.to_datetime(df_projects['end_date'])
        df_tasks['start_date'] = pd.to_datetime(df_tasks['start_date'])
        df_tasks['end_date'] = pd.to_datetime(df_tasks['end_date'])
        df_allocations['allocation_date'] = pd.to_datetime(df_allocations['allocation_date'])
        
        return df_projects, df_tasks, df_resources, df_allocations, df_dependencies
    
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar ficheiro: {e}. Certifique-se de que os 5 CSVs estão na pasta.")
        return None, None, None, None, None

@st.cache_data
def run_pre_mining_analysis(df_projects, df_tasks, df_resources, df_allocations):
    
    # 1. Merge de Dados para Análise Completa
    df_full_context = df_allocations.merge(df_tasks, on=['task_id', 'project_id'], how='left')
    df_full_context = df_full_context.merge(df_projects, on='project_id', how='left')
    df_full_context = df_full_context.merge(df_resources, on='resource_id', how='left')
    
    # 2. Cálculo do Custo Real por Alocação
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']
    
    # 3. Agregação de Custos e Criação de KPIs no nível do Projeto
    project_costs = df_full_context.groupby('project_id')['cost_of_work'].sum().reset_index()
    project_costs.rename(columns={'cost_of_work': 'real_process_cost'}, inplace=True)
    
    df_projects = df_projects.merge(project_costs, on='project_id', how='left')
    
    # 4. Cálculo de Desvios (Performance & Custos)
    df_projects['duration_diff_days'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    df_projects['cost_diff'] = df_projects['real_process_cost'] - df_projects['budget_impact'] # budget_impact = Orçamento (Baseline)
    
    # 5. Cálculo do Esforço Total por Tarefa (Ajuste Cirúrgico AQUI)
    # TAREFAS: Colunas de esforço agregadas (Crédito Habitação) em vez de detalhe (IT)
    df_tasks['estimated_effort_total'] = df_tasks['estimated_effort']
    df_tasks['actual_effort_total'] = df_tasks['actual_effort']
    
    # Cálculo de Variância de Esforço
    df_tasks['effort_variance'] = df_tasks['actual_effort_total'] - df_tasks['estimated_effort_total']
    
    # 6. Extração do Tipo de Projeto (Ajuste Cirúrgico AQUI)
    # PROJETOS: Novo esquema utiliza 'path_name' para categorização
    df_projects['project_type'] = df_projects['path_name']
    
    # 7. Criação do Log de Eventos para Process Mining (PM4PY)
    # Colunas de log para PM4PY
    log_df = df_full_context[[
        'project_id', 
        'task_name', 
        'allocation_date', 
        'resource_name'
    ]].copy()

    # Renomear para o padrão PM4PY
    log_df.rename(columns={
        'project_id': 'case:concept:name',
        'task_name': 'concept:name',
        'allocation_date': 'time:timestamp',
        'resource_name': 'org:resource'
    }, inplace=True)
    
    # Garantir que o log está ordenado
    log_df = log_df.sort_values(by=['case:concept:name', 'time:timestamp'])
    
    event_log_pm4py = pm4py.convert_to_event_log(log_df)
    
    return df_projects, df_tasks, df_resources, df_allocations, df_full_context, event_log_pm4py

# --- 4. FUNÇÕES DE VISUALIZAÇÃO ---

def plot_kpi_card(title, value, unit="", delta=None):
    """Cria um cartão de KPI simples."""
    st.metric(label=title, value=f"{value}{unit}", delta=delta)

def plot_process_mining_viz(event_log, title, visualizer_func, discovery_algo):
    """Gera e exibe visualizações de Process Mining."""
    try:
        st.subheader(title)
        
        # 1. Descoberta (e.g., DFG, Petri Net)
        if visualizer_func == dfg_visualizer:
            dfg = discovery_algo.apply(event_log)
            gviz = visualizer_func.apply(dfg, log=event_log, variant=visualizer_func.Variants.FREQUENCY)
        elif visualizer_func == pn_visualizer:
            process_tree = discovery_algo.apply(event_log)
            net, initial_marking, final_marking = pt_converter.apply(process_tree)
            gviz = visualizer_func.apply(net, initial_marking, final_marking, variant=visualizer_func.Variants.FREQUENCY)
        else:
            return

        # 2. Renderização
        st.graphviz_chart(gviz.source, use_container_width=True)
        
        # 3. Métricas de Conformance (Apenas para Petri Net)
        if visualizer_func == pn_visualizer:
            fitness = replay_fitness_evaluator.apply(event_log, net, initial_marking, final_marking, variant=alignments_miner.Variants.VERSION_STATE_EQUALS_ALWAYS)
            st.info(f"Métricas do Modelo ({discovery_algo.__name__.split('.')[-1]}):")
            col1, col2 = st.columns(2)
            col1.metric("Fitness", f"{fitness['average_trace_fitness']:.2f}", help="Mede quão bem o log de eventos corresponde ao modelo (Net).")
            col2.metric("Simplicidade", f"{simplicity_evaluator.apply(net):.2f}", help="Mede a complexidade estrutural do modelo (Net).")

    except Exception as e:
        st.warning(f"Não foi possível gerar a visualização {title}. (Erro: {e})")

# --- 5. LAYOUT DA PÁGINA (DASHBOARD) ---

def dashboard_page():
    st.title("🏛️ Dashboard de Análise de Processos")

    df_projects, df_tasks, df_resources, df_allocations, df_dependencies = load_data()
    
    if df_projects is None:
        return
        
    df_projects, df_tasks, df_resources, df_allocations, df_full_context, event_log_pm4py = run_pre_mining_analysis(
        df_projects, df_tasks, df_resources, df_allocations
    )
    
    # --- FILTROS ---
    st.sidebar.markdown("## ⚙️ Filtros Analíticos")
    
    project_types = df_projects['project_type'].unique()
    selected_types = st.sidebar.multiselect("Tipo de Processo (Path)", project_types, default=project_types)
    
    resource_types = df_resources['resource_type'].unique()
    selected_resources = st.sidebar.multiselect("Tipo de Recurso", resource_types, default=resource_types)

    # Filtragem de Dados
    df_projects_filtered = df_projects[df_projects['project_type'].isin(selected_types)]
    
    # Filtro de recursos impacta as alocações/custos
    filtered_resource_ids = df_resources[df_resources['resource_type'].isin(selected_resources)]['resource_id'].tolist()
    df_allocations_filtered = df_full_context[
        df_full_context['project_id'].isin(df_projects_filtered['project_id']) &
        df_full_context['resource_id'].isin(filtered_resource_ids)
    ]
    
    # ------------------------------------
    
    # --- 5.1. SECÇÃO 1: KPIs SUMÁRIOS ---
    st.header("Sumário Executivo e KPIs de Performance")
    
    total_projects = len(df_projects_filtered)
    total_cost = df_allocations_filtered['cost_of_work'].sum() / 1000
    avg_duration = df_projects_filtered['total_duration_days'].mean()
    
    # KPI de Desvio de Duração (Baseline: 31 dias)
    projects_on_time = df_projects_filtered[df_projects_filtered['duration_diff_days'] <= 0]
    percent_on_time = (len(projects_on_time) / total_projects) * 100 if total_projects else 0
    
    # KPI de Desvio de Custo
    avg_cost_diff = df_projects_filtered['cost_diff'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plot_kpi_card("Total de Dossiers", total_projects, unit="")
    with col2:
        plot_kpi_card("Custo Total (K€)", f"{total_cost:,.0f}", unit="K€")
    with col3:
        plot_kpi_card("Duração Média (Dias)", f"{avg_duration:.1f}", unit=" dias")
    with col4:
        plot_kpi_card("% Dentro do Prazo", f"{percent_on_time:.1f}", unit="%", delta=f"{avg_cost_diff:,.0f}€ (Desvio Custo Médio)")

    # --- 5.2. SECÇÃO 2: MATRIZ DE PERFORMANCE ---
    st.header("Matriz de Performance (Custo vs. Duração)")
    
    df_scatter = df_projects_filtered.groupby('project_id').agg(
        avg_cost_diff=('cost_diff', 'first'),
        total_duration=('total_duration_days', 'first'),
        status=('project_status', 'first'),
        type=('project_type', 'first')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='total_duration', 
        y='avg_cost_diff', 
        hue='status', 
        size='avg_cost_diff', 
        data=df_scatter, 
        palette={'Aprovado': 'g', 'Recusado': 'r', 'Concluída': 'b'},
        ax=ax
    )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title('Desvio de Custo vs. Duração Total por Dossier')
    ax.set_xlabel('Duração Total (Dias Úteis)')
    ax.set_ylabel('Desvio de Custo (Custo Real - Orçamento) [€]')
    st.pyplot(fig)

    # --- 5.3. SECÇÃO 3: ANÁLISE DE RECURSOS E REWORK ---
    st.header("Análise de Carga de Trabalho e Rework")

    col1, col2 = st.columns(2)
    
    # Gráfico 1: Carga de Trabalho por Recurso
    with col1:
        st.subheader("Carga de Trabalho por Tipo de Recurso")
        df_resource_load = df_allocations_filtered.groupby('resource_type')['hours_worked'].sum().sort_values(ascending=False).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='hours_worked', y='resource_type', data=df_resource_load, ax=ax, palette='viridis')
        ax.set_title('Total de Horas Trabalhadas por Tipo de Recurso')
        ax.set_xlabel('Horas Trabalhadas (H)')
        ax.set_ylabel('')
        st.pyplot(fig)

    # Gráfico 2: Esforço de Rework por Tarefa
    with col2:
        st.subheader("Variância Média de Esforço por Tarefa")
        # Filtra tarefas com rework (actual_effort > estimated_effort)
        df_rework = df_tasks[df_tasks['effort_variance'] > 0]
        df_rework_summary = df_rework.groupby('task_name')['effort_variance'].mean().sort_values(ascending=False).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='effort_variance', y='task_name', data=df_rework_summary.head(10), ax=ax, palette='inferno')
        ax.set_title('Variância Média de Esforço por Tarefa (Rework)')
        ax.set_xlabel('Média do Desvio de Esforço (Dias Úteis)')
        ax.set_ylabel('')
        st.pyplot(fig)

    # --- 5.4. SECÇÃO 4: PROCESS MINING ---
    st.header("Process Mining e Descoberta de Fluxo")

    mining_approach = st.selectbox(
        "Selecione o Algoritmo de Descoberta",
        ["Gráfico de Fluxo de Desempenho (DFG)", "Rede de Petri (Inductive Miner)", "Rede de Petri (Heuristics Miner)"]
    )
    
    # 4.1. Filtrar Log para PM (apenas projetos visíveis no filtro)
    case_ids_filtered = df_projects_filtered['project_id'].astype(str).tolist()
    
    # Filtragem do Log de Eventos
    filtered_log = pm4py.filter_log(lambda trace: trace.attributes['concept:name'] in case_ids_filtered, event_log_pm4py)
    
    if mining_approach == "Gráfico de Fluxo de Desempenho (DFG)":
        plot_process_mining_viz(filtered_log, "Descoberta DFG (Performance)", dfg_visualizer, dfg_discovery)
        
    elif mining_approach == "Rede de Petri (Inductive Miner)":
        plot_process_mining_viz(filtered_log, "Descoberta Rede de Petri (Inductive Miner)", pn_visualizer, inductive_miner)
        
    elif mining_approach == "Rede de Petri (Heuristics Miner)":
        plot_process_mining_viz(filtered_log, "Descoberta Rede de Petri (Heuristics Miner)", pn_visualizer, heuristics_miner)

# --- 6. PÁGINAS ADICIONAIS ---

def rl_page():
    st.title("🤖 Reinforcement Learning e Otimização")
    st.info("Esta secção conteria a interface para treinar e simular modelos de RL para otimização de alocação de recursos e sequenciação de tarefas.")
    st.image("rl_placeholder.png", caption="Placeholder de Gráfico de Recompensa de RL")

def settings_page():
    st.title("⚙️ Configurações e Gestão de Dados")
    st.info("Aqui seria possível gerir os ficheiros, configurar custos/taxas de risco ou ajustar os parâmetros da simulação.")
    st.markdown("### EDA - Verificação de Qualidade dos Dados")
    
    if st.button("Executar EDA"):
        df_projects, df_tasks, df_resources, df_allocations, df_dependencies = load_data()
        if df_projects is not None:
            st.markdown("#### Distribuição de Valores Nulos (Missing Values)")
            
            # Gráfico Missingno para Projetos
            fig_proj = msno.matrix(df_projects, figsize=(10, 4))
            st.pyplot(fig_proj.get_figure())
            st.markdown("---")
            
            # Gráfico Missingno para Tasks
            fig_tasks = msno.matrix(df_tasks, figsize=(10, 4))
            st.pyplot(fig_tasks.get_figure())
            
            st.success("Verificação de EDA concluída. A falta de barras escuras confirma a ausência de valores nulos nos campos essenciais.")


# --- 7. FLUXO PRINCIPAL DA APP ---

if __name__ == '__main__':
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.current_page = "Login"
    
    # Estilos CSS
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            display: flex; flex-direction: column; justify-content: center; align-items: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
    if not st.session_state.authenticated:
        st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            display: flex; flex-direction: column; justify-content: center; align-items: center;
        }
        </style>
        """, unsafe_allow_html=True)
        login_page()
    else:
        with st.sidebar:
            st.markdown(f"### 👤 {st.session_state.get('user_name', 'Admin')}")
            st.markdown("---")
            if st.button("🏠 Dashboard Geral", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.rerun()
            if st.button("🤖 Reinforcement Learning", use_container_width=True):
                st.session_state.current_page = "RL"
                st.rerun()
            if st.button("⚙️ Configurações", use_container_width=True):
                st.session_state.current_page = "Settings"
                st.rerun()
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🚪 Sair", use_container_width=True):
                st.session_state.authenticated = False
                for key in list(st.session_state.keys()):
                    if key not in ['authenticated']: del st.session_state[key]
                st.rerun()
                
        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "RL":
            rl_page()
        elif st.session_state.current_page == "Settings":
            settings_page()
