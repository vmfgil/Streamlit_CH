import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
import io
import base64
import time
import random
from datetime import timedelta
import textwrap
import html # <--- ADICIONADO PARA CORRIGIR O ERRO

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
    page_title="Transformação Inteligente de Processos",
    page_icon="✨",
    layout="wide"
)

# --- ESTILO CSS REFORMULADO (NOVO ESQUEMA DE CORES) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    
    /* Nova Paleta de Cores - Tema Claro e Profissional */
    :root {
        --primary-color: #0d6efd;       /* Azul Principal para botões e links */
        --secondary-color: #6c757d;      /* Cinza para texto secundário e bordas */
        --success-color: #198754;       /* Verde para indicadores positivos */
        --warning-color: #ffc107;       /* Amarelo para alertas e destaques */
        
        --background-color: #f8f9fa;      /* Fundo Principal (Cinza muito claro) */
        --sidebar-background: #ffffff;    /* Fundo da Sidebar (Branco) */
        --card-background-color: #ffffff; /* Fundo dos Cartões (Branco) */
        
        --text-color: #212529;            /* Texto Principal (Quase Preto) */
        --text-muted-color: #6c757d;     /* Texto Secundário (Cinza) */
        --border-color: #dee2e6;          /* Cor da Borda/Separador */
    }
    
    .stApp { background-color: var(--background-color); color: var(--text-color); }
    h1, h2, h3, h4 { color: var(--text-color); font-weight: 600; }
    
    /* Painel Lateral */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-background);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: transparent !important;
        color: var(--text-color) !important;
        border: 1px solid transparent !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #f8f9fa !important; /* Cinza claro no hover */
        border: 1px solid transparent !important;
    }
    
    /* --- CARTÕES (Cards) --- */
    .card {
        background-color: var(--card-background-color);
        color: var(--text-color);
        border-radius: 8px;
        padding: 20px 25px;
        border: 1px solid var(--border-color);
        height: 100%;
        display: flex;
        flex-direction: column;
        margin-bottom: 25px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Sombra suave */
    }
    .card-header { padding-bottom: 10px; border-bottom: 1px solid var(--border-color); }
    .card .card-header h4 {
        color: var(--text-color);
        font-size: 1.1rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px; /* Espaço entre ícone e texto */
    }
    .card .card-header h4 i { /* Estilo para os ícones */
        font-size: 1.2rem;
        color: var(--primary-color);
    }
    .card-body { flex-grow: 1; padding-top: 15px; }
    .dataframe-card-body {
        max-height: 300px;
        overflow-y: auto;
        overflow-x: auto;
        padding: 0;
    }
    
    /* --- CARTÕES DE MÉTRICAS (KPIs) --- */
    [data-testid="stMetric"] {
        background-color: var(--card-background-color);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color); /* Borda de destaque à esquerda */
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetric"] label { color: var(--text-muted-color) !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text-color) !important; }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] svg { display: none; } /* Opcional: Esconde setas padrão */

    /* --- ÍCONE DE TOOLTIP PARA OS CARTÕES --- */
    .card-header {
        position: relative; /* Necessário para posicionar o ícone */
    }
    .tooltip-icon {
        position: absolute;
        top: 15px;
        right: 15px;
        font-size: 1rem;
        color: var(--secondary-color);
        cursor: help; /* Muda o cursor para indicar que é clicável/informativo */
    }
    .tooltip-icon:hover {
        color: var(--primary-color);
    }
    
    /* Botões */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600;
    }

    /* Botão de Análise (cor de destaque) */
    .iniciar-analise-button .stButton>button {
        background-color: var(--warning-color) !important;
        color: var(--text-color) !important;
        border: 2px solid var(--warning-color) !important;
    }

</style>
""", unsafe_allow_html=True)


# --- FUNÇÕES AUXILIARES ---
def convert_fig_to_bytes(fig, format='png'):
    buf = io.BytesIO()
    # Cor de fundo do gráfico (branco, igual ao card)
    fig.patch.set_facecolor('#ffffff') 
    for ax in fig.get_axes():
        # Cor de fundo da área de plotagem
        ax.set_facecolor('#ffffff') 
        
        # Cores do texto e eixos (quase preto)
        ax.tick_params(colors='#212529', which='both') 
        ax.xaxis.label.set_color('#212529')
        ax.yaxis.label.set_color('#212529')
        ax.title.set_color('#212529')
        
        # Cores da legenda
        if ax.get_legend() is not None:
            plt.setp(ax.get_legend().get_texts(), color='#212529')
            ax.get_legend().get_frame().set_facecolor('#ffffff')
            ax.get_legend().get_frame().set_edgecolor('#dee2e6') # Cor da borda
            
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def convert_gviz_to_bytes(gviz, format='png'):
    return io.BytesIO(gviz.pipe(format=format))

# --- FUNÇÃO CORRIGIDA ---
def create_card(title, icon_html, chart_bytes=None, dataframe=None, use_container_width=False, tooltip=None):
    # Gera o HTML do ícone da tooltip, se um texto for fornecido
    tooltip_html = ""
    if tooltip:
        # Usa html.escape para garantir que o texto da tooltip é seguro para HTML e não quebra a string
        safe_tooltip = html.escape(tooltip, quote=True)
        tooltip_html = f'<i class="bi bi-question-circle-fill tooltip-icon" title="{safe_tooltip}"></i>'

    if chart_bytes:
        b64_image = base64.b64encode(chart_bytes.getvalue()).decode()
        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <h4>{icon_html} {title}</h4>
                {tooltip_html}
            </div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}" style="width: 100%; height: auto;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif dataframe is not None:
        df_html = dataframe.to_html(classes=['pandas-df-card'], index=False)
        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <h4>{icon_html} {title}</h4>
                {tooltip_html}
            </div>
            <div class="card-body dataframe-card-body">
                {df_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
if 'current_section' not in st.session_state: st.session_state.current_section = "visao_geral"
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'rl_analysis_run' not in st.session_state: st.session_state.rl_analysis_run = False
if 'rl_params_expanded' not in st.session_state: st.session_state.rl_params_expanded = True
if 'project_id_simulated' not in st.session_state: st.session_state.project_id_simulated = "25"
if 'plots_pre_mining' not in st.session_state: st.session_state.plots_pre_mining = {}
if 'plots_post_mining' not in st.session_state: st.session_state.plots_post_mining = {}
if 'tables_pre_mining' not in st.session_state: st.session_state.tables_pre_mining = {}
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'plots_eda' not in st.session_state: st.session_state.plots_eda = {}
if 'tables_eda' not in st.session_state: st.session_state.tables_eda = {}
if 'plots_rl' not in st.session_state: st.session_state.plots_rl = {}
if 'tables_rl' not in st.session_state: st.session_state.tables_rl = {}
if 'logs_rl' not in st.session_state: st.session_state.logs_rl = {}


# --- FUNÇÕES DE ANÁLISE (PROCESS MINING E EDA) ---
#@st.cache_data
def run_pre_mining_analysis(dfs):
    plots = {}
    tables = {}
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ['project_id', 'task_id', 'resource_id']:
        for df in [df_projects, df_tasks, df_resources, df_resource_allocations, df_dependencies]:
            if col in df.columns: df[col] = df[col].astype(str)

    df_projects['start_date'] = pd.to_datetime(df_projects['start_date'])
    df_projects['end_date'] = pd.to_datetime(df_projects['end_date'])
    df_projects['planned_end_date'] = pd.to_datetime(df_projects['planned_end_date'])
    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['path_name']
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M').astype(str)

    df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id')
    df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']
    
    project_aggregates = df_alloc_costs.groupby('project_id').agg(total_actual_cost=('cost_of_work', 'sum'), num_resources=('resource_id', 'nunique')).reset_index()
    df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    df_projects['total_actual_cost'] = df_projects['total_actual_cost'].fillna(0)
    df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
    df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)
    
    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
    df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']

    log_df_final = df_full_context[['project_id', 'task_name', 'allocation_date', 'resource_name']].copy()
    log_df_final.rename(columns={'project_id': 'case:concept:name', 'task_name': 'concept:name', 'allocation_date': 'time:timestamp', 'resource_name': 'org:resource'}, inplace=True)
    log_df_final['lifecycle:transition'] = 'complete'
    event_log_pm4py = pm4py.convert_to_event_log(log_df_final)
    
    tables['kpi_data'] = {
        'Total de Processos': len(df_projects),
        'Total de Tarefas': len(df_tasks),
        'Total de Recursos': len(df_resources),
        'Duração Média (dias)': f"{df_projects['actual_duration_days'].mean():.1f}"
    }
    tables['outlier_duration'] = df_projects.sort_values('actual_duration_days', ascending=False).head(5)
    tables['outlier_cost'] = df_projects.sort_values('total_actual_cost', ascending=False).head(5)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df_projects, x='days_diff', y='cost_diff', hue='project_type', s=80, alpha=0.7, ax=ax, palette='viridis'); ax.axhline(0, color='#FBBF24', ls='--'); ax.axvline(0, color='#FBBF24', ls='--'); ax.set_title("Matriz de Performance (PM)")
    plots['performance_matrix'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=df_projects['actual_duration_days'], ax=ax, color="#2563EB"); sns.stripplot(x=df_projects['actual_duration_days'], color="#FBBF24", size=4, jitter=True, alpha=0.7, ax=ax); ax.set_title("Distribuição da Duração dos Processos (PM)")
    plots['case_durations_boxplot'] = convert_fig_to_bytes(fig)
    
    lead_times = log_df_final.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"]).reset_index()
    lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
    def compute_avg_throughput(group):
        group = group.sort_values("time:timestamp"); deltas = group["time:timestamp"].diff().dropna()
        return deltas.mean().total_seconds() if not deltas.empty else 0
    throughput_per_case = log_df_final.groupby("case:concept:name").apply(compute_avg_throughput).reset_index(name="avg_throughput_seconds")
    throughput_per_case["avg_throughput_hours"] = throughput_per_case["avg_throughput_seconds"] / 3600
    perf_df = pd.merge(lead_times, throughput_per_case, on="case:concept:name")
    tables['perf_stats'] = perf_df[["lead_time_days", "avg_throughput_hours"]].describe()
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["lead_time_days"], bins=20, kde=True, ax=ax, color="#2563EB"); ax.set_title("Distribuição do Lead Time (dias)")
    plots['lead_time_hist'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(perf_df["avg_throughput_hours"], bins=20, kde=True, color='#06B6D4', ax=ax); ax.set_title("Distribuição do Throughput (horas)")
    plots['throughput_hist'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=perf_df["avg_throughput_hours"], color='#FBBF24', ax=ax); ax.set_title("Boxplot do Throughput (horas)")
    plots['throughput_boxplot'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(x="avg_throughput_hours", y="lead_time_days", data=perf_df, ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Relação Lead Time vs Throughput")
    plots['lead_time_vs_throughput'] = convert_fig_to_bytes(fig)
    
    service_times = df_full_context.groupby('task_name')['hours_worked'].mean().reset_index()
    service_times['service_time_days'] = service_times['hours_worked'] / 8
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), ax=ax, hue='task_name', legend=False, palette='coolwarm'); ax.set_title("Tempo Médio de Execução por Atividade")
    plots['activity_service_times'] = convert_fig_to_bytes(fig)
    
    df_handoff = log_df_final.sort_values(['case:concept:name', 'time:timestamp'])
    df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1)
    df_handoff['handoff_time_days'] = (df_handoff['time:timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600)
    df_handoff['previous_activity'] = df_handoff.groupby('case:concept:name')['concept:name'].shift(1)
    handoff_stats = df_handoff.groupby(['previous_activity', 'concept:name'])['handoff_time_days'].mean().reset_index().sort_values('handoff_time_days', ascending=False)
    handoff_stats['transition'] = handoff_stats['previous_activity'].fillna('') + ' -> ' + handoff_stats['concept:name'].fillna('')
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.head(10), y='transition', x='handoff_time_days', ax=ax, hue='transition', legend=False, palette='viridis'); ax.set_title("Top 10 Handoffs por Tempo de Espera")
    plots['top_handoffs'] = convert_fig_to_bytes(fig)
    
    handoff_stats['estimated_cost_of_wait'] = handoff_stats['handoff_time_days'] * df_projects['cost_per_day'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', ax=ax, hue='transition', legend=False, palette='magma'); ax.set_title("Top 10 Handoffs por Custo de Espera")
    plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)

    activity_counts = df_tasks["task_name"].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='YlGnBu'); ax.set_title("Atividades Mais Frequentes")
    plots['top_activities_plot'] = convert_fig_to_bytes(fig)
    
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), ax=ax, hue='resource_name', legend=False, palette='plasma'); ax.set_title("Top 10 Recursos por Horas Trabalhadas (PM)")
    plots['resource_workload'] = convert_fig_to_bytes(fig)
    
    resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index()
    resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_events_per_case', y='resource_name', data=resource_metrics.sort_values('avg_events_per_case', ascending=False).head(10), ax=ax, hue='resource_name', legend=False, palette='coolwarm'); ax.set_title("Recursos por Média de Tarefas por Processo")
    plots['resource_avg_events'] = convert_fig_to_bytes(fig)
    
    resource_activity_matrix_pivot = df_full_context.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum').fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 8)); sns.heatmap(resource_activity_matrix_pivot, cmap='Blues', annot=True, fmt=".0f", ax=ax, annot_kws={"size": 8}, linewidths=.5, linecolor='#374151'); ax.set_title("Heatmap de Esforço por Top 30 Recursos e Atividade")
    plots['resource_activity_matrix'] = convert_fig_to_bytes(fig)
    
    handoff_counts = Counter((trace[i]['org:resource'], trace[i+1]['org:resource']) for trace in event_log_pm4py for i in range(len(trace) - 1) if 'org:resource' in trace[i] and 'org:resource' in trace[i+1] and trace[i]['org:resource'] != trace[i+1]['org:resource'])
    df_resource_handoffs = pd.DataFrame(handoff_counts.most_common(10), columns=['Handoff', 'Contagem'])
    df_resource_handoffs['Handoff'] = df_resource_handoffs['Handoff'].apply(lambda x: f"{x[0]} -> {x[1]}")
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='Contagem', y='Handoff', data=df_resource_handoffs, ax=ax, hue='Handoff', legend=False, palette='rocket'); ax.set_title("Top 10 Handoffs entre Recursos")
    plots['resource_handoffs'] = convert_fig_to_bytes(fig)
    
    cost_by_resource_type = df_full_context.groupby('resource_type')['cost_of_work'].sum().sort_values(ascending=False).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cost_by_resource_type, x='cost_of_work', y='resource_type', ax=ax, hue='resource_type', legend=False, palette='magma')
    ax.set_title("Custo por Tipo de Recurso")

    # --- NOVO BLOCO DE CÓDIGO PARA FORMATAÇÃO ---
    # Formata o eixo do x para mostrar os números por extenso (ex: €1.500.000)
    formatter = FuncFormatter(lambda x, pos: f'€{x:,.0f}')
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=30, ha='right') # Roda os labels para evitar sobreposição
    fig.tight_layout() # Ajusta o layout para garantir que os labels ficam visíveis
    # --------------------------------------------

    plots['cost_by_resource_type'] = convert_fig_to_bytes(fig)
    
    variants_df = log_df_final.groupby('case:concept:name')['concept:name'].apply(list).reset_index(name='trace')
    variants_df['variant_str'] = variants_df['trace'].apply(lambda x: ' -> '.join(x))
    variant_analysis = variants_df['variant_str'].value_counts().reset_index(name='frequency')
    variant_analysis['percentage'] = (variant_analysis['frequency'] / variant_analysis['frequency'].sum()) * 100
    tables['variants_table'] = variant_analysis.head(10)
    
    # --- CORREÇÃO PARA O GRÁFICO DE FREQUÊNCIA ---
    data_plot_freq = variant_analysis.head(10)
    data_plot_freq['variant_str_wrapped'] = data_plot_freq['variant_str'].apply(
        lambda x: textwrap.fill(x, width=90)
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='frequency', y='variant_str_wrapped', data=data_plot_freq, ax=ax, 
                orient='h', hue='variant_str_wrapped', legend=False, palette='coolwarm')
    
    ax.tick_params(axis='y', labelsize=5)
    ax.set_title("Top 10 Variantes de Processo por Frequência")
    ax.set_xlabel("Frequência (Nº de Casos)")
    ax.set_ylabel("Variante do Processo")
    fig.tight_layout()
    # --- FIM DA CORREÇÃO ---
    
    plots['variants_frequency'] = convert_fig_to_bytes(fig)
    
    # --- INÍCIO DA CORREÇÃO: Lógica de Deteção de Rework Melhorada ---
    rework_loops = Counter()
    for trace in variants_df['trace']:
        # Encontra todas as atividades que se repetem na mesma trace
        seen_activities = defaultdict(list)
        for i, activity in enumerate(trace):
            seen_activities[activity].append(i)
        
        # Para cada atividade que aparece mais de uma vez, regista o loop
        for activity, indices in seen_activities.items():
            if len(indices) > 1:
                # Itera sobre os pares de ocorrências da mesma atividade (ex: a 1ª e a 2ª, a 2ª e a 3ª, etc.)
                for i in range(len(indices) - 1):
                    start_index = indices[i]
                    end_index = indices[i+1]
                    
                    # Ignora self-loops diretos (ex: A -> A), que não são rework
                    if end_index == start_index + 1:
                        continue
    
                    # O loop é a sequência de atividades entre as duas ocorrências (inclusive)
                    loop_sequence = trace[start_index : end_index + 1]
                    
                    # Formata a string do loop para ser legível e adiciona ao contador
                    loop_str = ' -> '.join(loop_sequence)
                    rework_loops[loop_str] += 1
    
    tables['rework_loops_table'] = pd.DataFrame(rework_loops.most_common(10), columns=['rework_loop', 'frequency'])
    # --- FIM DA CORREÇÃO ---

    delayed_projects = df_projects[df_projects['days_diff'] > 0].copy()

    if delayed_projects.empty:
        total_cost_delay = 0.0
        mean_delay_days = 0.0
        mean_cost_per_day = 0.0
    else:
        delayed_projects['total_actual_cost'] = delayed_projects['total_actual_cost'].fillna(0)
        total_cost_delay = delayed_projects['total_actual_cost'].sum()
        mean_delay_days = delayed_projects['days_diff'].mean()
        # evita divisão por zero / infinito
        delayed_projects['safe_days_diff'] = delayed_projects['days_diff'].replace(0, np.nan)
        mean_cost_per_day = (
            (delayed_projects['total_actual_cost'] / delayed_projects['safe_days_diff'])
            .replace([np.inf, -np.inf], np.nan)
            .mean()
        )
    
    # GUARDA NÚMEROS (nota: chave EXACTA com A maiúscula para casar com o dashboard)
    tables['cost_of_delay_kpis'] = {
        'Custo Total Processos Atrasados': total_cost_delay,
        'Atraso Médio (dias)': mean_delay_days,
        'Custo Médio/Dia Atraso': mean_cost_per_day
    }

    min_res, max_res = df_projects['num_resources'].min(), df_projects['num_resources'].max()
    bins = np.linspace(min_res, max_res, 5, dtype=int) if max_res > min_res else [min_res, max_res]
    df_projects['team_size_bin_dynamic'] = pd.cut(df_projects['num_resources'], bins=bins, include_lowest=True, duplicates='drop').astype(str)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_projects.dropna(subset=['team_size_bin_dynamic']), x='team_size_bin_dynamic', y='days_diff', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='flare'); ax.set_title("Impacto do Tamanho da Equipa no Atraso (PM)")
    plots['delay_by_teamsize'] = convert_fig_to_bytes(fig)
    
    median_duration_by_team_size = df_projects.groupby('team_size_bin_dynamic')['actual_duration_days'].median().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=median_duration_by_team_size, x='team_size_bin_dynamic', y='actual_duration_days', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='crest'); ax.set_title("Duração Mediana por Tamanho da Equipa")
    plots['median_duration_by_teamsize'] = convert_fig_to_bytes(fig)
    
    df_alloc_costs['day_of_week'] = df_alloc_costs['allocation_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=df_alloc_costs.groupby('day_of_week')['hours_worked'].sum().reindex(weekday_order).reset_index(), x='day_of_week', y='hours_worked', ax=ax, hue='day_of_week', legend=False, palette='viridis'); ax.set_title("Eficiência Semanal (Horas Trabalhadas)")
    plots['weekly_efficiency'] = convert_fig_to_bytes(fig)
    
    df_tasks_analysis = df_tasks.copy(); df_tasks_analysis['service_time_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis.sort_values(['project_id', 'start_date'], inplace=True); df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
    df_tasks_analysis['waiting_time_days'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds() / (24*60*60)
    df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['waiting_time_days'].apply(lambda x: x if x > 0 else 0)
    df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id', 'resource_name']], on='task_id', how='left').drop_duplicates()
    bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=bottleneck_by_resource, y='resource_name', x='waiting_time_days', ax=ax, hue='resource_name', legend=False, palette='rocket'); ax.set_title("Top 15 Recursos por Tempo Médio de Espera")
    plots['bottleneck_by_resource'] = convert_fig_to_bytes(fig)
    
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5)); bottleneck_by_activity.plot(kind='bar', stacked=True, color=['#2563EB', '#FBBF24'], ax=ax); ax.set_title("Gargalos: Tempo de Serviço vs. Espera")
    plots['service_vs_wait_stacked'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=bottleneck_by_activity.reset_index(), x='service_time_days', y='waiting_time_days', ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Tempo de Espera vs. Tempo de Execução")
    plots['wait_vs_service_scatter'] = convert_fig_to_bytes(fig)
    
    df_wait_over_time = df_tasks_analysis.merge(df_projects[['project_id', 'completion_month']], on='project_id')
    monthly_wait_time = df_wait_over_time.groupby('completion_month')['waiting_time_days'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4)); sns.lineplot(data=monthly_wait_time, x='completion_month', y='waiting_time_days', marker='o', ax=ax, color='#06B6D4'); plt.xticks(rotation=45); ax.set_title("Evolução do Tempo Médio de Espera")
    plots['wait_time_evolution'] = convert_fig_to_bytes(fig)
    
    df_perf_full = perf_df.merge(df_projects, left_on='case:concept:name', right_on='project_id')
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=df_perf_full, x='team_size_bin_dynamic', y='avg_throughput_hours', ax=ax, hue='team_size_bin_dynamic', legend=False, palette='plasma'); ax.set_title("Benchmark de Throughput por Tamanho da Equipa")
    plots['throughput_benchmark_by_teamsize'] = convert_fig_to_bytes(fig)
    
    def get_phase(task_type):
        if task_type in ['Onboarding', 'Validação KYC e Conformidade', 'Análise Documental']:
            return '1. Onboarding, KYC e Documentação'
        elif task_type in ['Análise de Risco e Proposta']:
            return '2. Análise de Risco'
        elif task_type in ['Avaliação da Imóvel']:
            return '3. Avaliação de imóvel'
        elif task_type in ['Decisão de Crédito e Condições']:
            return '4. Decisão de crédito'
        elif task_type in ['Fecho', 'Preparação Legal']:
            return '5. Contratação e Desembolso'
        return task_type
    df_tasks['phase'] = df_tasks['task_type'].apply(get_phase)
    phase_times = df_tasks.groupby(['project_id', 'phase']).agg(start=('start_date', 'min'), end=('end_date', 'max')).reset_index()
    phase_times['cycle_time_days'] = (phase_times['end'] - phase_times['start']).dt.days
    avg_cycle_time_by_phase = phase_times.groupby('phase')['cycle_time_days'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 4)); avg_cycle_time_by_phase.plot(kind='bar', color=sns.color_palette('tab10'), ax=ax); ax.set_title("Duração Média por Fase do Processo"); plt.xticks(rotation=45,ha='right')
    plots['cycle_time_breakdown'] = convert_fig_to_bytes(fig)
    
    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

#@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    plots = {}
    metrics = {}
    
    # 1. Preparação de um log com ciclo de vida completo (start/complete) para certas análises
    df_start_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'start_date']].rename(columns={'start_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_start_events['lifecycle:transition'] = 'start'
    df_complete_events = _df_tasks_raw[['project_id', 'task_id', 'task_name', 'end_date']].rename(columns={'end_date': 'time:timestamp', 'task_name': 'concept:name', 'project_id': 'case:concept:name'})
    df_complete_events['lifecycle:transition'] = 'complete'
    log_df_full_lifecycle = pd.concat([df_start_events, df_complete_events]).sort_values('time:timestamp')
    log_full_pm4py = pm4py.convert_to_event_log(log_df_full_lifecycle)

    # 2. Descoberta de Modelos e Métricas (usando uma amostra das 3 variantes principais para performance)
    variants_dict = variants_filter.get_variants(_event_log_pm4py)
    top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    top_variant_names = [v[0] for v in top_variants_list]
    log_top_3_variants = variants_filter.apply(_event_log_pm4py, top_variant_names)
    
    pt_inductive = inductive_miner.apply(log_top_3_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)
    
    def plot_metrics_chart(metrics_dict, title):
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor'])
        fig, ax = plt.subplots(figsize=(8, 4)); barplot = sns.barplot(data=df_metrics, x='Métrica', y='Valor', ax=ax, hue='Métrica', legend=False, palette='coolwarm')
        for p in barplot.patches: ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', color='#E5E7EB')
        ax.set_title(title); ax.set_ylim(0, 1.05); return fig
        
    metrics_im = {"Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im), "Generalização": generalization_evaluator.apply(log_top_3_variants, net_im, im_im, fm_im), "Simplicidade": simplicity_evaluator.apply(net_im)}
    plots['metrics_inductive'] = convert_fig_to_bytes(plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)'))
    metrics['inductive_miner'] = metrics_im

    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_3_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm)
    
    metrics_hm = {"Fitness": replay_fitness_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm), "Generalização": generalization_evaluator.apply(log_top_3_variants, net_hm, im_hm, fm_hm), "Simplicidade": simplicity_evaluator.apply(net_hm)}
    plots['metrics_heuristic'] = convert_fig_to_bytes(plot_metrics_chart(metrics_hm, 'Métricas de Qualidade (Heuristics Miner)'))
    metrics['heuristics_miner'] = metrics_hm
    
    # 3. Análises de Performance Rápidas (correm com todos os dados)
    kpi_temporal = _df_projects.groupby('completion_month').agg(avg_lead_time=('actual_duration_days', 'mean'), throughput=('project_id', 'count')).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 6)); ax1.plot(kpi_temporal['completion_month'], kpi_temporal['avg_lead_time'], marker='o', color='#2563EB', label='Lead Time'); ax2 = ax1.twinx(); ax2.bar(kpi_temporal['completion_month'], kpi_temporal['throughput'], color='#06B6D4', alpha=0.6, label='Throughput'); fig.suptitle('Séries Temporais de KPIs de Performance')
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9)); ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.label.set_color('#2563EB'); ax2.yaxis.label.set_color('#06B6D4'); ax1.tick_params(axis='y', colors='#2563EB'); ax2.tick_params(axis='y', colors='#06B6D4')
    plots['kpi_time_series'] = convert_fig_to_bytes(fig)
    
    dfg_perf, _, _ = pm4py.discover_performance_dfg(log_full_pm4py)
    gviz_dfg = dfg_visualizer.apply(dfg_perf, log=log_full_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['performance_heatmap'] = convert_gviz_to_bytes(gviz_dfg)
    
    fig, ax = plt.subplots(figsize=(8, 4)); log_df_full_lifecycle['weekday'] = log_df_full_lifecycle['time:timestamp'].dt.day_name(); weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = log_df_full_lifecycle.groupby('weekday')['case:concept:name'].count().reindex(weekday_order).fillna(0); sns.barplot(x=heatmap_data.index, y=heatmap_data.values, ax=ax, hue=heatmap_data.index, legend=False, palette='coolwarm'); ax.set_title('Ocorrências de Atividades por Dia da Semana'); plt.xticks(rotation=45)
    plots['temporal_heatmap_fixed'] = convert_fig_to_bytes(fig)

    # 4. Otimização do Gantt Chart (usa amostra de 50 se os dados forem grandes)
    if len(_df_projects) > 50:
        ids_amostra_gantt = _df_projects['project_id'].sample(n=50, random_state=42).tolist()
        df_projects_gantt = _df_projects[_df_projects['project_id'].isin(ids_amostra_gantt)]
        df_tasks_gantt = _df_tasks_raw[_df_tasks_raw['project_id'].isin(ids_amostra_gantt)]
        gantt_title = 'Linha do Tempo de 50 processos (Amostra)'
    else:
        df_projects_gantt = _df_projects
        df_tasks_gantt = _df_tasks_raw
        gantt_title = 'Linha do Tempo de Todos os processos (Gantt Chart)'

    fig_gantt, ax_gantt = plt.subplots(figsize=(20, max(10, len(df_projects_gantt) * 0.4)))
    all_projects = df_projects_gantt.sort_values('start_date')['project_id'].tolist()
    gantt_data = df_tasks_gantt[df_tasks_gantt['project_id'].isin(all_projects)].sort_values(['project_id', 'start_date'])
    project_y_map = {proj_id: i for i, proj_id in enumerate(all_projects)}
    if not gantt_data.empty:
        color_map = {task_name: plt.get_cmap('tab10', gantt_data['task_name'].nunique())(i) for i, task_name in enumerate(gantt_data['task_name'].unique())}
        for _, task in gantt_data.iterrows():
            ax_gantt.barh(project_y_map[task['project_id']], (task['end_date'] - task['start_date']).days + 1, left=task['start_date'], height=0.6, color=color_map.get(task['task_name']), edgecolor='#E5E7EB')
        handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in color_map]
        ax_gantt.legend(handles, color_map.keys(), title='Tipo de Tarefa', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_gantt.set_yticks(list(project_y_map.values())); ax_gantt.set_yticklabels([f"Processo {pid}" for pid in project_y_map.keys()]); ax_gantt.invert_yaxis(); ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); plt.xticks(rotation=45)
    ax_gantt.set_title(gantt_title); fig_gantt.tight_layout()
    plots['gantt_chart_all_projects'] = convert_fig_to_bytes(fig_gantt)
    
    # 5. Análises de Recursos (correm com todos os dados)
    log_df_complete = pm4py.convert_to_dataframe(_event_log_pm4py)
    handover_edges = Counter((log_df_complete.iloc[i]['org:resource'], log_df_complete.iloc[i+1]['org:resource']) for i in range(len(log_df_complete)-1) if log_df_complete.iloc[i]['case:concept:name'] == log_df_complete.iloc[i+1]['case:concept:name'] and log_df_complete.iloc[i]['org:resource'] != log_df_complete.iloc[i+1]['org:resource'])
    
    # 1. CRIAÇÃO DO GRAFO (G) e FIGURA: Estas linhas têm de estar sempre aqui
    fig_net, ax_net = plt.subplots(figsize=(18, 12)); 
    G = nx.DiGraph();
    for (source, target), weight in handover_edges.items(): G.add_edge(str(source), str(target), weight=weight)

        # Filtrar o grafo para mostrar apenas os nós mais relevantes
    recursos_importantes = {"ExCo", "Comité de Crédito", "Diretor de Risco"}
    node_degrees = dict(G.degree())
    recursos_ordenados = sorted(node_degrees, key=node_degrees.get, reverse=True)
    top_recursos = set(recursos_ordenados[:30])
    nos_para_manter = top_recursos.union(recursos_importantes)
    G_filtrado = G.subgraph(nos_para_manter).copy()
    
    # Desenhar o grafo filtrado
    if G_filtrado.nodes():
        pos = nx.spring_layout(G_filtrado, k=0.8, iterations=50, seed=42)
        weights = [G_filtrado[u][v]['weight'] for u, v in G_filtrado.edges()]
    
        nx.draw(G_filtrado, pos, with_labels=True, node_color='#0d6efd', node_size=2500,
                edgecolors='#dee2e6', linewidths=1, edge_color='#6c757d',
                width=[w * 0.2 for w in weights], ax=ax_net, font_size=9,
                font_color='black', alpha=1.0, connectionstyle='arc3,rad=0.1',
                labels={node: str(node).replace(" ", "\n") for node in G_filtrado.nodes()})
    
        nx.draw_networkx_edge_labels(G_filtrado, pos, edge_labels=nx.get_edge_attributes(G_filtrado, 'weight'),
                                     ax=ax_net, font_color='#dc3545', font_size=8,
                                     bbox={'facecolor': 'white', 'alpha': 0.6, 'edgecolor': 'none'})
    
        ax_net.set_title('Rede Social de Recursos (Top 30 por Conexões + Principais)')
        plots['resource_network_adv'] = convert_fig_to_bytes(fig_net)
    

    
    if 'skill_level' in _df_resources.columns:
        perf_recursos = _df_full_context.groupby('resource_id').agg(total_hours=('hours_worked', 'sum'), total_tasks=('task_id', 'nunique')).reset_index()
        perf_recursos['avg_hours_per_task'] = perf_recursos['total_hours'] / perf_recursos['total_tasks']
        perf_recursos = perf_recursos.merge(_df_resources[['resource_id', 'skill_level', 'resource_name']], on='resource_id')
        fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=perf_recursos, x='skill_level', y='avg_hours_per_task', ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Relação entre Skill e Performance")
        plots['skill_vs_performance_adv'] = convert_fig_to_bytes(fig)
        
        resource_role_counts = _df_full_context.groupby(['resource_name', 'skill_level']).size().reset_index(name='count')

        # Filtrar o dataframe antes de construir o grafo
        recursos_importantes = {"ExCo", "Comité de Crédito", "Diretor de Risco"}
        recursos_ordenados_df = resource_role_counts.sort_values('count', ascending=False)
        top_30_recursos = set(recursos_ordenados_df['resource_name'].head(30))
        recursos_para_manter = top_30_recursos.union(recursos_importantes)
        df_filtrado = resource_role_counts[resource_role_counts['resource_name'].isin(recursos_para_manter)]
        
        # Construir o grafo bipartido a partir dos dados JÁ FILTRADOS
        G_bipartite = nx.Graph()
        resources_nodes = df_filtrado['resource_name'].unique()
        roles_nodes = df_filtrado['skill_level'].unique()
        
        G_bipartite.add_nodes_from(resources_nodes, bipartite=0)
        G_bipartite.add_nodes_from(roles_nodes, bipartite=1)
        for _, row in df_filtrado.iterrows():
            G_bipartite.add_edge(row['resource_name'], row['skill_level'], weight=row['count'])
        
        # Desenhar o grafo filtrado
        fig, ax = plt.subplots(figsize=(12, max(8, len(resources_nodes) * 0.3))) # Altura dinâmica
        pos = nx.bipartite_layout(G_bipartite, resources_nodes)
        
        nx.draw(G_bipartite, pos, with_labels=True,
                node_color=['#0d6efd' if node in resources_nodes else '#ffc107' for node in G_bipartite.nodes()],
                node_size=2500, ax=ax, font_size=9, edge_color='#dee2e6', font_color='black',
                labels={node: str(node).replace(" ", "\n") for node in G_bipartite.nodes()})
        
        edge_labels = nx.get_edge_attributes(G_bipartite, 'weight')
        nx.draw_networkx_edge_labels(G_bipartite, pos, edge_labels=edge_labels, ax=ax, font_color='#dc3545', font_size=8)
        ax.set_title('Rede de Top 30 Recursos por Função (Atividade + Principais)')
        plots['resource_network_bipartite'] = convert_fig_to_bytes(fig)

    variants_df = log_df_full_lifecycle.groupby('case:concept:name').agg(variant=('concept:name', lambda x: tuple(x)), start_timestamp=('time:timestamp', 'min'), end_timestamp=('time:timestamp', 'max')).reset_index()
    variants_df['duration_hours'] = (variants_df['end_timestamp'] - variants_df['start_timestamp']).dt.total_seconds() / 3600
    variant_durations = variants_df.groupby('variant').agg(count=('case:concept:name', 'count'), avg_duration_hours=('duration_hours', 'mean')).reset_index().sort_values(by='count', ascending=False).head(10)
    
    # --- CORREÇÃO FINAL PARA O GRÁFICO DE DURAÇÃO ---
    variant_durations['variant_str_wrapped'] = variant_durations['variant'].apply(
        lambda x: textwrap.fill(' -> '.join(map(str, x)), width=90) # Aumentado para 90
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.barplot(x='avg_duration_hours', y='variant_str_wrapped',
                data=variant_durations.astype({'avg_duration_hours':'float'}),
                ax=ax, hue='variant_str_wrapped', legend=False, palette='plasma')
    
    # Diminuir ainda mais o tamanho da fonte
    ax.tick_params(axis='y', labelsize=5) 
    
    ax.set_title('Duração Média das 10 Variantes Mais Comuns')
    ax.set_ylabel('Variante do Processo')
    ax.set_xlabel('Duração Média (horas)')
    fig.tight_layout()
    plots['variant_duration_plot'] = convert_fig_to_bytes(fig)
    # --- FIM DA CORREÇÃO FINAL ---
    
    ax.set_title('Duração Média das 10 Variantes Mais Comuns')
    ax.set_ylabel('Variante do Processo')
    ax.set_xlabel('Duração Média (horas)')
    fig.tight_layout()
    plots['variant_duration_plot'] = convert_fig_to_bytes(fig)
    
    # 6. ABORDAGEM DEFINITIVA PARA ANÁLISE DE ALINHAMENTOS (usa amostra de 50 se os dados forem grandes)
    log_df_para_alinhar = pm4py.convert_to_dataframe(log_full_pm4py)
    num_cases = log_df_para_alinhar['case:concept:name'].nunique()
    
    log_para_alinhar_obj = log_full_pm4py
    align_title_suffix = ''

    if num_cases > 50:
        case_ids_todos = log_df_para_alinhar['case:concept:name'].unique().tolist()
        case_ids_amostra = random.sample(case_ids_todos, 50)
        df_amostra = log_df_para_alinhar[log_df_para_alinhar['case:concept:name'].isin(case_ids_amostra)]
        log_para_alinhar_obj = pm4py.convert_to_event_log(df_amostra)
        align_title_suffix = ' (Amostra de 50 Casos)'

    aligned_traces = alignments_miner.apply(log_para_alinhar_obj, net_im, im_im, fm_im)
    deviations_list = [{'fitness': trace['fitness'], 'deviations': sum(1 for move in trace['alignment'] if '>>' in move[0] or '>>' in move[1])} for trace in aligned_traces if 'fitness' in trace]
    
    if deviations_list:
        deviations_df = pd.DataFrame(deviations_list)
        fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(x='fitness', y='deviations', data=deviations_df, alpha=0.6, ax=ax, color='#FBBF24'); ax.set_title(f'Fitness vs. Desvios{align_title_suffix}'); fig.tight_layout()
        plots['deviation_scatter_plot'] = convert_fig_to_bytes(fig)

    case_fitness_data = [{'project_id': str(trace.attributes['concept:name']), 'fitness': alignment['fitness']} for trace, alignment in zip(log_para_alinhar_obj, aligned_traces) if 'concept:name' in trace.attributes and 'fitness' in alignment]
    
    if case_fitness_data:
        case_fitness_df = pd.DataFrame(case_fitness_data).merge(_df_projects[['project_id', 'end_date']], on='project_id')
        case_fitness_df['end_month'] = case_fitness_df['end_date'].dt.to_period('M').astype(str)
        monthly_fitness = case_fitness_df.groupby('end_month')['fitness'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=monthly_fitness, x='end_month', y='fitness', marker='o', ax=ax, color='#2563EB'); ax.set_title(f'Score de Conformidade ao Longo do Tempo{align_title_suffix}'); ax.set_ylim(0, 1.05); ax.tick_params(axis='x', rotation=45); fig.tight_layout()
        plots['conformance_over_time_plot'] = convert_fig_to_bytes(fig)

    # 7. Análises Finais (correm com todos os dados)
    kpi_daily = _df_projects.groupby(_df_projects['end_date'].dt.date).agg(avg_cost_per_day=('cost_per_day', 'mean')).reset_index()
    kpi_daily.rename(columns={'end_date': 'completion_date'}, inplace=True)
    kpi_daily['completion_date'] = pd.to_datetime(kpi_daily['completion_date'])
    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=kpi_daily, x='completion_date', y='avg_cost_per_day', ax=ax, color='#FBBF24'); ax.set_title('Custo Médio por Dia ao Longo do Tempo'); fig.tight_layout()
    plots['cost_per_day_time_series'] = convert_fig_to_bytes(fig)

    df_projects_sorted = _df_projects.sort_values(by='end_date'); df_projects_sorted['cumulative_throughput'] = range(1, len(df_projects_sorted) + 1)
    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(x='end_date', y='cumulative_throughput', data=df_projects_sorted, ax=ax, color='#06B6D4'); ax.set_title('Gráfico Acumulado de Throughput'); fig.tight_layout()
    plots['cumulative_throughput_plot'] = convert_fig_to_bytes(fig)
    
    def generate_custom_variants_plot(event_log):
        variants = variants_filter.get_variants(event_log)
        top_variants = sorted(variants.items(), key=lambda item: len(item[1]), reverse=True)[:10]
        variant_sequences = {f"V{i+1} ({len(v)} casos)": [str(a) for a in k] for i, (k, v) in enumerate(top_variants)}
        fig, ax = plt.subplots(figsize=(12, 6)) 
        all_activities = sorted(list(set([act for seq in variant_sequences.values() for act in seq])))
        if all_activities:
            activity_to_y = {activity: i for i, activity in enumerate(all_activities)}
            colors = plt.cm.get_cmap('tab10', len(variant_sequences.keys()))
            for i, (variant_name, sequence) in enumerate(variant_sequences.items()):
                ax.plot(range(len(sequence)), [activity_to_y[activity] for activity in sequence], marker='o', linestyle='-', label=variant_name, color=colors(i))
            ax.set_yticks(list(activity_to_y.values()))
            ax.set_yticklabels(list(activity_to_y.keys()))
            ax.set_title('Sequência de Atividades das 10 Variantes Mais Comuns')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
        return fig
    plots['custom_variants_sequence_plot'] = convert_fig_to_bytes(generate_custom_variants_plot(log_full_pm4py))
    
    milestones = ['Onboarding/Recolha de Dados', 'Análise de Risco e Proposta', 'Decisão de Crédito e Condições', 'Fecho/Desembolso']
    df_milestones = _df_tasks_raw[_df_tasks_raw['task_name'].isin(milestones)].copy()
    milestone_pairs = []
    if not df_milestones.empty:
        for project_id, group in df_milestones.groupby('project_id'):
            sorted_tasks = group.sort_values('start_date')
            for i in range(len(sorted_tasks) - 1):
                duration = (sorted_tasks.iloc[i+1]['start_date'] - sorted_tasks.iloc[i]['end_date']).total_seconds() / 3600
                if duration >= 0: milestone_pairs.append({'transition': f"{sorted_tasks.iloc[i]['task_name']} -> {sorted_tasks.iloc[i+1]['task_name']}", 'duration_hours': duration})
    df_milestone_pairs = pd.DataFrame(milestone_pairs)
    if not df_milestone_pairs.empty:
        fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=df_milestone_pairs, x='duration_hours', y='transition', ax=ax, orient='h', hue='transition', legend=False, palette='coolwarm'); ax.set_title('Análise de Tempo entre Marcos do Processo'); fig.tight_layout()
        plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)

    df_tasks_sorted = _df_tasks_raw.sort_values(['project_id', 'start_date']); df_tasks_sorted['previous_end_date'] = df_tasks_sorted.groupby('project_id')['end_date'].shift(1)
    df_tasks_sorted['waiting_time_days'] = (df_tasks_sorted['start_date'] - df_tasks_sorted['previous_end_date']).dt.total_seconds() / (24 * 3600)
    df_tasks_sorted.loc[df_tasks_sorted['waiting_time_days'] < 0, 'waiting_time_days'] = 0
    df_tasks_sorted['previous_task_name'] = df_tasks_sorted.groupby('project_id')['task_name'].shift(1)
    waiting_times_matrix = df_tasks_sorted.pivot_table(index='previous_task_name', columns='task_name', values='waiting_time_days', aggfunc='mean').fillna(0)
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(waiting_times_matrix * 24, cmap='Blues', annot=True, fmt='.1f', ax=ax, annot_kws={"size": 8}, linewidths=.5, linecolor='#374151'); ax.set_title('Matriz de Tempo de Espera entre Atividades (horas)'); fig.tight_layout()
    plots['waiting_time_matrix_plot'] = convert_fig_to_bytes(fig)
    
    resource_efficiency = _df_full_context.groupby('resource_name').agg(
        total_hours_worked=('hours_worked', 'sum'),
        total_tasks_completed=('task_name', 'count')
    ).reset_index()
    resource_efficiency['avg_hours_per_task'] = resource_efficiency['total_hours_worked'] / resource_efficiency['total_tasks_completed']
    
    # --- INÍCIO DA NOVA LÓGICA: TOP 10 MELHORES E PIORES ---
    # Garantir que temos pelo menos 20 recursos para mostrar
    if len(resource_efficiency) > 20:
        # Piores: os que demoram MAIS horas por tarefa
        piores = resource_efficiency.sort_values(by='avg_hours_per_task', ascending=False).head(10)
        piores['Desempenho'] = '10 Piores (Mais Horas/Tarefa)'
        
        # Melhores: os que demoram MENOS horas por tarefa
        melhores = resource_efficiency.sort_values(by='avg_hours_per_task', ascending=True).head(10)
        melhores['Desempenho'] = '10 Melhores (Menos Horas/Tarefa)'
        
        # Combinar os dois dataframes
        data_para_plot = pd.concat([piores, melhores]).sort_values(by='avg_hours_per_task', ascending=False)
        
        # Definir uma altura fixa, pois agora temos sempre 20 itens
        altura_figura = 10
        titulo_grafico = 'Métricas de Eficiência: Top 10 Melhores e Piores Recursos'
    else:
        # Se houver 20 ou menos recursos, mostrar todos
        data_para_plot = resource_efficiency.sort_values(by='avg_hours_per_task', ascending=False)
        data_para_plot['Desempenho'] = 'Recursos'
        altura_figura = max(6, len(data_para_plot) * 0.4)
        titulo_grafico = 'Métricas de Eficiência Individual por Recurso'
    
    fig, ax = plt.subplots(figsize=(10, altura_figura))
    sns.barplot(data=data_para_plot, x='avg_hours_per_task', y='resource_name',
                hue='Desempenho', palette={'10 Piores (Mais Horas/Tarefa)': '#dc3545', '10 Melhores (Menos Horas/Tarefa)': '#198754', 'Recursos': '#0d6efd'},
                ax=ax, dodge=False) # dodge=False para não separar as barras
    
    ax.set_title(titulo_grafico)
    ax.set_xlabel("Média de Horas por Tarefa")
    ax.set_ylabel("Recurso")
    fig.tight_layout()
    plots['resource_efficiency_plot'] = convert_fig_to_bytes(fig)

    df_tasks_sorted['sojourn_time_hours'] = df_tasks_sorted['waiting_time_days'] * 24
    waiting_time_by_task = df_tasks_sorted.groupby('task_name')['sojourn_time_hours'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=waiting_time_by_task.sort_values(by='sojourn_time_hours', ascending=False), x='sojourn_time_hours', y='task_name', ax=ax, hue='task_name', legend=False, palette='magma'); ax.set_title('Tempo Médio de Espera por Atividade'); fig.tight_layout()
    plots['avg_waiting_time_by_activity_plot'] = convert_fig_to_bytes(fig)

    return plots, metrics
# --- NOVA FUNÇÃO DE ANÁLISE (EDA) ---
#@st.cache_data
def run_eda_analysis(dfs):
    plots = {}
    tables = {}
    
    # --- Pré-processamento e Feature Engineering (da célula 6 do notebook) ---
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = dfs['resources'].copy()
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- Normalizar project_id e remover duplicados por project_id para evitar contagens infladas ---
    if 'project_id' in df_projects.columns:
        # força project_id para string em todos os dataframes relevantes
        df_projects['project_id'] = df_projects['project_id'].astype(str)
    for tmp_df in [df_tasks, df_resource_allocations, df_dependencies]:
        if 'project_id' in tmp_df.columns:
            tmp_df['project_id'] = tmp_df['project_id'].astype(str)
    
    # Remove duplicados por project_id em df_projects mantendo a primeira ocorrência
    df_projects = df_projects.drop_duplicates(subset=['project_id']).reset_index(drop=True)
    
    # --- Calcular days_diff de forma robusta (mantém NaN para inspeção) ---
    # garante parsing consistente (usa o parsing que aplicaste acima na função; se trocaste dayfirst, fica consistente)
    df_projects['end_date'] = pd.to_datetime(df_projects['end_date'], errors='coerce')
    df_projects['planned_end_date'] = pd.to_datetime(df_projects['planned_end_date'], errors='coerce')
    
    # calcula diferença em dias; preserva NaN quando faltar data
    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    
    # coluna indicadora de validade (para usar nas visualizações e evitar mascarar NaNs)
    df_projects['days_diff_valid'] = df_projects['days_diff'].notna()
    
    # opcional: coluna auxiliar com módulo do desvio (para detectar outliers sem os mascarar)
    df_projects['days_diff_abs'] = df_projects['days_diff'].abs()
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['path_name']
    df_tasks['task_duration_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.days
    df_projects['completion_quarter'] = df_projects['end_date'].dt.to_period('Q')
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M')

    df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id')
    df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']

    # --- CORREÇÃO DEFINITIVA ---
    # Este bloco, que existe na outra função, estava em falta aqui.
    project_aggregates = df_alloc_costs.groupby('project_id').agg(
        total_actual_cost=('cost_of_work', 'sum'),
        avg_hourly_rate=('cost_per_hour', 'mean'),
        num_resources=('resource_id', 'nunique')
    ).reset_index()
    
    df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    df_projects['total_actual_cost'] = df_projects['total_actual_cost'].fillna(0)
    # --- FIM DA CORREÇÃO ---

    # --- NOVA CORREÇÃO: Restaurar a coluna 'dependency_count' ---
    dep_counts = df_dependencies.groupby('project_id').size().reset_index(name='dependency_count')
    df_projects = df_projects.merge(dep_counts, on='project_id', how='left')
    df_projects['dependency_count'] = df_projects['dependency_count'].fillna(0)
    # --- FIM DA NOVA CORREÇÃO ---
    
    df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
    df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)

    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
    df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']

    # ... (linha que cria df_full_context)
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']
    
    # --- INÍCIO DA NOVA LÓGICA DE COMPLEXIDADE ---
    # 1. Mapear o Risco para um valor numérico
    risk_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    df_full_context['risk_score'] = df_full_context['risk_rating'].map(risk_map)
    
    # 2. Calcular o número de funções únicas e o risco por projeto
    project_complexity_metrics = df_full_context.groupby('project_id').agg(
        num_unique_roles=('resource_type', 'nunique'),
        risk_score=('risk_score', 'first') # O risco é o mesmo para todo o projeto
    ).reset_index()
    
    # 3. Criar o novo Índice de Complexidade (Risco * Nº de Funções)
    project_complexity_metrics['complexity_score'] = project_complexity_metrics['num_unique_roles'] * project_complexity_metrics['risk_score']
    
    # 4. Juntar o novo índice ao dataframe principal de projetos
    df_projects = df_projects.merge(project_complexity_metrics[['project_id', 'complexity_score']], on='project_id', how='left')
    df_projects['complexity_score'] = df_projects['complexity_score'].fillna(0)
    # --- FIM DA NOVA LÓGICA DE COMPLEXIDADE ---   
    tables['stats_table'] = df_projects[['actual_duration_days', 'days_diff', 'budget_impact', 'total_actual_cost', 'cost_diff', 'num_resources', 'avg_hourly_rate', 'complexity_score']].describe().round(2)
    

    # --- Geração dos Gráficos (da célula 6 do notebook) ---
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_projects, x='project_status', ax=ax, palette='viridis'); ax.set_title('Distribuição do Status dos Processos')
    plots['plot_01'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_projects, x='days_diff', kde=True, color='salmon', ax=ax); ax.set_title('Diferença entre Data Real e Planeada')
    plots['plot_03'] = convert_fig_to_bytes(fig)

    #Custo Real vs. Orçamento por Processo (plot_04) ---
    df_projects_sorted = df_projects.sort_values('budget_impact', ascending=False)
    
    # Regra: Se houver mais de 50 processos, mostrar apenas os 50 maiores por orçamento
    if len(df_projects_sorted) > 50:
        data_para_plot = df_projects_sorted.head(50)
        titulo_grafico = 'Custo Real vs. Orçamento (Top 50 Processos por Orçamento)'
    else:
        data_para_plot = df_projects_sorted
        titulo_grafico = 'Custo Real vs. Orçamento por Processo'
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(data=data_para_plot, x='project_name', y='budget_impact', color='lightblue', label='Orçamento', ax=ax)
    sns.barplot(data=data_para_plot, x='project_name', y='total_actual_cost', color='salmon', alpha=0.8, label='Custo Real', ax=ax)
    
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    ax.set_title(titulo_grafico)
    fig.tight_layout() # Adicionado para garantir que os labels do eixo X não são cortados
    plots['plot_04'] = convert_fig_to_bytes(fig)
    
    df_projects_q = df_projects.dropna(subset=['completion_quarter']).copy()
    df_projects_q['completion_quarter'] = df_projects_q['completion_quarter'].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(data=df_projects_q, x='completion_quarter', y='days_diff', ax=ax, palette='coolwarm'); ax.set_title('Performance de Prazos por Trimestre')
    plots['plot_05'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_projects_q.groupby('completion_quarter')['total_actual_cost'].mean().reset_index(), x='completion_quarter', y='total_actual_cost', ax=ax, palette='viridis'); ax.set_title('Custo Médio dos Processos por Trimestre')
    plots['plot_06'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_projects_q.groupby('completion_quarter')['num_resources'].mean().reset_index(), x='completion_quarter', y='num_resources', ax=ax, palette='crest'); ax.set_title('Nº Médio de Recursos por Processo a Cada Trimestre')
    plots['plot_07'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_tasks, y='task_type', order=df_tasks['task_type'].value_counts().index, ax=ax, palette='crest'); ax.set_title('Distribuição de Tarefas por Tipo')
    plots['plot_08'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_tasks, x='priority', ax=ax, palette='magma'); ax.set_title('Distribuição de Tarefas por Prioridade')
    plots['plot_09'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_tasks, x='task_duration_days', kde=True, color='indigo', ax=ax); ax.set_title('Distribuição da Duração das Tarefas')
    plots['plot_10'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_tasks.sort_values('task_duration_days', ascending=False).head(10), x='task_duration_days', y='task_name', ax=ax, palette='rocket'); ax.set_title('Top 10 Tarefas Específicas Mais Demoradas')
    plots['plot_11'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.countplot(data=df_resources, x='resource_type', ax=ax, palette='cubehelix'); ax.set_title('Distribuição de Recursos por Tipo')
    plt.xticks(rotation=45, ha='right')
    plots['plot_12'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(data=df_full_context.groupby('resource_name')['days_diff'].mean().sort_values(ascending=False).reset_index().head(20), y='resource_name', x='days_diff', ax=ax, palette='cividis'); ax.set_title('Top 20 Recursos com Maior Atraso Médio (dias)')
    plots['plot_14'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_projects, x='cost_per_day', kde=True, color='teal', ax=ax); ax.set_title('Distribuição do Custo por Dia (Eficiência)')
    plots['plot_16'] = convert_fig_to_bytes(fig)
    
    df_projects['budget_bin'] = pd.cut(df_projects['budget_impact'], bins=4)
    data_plot_17 = df_full_context.merge(df_projects[['project_id', 'budget_bin']], on='project_id').groupby(['budget_bin', 'resource_type'], observed=False)['cost_of_work'].sum().unstack()
    fig = data_plot_17.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 8)).get_figure()
    plots['plot_17'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='total_actual_cost', y='days_diff', color='crimson', ax=ax); ax.set_title('Custo Real vs. Atraso')
    plots['plot_18'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='avg_hourly_rate', y='days_diff', color='olivedrab', ax=ax); ax.set_title('Rate Horário Médio vs. Atraso')
    plots['plot_19'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='num_resources', y='total_actual_cost', color='darkorange', ax=ax); ax.set_title('Nº de Recursos vs. Custo Total')
    plots['plot_20'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.boxplot(x=df_projects['budget_bin'], y=df_projects['days_diff'], ax=ax, palette='pastel'); ax.set_title('Atraso por Faixa de Orçamento')
    plots['plot_22'] = convert_fig_to_bytes(fig)
    
    df_skill_delay = df_full_context[['skill_level', 'project_id', 'days_diff']].drop_duplicates().dropna()
    if not df_skill_delay.empty:
        fig, ax = plt.subplots(figsize=(10, 6)); sns.violinplot(data=df_skill_delay, x='skill_level', y='days_diff', ax=ax, palette='muted'); ax.set_title('Atraso por Nível de Competência')
        plots['plot_23'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_projects, x='complexity_score', kde=True, color='darkslateblue', ax=ax); ax.set_title('Distribuição da Complexidade dos Processos')
    plots['plot_24'] = convert_fig_to_bytes(fig)
    
    # ======= Normalizar tipos antes do merge (corrige ValueError int64 vs object) =======
    # Garante que task_id em df_tasks e task_id_predecessor / task_id_successor em df_dependencies têm o mesmo dtype (string)
    if 'task_id' in df_tasks.columns:
        df_tasks['task_id'] = df_tasks['task_id'].astype(str)
    if 'task_id_predecessor' in df_dependencies.columns:
        df_dependencies['task_id_predecessor'] = df_dependencies['task_id_predecessor'].astype(str)
    if 'task_id_successor' in df_dependencies.columns:
        df_dependencies['task_id_successor'] = df_dependencies['task_id_successor'].astype(str)
    
    # Remove linhas inválidas/NaN nas colunas de ligação para evitar merges indesejados
    df_dependencies = df_dependencies.dropna(subset=[c for c in ['task_id_predecessor', 'task_id_successor'] if c in df_dependencies.columns])
    
    # Agora os merges funcionam sem erro de dtype
    predecessor_counts = df_dependencies.merge(df_tasks, left_on='task_id_predecessor', right_on='task_id', how='left')['task_type'].value_counts()
    successor_counts = df_dependencies.merge(df_tasks, left_on='task_id_successor', right_on='task_id', how='left')['task_type'].value_counts()
    # ====================================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)); predecessor_counts.plot(kind='bar', color=sns.color_palette("Paired")[1], title='Mais Comuns como Predecessoras', ax=ax1); successor_counts.plot(kind='bar', color=sns.color_palette("Paired")[3], title='Mais Comuns como Sucessoras', ax=ax2);
    plots['plot_25'] = convert_fig_to_bytes(fig)

    PROJECT_ID_EXAMPLE = "25"
    project_deps = df_dependencies[df_dependencies['project_id'] == PROJECT_ID_EXAMPLE]
    if not project_deps.empty:
        G = nx.from_pandas_edgelist(project_deps, 'task_id_predecessor', 'task_id_successor', create_using=nx.DiGraph()); pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(14, 9)); nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, ax=ax); ax.set_title(f'Gráfico de Dependências: Processo {PROJECT_ID_EXAMPLE}')
        plots['plot_26'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='complexity_score', y='days_diff', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax); ax.set_title('Relação entre Complexidade e Atraso')
    plots['plot_27'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='dependency_count', y='cost_diff', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax); ax.set_title('Relação entre Dependências e Desvio de Custo')
    plots['plot_28'] = convert_fig_to_bytes(fig)

    df_numeric = df_full_context[['budget_impact', 'total_actual_cost', 'days_diff', 'skill_level', 'cost_per_hour', 'priority']].dropna()
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax); ax.set_title('Matriz de Correlação')
    plots['plot_29'] = convert_fig_to_bytes(fig)
    
    monthly_kpis = df_projects.groupby('completion_month').agg(mean_days_diff=('days_diff', 'mean'), mean_cost_diff=('cost_diff', 'mean'), completed_projects=('project_id', 'count'), mean_duration=('actual_duration_days', 'mean')).reset_index()
    monthly_kpis['completion_month'] = monthly_kpis['completion_month'].astype(str)
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10));
    ax1.plot(monthly_kpis['completion_month'], monthly_kpis['mean_days_diff'], marker='o', color='royalblue',linewidth=3); ax1.set_title('Atraso Médio Mensal'); ax1.grid(True)
    ax2.plot(monthly_kpis['completion_month'], monthly_kpis['mean_cost_diff'], marker='o', color='firebrick',linewidth=3); ax2.set_title('Desvio de Custo Médio Mensal'); ax2.grid(True)
    plots['plot_30'] = convert_fig_to_bytes(fig)
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10));
    ax1.bar(monthly_kpis['completion_month'], monthly_kpis['completed_projects'], color='seagreen'); ax1.set_title('Nº de Processos Concluídos por Mês'); ax1.grid(True)
    ax2.plot(monthly_kpis['completion_month'], monthly_kpis['mean_duration'], marker='o', color='#0d6efd',linewidth=3); ax2.set_title('Duração Média dos Processos Concluídos'); ax2.grid(True)
    plots['plot_31'] = convert_fig_to_bytes(fig)

    return plots, tables

# --- NOVA FUNÇÃO DE ANÁLISE (REINFORCEMENT LEARNING) ---
#@st.cache_data # Removido para permitir interatividade e barra de progresso
def run_rl_analysis(dfs, project_id_to_simulate, num_episodes, reward_config, progress_bar, status_text, agent_params=None):
    if agent_params is None:
        agent_params = {}    
    dfs = {key: df.copy() for key, df in dfs.items()}
    
    # --- PASSO 1 (CORREÇÃO): CONVERTER TODAS AS DATAS NOS DADOS ORIGINAIS PRIMEIRO ---
    for df_name in ['projects', 'tasks', 'resource_allocations', 'dependencies']:
        df = dfs[df_name]
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    # -------------------------------------------------------------------------------------

    # --- PASSO 2: CALCULAR O CUSTO REAL NO CONJUNTO DE DADOS COMPLETO ---
    # garantir tipos consistentes antes de agregações
    for id_col in ['project_id', 'task_id', 'resource_id']:
        for dname in ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']:
            if dname in dfs and id_col in dfs[dname].columns:
                dfs[dname][id_col] = dfs[dname][id_col].astype(str)
    
    # garantir colunas numericas nos recursos e alocações
    if 'resources' in dfs:
        for col in ['cost_per_hour', 'daily_capacity']:
            if col in dfs['resources'].columns:
                dfs['resources'][col] = pd.to_numeric(dfs['resources'][col], errors='coerce').fillna(0)
    if 'resource_allocations' in dfs:
        for col in ['hours_worked']:
            if col in dfs['resource_allocations'].columns:
                dfs['resource_allocations'][col] = pd.to_numeric(dfs['resource_allocations'][col], errors='coerce').fillna(0)
    
    # recalcula custos reais de forma segura e numérica
    df_real_costs = (
        dfs['resource_allocations']
        .merge(dfs['resources'][['resource_id', 'cost_per_hour']], on='resource_id', how='left')
    )
    df_real_costs['cost_per_hour'] = pd.to_numeric(df_real_costs['cost_per_hour'], errors='coerce').fillna(0)
    df_real_costs['hours_worked'] = pd.to_numeric(df_real_costs['hours_worked'], errors='coerce').fillna(0)
    df_real_costs['cost'] = df_real_costs['hours_worked'] * df_real_costs['cost_per_hour']
    df_real_costs = df_real_costs.groupby('project_id', dropna=False)['cost'].sum().reset_index().rename(columns={'cost': 'total_actual_cost'})
    
    dfs['projects'] = dfs['projects'].merge(df_real_costs, on='project_id', how='left')
    dfs['projects']['total_actual_cost'] = pd.to_numeric(dfs['projects'].get('total_actual_cost', pd.Series(dtype=float)), errors='coerce').fillna(0)

    
    # --- PASSO 3: CRIAR A AMOSTRA ---
    st.info("A componente de RL irá correr numa amostra de 500 processos para garantir a performance.")
    ids_amostra = st.session_state['rl_sample_ids']
    dfs_rl = {}
    ids_amostra = [str(i) for i in ids_amostra]  # garante tipo string
    dfs_rl = {}
    for nome_df, df in dfs.items():
        if 'project_id' in df.columns:
            dfs_rl[nome_df] = df[df['project_id'].astype(str).isin(ids_amostra)].copy()
        else:
            dfs_rl[nome_df] = df.copy()
            
    plots = {}
    tables = {}
    logs = {}

    df_projects = dfs_rl['projects'].copy()
    df_tasks = dfs_rl['tasks'].copy()
    df_resources = dfs_rl['resources'].copy()
    df_resource_allocations = dfs_rl['resource_allocations'].copy()
    df_dependencies = dfs_rl['dependencies'].copy()


    def calculate_business_days(start, end):
        return np.busday_count(start.date(), end.date()) if pd.notna(start) and pd.notna(end) else 0

    df_projects['planned_duration_days'] = df_projects.apply(lambda row: calculate_business_days(row['start_date'], row['planned_end_date']), axis=1)
    df_projects['total_duration_days'] = df_projects.apply(lambda row: calculate_business_days(row['start_date'], row['end_date']), axis=1)
    
    # --- AMBIENTE E AGENTE (CLASSES) --- (VERSÃO PORTFÓLIO)
    class PortfolioManagementEnv:
        def __init__(self, dfs, reward_config):
            self.rewards = reward_config

            # 1. Carregar e pré-processar TODOS os dados de uma vez
            self.df_projects = dfs['projects'].sort_values('start_date').reset_index(drop=True)
            self.df_resources = dfs['resources']
            self.df_dependencies = dfs['dependencies']
            
            # 2. Criar dicionários para acesso rápido (MUITO mais rápido que filtrar DataFrames em loop)
            self.tasks_by_project = {pid: df_group.to_dict('records') for pid, df_group in dfs['tasks'].groupby('project_id')}
            self.dependencies_by_project = {pid: {str(row['task_id_successor']): str(row['task_id_predecessor']) for _, row in df_group.iterrows()} for pid, df_group in self.df_dependencies.groupby('project_id')}
            
            self.resource_types = sorted(self.df_resources['resource_type'].unique().tolist())
            self.resources_by_type = {rt: self.df_resources[self.df_resources['resource_type'] == rt] for rt in self.resource_types}
            self.resource_capacity_map = pd.Series(self.df_resources.daily_capacity.values, index=self.df_resources.resource_id).to_dict()
            self.TASK_TYPE_RESOURCE_MAP = {
                'Onboarding': ['Analista Comercial', 'Gerente Comercial'],
                'Validação KYC e Conformidade': ['Analista de Risco', 'Analista Operações/Legal'],
                'Análise Documental': ['Analista Comercial', 'Gerente Comercial'],
                'Análise de Risco e Proposta': ['Analista de Risco'],
                'Avaliação da Imóvel': ['Avaliador Externo'],
                'Preparação Legal': ['Analista Operações/Legal'],
                'Fecho': ['Analista Operações/Legal']
            }
            self.RISK_ESCALATION_MAP = {'A': ['Analista de Risco'], 'B': ['Analista de Risco', 'Diretor de Risco'],'C': ['Analista de Risco', 'Diretor de Risco', 'Comité de Crédito'],'D': ['Analista de Risco', 'Diretor de Risco', 'Comité de Crédito', 'ExCo']}
        
        def reset(self):
            # 3. Reset GERAL da simulação
            self.current_date = self.df_projects['start_date'].min()
            self.resource_calendar = {} # Calendário de ocupação de recursos (essencial para a contenção)
            
            self.future_projects = self.df_projects.to_dict('records')
            self.active_projects = {} # Dicionário para guardar o estado de cada projeto ativo
            self.completed_projects = {} # Para guardar os resultados finais
            self.episode_logs = []
            self.detailed_logs = {} # Para guardar o progresso diário de cada projeto
            
            return self.get_state()

        def get_state(self):
            # 4. O estado agora descreve o PORTFÓLIO, não um projeto
            num_active = len(self.active_projects)
            pending_tasks = sum(1 for proj in self.active_projects.values() for task in proj['tasks'].values() if task['status'] == 'Pendente' and self._is_task_eligible(task, proj['risk_rating'], proj['tasks'], proj['dependencies']))
            high_prio_tasks = sum(1 for proj in self.active_projects.values() for task in proj['tasks'].values() if task['status'] == 'Pendente' and task['priority'] >= 4 and self._is_task_eligible(task, proj['risk_rating'], proj['tasks'], proj['dependencies']))
            
            # O dia da semana é um bom indicador para o agente
            day_of_week = self.current_date.weekday()
            
            return (num_active, pending_tasks, high_prio_tasks, day_of_week)

        def _is_task_eligible(self, task, risk_rating, all_project_tasks, project_dependencies, check_resource_type=None):
            if task['status'] == 'Concluída': return False

            pred_id = project_dependencies.get(str(task['task_id']))
            if pred_id and all_project_tasks.get(pred_id, {}).get('status') != 'Concluída':
                return False
            
            # Se a verificação for para um tipo de recurso específico
            if check_resource_type:
                task_type = task['task_type']
                if task_type == 'Decisão de Crédito e Condições':
                    required_resources = self.RISK_ESCALATION_MAP.get(risk_rating, [])
                    return check_resource_type in required_resources
                else:
                    allowed_resources = self.TASK_TYPE_RESOURCE_MAP.get(task_type, [])
                    return check_resource_type in allowed_resources
            
            return True # Retorna True se não estiver a verificar um recurso específico (elegibilidade geral)

        def get_possible_actions_for_state(self):
            # 5. Procura tarefas elegíveis em TODOS os projetos ativos
            possible_actions = set()
            for proj_id, proj_state in self.active_projects.items():
                for task_id, task_data in proj_state['tasks'].items():
                    if task_data['status'] == 'Pendente':
                        # Verifica para cada tipo de recurso se a tarefa é elegível
                        for res_type in self.resource_types:
                             if self._is_task_eligible(task_data, proj_state['risk_rating'], proj_state['tasks'], proj_state['dependencies'], check_resource_type=res_type):
                                 # A ação agora inclui o ID do projeto e da tarefa para ser única
                                 possible_actions.add((res_type, task_data['task_type'], proj_id, task_id))
            
            # Adicionar sempre a opção 'idle' para cada tipo de recurso
            for res_type in self.resource_types:
                possible_actions.add((res_type, 'idle', None, None))
                
            return list(possible_actions)

        def step(self, action_list):
            # 6. O 'step' agora representa um DIA de trabalho para a empresa inteira
            if self.current_date.weekday() >= 5: # Pular fim de semana
                self.current_date += timedelta(days=1)
                return self.get_state(), 0, False 

            # 6a. Ativar novos projetos que começam hoje
            projects_to_activate = [p for p in self.future_projects if p['start_date'] == self.current_date]
            for proj_data in projects_to_activate:
                proj_id = proj_data['project_id']
                self.active_projects[proj_id] = {
                    'tasks': {str(t['task_id']): {'status': 'Pendente', 'progress': 0.0, 'estimated_effort': t['estimated_effort'] * 8, 'priority': t['priority'], 'task_type': t['task_type'], 'task_id': str(t['task_id'])} for t in self.tasks_by_project.get(proj_id, [])},
                    'dependencies': self.dependencies_by_project.get(proj_id, {}),
                    'risk_rating': proj_data['risk_rating'],
                    'current_cost': 0.0,
                    'start_date': self.current_date
                }
            self.future_projects = [p for p in self.future_projects if p['start_date'] != self.current_date]

            # 6b. Processar ações do agente e alocar recursos
            daily_cost = 0
            reward_from_tasks = 0
            resources_hours_today = defaultdict(float)
            daily_cost_by_project = defaultdict(float) # Necessário para os logs detalhados
            daily_hours_by_project = defaultdict(float) # Necessário para os logs detalhados

            for res_type, task_type, proj_id, task_id in action_list:
                if task_type == "idle":
                    reward_from_tasks -= self.rewards['idle_penalty']
                    continue

                # CÓDIGO CORRIGIDO E OTIMIZADO
                pool = self.resources_by_type[res_type]
                
                # Identificar recursos do pool que ainda têm capacidade hoje
                ids_in_pool = pool['resource_id'].tolist()
                available_ids = [rid for rid in ids_in_pool if resources_hours_today[rid] < self.resource_capacity_map.get(rid, 8)]

                if not available_ids: continue
                
                # Escolher um recurso aleatório dos que estão disponíveis
                chosen_res_id = random.choice(available_ids)
                res_info = pool[pool['resource_id'] == chosen_res_id].iloc[0]

                # Avançar o trabalho na tarefa escolhida
                task_data = self.active_projects[proj_id]['tasks'][task_id]
                if task_data['status'] == 'Concluída': continue
                if task_data['status'] == 'Pendente': task_data['status'] = 'Em Andamento'
                
                remaining_effort = task_data['estimated_effort'] - task_data['progress']
                capacity_today = self.resource_capacity_map.get(chosen_res_id, 8)
                workable_hours_today = capacity_today - resources_hours_today[chosen_res_id]
                
                hours_to_work = min(workable_hours_today, remaining_effort)
                if hours_to_work <= 0: continue

                cost_today = hours_to_work * float(res_info['cost_per_hour'])
                self.active_projects[proj_id]['current_cost'] += cost_today
                task_data['progress'] += hours_to_work
                resources_hours_today[chosen_res_id] += hours_to_work

                # Para os logs detalhados
                daily_cost_by_project[proj_id] += cost_today
                daily_hours_by_project[proj_id] += hours_to_work

                if task_data['progress'] >= task_data['estimated_effort']:
                    task_data['status'] = 'Concluída'
                    task_data['completion_date'] = self.current_date
                    reward_from_tasks += task_data['priority'] * self.rewards['priority_task_bonus_factor']
                    
            # NOVO BLOCO PARA LOGGING DETALHADO
            for proj_id, proj_state in self.active_projects.items():
                if proj_id not in self.detailed_logs:
                    self.detailed_logs[proj_id] = []
                
                # Obter o último estado de progresso ou começar do zero
                last_log = self.detailed_logs[proj_id][-1] if self.detailed_logs[proj_id] else {'cumulative_cost': 0.0, 'cumulative_hours': 0.0}
                
                self.detailed_logs[proj_id].append({
                    'date': self.current_date,
                    'day': (self.current_date - proj_state['start_date']).days,
                    'cumulative_cost': last_log['cumulative_cost'] + daily_cost_by_project[proj_id],
                    'cumulative_hours': last_log['cumulative_hours'] + daily_hours_by_project[proj_id]
                })


            # 6c. Atualizar estados dos projetos e calcular recompensas de conclusão
            projects_to_remove = []
            for proj_id, proj_state in self.active_projects.items():
                if all(t['status'] == 'Concluída' for t in proj_state['tasks'].values()):
                    projects_to_remove.append(proj_id)
                    
                    # Calcular métricas finais e recompensa de conclusão
                    original_proj_info = self.df_projects[self.df_projects['project_id'] == proj_id].iloc[0]
                    final_duration = (self.current_date - proj_state['start_date']).days
                    real_duration = original_proj_info['total_duration_days']
                    
                    time_diff = real_duration - final_duration
                    reward_from_tasks += self.rewards['completion_base']
                    reward_from_tasks += time_diff * self.rewards['per_day_early_bonus'] if time_diff >= 0 else time_diff * self.rewards['per_day_late_penalty']
                    
                    # Guardar resultados
                    self.completed_projects[proj_id] = {'simulated_duration': final_duration, 'simulated_cost': proj_state['current_cost']}

            for proj_id in projects_to_remove:
                del self.active_projects[proj_id]

            # 6d. Finalizar o dia
            self.current_date += timedelta(days=1)
            total_reward = reward_from_tasks - self.rewards['daily_time_penalty']
            
            done = not self.active_projects and not self.future_projects
            
            return self.get_state(), total_reward, done
            
    class QLearningAgent:
        def __init__(self, actions, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
            self.actions = actions; self.action_to_index = {action: i for i, action in enumerate(actions)}; self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
            self.lr, self.gamma, self.epsilon = lr, gamma, epsilon; self.epsilon_decay, self.min_epsilon = epsilon_decay, min_epsilon
            self.epsilon_history, self.episode_rewards, self.episode_durations, self.episode_costs = [], [], [], []
        def choose_action(self, state, possible_actions):
            if not possible_actions: return None
            if random.uniform(0, 1) < self.epsilon: return random.choice(possible_actions)
            else:
                q_values = {action: self.q_table[state][self.action_to_index[action]] for action in possible_actions if action in self.action_to_index}
                if not q_values: return random.choice(possible_actions)
                max_q = max(q_values.values()); best_actions = sorted([action for action, q_val in q_values.items() if q_val == max_q]); return best_actions[0]
        def update_q_table(self, state, action, reward, next_state):
            action_index = self.action_to_index.get(action);
            if action_index is None: return
            old_value = self.q_table[state][action_index]; next_max = np.max(self.q_table[next_state]) if next_state in self.q_table else 0.0
            new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value); self.q_table[state][action_index] = new_value
        def decay_epsilon(self): self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay); self.epsilon_history.append(self.epsilon)
    
    
    # --- INÍCIO DO NOVO BLOCO ---

    # --- Lógica de Simulação de Portfólio ---
    
    # 1. Instanciar o novo ambiente de portfólio.
    # Usamos a amostra completa de projetos para simular o portfólio.
    env = PortfolioManagementEnv(dfs={'projects': df_projects, 'tasks': df_tasks, 'resources': df_resources, 'dependencies': df_dependencies}, reward_config=reward_config)

    # 2. Simplificar as ações para o agente.
    # O agente aprende uma política geral (ex: 'para um Analista de Risco, é melhor focar em Análise de Risco').
    # Isto mantém a Q-Table com um tamanho gerenciável.
    simplified_actions = set()
    for res_type in env.resource_types:
        simplified_actions.add((res_type, 'idle'))
        for task_type in df_tasks['task_type'].unique():
            simplified_actions.add((res_type, task_type))
    
    # 3. Extrair parâmetros e instanciar o Agente.
    lr = float(agent_params.get('lr', 0.1)); gamma = float(agent_params.get('gamma', 0.9))
    epsilon = float(agent_params.get('epsilon', 1.0)); epsilon_decay = float(agent_params.get('epsilon_decay', 0.9995))
    min_epsilon = float(agent_params.get('min_epsilon', 0.01))

    agent = QLearningAgent(actions=tuple(sorted(list(simplified_actions))), lr=lr, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
    
    # 4. NOVO CICLO DE TREINO
    # Cada episódio é uma simulação completa de TODO o portfólio de projetos.
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward, done = 0, False
        
        while not done:
            possible_actions_full = env.get_possible_actions_for_state()
            action_list_for_step = []

            # Para cada tipo de recurso, o agente decide qual o tipo de tarefa a priorizar.
            for res_type in env.resource_types:
                simplified_options = list(set([(a[0], a[1]) for a in possible_actions_full if a[0] == res_type]))
                if not simplified_options: continue

                num_resources_of_type = len(env.resources_by_type.get(res_type, []))

                # Tomar uma decisão para cada recurso disponível daquele tipo.
                for _ in range(num_resources_of_type):
                    chosen_simplified_action = agent.choose_action(state, simplified_options)
                    if not chosen_simplified_action or chosen_simplified_action[1] == 'idle':
                        continue
                    
                    # Encontrar a tarefa real mais prioritária que corresponde à decisão do agente.
                    candidate_tasks = [a for a in possible_actions_full if (a[0], a[1]) == chosen_simplified_action]
                    if not candidate_tasks: continue

                    best_task_action = max(candidate_tasks, key=lambda a: env.active_projects[a[2]]['tasks'][a[3]]['priority'])
                    
                    action_list_for_step.append(best_task_action)
                    # Remover a tarefa escolhida da lista de possibilidades para não ser alocada duas vezes no mesmo dia.
                    possible_actions_full = [a for a in possible_actions_full if a[3] != best_task_action[3]]

            next_state, reward, done = env.step(action_list_for_step)
            
            # Atualizar a Q-table com base nas ações simplificadas que foram tomadas.
            for action in action_list_for_step:
                simplified_action = (action[0], action[1])
                # A recompensa é distribuída pelas ações que a geraram.
                reward_per_action = reward / len(action_list_for_step) if action_list_for_step else reward
                agent.update_q_table(state, simplified_action, reward_per_action, next_state)

            state = next_state
            episode_reward += reward

            # Safety break para evitar loops infinitos.
            if env.current_date > env.df_projects['start_date'].max() + timedelta(days=1000):
                print("Safety break: Simulação excedeu o tempo limite.")
                break

        agent.decay_epsilon()
        # As métricas de duração e custo agora referem-se ao desempenho do portfólio inteiro.
        total_duration = sum(p['simulated_duration'] for p in env.completed_projects.values())
        total_cost = sum(p['simulated_cost'] for p in env.completed_projects.values())
        agent.episode_rewards.append(episode_reward)
        agent.episode_durations.append(total_duration)
        agent.episode_costs.append(total_cost)
        
        progress = (episode + 1) / num_episodes
        progress_bar.progress(progress)
        status_text.info(f"A treinar... Episódio {episode + 1}/{num_episodes}. Data final da simulação: {env.current_date.strftime('%Y-%m-%d')}")

    status_text.success("Treino concluído!")
    status_text.info("A preparar os gráficos e análises finais...")
    
    # --- INÍCIO DO BLOCO DE CÓDIGO FINAL ---

    # 5. Geração de Gráficos de Treino
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # ... (O código para gerar os 4 gráficos de treino permanece o mesmo, por isso não o vou repetir aqui, apenas a sua continuação)
    rewards, durations, costs, epsilon_history = agent.episode_rewards, agent.episode_durations, agent.episode_costs, agent.epsilon_history
    rolling_avg_window = 50
    axes[0, 0].plot(rewards, alpha=0.3, label='Recompensa do Episódio'); axes[0, 0].plot(pd.Series(rewards).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)'); axes[0, 0].set_title('Recompensa por Episódio'); axes[0, 0].set_xlabel('Episódio'); axes[0, 0].set_ylabel('Recompensa Total'); axes[0, 0].legend(); axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].plot(durations, alpha=0.3, label='Duração do Portfólio'); axes[0, 1].plot(pd.Series(durations).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)'); axes[0, 1].set_title('Duração Total do Portfólio'); axes[0, 1].set_xlabel('Episódio'); axes[0, 1].set_ylabel('Soma das Durações (dias)'); axes[0, 1].legend(); axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].plot(epsilon_history, color='green'); axes[1, 0].set_title('Decaimento do Epsilon (Exploração)'); axes[1, 0].set_xlabel('Episódio'); axes[1, 0].set_ylabel('Valor de Epsilon'); axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].plot(costs, alpha=0.3, label='Custo do Portfólio'); axes[1, 1].plot(pd.Series(costs).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)'); axes[1, 1].set_title('Custo Total do Portfólio'); axes[1, 1].set_xlabel('Episódio'); axes[1, 1].set_ylabel('Custo Total (€)'); axes[1, 1].legend(); axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    plots['training_metrics'] = convert_fig_to_bytes(fig)

    # 6. AVALIAÇÃO OTIMIZADA FINAL
    status_text.info("A correr a simulação final otimizada para avaliação...")
    agent.epsilon = 0
    state = env.reset()
    done = False
    while not done:
        possible_actions_full = env.get_possible_actions_for_state()
        action_list_for_step = []
        for res_type in env.resource_types:
            simplified_options = list(set([(a[0], a[1]) for a in possible_actions_full if a[0] == res_type]))
            if not simplified_options: continue
            num_resources_of_type = len(env.resources_by_type.get(res_type, []))
            for _ in range(num_resources_of_type):
                chosen_simplified_action = agent.choose_action(state, simplified_options)
                if not chosen_simplified_action or chosen_simplified_action[1] == 'idle': continue
                candidate_tasks = [a for a in possible_actions_full if (a[0], a[1]) == chosen_simplified_action]
                if not candidate_tasks: continue
                best_task_action = max(candidate_tasks, key=lambda a: env.active_projects[a[2]]['tasks'][a[3]]['priority'])
                action_list_for_step.append(best_task_action)
                possible_actions_full = [a for a in possible_actions_full if a[3] != best_task_action[3]]
        next_state, _, done = env.step(action_list_for_step)
        state = next_state
        if env.current_date > env.df_projects['start_date'].max() + timedelta(days=1000): break

    simulated_results = env.completed_projects
    results_list = []
    for proj_id, sim_data in simulated_results.items():
        real_data = df_projects[df_projects['project_id'] == proj_id].iloc[0]
        results_list.append({
            'project_id': proj_id, 'simulated_duration': sim_data['simulated_duration'], 'simulated_cost': sim_data['simulated_cost'],
            'real_duration': real_data['total_duration_days'], 'real_cost': real_data['total_actual_cost']
        })
    test_results_df = pd.DataFrame(results_list)

    if test_results_df.empty:
        st.error("A simulação não concluiu nenhum projeto. Verifique os parâmetros.")
        return plots, tables, logs

    # 7. Geração de Gráficos de COMPARAÇÃO DE PORTFÓLIO
    df_plot_test = test_results_df.sort_values(by='real_duration').reset_index(drop=True).head(50)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)); index_test = np.arange(len(df_plot_test)); bar_width = 0.35
    axes[0].bar(index_test - bar_width/2, df_plot_test['real_duration'], bar_width, label='Real', color='orangered'); axes[0].bar(index_test + bar_width/2, df_plot_test['simulated_duration'], bar_width, label='Simulado (RL)', color='dodgerblue'); axes[0].set_title('Duração do Processo (Amostra de Teste)'); axes[0].set_xlabel('ID do Processo'); axes[0].set_ylabel('Duração (dias)'); axes[0].set_xticks(index_test); axes[0].set_xticklabels(df_plot_test['project_id'], rotation=90, ha="right"); axes[0].legend(); axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].bar(index_test - bar_width/2, df_plot_test['real_cost'], bar_width, label='Real', color='orangered'); axes[1].bar(index_test + bar_width/2, df_plot_test['simulated_cost'], bar_width, label='Simulado (RL)', color='dodgerblue'); axes[1].set_title('Custo do Processo (Amostra de Teste)'); axes[1].set_xlabel('ID do Processo'); axes[1].set_ylabel('Custo (€)'); axes[1].set_xticks(index_test); axes[1].set_xticklabels(df_plot_test['project_id'], rotation=90, ha="right"); axes[1].legend(); axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    plots['evaluation_comparison_test'] = convert_fig_to_bytes(fig)

    def get_global_performance_df(results_df):
        real_duration = results_df['real_duration'].sum(); sim_duration = results_df['simulated_duration'].sum(); real_cost = results_df['real_cost'].sum(); sim_cost = results_df['simulated_cost'].sum()
        dur_improv = real_duration - sim_duration; cost_improv = real_cost - sim_cost
        dur_improv_perc = (dur_improv / real_duration) * 100 if real_duration > 0 else 0; cost_improv_perc = (cost_improv / real_cost) * 100 if real_cost > 0 else 0
        perf_data = {'Métrica': ['Duração Total (dias)', 'Custo Total (€)'], 'Real (Histórico)': [f"{real_duration:.0f}", f"€{real_cost:,.2f}"], 'Simulado (RL)': [f"{sim_duration:.0f}", f"€{sim_cost:,.2f}"], 'Melhoria': [f"{dur_improv:.0f} ({dur_improv_perc:.1f}%)", f"€{cost_improv:,.2f} ({cost_improv_perc:.1f}%)"]}
        return pd.DataFrame(perf_data)
    tables['global_performance_test'] = get_global_performance_df(test_results_df)

    # 8. REINTRODUÇÃO DA ANÁLISE DETALHADA
    project_info_real = df_projects.loc[df_projects['project_id'] == project_id_to_simulate].iloc[0]
    project_info_sim = test_results_df.loc[test_results_df['project_id'] == project_id_to_simulate].iloc[0]
    
    tables['project_summary'] = pd.DataFrame({
        'Métrica': ['Duração (dias)', 'Custo (€)'],
        'Real (Histórico)': [project_info_real['total_duration_days'], project_info_real['total_actual_cost']],
        'Simulado (RL)': [project_info_sim['simulated_duration'], project_info_sim['simulated_cost']]
    })

    # Preparar dados para o gráfico detalhado
    sim_log_df = pd.DataFrame(env.detailed_logs.get(project_id_to_simulate, []))
    real_allocations = df_resource_allocations[df_resource_allocations['project_id'] == project_id_to_simulate].copy()
    real_allocations['allocation_date'] = pd.to_datetime(real_allocations['allocation_date'])
    real_log_merged = real_allocations.merge(df_resources[['resource_id', 'cost_per_hour']], on='resource_id', how='left')
    real_log_merged['daily_cost'] = real_log_merged['hours_worked'] * real_log_merged['cost_per_hour']
    
    project_start_date = pd.to_datetime(project_info_real['start_date'])
    real_log_merged['day'] = (real_log_merged['allocation_date'] - project_start_date).dt.days

    # Calcular cumulativos para os dados reais
    real_daily_cost = real_log_merged.groupby('day')['daily_cost'].sum().cumsum().reset_index().rename(columns={'daily_cost': 'cumulative_cost'})
    real_daily_hours = real_log_merged.groupby('day')['hours_worked'].sum().cumsum().reset_index().rename(columns={'hours_worked': 'cumulative_hours'})
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # Gráfico de Custo Acumulado
    axes[0].plot(sim_log_df['day'], sim_log_df['cumulative_cost'], label='Custo Simulado', marker='o', linestyle='--', color='b')
    axes[0].plot(real_daily_cost['day'], real_daily_cost['cumulative_cost'], label='Custo Real', marker='x', linestyle='-', color='r')
    axes[0].axvline(x=project_info_real['total_duration_days'], color='k', linestyle=':', label=f"Fim Real ({project_info_real['total_duration_days']} dias)")
    axes[0].set_title('Custo Acumulado'); axes[0].set_xlabel('Dias desde o início'); axes[0].set_ylabel('Custo Acumulado (€)'); axes[0].legend(); axes[0].grid(True)
    # Gráfico de Progresso Acumulado
    axes[1].plot(sim_log_df['day'], sim_log_df['cumulative_hours'], label='Progresso Simulado', marker='o', linestyle='--', color='b')
    axes[1].plot(real_daily_hours['day'], real_daily_hours['cumulative_hours'], label='Progresso Real', marker='x', linestyle='-', color='r')
    axes[1].set_title('Progresso Acumulado (Horas)'); axes[1].set_xlabel('Dias desde o início'); axes[1].set_ylabel('Horas Acumuladas'); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    plots['project_detailed_comparison'] = convert_fig_to_bytes(fig)
    
    return plots, tables, logs
    
    # --- FIM DO BLOCO DE CÓDIGO FINAL ---
    
    # --- FIM DO NOVO BLOCO ---

# --- PÁGINA DE LOGIN ---
def login_page():
    st.markdown("<h2>✨ Transformação Inteligente de Processos</h2>", unsafe_allow_html=True)
    username = st.text_input("Utilizador", placeholder="Vasco", value="Vasco")
    password = st.text_input("Senha", type="password", placeholder="1234", value="1234")
    if st.button("Entrar", use_container_width=True):
        if username == "Vasco" and password == "1234":
            st.session_state.authenticated = True
            st.session_state.user_name = "Vasco"
            st.session_state.show_welcome_message = True
            st.rerun()
        else:
            st.error("Utilizador ou senha inválidos.")


# --- PÁGINA DE CONFIGURAÇÕES / UPLOAD ---
def settings_page():
    st.title("⚙️ Configurações e Upload de Dados")
    st.warning("Se carregou novos ficheiros CSV, clique primeiro neste botão para limpar a memória da aplicação antes de iniciar a nova análise.")
    if st.button("🔴 Limpar Cache e Recomeçar Análise"):
        st.cache_data.clear()
        st.success("Cache limpa com sucesso! A página será recarregada. Por favor, carregue os seus ficheiros novamente.")
        st.rerun()
    ############################################

    st.markdown("---") # Para separar visualmente
    st.markdown("---")
    st.subheader("Upload dos Ficheiros de Dados (.csv)")
    st.info("Por favor, carregue os 5 ficheiros CSV necessários para a análise.")
    # Dicionário com os textos para as tooltips
    tooltips = {
        'projects': "Ficheiro mestre que define cada processo. Colunas: project_id (ID único do processo), project_name (Nome descritivo), path_name (Tipo de processo, ex: CH_Jovem), start_date (Início real), end_date (Fim real), planned_end_date (Fim planeado), total_duration_days (Duração real em dias úteis), project_status (Estado final), loan_amount_eur (Valor do crédito), risk_rating (Nível de risco A-D), budget_impact (Orçamento total estimado).",
        'tasks': "O log de eventos, onde cada linha é uma tarefa executada. Colunas: task_id (ID único da tarefa), project_id (ID do processo-pai), task_name (Nome da atividade, ex: Análise Documental), task_type (Categoria da atividade), estimated_effort (Duração planeada em dias), actual_effort (Duração real em dias), task_status (Estado final, ex: Concluída), priority (Prioridade 1-5), start_date (Início real), end_date (Fim real).",
        'resources': "Define quem executa o trabalho. Colunas: resource_id (ID único do recurso), resource_name (Nome do colaborador/equipa), resource_type (Função, ex: Analista de Risco), skill_level (Nível de competência), daily_capacity (Horas de trabalho por dia), cost_per_hour (Custo por hora).",
        'resource_allocations': "Registo detalhado do trabalho diário, ligando recursos a tarefas. Colunas: allocation_id (ID único do registo de trabalho), task_id (ID da tarefa executada), resource_id (ID de quem executou), allocation_date (Data do trabalho), hours_worked (Horas trabalhadas nesse dia), project_id (ID do processo global).",
        'dependencies': "Mapeia a sequência e as regras do fluxo de trabalho. Colunas: dependency_id (ID único da regra de sequência), project_id (ID do processo), task_id_predecessor (ID da tarefa que deve ser concluída primeiro), task_id_successor (ID da tarefa que depende da anterior)."
    }
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    
    upload_cols = st.columns(5)
    for i, name in enumerate(file_names):
        with upload_cols[i]:
            # A alteração está na linha seguinte, com a adição do parâmetro 'help'
            uploaded_file = st.file_uploader(
                f"Carregar `{name}.csv`", 
                type="csv", 
                key=f"upload_{name}",
                help=tooltips[name]  # <-- LINHA ADICIONADA
            )
            if uploaded_file:
                # normalização mínima após upload
                df = pd.read_csv(uploaded_file)
                
                # ids como string onde existam
                for id_col in ['project_id', 'task_id', 'resource_id']:
                    if id_col in df.columns:
                        df[id_col] = df[id_col].astype(str)
                
                # colunas numéricas importantes: forçar numeric e preencher NaN com 0
                for num_col in ['hours_worked', 'cost_per_hour', 'daily_capacity', 'estimated_effort', 'priority', 'budget_impact']:
                    if num_col in df.columns:
                        df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0)
                
                # [CORREÇÃO PONTO 10] Adiciona performance_factor se não existir
                if name == 'resources' and 'performance_factor' not in df.columns:
                    df['performance_factor'] = 1.0

                # datas: parse imediato
                for date_col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                st.session_state.dfs[name] = df
                st.markdown(f'<p style="font-size: small; color: #06B6D4;">`{name}.csv` carregado e normalizado.</p>', unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)
    all_files_uploaded = all(st.session_state.dfs.get(name) is not None for name in file_names)
    
    if all_files_uploaded:
        if st.toggle("Visualizar as primeiras 5 linhas dos ficheiros", value=False):
            for name, df in st.session_state.dfs.items():
                st.markdown(f"**Ficheiro: `{name}.csv`**")
                st.dataframe(df.head())
        
        st.subheader("Execução da Análise")
        st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
        if st.button("🚀 Iniciar Análise Inicial (PM & EDA)", use_container_width=True):
            with st.spinner("A executar a análise... Este processo pode demorar alguns minutos."):
                plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(st.session_state.dfs)
                st.session_state.plots_pre_mining = plots_pre
                st.session_state.tables_pre_mining = tables_pre
                log_from_df = pm4py.convert_to_event_log(pm4py.convert_to_dataframe(event_log))
                plots_post, metrics = run_post_mining_analysis(log_from_df, df_p, df_t, df_r, df_fc)
                st.session_state.plots_post_mining = plots_post
                st.session_state.metrics = metrics
                plots_eda, tables_eda = run_eda_analysis(st.session_state.dfs)
                st.session_state.plots_eda = plots_eda
                st.session_state.tables_eda = tables_eda

            st.session_state.analysis_run = True
            st.success("✅ Análise concluída! Navegue para o 'Process Mining' ou para a página de 'Reinforcement Learning'.")
            st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Aguardando o carregamento de todos os ficheiros CSV para poder iniciar a análise.")


# --- PÁGINA DO DASHBOARD ---
def dashboard_page():
    st.title("🏠 Process Mining")

    if st.session_state.get('show_welcome_message', False):
        st.success(f"Bem-vindo, {st.session_state.user_name}!")
        st.session_state.show_welcome_message = False

    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Vá à página de 'Configurações' para carregar os dados e iniciar.")
        return
        
    sections = {
        "visao_geral": "1. Visão Geral e Custos",
        "performance": "2. Performance e Prazos",
        "recursos": "3. Recursos e Equipa",
        "gargalos": "4. Handoffs e Espera",
        "fluxo": "5. Fluxo e Conformidade"
    }
    
    nav_cols = st.columns(len(sections))
    for i, (key, name) in enumerate(sections.items()):
        with nav_cols[i]:
            st.markdown(f'<div class="{"active-button" if st.session_state.current_section == key else ""}">', unsafe_allow_html=True)
            if st.button(name, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_section = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-top: 25px; margin-bottom: 25px; border-color: var(--border-color);'>", unsafe_allow_html=True)
    
    plots_pre = st.session_state.plots_pre_mining
    tables_pre = st.session_state.tables_pre_mining
    plots_post = st.session_state.plots_post_mining
    plots_eda = st.session_state.plots_eda
    tables_eda = st.session_state.tables_eda

    if st.session_state.current_section == "visao_geral":
        st.subheader("1. Visão Geral e Custos")
        kpi_data = tables_pre.get('kpi_data', {})
        kpi_cols = st.columns(4)
        kpi_cols[0].metric(label="Total de Processos", value=kpi_data.get('Total de Processos'))
        kpi_cols[1].metric(label="Total de Tarefas", value=kpi_data.get('Total de Tarefas'))
        kpi_cols[2].metric(label="Total de Recursos", value=kpi_data.get('Total de Recursos'))
        kpi_cols[3].metric(label="Duração Média", value=f"{kpi_data.get('Duração Média (dias)')} dias")
        
        kpi_delay_data = tables_pre.get('cost_of_delay_kpis', {})

        total_cost = float(kpi_delay_data.get('Custo Total Processos Atrasados', 0.0) or 0.0)
        mean_delay = float(kpi_delay_data.get('Atraso Médio (dias)', 0.0) or 0.0)
        mean_cost_day = float(kpi_delay_data.get('Custo Médio/Dia Atraso', 0.0) or 0.0)
        
        kpi_cols_2 = st.columns(3)
        kpi_cols_2[0].metric(label="Custo Total em Atraso", value=f"€{total_cost:,.2f}")
        kpi_cols_2[1].metric(label="Atraso Médio (dias)", value=f"{mean_delay:.1f}")
        kpi_cols_2[2].metric(label="Custo Médio/Dia de Atraso", value=f"€{mean_cost_day:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            create_card("Matriz de Performance (Custo vs Prazo) (PM)", '<i class="bi bi-bullseye"></i>', chart_bytes=plots_pre.get('performance_matrix'), tooltip="Este gráfico cruza o desvio de custo (eixo Y) com o desvio de prazo (eixo X). O objetivo é identificar rapidamente processos problemáticos. Quadrante Superior Direito: Processos que custaram mais e demoraram mais que o planeado. Quadrante Inferior Esquerdo: Processos eficientes, que terminaram abaixo do custo e antes do prazo.")
            create_card("Top 5 Processos Mais Caros", '<i class="bi bi-cash-coin"></i>', dataframe=tables_pre.get('outlier_cost'), tooltip="Lista os cinco processos individuais que tiveram o maior custo real. Útil para focar a análise em outliers de custo significativo.")
            create_card("Séries Temporais de KPIs de Performance", '<i class="bi bi-graph-up-arrow"></i>', chart_bytes=plots_post.get('kpi_time_series'), tooltip="Apresenta a evolução do Lead Time (tempo médio do processo, em azul) e do Throughput (nº de processos concluídos, em cinza) ao longo do tempo. Permite analisar se a performance geral está a melhorar ou a piorar.")
            create_card("Distribuição do Status dos Processos", '<i class="bi bi-bar-chart-line-fill"></i>', chart_bytes=plots_eda.get('plot_01'), tooltip="Mostra a contagem de processos por estado (ex: 'Concluído', 'Em Andamento'). Oferece uma visão rápida do volume de trabalho atual e do histórico.")
            create_card("Custo Médio dos Processos por Trimestre", '<i class="bi bi-currency-euro"></i>', chart_bytes=plots_eda.get('plot_06'), tooltip="Agrega o custo médio de todos os processos concluídos em cada trimestre. Ajuda a identificar tendências sazonais ou mudanças de custo ao longo do ano.")
            create_card("Alocação de Custos por Orçamento e Recurso", '<i class="bi bi-pie-chart-fill"></i>', chart_bytes=plots_eda.get('plot_17'), tooltip="Gráfico de barras empilhadas que mostra como o custo total, segmentado por tipo de recurso, se distribui por diferentes faixas de orçamento dos processos. Ajuda a perceber se processos mais caros consomem tipos de recursos diferentes.")
        with c2:
            create_card("Custo por Tipo de Recurso", '<i class="bi bi-tags-fill"></i>', chart_bytes=plots_pre.get('cost_by_resource_type'), tooltip="Mostra o custo total acumulado para cada função ou tipo de recurso envolvido nos processos. Ajuda a identificar quais são as funções mais dispendiosas para a operação.")
            create_card("Top 5 Processos Mais Longos", '<i class="bi bi-hourglass-split"></i>', dataframe=tables_pre.get('outlier_duration'), tooltip="Lista os cinco processos individuais que tiveram a maior duração real (em dias). Útil para focar a análise em outliers de tempo e identificar possíveis gargalos crónicos.")
            create_card("Custo Médio por Dia ao Longo do Tempo", '<i class="bi bi-wallet2"></i>', chart_bytes=plots_post.get('cost_per_day_time_series'), tooltip="Analisa a evolução do custo médio diário dos processos. Picos neste gráfico podem indicar períodos de menor eficiência ou a execução de processos mais caros.")
            create_card("Custo Real vs. Orçamento por Processo", '<i class="bi bi-credit-card"></i>', chart_bytes=plots_eda.get('plot_04'), tooltip="Compara o custo planeado (orçamento) com o custo real para cada processo. Ideal para detetar desvios de orçamento e analisar a precisão das estimativas.")
            create_card("Distribuição do Custo por Dia (Eficiência)", '<i class="bi bi-lightbulb"></i>', chart_bytes=plots_eda.get('plot_16'), tooltip="Este histograma mostra a frequência de diferentes níveis de 'custo por dia'. Uma concentração em valores mais baixos indica maior eficiência de custos na execução dos processos.")
            create_card("Evolução do Volume e Tamanho dos Processos", '<i class="bi bi-reception-4"></i>', chart_bytes=plots_eda.get('plot_31'), tooltip="Apresenta o número de processos concluídos por mês (barras) e a evolução da sua duração média (linha). Permite correlacionar o volume de trabalho com a eficiência.")

    elif st.session_state.current_section == "performance":
        st.subheader("2. Performance e Prazos")
        c1, c2 = st.columns(2)
        with c1:
            create_card("Relação Lead Time vs Throughput", '<i class="bi bi-link-45deg"></i>', chart_bytes=plots_pre.get('lead_time_vs_throughput'), tooltip="Este gráfico de dispersão explora a relação entre o tempo médio para concluir uma tarefa (Throughput) e o tempo total do processo (Lead Time). Idealmente, um throughput mais rápido deveria levar a um lead time menor.")
            create_card("Distribuição do Lead Time", '<i class="bi bi-stopwatch"></i>', chart_bytes=plots_pre.get('lead_time_hist'), tooltip="Mostra a frequência das diferentes durações totais dos processos (do início ao fim). Permite perceber se a maioria dos processos termina dentro de um prazo esperado ou se existem muitos outliers.")
            create_card("Distribuição da Duração dos Processos (PM)", '<i class="bi bi-distribute-vertical"></i>', chart_bytes=plots_pre.get('case_durations_boxplot'), tooltip="Um boxplot que resume estatisticamente a duração dos processos, mostrando a mediana, os quartis e os outliers. Oferece uma visão rápida da consistência dos prazos.")
            create_card("Gráfico Acumulado de Throughput", '<i class="bi bi-graph-up"></i>', chart_bytes=plots_post.get('cumulative_throughput_plot'), tooltip="Mostra o número total de processos concluídos ao longo do tempo. A inclinação da linha indica a velocidade de conclusão: quanto mais íngreme, maior o ritmo de entrega.")
            create_card("Performance de Prazos por Trimestre", '<i class="bi bi-graph-down-arrow"></i>', chart_bytes=plots_eda.get('plot_05'), tooltip="Analisa o desvio de prazo (dias de atraso ou adiantamento) por trimestre. Permite ver se a pontualidade tem vindo a melhorar ou piorar ao longo do tempo.")
        with c2:
            create_card("Duração Média por Fase do Processo", '<i class="bi bi-folder2-open"></i>', chart_bytes=plots_pre.get('cycle_time_breakdown'), tooltip="Decompõe a duração total do processo, mostrando o tempo médio gasto em cada fase principal (ex: Onboarding, Análise de Risco, etc.). Essencial para identificar as fases mais demoradas.")
            create_card("Distribuição do Throughput (horas)", '<i class="bi bi-rocket-takeoff"></i>', chart_bytes=plots_pre.get('throughput_hist'), tooltip="Apresenta a distribuição do tempo médio entre a conclusão de tarefas consecutivas dentro de um mesmo processo. Valores mais baixos indicam maior velocidade e fluidez.")
            create_card("Boxplot do Throughput (horas)", '<i class="bi bi-box-seam"></i>', chart_bytes=plots_pre.get('throughput_boxplot'), tooltip="Um boxplot que resume estatisticamente o tempo de throughput. Ajuda a visualizar a variabilidade e a identificar casos com tempos de espera anormais entre tarefas.")
            create_card("Atividades por Dia da Semana", '<i class="bi bi-calendar-week"></i>', chart_bytes=plots_post.get('temporal_heatmap_fixed'), tooltip="Conta o número de eventos (início/fim de tarefas) que ocorrem em cada dia da semana. Útil para identificar padrões de trabalho e possíveis sobrecargas ou ociosidades.")
            create_card("Evolução da Performance (Prazo e Custo)", '<i class="bi bi-activity"></i>', chart_bytes=plots_eda.get('plot_30'), tooltip="Apresenta a evolução mensal do desvio médio de prazo e do desvio médio de custo. Permite uma análise de alto nível sobre a saúde e eficiência do processo ao longo do tempo.")
        c3, c4 = st.columns(2)
        with c3:
                create_card("Diferença entre Data Real e Planeada", '<i class="bi bi-calendar-range"></i>', chart_bytes=plots_eda.get('plot_03'), tooltip="Histograma que mostra a distribuição dos desvios de prazo. Uma concentração em torno do zero indica boa previsibilidade. Valores à direita indicam atrasos; à esquerda, adiantamentos.")
        with c4:
            create_card("Estatísticas de Performance", '<i class="bi bi-table"></i>', dataframe=tables_pre.get('perf_stats'), tooltip="Tabela com um resumo descritivo do Lead Time (duração total do processo) e do Throughput (tempo entre tarefas), incluindo média, mediana, desvio padrão, etc.")
        create_card("Linha do Tempo de Todos os Processos (Gantt Chart)", '<i class="bi bi-kanban"></i>', chart_bytes=plots_post.get('gantt_chart_all_projects'), tooltip="Visualiza a duração e o sequenciamento das tarefas para cada processo. É uma ferramenta poderosa para entender a sobreposição de trabalho e a linha do tempo de projetos específicos.")

    elif st.session_state.current_section == "recursos":
        st.subheader("3. Recursos e Equipa")
        
        # Primeira linha de cartões
        c1, c2 = st.columns(2)
        with c1:
            create_card("Distribuição de Recursos por Tipo", '<i class="bi bi-tools"></i>', chart_bytes=plots_eda.get('plot_12'), tooltip="Mostra quantos colaboradores existem em cada função. Ajuda a entender a composição da equipa.")
            create_card("Recursos por Média de Tarefas/Processo", '<i class="bi bi-person-workspace"></i>', chart_bytes=plots_pre.get('resource_avg_events'), tooltip="Classifica os recursos pela quantidade média de tarefas que executam por cada processo em que participam. Pode revelar especialistas ou recursos que são envolvidos em muitas etapas.")
            create_card("Eficiência Semanal (Horas Trabalhadas)", '<i class="bi bi-calendar3-week"></i>', chart_bytes=plots_pre.get('weekly_efficiency'), tooltip="Soma das horas de trabalho registadas em cada dia da semana. Permite identificar os dias de maior e menor produtividade ou esforço.")
            create_card("Impacto do Tamanho da Equipa no Atraso (PM)", '<i class="bi bi-people"></i>', chart_bytes=plots_pre.get('delay_by_teamsize'), tooltip="Analisa como o número de pessoas alocadas a um processo impacta o desvio de prazo. Ajuda a responder se equipas maiores são, de facto, mais rápidas ou se geram mais custos de coordenação.")
            create_card("Benchmark de Throughput por Equipa", '<i class="bi bi-trophy"></i>', chart_bytes=plots_pre.get('throughput_benchmark_by_teamsize'), tooltip="Compara a velocidade de execução (throughput) entre equipas de diferentes tamanhos. Permite identificar o 'tamanho ideal' de equipa para máxima eficiência.")
        with c2:
            create_card("Top 10 Recursos por Horas Trabalhadas (PM)", '<i class="bi bi-lightning-charge-fill"></i>', chart_bytes=plots_pre.get('resource_workload'), tooltip="Lista os 10 colaboradores que registaram o maior número de horas de trabalho. Pode indicar tanto os mais produtivos como os mais sobrecarregados.")
            create_card("Top 10 Handoffs entre Recursos", '<i class="bi bi-arrow-repeat"></i>', chart_bytes=plots_pre.get('resource_handoffs'), tooltip="Mostra as transições de trabalho mais frequentes entre dois colaboradores diferentes. Ajuda a visualizar as principais interações e dependências na equipa.")
            create_card("Métricas de Eficiência: Top 10 Melhores e Piores Recursos", '<i class="bi bi-person-check"></i>', chart_bytes=plots_post.get('resource_efficiency_plot'), tooltip="Compara os recursos com base na média de horas que demoram a completar uma tarefa. 'Melhores' são os mais rápidos (menos horas/tarefa), e 'Piores' são os mais lentos.")
            create_card("Duração Mediana por Tamanho da Equipa", '<i class="bi bi-speedometer"></i>', chart_bytes=plots_pre.get('median_duration_by_teamsize'), tooltip="Mostra a duração mediana dos processos com base no tamanho da equipa. Complementa o gráfico de atraso, focando na duração total e não no desvio.")
            create_card("Nº Médio de Recursos por Processo a Cada Trimestre", '<i class="bi bi-person-plus"></i>', chart_bytes=plots_eda.get('plot_07'), tooltip="Analisa a evolução do tamanho médio das equipas alocadas aos processos ao longo do tempo.")
            create_card("Atraso Médio por Recurso", '<i class="bi bi-person-exclamation"></i>', chart_bytes=plots_eda.get('plot_14'), tooltip="Lista os 20 recursos associados ao maior atraso médio nos processos em que participaram.")

        # Segunda linha de cartões, para os gráficos de análise de Skill
        c3, c4 = st.columns(2)
        with c3:
            if 'skill_vs_performance_adv' in plots_post:
                create_card("Relação entre Skill e Performance", '<i class="bi bi-graph-up-arrow"></i>', chart_bytes=plots_post.get('skill_vs_performance_adv'), tooltip="Explora se existe uma correlação entre o nível de competência (skill level) de um recurso e a sua performance (medida em horas médias por tarefa).")
        with c4:
            create_card("Atraso por Nível de Competência", '<i class="bi bi-mortarboard"></i>', chart_bytes=plots_eda.get('plot_23'), tooltip="Analisa a distribuição dos atrasos dos processos com base no nível de competência dos recursos envolvidos.")

        # Gráficos complexos que ocupam a largura total
        if 'resource_network_bipartite' in plots_post:
            create_card("Rede de Recursos por Função", '<i class="bi bi-node-plus-fill"></i>', chart_bytes=plots_post.get('resource_network_bipartite'), tooltip="Grafo que conecta os recursos às suas funções/skills. Útil para visualizar a polivalência dos colaboradores e a distribuição de competências na equipa.")

        create_card("Rede Social de Recursos (Handovers)", '<i class="bi bi-diagram-3-fill"></i>', chart_bytes=plots_post.get('resource_network_adv'), tooltip="Grafo onde os nós são os recursos e as arestas representam a passagem de trabalho (handoff) entre eles. A espessura da aresta indica a frequência. Mostra os fluxos de comunicação e colaboração centrais.")
        
        create_card("Heatmap de Esforço (Recurso vs Atividade)", '<i class="bi bi-map"></i>', chart_bytes=plots_pre.get('resource_activity_matrix'), tooltip="Matriz que cruza recursos com atividades, mostrando as horas totais trabalhadas. Permite identificar rapidamente quem são os especialistas em cada tipo de tarefa.")
    
    elif st.session_state.current_section == "gargalos":
        st.subheader("4. Handoffs e Espera")
        create_card("Heatmap de Performance no Processo (Gargalos)", '<i class="bi bi-fire"></i>', chart_bytes=plots_post.get('performance_heatmap'), tooltip="Este é um mapa visual do seu processo, focado em identificar os tempos de espera, que são os verdadeiros gargalos. Como ler o gráfico: ● Caixas (Nós): Representam as atividades. ● Setas (Transições): Mostram o fluxo de trabalho. ● Números e Cores nas Setas: Este é o ponto mais importante. O valor (ex: 2d 4h 15m) e a cor da seta representam o tempo médio de espera entre a conclusão da atividade de origem e o início da atividade de destino. Análise Prática: Procure as setas com as cores mais quentes (ex: laranja, vermelho) e os valores de tempo mais altos. Estes são os seus maiores gargalos.")
        
        c1, c2 = st.columns(2)
        with c1:
            create_card("Atividades Mais Frequentes", '<i class="bi bi-speedometer2"></i>', chart_bytes=plots_pre.get('top_activities_plot'), tooltip="Gráfico de barras que mostra as atividades que ocorrem com maior frequência em todos os processos. Ajuda a identificar os passos mais comuns e centrais do fluxo de trabalho.")
            create_card("Gargalos: Tempo de Serviço vs. Espera", '<i class="bi bi-traffic-light"></i>', chart_bytes=plots_pre.get('service_vs_wait_stacked'), tooltip="Compara o tempo em que uma tarefa está a ser ativamente trabalhada (Tempo de Serviço) com o tempo em que fica parada à espera (Tempo de Espera). Atividades com alto tempo de espera são os principais gargalos.")
            create_card("Top 10 Handoffs por Custo de Espera", '<i class="bi bi-currency-exchange"></i>', chart_bytes=plots_pre.get('top_handoffs_cost'), tooltip="Estima o custo financeiro do tempo de espera entre as 10 transições mais lentas. Quantifica o impacto dos gargalos, traduzindo tempo perdido em perdas financeiras.")
            create_card("Top Recursos por Tempo de Espera Gerado", '<i class="bi bi-sign-stop"></i>', chart_bytes=plots_pre.get('bottleneck_by_resource'), tooltip="Identifica os recursos que, em média, geram o maior tempo de espera para a tarefa seguinte após concluírem o seu trabalho. Pode indicar sobrecarga ou necessidade de otimização no trabalho desse recurso.")
            create_card("Custo Real vs. Atraso", '<i class="bi bi-cash-stack"></i>', chart_bytes=plots_eda.get('plot_18'), tooltip="Analisa a correlação entre o custo total de um processo e o seu atraso em dias.")
            create_card("Nº de Recursos vs. Custo Total", '<i class="bi bi-people-fill"></i>', chart_bytes=plots_eda.get('plot_20'), tooltip="Analisa a correlação entre o número de recursos alocados a um processo e o seu custo final.")
            
        with c2:
            create_card("Tempo Médio de Execução por Atividade", '<i class="bi bi-hammer"></i>', chart_bytes=plots_pre.get('activity_service_times'), tooltip="Mostra o tempo médio que cada tipo de atividade leva para ser concluída (tempo de serviço). Permite identificar as tarefas que, isoladamente, são as mais demoradas.")
            create_card("Espera vs. Execução (Dispersão)", '<i class="bi bi-search"></i>', chart_bytes=plots_pre.get('wait_vs_service_scatter'), tooltip="Analisa a correlação entre o tempo de execução de uma tarefa e o tempo de espera que a sucede. Pode revelar se tarefas longas tendem a gerar mais espera a jusante.")
            create_card("Evolução do Tempo Médio de Espera", '<i class="bi bi-clock-history"></i>', chart_bytes=plots_pre.get('wait_time_evolution'), tooltip="Mostra como o tempo médio de espera entre tarefas tem evoluído ao longo dos meses. Permite avaliar o impacto de melhorias ou a degradação da eficiência do processo.")
            create_card("Top 10 Handoffs por Tempo de Espera", '<i class="bi bi-pause-circle"></i>', chart_bytes=plots_pre.get('top_handoffs'), tooltip="Lista as 10 transições entre atividades que têm o maior tempo médio de espera. Aponta diretamente para os maiores gargalos do processo em termos de tempo.")
            create_card("Rate Horário Médio vs. Atraso", '<i class="bi bi-alarm"></i>', chart_bytes=plots_eda.get('plot_19'), tooltip="Analisa a correlação entre o custo médio por hora dos recursos de um processo e o seu atraso final.")
            create_card("Atraso por Faixa de Orçamento", '<i class="bi bi-layers-half"></i>', chart_bytes=plots_eda.get('plot_22'), tooltip="Mostra a distribuição dos atrasos para processos agrupados por diferentes faixas de orçamento.")

        if 'milestone_time_analysis_plot' in plots_post:
            create_card("Análise de Tempo entre Marcos do Processo", '<i class="bi bi-flag"></i>', chart_bytes=plots_post.get('milestone_time_analysis_plot'), tooltip="Mede o tempo de espera entre as fases mais importantes do processo (marcos). Dá uma visão de alto nível sobre onde o processo fica parado por mais tempo entre etapas críticas.")
        
        c3, c4 = st.columns(2)
        with c3:
            create_card("Matriz de Correlação", '<i class="bi bi-bounding-box-circles"></i>', chart_bytes=plots_eda.get('plot_29'), tooltip="Heatmap que mostra a correlação estatística entre diferentes variáveis numéricas (custo, duração, prioridade, etc.). Ajuda a descobrir relações inesperadas, como 'custo por hora' e 'atraso'.")
        with c4:
            create_card("Tempo Médio de Espera por Atividade", '<i class="bi bi-hourglass-bottom"></i>', chart_bytes=plots_post.get('avg_waiting_time_by_activity_plot'), tooltip="Mostra o tempo médio que cada atividade fica em espera *antes* de ser iniciada. Complementa a análise de gargalos, focando no tempo de fila de cada tarefa.")
            
        create_card("Matriz de Tempo de Espera entre Atividades (horas)", '<i class="bi bi-grid-3x3-gap"></i>', chart_bytes=plots_post.get('waiting_time_matrix_plot'), tooltip="Uma matriz detalhada que mostra o tempo médio de espera (em horas) para cada transição possível entre duas atividades. Permite uma análise granular de todos os handoffs.")

    elif st.session_state.current_section == "fluxo":
        st.subheader("5. Fluxo e Conformidade")

        create_card("Modelo - Inductive Miner", '<i class="bi bi-compass"></i>', chart_bytes=plots_post.get('model_inductive_petrinet'), tooltip="Este é um modelo do processo descoberto automaticamente a partir dos dados. O Inductive Miner gera modelos estruturados e fáceis de ler, que representam o fluxo de trabalho 'ideal' ou mais comum.")
        create_card("Modelo - Heuristics Miner", '<i class="bi bi-gear"></i>', chart_bytes=plots_post.get('model_heuristic_petrinet'), tooltip="Este é outro modelo do processo, focado em mostrar as conexões mais frequentes, mesmo que isso resulte num modelo menos estruturado. É útil para explorar os caminhos mais comuns e ignorar os raros.")

        c1, c2 = st.columns(2)
        with c1:
            create_card("Métricas (Inductive Miner)", '<i class="bi bi-clipboard-data"></i>', chart_bytes=plots_post.get('metrics_inductive'), tooltip="Avalia a qualidade do modelo Inductive Miner. Fitness: Quão bem o modelo representa a realidade. Precisão: Quão bem o modelo evita comportamentos não existentes. Generalização: Capacidade de representar comportamentos não vistos. Simplicidade: Facilidade de leitura do modelo.")
        with c2:
            create_card("Métricas (Heuristics Miner)", '<i class="bi bi-clipboard-check"></i>', chart_bytes=plots_post.get('metrics_heuristic'), tooltip="Avalia a qualidade do modelo Heuristics Miner, usando os mesmos critérios: Fitness, Precisão, Generalização e Simplicidade.")
        
        create_card("Sequência de Atividades das 10 Variantes Mais Comuns", '<i class="bi bi-music-note-list"></i>', chart_bytes=plots_post.get('custom_variants_sequence_plot'), tooltip="Este gráfico mostra o fluxo de atividades, passo a passo, para cada uma das 10 variações de processo mais frequentes. Permite comparar visualmente os diferentes caminhos que os processos seguem.")
        create_card("Duração Média das Variantes Mais Comuns", '<i class="bi bi-clock-history"></i>', chart_bytes=plots_post.get('variant_duration_plot'), tooltip="Apresenta a duração média de cada uma das 10 variantes de processo mais comuns. Ajuda a identificar quais são os fluxos de trabalho mais rápidos e os mais lentos.")
        create_card("Top 10 Variantes de Processo por Frequência", '<i class="bi bi-sort-numeric-down"></i>', chart_bytes=plots_pre.get('variants_frequency'), tooltip="Mostra, em barras, a frequência das 10 variações de processo mais comuns. Permite entender qual é o 'caminho feliz' e quais são as exceções mais recorrentes.")

        c3, c4 = st.columns(2)
        with c3:
            create_card("Frequência das 10 Principais Variantes", '<i class="bi bi-masks"></i>', dataframe=tables_pre.get('variants_table'), tooltip="Tabela com os dados detalhados das variantes de processo mais comuns, incluindo a sua frequência absoluta e o seu peso percentual em relação ao total.")
            create_card("Distribuição de Tarefas por Tipo", '<i class="bi bi-card-list"></i>', chart_bytes=plots_eda.get('plot_08'), tooltip="Mostra a contagem de todas as tarefas executadas, agrupadas pelo seu tipo. Dá uma visão geral do volume de trabalho para cada categoria de atividade.")
            create_card("Distribuição da Duração das Tarefas", '<i class="bi bi-hourglass"></i>', chart_bytes=plots_eda.get('plot_10'), tooltip="Histograma que mostra a distribuição dos tempos de execução de todas as tarefas. Ajuda a entender a variabilidade e a previsibilidade do esforço necessário para cada atividade.")
            create_card("Centralidade dos Tipos de Tarefa", '<i class="bi bi-arrows-angle-contract"></i>', chart_bytes=plots_eda.get('plot_25'), tooltip="Analisa quais tipos de tarefas são mais frequentemente 'predecessoras' (ocorrem antes de outras) e quais são mais 'sucessoras' (ocorrem depois de outras). Ajuda a entender as dependências centrais no fluxo.")
        with c4:
            create_card("Principais Loops de Rework", '<i class="bi bi-arrow-clockwise"></i>', dataframe=tables_pre.get('rework_loops_table'), tooltip="Identifica e conta as sequências de atividades que representam retrabalho (ex: A -> B -> C -> A). Essencial para encontrar ineficiências e ciclos de correção.")
            create_card("Score de Conformidade ao Longo do Tempo", '<i class="bi bi-check2-circle"></i>', chart_bytes=plots_post.get('conformance_over_time_plot'), tooltip="Mede a aderência dos processos ao modelo descoberto (Fitness) ao longo do tempo. Uma linha ascendente indica que os processos estão a tornar-se mais padronizados e conformes.")
            create_card("Distribuição de Tarefas por Prioridade", '<i class="bi bi-award"></i>', chart_bytes=plots_eda.get('plot_09'), tooltip="Conta o número de tarefas para cada nível de prioridade definido. Permite analisar se a atribuição de prioridades está bem distribuída ou concentrada.")
            create_card("Top 10 Tarefas Específicas Mais Demoradas", '<i class="bi bi-sort-down"></i>', chart_bytes=plots_eda.get('plot_11'), tooltip="Lista as 10 tarefas individuais (não o tipo, mas a instância específica) que mais tempo demoraram a ser concluídas.")

        c5, c6 = st.columns(2)
        with c5:
            create_card("Distribuição da Complexidade dos Processos", '<i class="bi bi-bezier"></i>', chart_bytes=plots_eda.get('plot_24'), tooltip="Mostra a distribuição do 'índice de complexidade' calculado para os processos. Este índice combina o risco do projeto com o número de funções diferentes envolvidas.")
            create_card("Gráfico de Dependências: Processo 25", '<i class="bi bi-diagram-2"></i>', chart_bytes=plots_eda.get('plot_26'), tooltip="Exemplo visual da rede de dependências entre as tarefas de um único processo. Ajuda a entender o caminho crítico e a estrutura de um projeto específico.")
        with c6:
            create_card("Relação entre Complexidade e Atraso", '<i class="bi bi-arrows-collapse"></i>', chart_bytes=plots_eda.get('plot_27'), tooltip="Analisa se processos considerados mais 'complexos' (maior score de complexidade) tendem a ter maiores atrasos.")
            create_card("Relação entre Dependências e Desvio de Custo", '<i class="bi bi-arrows-expand"></i>', chart_bytes=plots_eda.get('plot_28'), tooltip="Analisa se processos com maior número de dependências entre tarefas tendem a ter maiores desvios de custo.")

# --- NOVA PÁGINA (REINFORCEMENT LEARNING) ---
def rl_page():
    st.title("🤖 Simulação com Reinforcement Learning")

    if not st.session_state.analysis_run:
        st.warning("É necessário executar a análise inicial primeiro. Vá à página de 'Configurações' para carregar os dados.")
        return

    # --- LÓGICA CORRIGIDA: CRIAR A AMOSTRA DE RL APENAS UMA VEZ ---
    # Se a amostra de IDs ainda não foi criada, cria-a e guarda-a no estado da sessão.
    if 'rl_sample_ids' not in st.session_state:
        proj_ids = st.session_state.dfs['projects']['project_id'].astype(str)
        n = min(500, len(proj_ids))
        st.session_state['rl_sample_ids'] = proj_ids.sample(n=n, random_state=42).tolist()

    # -----------------------------------------------------------------

    st.info("Esta secção permite treinar um agente de IA para otimizar a gestão de processos. O treino e a análise correm sobre uma amostra de 500 processos para garantir a performance.")

    with st.expander("⚙️ Parâmetros da Simulação", expanded=st.session_state.rl_params_expanded):
        st.markdown("<p><strong>Parâmetros Gerais</strong></p>", unsafe_allow_html=True)
        
        # As opções agora vêm da amostra que acabámos de criar.
        project_ids_elegiveis = st.session_state.get('rl_sample_ids', [])
        
        c1, c2 = st.columns(2)
        with c1:
            project_id_to_simulate = st.selectbox(
            "Selecione o Processo para Simulação Detalhada (Amostra)",
            options=project_ids_elegiveis,
            index=0  # Garante que o primeiro item da lista é sempre o default
        )
        with c2:
            num_episodes = st.number_input("Número de Episódios de Treino", min_value=5, max_value=10000, value=1000, step=100)

        st.markdown("<p><strong>Parâmetros de Recompensa e Penalização do Agente</strong></p>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            cost_impact_factor = st.number_input("Fator de Impacto do Custo", value=1.0)
            daily_time_penalty = st.number_input("Penalização Diária por Tempo", value=20.0)
            idle_penalty = st.number_input("Penalização por Inatividade", value=10.0)
        with rc2:
            per_day_early_bonus = st.number_input("Bónus por Dia de Adiantamento", value=500.0)
            completion_base = st.number_input("Recompensa Base por Conclusão", value=5000.0)
            per_day_late_penalty = st.number_input("Penalização por Dia de Atraso", value=1500.0)
        with rc3:
            priority_task_bonus_factor = st.number_input("Bónus por Tarefa Prioritária", value=500)
            pending_task_penalty_factor = st.number_input("Penalização por Tarefa Pendente", value=20)
        
        st.markdown("<p><strong>Parâmetros do Agente</strong></p>", unsafe_allow_html=True)
        c_ag1, c_ag2, c_ag3 = st.columns(3)
        with c_ag1:
            agent_lr = st.number_input("Learning rate (lr)", min_value=0.0001, max_value=1.0, value=0.1, step=0.0001, format="%.4f")
            agent_gamma = st.number_input("Gamma (discount)", min_value=0.0, max_value=1.0, value=0.9, step=0.01, format="%.2f")
        with c_ag2:
            agent_epsilon = st.number_input("Epsilon (start)", min_value=0.0, max_value=1.0, value=1.0, step=0.01, format="%.2f")
            agent_epsilon_decay = st.number_input("Epsilon decay", min_value=0.0, max_value=1.0, value=0.9995, step=0.0001, format="%.4f")
        with c_ag3:
            agent_min_epsilon = st.number_input("Epsilon min", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f")

    status_container = st.empty()

    if st.button("▶️ Iniciar Treino e Simulação do Agente", use_container_width=True):
        st.session_state.rl_params_expanded = False
        
        # CORREÇÃO: Criar reward_config DENTRO do botão para usar valores atuais
        reward_config = {
            'cost_impact_factor': cost_impact_factor, 'daily_time_penalty': daily_time_penalty, 'idle_penalty': idle_penalty,
            'per_day_early_bonus': per_day_early_bonus, 'completion_base': completion_base, 'per_day_late_penalty': per_day_late_penalty,
            'priority_task_bonus_factor': priority_task_bonus_factor, 'pending_task_penalty_factor': pending_task_penalty_factor
        }
        st.session_state.project_id_simulated = project_id_to_simulate
        
        with status_container.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("A iniciar o treino do agente de RL...")
    
            plots_rl, tables_rl, logs_rl = run_rl_analysis(
                st.session_state.dfs,
                project_id_to_simulate,
                num_episodes,
                reward_config,
                progress_bar,
                status_text,
                agent_params={
                    'lr': float(agent_lr),
                    'gamma': float(agent_gamma),
                    'epsilon': float(agent_epsilon),
                    'epsilon_decay': float(agent_epsilon_decay),
                    'min_epsilon': float(agent_min_epsilon)
                }
            )
            
            # Localização Correta das Novas Linhas
            
    
        st.session_state.plots_rl = plots_rl
        st.session_state.tables_rl = tables_rl
        st.session_state.logs_rl = logs_rl
        st.session_state.rl_analysis_run = True
        st.rerun()
    if st.session_state.rl_analysis_run:
        # (O resto da sua função rl_page para mostrar os resultados continua aqui, sem alterações)
        st.markdown("---")
        st.subheader("Resultados da Simulação")
        
        plots_rl = st.session_state.plots_rl
        tables_rl = st.session_state.tables_rl
        
        st.markdown("<h4>Desempenho Global</h4>", unsafe_allow_html=True)
        create_card(
            "Performance Global (Conjunto de Teste)", 
            '<i class="bi bi-clipboard-data-fill"></i>', 
            dataframe=tables_rl.get('global_performance_test'),
            tooltip="Esta tabela resume o desempenho agregado do agente de IA no conjunto de teste. Compara a soma total de dias e custos dos processos simulados com os valores históricos reais, calculando a melhoria percentual."
        )

        st.markdown("<h4>Métricas de Treino do Agente</h4>", unsafe_allow_html=True)
        create_card(
            "Evolução do Treino", 
            '<i class="bi bi-robot"></i>', 
            chart_bytes=plots_rl.get('training_metrics'),
            tooltip="Estes gráficos mostram as curvas de aprendizagem do agente. Recompensa: Deve ter uma tendência crescente. Duração/Custo: Devem ter uma tendência decrescente. Epsilon: Mostra a taxa de exploração, que diminui à medida que o agente fica mais confiante."
        )
        
        st.markdown("<h4>Comparação de Desempenho (Simulado vs. Real)</h4>", unsafe_allow_html=True)
        create_card(
            "Comparação do Desempenho (Conjunto de Teste da Amostra)", 
            '<i class="bi bi-bullseye"></i>', 
            chart_bytes=plots_rl.get('evaluation_comparison_test'),
            tooltip="Estes gráficos comparam, processo a processo, a performance do agente (barras azuis) com os dados históricos (barras vermelhas) para o conjunto de teste. Permite uma análise visual de onde o agente teve melhor ou pior desempenho."
        )
        
        st.markdown(f"<h4>Análise Detalhada da Simulação (Processo {st.session_state.project_id_simulated})</h4>", unsafe_allow_html=True)
        summary_df = tables_rl.get('project_summary')
        if summary_df is not None:
            metric_cols = st.columns(2)
            with metric_cols[0]:
                real_duration = summary_df.loc[summary_df['Métrica'] == 'Duração (dias úteis)', 'Real (Histórico)'].iloc[0]
                sim_duration = summary_df.loc[summary_df['Métrica'] == 'Duração (dias úteis)', 'Simulado (RL)'].iloc[0]
                st.metric(label="Duração (dias úteis)", value=f"{sim_duration:.0f}", delta=f"{sim_duration - real_duration:.0f} vs Real",delta_color="inverse")
            with metric_cols[1]:
                real_cost = summary_df.loc[summary_df['Métrica'] == 'Custo (€)', 'Real (Histórico)'].iloc[0]
                sim_cost = summary_df.loc[summary_df['Métrica'] == 'Custo (€)', 'Simulado (RL)'].iloc[0]
                st.metric(label="Custo (€)", value=f"€{sim_cost:,.2f}", delta=f"€{sim_cost - real_cost:,.2f} vs Real",delta_color="inverse")

        create_card(
        f"Comparação Detalhada (Processo {st.session_state.project_id_simulated})", 
        '<i class="bi bi-search"></i>', 
        chart_bytes=plots_rl.get('project_detailed_comparison'),
        tooltip="Análise detalhada de um único processo. O gráfico da esquerda compara o custo acumulado (Real vs. Simulado). O da direita compara o progresso acumulado (em horas trabalhadas). Ideal para entender a estratégia do agente no dia a dia."
    )

# --- CONTROLO PRINCIPAL DA APLICAÇÃO ---
def main():
    if not st.session_state.authenticated:
        st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] > .main {
                display: flex; flex-direction: column; justify-content: center; align-items: center;
            }
        </style>
        """, unsafe_allow_html=True)
        login_page()
    else:
        with st.sidebar:
            st.markdown(f'<h3><i class="bi bi-person-circle"></i> {st.session_state.get("user_name", "Admin")}</h3>', unsafe_allow_html=True)
            st.markdown("---")
            if st.button("🧠 Process Mining", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.rerun()
            if st.button("🤖 Reinforcement Learning", use_container_width=True):
                st.session_state.current_page = "RL"
                st.rerun()
            if st.button("⚙️ Configurações", use_container_width=True):
                st.session_state.current_page = "Settings"
                st.rerun()
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button(" ⏻ Terminar sessão", use_container_width=True):
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

if __name__ == "__main__":
    main()
