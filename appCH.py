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
from scipy import stats

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
if 'data_frames_processed' not in st.session_state: st.session_state.data_frames_processed = {}

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
    tables['kpi_data']['Duração Média Num'] = df_projects['actual_duration_days'].mean()
    tables['kpi_data']['Custo Médio'] = df_projects['total_actual_cost'].mean()
    tables['kpi_data']['Desvio de Custo Médio'] = df_projects['cost_diff'].mean()
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
    tables['handoff_stats_data'] = handoff_stats
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=handoff_stats.sort_values('estimated_cost_of_wait', ascending=False).head(10), y='transition', x='estimated_cost_of_wait', ax=ax, hue='transition', legend=False, palette='magma'); ax.set_title("Top 10 Handoffs por Custo de Espera")
    plots['top_handoffs_cost'] = convert_fig_to_bytes(fig)

    activity_counts = df_tasks["task_name"].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='YlGnBu'); ax.set_title("Atividades Mais Frequentes")
    plots['top_activities_plot'] = convert_fig_to_bytes(fig)
    
    resource_workload = df_full_context.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
    tables['resource_workload_data'] = resource_workload
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='hours_worked', y='resource_name', data=resource_workload.head(10), ax=ax, hue='resource_name', legend=False, palette='plasma'); ax.set_title("Top 10 Recursos por Horas Trabalhadas (PM)")
    plots['resource_workload'] = convert_fig_to_bytes(fig)
    
    resource_metrics = df_full_context.groupby("resource_name").agg(unique_cases=('project_id', 'nunique'), event_count=('task_id', 'count')).reset_index()
    resource_metrics["avg_events_per_case"] = resource_metrics["event_count"] / resource_metrics["unique_cases"]
    tables['resource_avg_events_data'] = resource_metrics
    
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
    tables['cost_by_resource_type_data'] = cost_by_resource_type
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
    tables['kpi_data']['Espera Média (dias)'] = df_tasks_analysis['waiting_time_days'].mean()

    df_tasks_with_resources = df_tasks_analysis.merge(df_full_context[['task_id', 'resource_name']], on='task_id', how='left').drop_duplicates()
    bottleneck_by_resource = df_tasks_with_resources.groupby('resource_name')['waiting_time_days'].mean().sort_values(ascending=False).head(15).reset_index()
    tables['bottleneck_by_resource_data'] = bottleneck_by_resource
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=bottleneck_by_resource, y='resource_name', x='waiting_time_days', ax=ax, hue='resource_name', legend=False, palette='rocket'); ax.set_title("Top 15 Recursos por Tempo Médio de Espera")
    plots['bottleneck_by_resource'] = convert_fig_to_bytes(fig)
    
    bottleneck_by_activity = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean()
    tables['bottleneck_by_activity_data'] = bottleneck_by_activity.reset_index()
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
    tables['avg_cycle_time_by_phase_data'] = avg_cycle_time_by_phase.reset_index()
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
    metrics['resource_efficiency_data'] = data_para_plot

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
    st.info("A componente de RL irá correr numa amostra de 100 processos para garantir a performance.")
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
            self.last_day_hours_worked_per_resource = defaultdict(float)
            self.current_day_hours_worked_per_resource = defaultdict(float)
            return self.get_state()

        def get_state(self):
            # 4. O estado agora descreve o PORTFÓLIO, não um projeto
            num_active = len(self.active_projects)
            pending_tasks_count = 0
            high_prio_tasks_count = 0
        
            # --- INÍCIO CÁLCULO max_pending_days ---
            max_pending_days = 0 
            # ---------------------------------------
        
            # Calcular tarefas pendentes, alta prioridade e idade máxima
            for proj in self.active_projects.values():
                for task in proj['tasks'].values():
                     # Usamos _is_task_eligible sem check_resource_type para elegibilidade geral
                     if task['status'] in ['Pendente', 'Em Andamento'] and \
                        self._is_task_eligible(task, proj['risk_rating'], proj['tasks'], proj['dependencies']):
        
                          # Contar Pendentes (Apenas Status 'Pendente')
                          if task['status'] == 'Pendente':
                               pending_tasks_count += 1
                               if task['priority'] >= 4:
                                    high_prio_tasks_count += 1
        
                          # --- LÓGICA PARA max_pending_days ---
                          eligibility_date = task.get('eligibility_date')
                          if eligibility_date:
                              # Calcula dias de CALENDÁRIO desde que ficou elegível
                              pending_days_calendar = (self.current_date - eligibility_date).days 
                              if pending_days_calendar > max_pending_days:
                                  max_pending_days = pending_days_calendar
                          # ---------------------------------------
        
            day_of_week = self.current_date.weekday()
        
            # --- Lógica existente para Utilização de Recursos ---
            utilization_rates = []
            for res_type in self.resource_types: 
                total_capacity_today = 0
                if self.current_date.weekday() < 5:
                     pool = self.resources_by_type.get(res_type, pd.DataFrame())
                     if not pool.empty:
                          total_capacity_today = pool['daily_capacity'].sum()
        
                hours_worked_today = 0
                if hasattr(self, 'last_day_hours_worked_per_resource'):
                    pool_ids = self.resources_by_type.get(res_type, pd.DataFrame())['resource_id'].tolist()
                    hours_worked_today = sum(self.last_day_hours_worked_per_resource.get(res_id, 0) for res_id in pool_ids)
        
                if total_capacity_today > 0:
                    utilization = hours_worked_today / total_capacity_today
                else:
                    utilization = 0.0
                utilization_rates.append(round(utilization, 1)) 
            # --- Fim Lógica Utilização ---
        
            # O novo estado inclui max_pending_days ANTES das taxas de utilização
            new_state_tuple = (num_active, pending_tasks_count, high_prio_tasks_count, day_of_week, max_pending_days) + tuple(utilization_rates)
        
            return new_state_tuple

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
                    if task_data['status'] in ['Pendente', 'Em Andamento']:
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
            # Guarda as horas do dia ANTERIOR antes de calcular as de hoje
            self.last_day_hours_worked_per_resource = getattr(self, 'current_day_hours_worked_per_resource', defaultdict(float))
            # 6a. Ativar novos projetos que começam hoje
            projects_to_activate = [p for p in self.future_projects if p['start_date'] == self.current_date]
            for proj_data in projects_to_activate:
                proj_id = proj_data['project_id']
                self.active_projects[proj_id] = {
                    'tasks': {str(t['task_id']): {
                    'status': 'Pendente', 'progress': 0.0,
                    'estimated_effort': real_hours_per_task.get(str(t['task_id']), t['estimated_effort'] * 8.0),
                    'priority': t['priority'], 'task_type': t['task_type'], 'task_id': str(t['task_id']),
                    'eligibility_date': self.current_date if not self.dependencies_by_project.get(proj_id, {}).get(str(t['task_id'])) else None
                } for t in self.tasks_by_project.get(proj_id, [])},
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

            # O loop agora processa a lista de ações detalhada
            for res_type, task_type, proj_id, task_id, res_id, hours_to_work in action_list:
                
                task_data = self.active_projects[proj_id]['tasks'][task_id]
                if task_data['status'] == 'Concluída':
                    continue
                if task_data['status'] == 'Pendente':
                    task_data['status'] = 'Em Andamento'

                # O esforço e progresso são medidos em HORAS.
                remaining_effort = task_data['estimated_effort'] - task_data['progress']
                
                # As horas a trabalhar já foram calculadas, mas não podem exceder o que falta
                actual_hours_worked = min(hours_to_work, remaining_effort)
                if actual_hours_worked <= 0:
                    continue

                res_info = self.df_resources[self.df_resources['resource_id'] == res_id].iloc[0]
                cost_today = actual_hours_worked * float(res_info['cost_per_hour'])

                self.active_projects[proj_id]['current_cost'] += cost_today
                task_data['progress'] += actual_hours_worked
                
                # Para os logs detalhados
                daily_cost_by_project[proj_id] += cost_today
                daily_hours_by_project[proj_id] += actual_hours_worked

                if task_data['progress'] >= task_data['estimated_effort']:
                    task_data['status'] = 'Concluída'
                    task_data['completion_date'] = self.current_date
                    reward_from_tasks += task_data['priority'] * self.rewards['priority_task_bonus_factor']

                    # --- INÍCIO DO NOVO BLOCO: Atualizar Elegibilidade de Sucessoras ---
                    completed_task_id = task_id # ID da tarefa que acabou de ser concluída
                    # Iterar sobre todos os projetos ativos para encontrar sucessoras
                    for check_proj_id, check_proj_state in self.active_projects.items():
                        for check_task_id, check_task_data in check_proj_state['tasks'].items():
                            # Verificar se esta tarefa é sucessora da que acabou de terminar
                            predecessor_id = check_proj_state['dependencies'].get(check_task_id)
                            if predecessor_id == completed_task_id:
                                # Se a tarefa sucessora ainda não tinha data de elegibilidade, define-a para hoje
                                if check_task_data['eligibility_date'] is None:
                                    check_task_data['eligibility_date'] = self.current_date
                    # --- FIM DO NOVO BLOCO ---
                
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
                    final_duration = np.busday_count(proj_state['start_date'].date(), self.current_date.date())
                    real_duration = original_proj_info['total_duration_days']
                    
                    time_diff = real_duration - final_duration
                    reward_from_tasks += self.rewards['completion_base']
                    reward_from_tasks += time_diff * self.rewards['per_day_early_bonus'] if time_diff >= 0 else time_diff * self.rewards['per_day_late_penalty']
                    reward_from_tasks -= proj_state['current_cost'] * self.rewards['cost_impact_factor']
                    # Guardar resultados
                    self.completed_projects[proj_id] = {'simulated_duration': final_duration, 'simulated_cost': proj_state['current_cost']}

            for proj_id in projects_to_remove:
                del self.active_projects[proj_id]

            # 6d. Finalizar o dia
            self.current_date += timedelta(days=1)
            # --- Alterado: Calcular Penalização DINÂMICA por Tarefas Pendentes/Atrasadas ---
            pending_penalty = 0
            for proj in self.active_projects.values():
                for task in proj['tasks'].values():
                    # Considera tarefas PENDENTES OU EM ANDAMENTO que estão ELEGÍVEIS mas não concluídas
                    if task['status'] in ['Pendente', 'Em Andamento'] and \
                       self._is_task_eligible(task, proj['risk_rating'], proj['tasks'], proj['dependencies']):
    
                        eligibility_date = task.get('eligibility_date')
                        if eligibility_date: 
                            # Calcula há quantos dias de CALENDÁRIO está elegível
                            pending_days_calendar = (self.current_date - eligibility_date).days
                            # Penalização aumenta linearmente com os dias pendente (mínimo 0)
                            penalty_for_this_task = self.rewards['pending_task_penalty_factor'] * max(0, pending_days_calendar)
                            pending_penalty += penalty_for_this_task
            # --- Fim Alterado ---
        
            total_reward = reward_from_tasks - self.rewards['daily_time_penalty'] - pending_penalty # Modificado para incluir a nova penalização
            
            done = not self.active_projects and not self.future_projects

            self.current_day_hours_worked_per_resource = resources_hours_today
            
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
    # Calcular o total de horas reais trabalhadas por tarefa a partir das alocações
    # Usamos df_resource_allocations (que já é a cópia filtrada da amostra)
    real_hours_per_task_series = df_resource_allocations.groupby('task_id')['hours_worked'].sum()
    real_hours_per_task = real_hours_per_task_series.to_dict()
    # --- FIM DO NOVO BLOCO ---
            
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
            # Se for fim de semana, apenas avança o dia e continua para a próxima iteração do loop
            if env.current_date.weekday() >= 5:
                env.current_date += timedelta(days=1)
                continue
            
            possible_actions_full = env.get_possible_actions_for_state()
            
            action_list_for_step = []
            
            # --- NOVA LÓGICA DE ALOCAÇÃO DIÁRIA ---
            # 1. Obter todas as tarefas de trabalho possíveis para hoje
            work_actions = [a for a in possible_actions_full if a[1] != 'idle']
            simplified_options = list(set([(a[0], a[1]) for a in work_actions]))

            # 2. Criar um pool de horas disponíveis para todos os recursos hoje
            available_hours_per_resource = {res_id: cap for res_id, cap in env.resource_capacity_map.items()}

            # 3. Enquanto houver trabalho a fazer e recursos com horas disponíveis
            while any(h > 0 for h in available_hours_per_resource.values()) and work_actions:
                
                # Pergunta ao agente qual o melhor TIPO de tarefa a fazer AGORA
                chosen_simplified_action = agent.choose_action(state, simplified_options)
                if not chosen_simplified_action:
                    break # Se o agente não escolher nada, paramos por hoje

                # Encontrar a tarefa real de maior prioridade que corresponde à escolha
                candidate_tasks = [t for t in work_actions if (t[0], t[1]) == chosen_simplified_action]
                if not candidate_tasks:
                    # Se não houver mais tarefas deste tipo, remove a opção e tenta de novo
                    simplified_options.remove(chosen_simplified_action)
                    continue

                best_task_action = max(candidate_tasks, key=lambda a: env.active_projects[a[2]]['tasks'][a[3]]['priority'])
                res_type, task_type, proj_id, task_id = best_task_action
                
                # Encontrar um recurso DESOCUPADO para esta tarefa
                pool = env.resources_by_type[res_type]
                available_resources = [rid for rid in pool['resource_id'] if available_hours_per_resource.get(rid, 0) > 0]
                
                if not available_resources:
                    # Não há mais recursos deste tipo disponíveis, remove a opção e tenta de novo
                    simplified_options = [opt for opt in simplified_options if opt[0] != res_type]
                    continue

                # Atribui a tarefa a um recurso disponível
                chosen_res_id = random.choice(available_resources)
                
                # Calcula as horas que o recurso ainda pode trabalhar hoje
                hours_resource_can_work = available_hours_per_resource[chosen_res_id]
                # Calcula as horas que a tarefa ainda precisa (total)
                hours_task_needs = env.active_projects[proj_id]['tasks'][task_id]['estimated_effort'] - env.active_projects[proj_id]['tasks'][task_id]['progress']
                # Atribui o mínimo entre o que o recurso pode dar e o que a tarefa precisa
                hours_to_assign = min(hours_resource_can_work, hours_task_needs)
                
                # Adiciona a atribuição à lista de ações (agora com resource_id)
                action_list_for_step.append((res_type, task_type, proj_id, task_id, chosen_res_id, hours_to_assign))
                
                # Atualiza as horas disponíveis do recurso
                available_hours_per_resource[chosen_res_id] -= hours_to_assign

                # Remove a tarefa escolhida da lista de trabalho para este dia.
                work_actions.remove(best_task_action)
                simplified_options = list(set([(a[0], a[1]) for a in work_actions]))
            
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
        # Pular fim de semana
        if env.current_date.weekday() >= 5:
            env.current_date += timedelta(days=1)
            continue
            
        possible_actions_full = env.get_possible_actions_for_state()
        action_list_for_step = []
        
        # --- Lógica de Alocação Diária (idêntica ao treino) ---
        work_actions = [a for a in possible_actions_full if a[1] != 'idle']
        simplified_options = list(set([(a[0], a[1]) for a in work_actions]))

        available_hours_per_resource = {res_id: cap for res_id, cap in env.resource_capacity_map.items()}

        while any(h > 0 for h in available_hours_per_resource.values()) and work_actions:
            chosen_simplified_action = agent.choose_action(state, simplified_options)
            if not chosen_simplified_action:
                break 

            candidate_tasks = [t for t in work_actions if (t[0], t[1]) == chosen_simplified_action]
            if not candidate_tasks:
                simplified_options.remove(chosen_simplified_action)
                continue

            best_task_action = max(candidate_tasks, key=lambda a: env.active_projects[a[2]]['tasks'][a[3]]['priority'])
            res_type, task_type, proj_id, task_id = best_task_action
            
            pool = env.resources_by_type[res_type]
            available_resources = [rid for rid in pool['resource_id'] if available_hours_per_resource.get(rid, 0) > 0]
            
            if not available_resources:
                simplified_options = [opt for opt in simplified_options if opt[0] != res_type]
                continue

            chosen_res_id = random.choice(available_resources)
            
            # Calcula as horas que o recurso ainda pode trabalhar hoje
            hours_resource_can_work = available_hours_per_resource[chosen_res_id]
            # Calcula as horas que a tarefa ainda precisa (total)
            hours_task_needs = env.active_projects[proj_id]['tasks'][task_id]['estimated_effort'] - env.active_projects[proj_id]['tasks'][task_id]['progress']
            # Atribui o mínimo entre o que o recurso pode dar e o que a tarefa precisa
            hours_to_assign = min(hours_resource_can_work, hours_task_needs)
            
            action_list_for_step.append((res_type, task_type, proj_id, task_id, chosen_res_id, hours_to_assign))
            
            available_hours_per_resource[chosen_res_id] -= hours_to_assign
            # --- LINHA DE CORREÇÃO CRÍTICA (AVALIAÇÃO) ---
            # Remove a tarefa escolhida da lista de trabalho para este dia.
            work_actions.remove(best_task_action)
            simplified_options = list(set([(a[0], a[1]) for a in work_actions]))
            
            # Se a tarefa ficar sem esforço restante, removemo-la das opções para o resto do dia
            # (Pequena otimização para não continuar a tentar atribuir uma tarefa "quase completa")
            task_state = env.active_projects[proj_id]['tasks'][task_id]
            if (task_state['estimated_effort'] - task_state['progress']) < 1:
                 work_actions = [a for a in work_actions if a[3] != task_id]
                 simplified_options = list(set([(a[0], a[1]) for a in work_actions]))


        next_state, _, done = env.step(action_list_for_step)
        state = next_state
        if env.current_date > env.df_projects['start_date'].max() + timedelta(days=1000): 
            break

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
        'Métrica': ['Duração (dias úteis)', 'Custo (€)'],
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
    

# --- INÍCIO DO NOVO MOTOR DE DIAGNÓSTICO (V8 - LÓGICA DE INVESTIGAÇÃO FLEXÍVEL) ---
from scipy import stats
import matplotlib.dates as mdates
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import math # Para verificar isnan

# --- MAPEAMENTO DE REFERÊNCIAS (MANUAL - REVER/COMPLETAR) ---
CARD_TITLE_MAP = {
    "Cartão 1": "Matriz de Performance (Custo vs Prazo)", "Cartão 2": "Top 5 Processos Mais Caros",
    "Cartão 3": "Séries Temporais de KPIs de Performance", "Cartão 4": "Distribuição do Status dos Processos",
    "Cartão 5": "Custo Médio dos Processos por Trimestre", "Cartão 6": "Alocação de Custos por Orçamento e Recurso",
    "Cartão 7": "Custo por Tipo de Recurso", "Cartão 8": "Top 5 Processos Mais Longos",
    "Cartão 9": "Custo Médio por Dia ao Longo do Tempo", "Cartão 10": "Custo Real vs. Orçamento por Processo",
    "Cartão 11": "Distribuição do Custo por Dia (Eficiência)", "Cartão 12": "Evolução do Volume e Tamanho dos Processos",
    "Cartão 13": "Relação Lead Time vs Throughput", "Cartão 14": "Distribuição do Lead Time",
    "Cartão 15": "Distribuição da Duração dos Processos (PM)", "Cartão 16": "Gráfico Acumulado de Throughput",
    "Cartão 17": "Performance de Prazos por Trimestre", "Cartão 18": "Duração Média por Fase do Processo",
    "Cartão 19": "Distribuição do Throughput (horas)", "Cartão 20": "Boxplot do Throughput (horas)",
    "Cartão 21": "Atividades por Dia da Semana", "Cartão 22": "Evolução da Performance (Prazo e Custo)",
    "Cartão 23": "Diferença entre Data Real e Planeada", "Cartão 26": "Distribuição de Recursos por Tipo",
    "Cartão 27": "Recursos por Média de Tarefas/Processo", "Cartão 29": "Impacto do Tamanho da Equipa no Atraso (PM)",
    "Cartão 31": "Top 10 Recursos por Horas Trabalhadas (PM)", "Cartão 32": "Top 10 Handoffs entre Recursos",
    "Cartão 33": "Métricas de Eficiência: Top Recursos", "Cartão 36": "Atraso Médio por Recurso",
    "Cartão 37": "Relação entre Skill e Performance", "Cartão 39": "Rede de Recursos por Função",
    "Cartão 40": "Rede Social de Recursos (Handovers)", "Cartão 41": "Heatmap de Esforço (Recurso vs Atividade)",
    "Cartão 42": "Heatmap de Performance no Processo (Gargalos)", "Cartão 44": "Gargalos: Tempo de Serviço vs. Espera",
    "Cartão 45": "Top 10 Handoffs por Custo de Espera", "Cartão 46": "Top Recursos por Tempo de Espera Gerado",
    "Cartão 49": "Tempo Médio de Execução por Atividade", "Cartão 51": "Evolução do Tempo Médio de Espera",
    "Cartão 52": "Top 10 Handoffs por Tempo de Espera",
    "Cartão 55": "Análise de Tempo entre Marcos do Processo",
    "Cartão 57": "Tempo Médio de Espera por Atividade", "Cartão 58": "Matriz de Tempo de Espera entre Atividades (horas)",
    "Cartão 61": "Métricas (Inductive Miner)", "Cartão 65": "Top 10 Variantes de Processo por Frequência",
    "Cartão 70": "Principais Loops de Rework", "Cartão 71": "Score de Conformidade ao Longo do Tempo",
    "Cartão 72": "Distribuição de Tarefas por Prioridade", "Cartão 74": "Distribuição da Complexidade dos Processos",
    "Cartão 76": "Relação entre Complexidade e Atraso", "Cartão 77": "Relação entre Dependências e Desvio de Custo",
    "KPIs Saúde Geral": "1. Saúde Geral (KPIs de Baseline)",
    "Análise Pareto": "Análise Geral (Princípio de Pareto)",
    "Análise Rede Social": "Análise Geral (Rede Social)",
}

SECTION_MAP = {
    'resumo_executivo': 'Resumo Executivo',
    'diagnostico_custos_atrasos': 'Análise: Custos e Prazos',
    'diagnostico_recursos_equipas': 'Análise: Recursos e Equipas',
    'diagnostico_gargalos_esperas': 'Análise: Gargalos e Rework',
    'diagnostico_fluxo_conformidade': 'Análise: Fluxo e Variabilidade',
}

class DiagnosticEngineV5:
    """ Motor V8: Lógica de investigação flexível, focada em padrões e não em valores fixos. """

    def __init__(self, tables_pre, metrics, data_frames, tables_eda):
        self.tables_pre_orig = tables_pre if tables_pre else {}
        self.metrics_orig = metrics if metrics else {}
        self.data_frames_orig = data_frames if data_frames else {}
        self.tables_eda_orig = tables_eda if tables_eda else {}

        # DataFrames base
        self.df_projects_base = self.data_frames_orig.get('projects', pd.DataFrame()).copy()
        self.df_tasks_base = self.data_frames_orig.get('tasks', pd.DataFrame()).copy()
        self.df_resources_base = self.data_frames_orig.get('resources', pd.DataFrame()).copy()
        self.df_full_context_base = self.data_frames_orig.get('full_context', pd.DataFrame()).copy()

        # KPIs e Métricas Chave
        self.kpis = self.tables_pre_orig.get('kpi_data', {})
        self.delay_kpis = self.tables_pre_orig.get('cost_of_delay_kpis', {})
        self.df_variants = self.tables_pre_orig.get('variants_table', pd.DataFrame())
        self.metrics_im = self.metrics_orig.get('inductive_miner', {})
        self.df_efficiency_metrics = self.metrics_orig.get('resource_efficiency_data', pd.DataFrame())
        
        # Mapeamento robusto da ordem das tarefas
        self.task_order_map = {
            'Onboarding/Recolha de Dados': 1,
            'Validação KYC e Conformidade': 2,
            'Análise Documental': 3,
            'Análise de Risco e Proposta': 4,
            'Avaliação da Imóvel': 5,
            'Decisão de Crédito e Condições': 6,
            'Preparação Legal': 7,
            'Fecho/Desembolso': 8,
            # Fallbacks para nomes comuns
            'Onboarding': 1, 'KYC': 2, 'Risco': 4, 'Avaliação': 5, 'Decisão': 6, 'Legal': 7, 'Fecho': 8
        }

        self.insights = { k: [] for k in ['saude_geral'] + list(SECTION_MAP.keys()) }
        self.narrative_flags = {} # O cérebro da V8

        self._preprocess_data()
        self._prepare_aggregated_data() # Esta função continha o bug

    def _initialize_aggregated_dfs(self):
        # (Mantém-se igual)
        attrs = [
            'df_handoffs', 'df_resource_avg_events', 'df_workload', 'df_bottleneck_res',
            'df_activity_wait_stats', 'df_cost_by_resource_type', 'df_avg_cycle_time_phase',
            'df_kpi_temporal', 'df_monthly_fitness', 'df_monthly_kpis_eda',
            'df_monthly_wait_time', 'df_wait_by_activity', 'df_milestone_transitions',
            'df_wait_matrix', 'df_perf_stats_data', 'df_activity_service_times'
        ]
        for attr in attrs: setattr(self, attr, pd.DataFrame())
        self.resource_handoffs_counts = Counter()

    def _preprocess_data(self):
        # (Mantém-se igual)
        try:
            # Projetos
            if not self.df_projects_base.empty:
                for col in ['start_date', 'end_date', 'planned_end_date']:
                    if col in self.df_projects_base.columns: self.df_projects_base[col] = pd.to_datetime(self.df_projects_base[col], errors='coerce')
                num_cols_proj = ['total_actual_cost', 'cost_diff', 'days_diff', 'actual_duration_days', 'num_resources', 'complexity_score', 'dependency_count', 'budget_impact', 'cost_per_day']
                for col in num_cols_proj:
                     if col in self.df_projects_base.columns: self.df_projects_base[col] = pd.to_numeric(self.df_projects_base[col], errors='coerce')
                if 'end_date' in self.df_projects_base.columns:
                    end_date_dt = pd.to_datetime(self.df_projects_base['end_date'], errors='coerce')
                    if not end_date_dt.isna().all():
                        self.df_projects_base['completion_month'] = end_date_dt.dt.to_period('M').astype(str)
                        self.df_projects_base['completion_quarter'] = end_date_dt.dt.to_period('Q').astype(str)
                    else:
                         self.df_projects_base['completion_month'] = ""
                         self.df_projects_base['completion_quarter'] = ""

            # Tarefas
            if not self.df_tasks_base.empty:
                for col in ['start_date', 'end_date']:
                     if col in self.df_tasks_base.columns: self.df_tasks_base[col] = pd.to_datetime(self.df_tasks_base[col], errors='coerce')
                if 'task_duration_days' not in self.df_tasks_base.columns and 'start_date' in self.df_tasks_base.columns and 'end_date' in self.df_tasks_base.columns:
                     self.df_tasks_base['task_duration_days'] = (self.df_tasks_base['end_date'] - self.df_tasks_base['start_date']).dt.days
                for col in ['priority', 'task_duration_days']:
                    if col in self.df_tasks_base.columns: self.df_tasks_base[col] = pd.to_numeric(self.df_tasks_base[col], errors='coerce')

            # Contexto Completo
            if not self.df_full_context_base.empty:
                 num_cols_fc = ['cost_of_work', 'hours_worked', 'skill_level', 'days_diff', 'priority', 'total_actual_cost', 'budget_impact', 'num_resources']
                 for col in num_cols_fc:
                     if col in self.df_full_context_base.columns: self.df_full_context_base[col] = pd.to_numeric(self.df_full_context_base[col], errors='coerce')
                 date_cols_fc = [c for c in ['allocation_date', 'start_date', 'end_date', 'planned_end_date'] if c in self.df_full_context_base.columns]
                 for dc in date_cols_fc: self.df_full_context_base[dc] = pd.to_datetime(self.df_full_context_base[dc], errors='coerce')
        except Exception as e: print(f"Erro _preprocess_data: {e}")

    def _prepare_aggregated_data(self):
        # (Função com a CORREÇÃO CRÍTICA)
         # --- Cálculos baseados em df_full_context_base ---
        if not self.df_full_context_base.empty:
            try: self.df_cost_by_resource_type = self.df_full_context_base.groupby('resource_type')['cost_of_work'].sum().reset_index()
            except Exception as e: print(f"Erro _prepare: df_cost_by_resource_type: {e}")
            try:
                agg_data = self.df_full_context_base.groupby("resource_name").agg(uc=('project_id', 'nunique'), ec=('task_id', 'count')).reset_index()
                agg_data["avg_events_per_case"] = agg_data.apply(lambda r: r["ec"] / r["uc"] if r["uc"] > 0 else 0, axis=1)
                self.df_resource_avg_events = agg_data
            except Exception as e: print(f"Erro _prepare: df_resource_avg_events: {e}")
            try: self.df_workload = self.df_full_context_base.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).reset_index()
            except Exception as e: print(f"Erro _prepare: df_workload: {e}")
            try:
                 self.df_activity_service_times = self.df_full_context_base.groupby('task_name')['hours_worked'].mean().reset_index()
                 self.df_activity_service_times['service_time_days'] = self.df_activity_service_times['hours_worked'] / 8
            except Exception as e: print(f"Erro _prepare: df_activity_service_times: {e}")

        # --- Cálculos baseados em df_tasks_base e df_projects_base ---
        if not self.df_tasks_base.empty and not self.df_projects_base.empty:
            try:
                df_tasks_analysis = self.df_tasks_base.dropna(subset=['project_id', 'start_date', 'end_date']).sort_values(['project_id', 'start_date']).copy()
                df_tasks_analysis['previous_task_end'] = df_tasks_analysis.groupby('project_id')['end_date'].shift(1)
                df_tasks_analysis['wait_sec'] = (df_tasks_analysis['start_date'] - df_tasks_analysis['previous_task_end']).dt.total_seconds()
                df_tasks_analysis['waiting_time_days'] = df_tasks_analysis['wait_sec'].apply(lambda x: x/(24*3600) if pd.notna(x) and x > 0 else 0)
                df_tasks_analysis['service_time_days'] = (df_tasks_analysis['end_date'] - df_tasks_analysis['start_date']).dt.total_seconds() / (24*3600)
                df_tasks_analysis.loc[df_tasks_analysis['service_time_days'] < 0, 'service_time_days'] = 0

                if 'task_type' in df_tasks_analysis.columns: self.df_activity_wait_stats = df_tasks_analysis.groupby('task_type')[['service_time_days', 'waiting_time_days']].mean().reset_index()
                self.df_wait_by_activity = df_tasks_analysis.groupby('task_name')['waiting_time_days'].mean().reset_index()
                self.df_wait_by_activity['sojourn_time_hours'] = self.df_wait_by_activity['waiting_time_days'] * 24

                # Usa 'completion_month' de df_projects_base (string YYYY-MM)
                if 'completion_month' in self.df_projects_base.columns:
                     df_wait_evo = df_tasks_analysis.merge(self.df_projects_base[['project_id', 'completion_month']], on='project_id', how='left')
                     if not df_wait_evo.empty and 'completion_month' in df_wait_evo.columns and df_wait_evo['completion_month'].nunique() > 1:
                          # --- INÍCIO DA CORREÇÃO DE ORDENAÇÃO ---
                          self.df_monthly_wait_time = df_wait_evo.groupby('completion_month')['waiting_time_days'].mean().reset_index().sort_values('completion_month')
                          # --- FIM DA CORREÇÃO DE ORDENAÇÃO ---


                df_tasks_analysis['previous_task_name'] = df_tasks_analysis.groupby('project_id')['task_name'].shift(1)
                if 'previous_task_name' in df_tasks_analysis.columns: self.df_wait_matrix = df_tasks_analysis.pivot_table(index='previous_task_name', columns='task_name', values='waiting_time_days', aggfunc='mean')

            except Exception as e: print(f"Erro _prepare: Cálculos Tempo Espera: {e}")

            # --- Duração Média por Fase (Regra 18) ---
            try:
                 def get_phase(task_type):
                     if not isinstance(task_type, str): return 'Fase Desconhecida'
                     if task_type in ['Onboarding', 'Validação KYC e Conformidade', 'Análise Documental']: return '1. Onboarding'
                     elif task_type in ['Análise de Risco e Proposta']: return '2. Análise Risco'
                     elif task_type in ['Avaliação da Imóvel']: return '3. Avaliação'
                     elif task_type in ['Decisão de Crédito e Condições']: return '4. Decisão'
                     elif task_type in ['Fecho', 'Preparação Legal']: return '5. Contratação'
                     return 'Outra Fase'

                 if 'task_type' in self.df_tasks_base.columns:
                     df_ph = self.df_tasks_base.dropna(subset=['project_id', 'start_date', 'end_date']).copy()
                     df_ph['phase'] = df_ph['task_type'].apply(get_phase)
                     ph_times = df_ph.groupby(['project_id', 'phase']).agg(start=('start_date', 'min'), end=('end_date', 'max')).reset_index()
                     ph_times['cycle_time_days'] = (ph_times['end'] - ph_times['start']).dt.days
                     ph_times = ph_times[ph_times['cycle_time_days'] >= 0]
                     if not ph_times.empty: self.df_avg_cycle_time_phase = ph_times.groupby('phase')['cycle_time_days'].mean().reset_index()
            except Exception as e: print(f"Erro _prepare: Duração Fase: {e}")

        # --- Cálculos de Handoffs e Rede Social (log reconstruído) ---
        try:
             if not self.df_full_context_base.empty and 'project_id' in self.df_full_context_base.columns:
                log_cols = ['project_id', 'task_name', 'allocation_date', 'resource_name']
                if all(c in self.df_full_context_base.columns for c in log_cols):
                    log_df = self.df_full_context_base[log_cols].dropna().copy()
                    log_df.rename(columns={'project_id': 'case', 'task_name': 'act', 'allocation_date': 'ts', 'resource_name': 'res'}, inplace=True)
                    log_df['ts'] = pd.to_datetime(log_df['ts'], errors='coerce')
                    log_df.dropna(subset=['ts', 'case', 'res'], inplace=True)
                    log_df.sort_values(['case', 'ts'], inplace=True)

                    if not log_df.empty:
                        # Handoffs Atividade
                        log_df['prev_end'] = log_df.groupby('case')['ts'].shift(1)
                        log_df['wait_sec']=(log_df['ts'] - log_df['prev_end']).dt.total_seconds()
                        log_df['h_days']=log_df['wait_sec'].apply(lambda x: x/(24*3600) if pd.notna(x) and x > 0 else 0)
                        log_df['prev_act']=log_df.groupby('case')['act'].shift(1)
                        h_stats = log_df.groupby(['prev_act', 'act'])['h_days'].mean().reset_index().sort_values('h_days', ascending=False)
                        h_stats['transition'] = h_stats['prev_act'].fillna('Start') + ' -> ' + h_stats['act'].fillna('End')
                        cost_day = self.df_projects_base['cost_per_day'].mean() if 'cost_per_day' in self.df_projects_base.columns else 0
                        h_stats['cost_wait'] = h_stats['h_days'] * (cost_day if not pd.isna(cost_day) else 0)
                        h_stats.rename(columns={'h_days': 'handoff_time_days', 'cost_wait': 'estimated_cost_of_wait'}, inplace=True)
                        self.df_handoffs = h_stats

                        # Handoffs Recursos
                        res_handoffs = Counter()
                        last_case, last_res = None, None
                        for _, row in log_df.iterrows():
                            case, res = row['case'], row['res']
                            if case == last_case and pd.notna(last_res) and pd.notna(res) and res != last_res: res_handoffs[(last_res, res)] += 1
                            last_case, last_res = case, res
                        self.resource_handoffs_counts = res_handoffs
                else: print("Aviso _prepare: Colunas essenciais em falta em df_full_context.")
             else: print("Aviso _prepare: df_full_context vazio ou sem project_id.")
        except Exception as e: print(f"Erro _prepare: Cálculos Log: {e}")

        # --- Obter dados pré-calculados que não são recalculados ---
        try:
             # --- INÍCIO DA CORREÇÃO DE ORDENAÇÃO ---
             # Tenta obter dos dados processados pela EDA, garantindo a ordenação
             if not self.df_projects_base.empty and 'completion_month' in self.df_projects_base.columns:
                monthly_kpis_calc = self.df_projects_base.groupby('completion_month').agg(
                    mean_days_diff=('days_diff', 'mean'), 
                    mean_cost_diff=('cost_diff', 'mean'), 
                    completed_projects=('project_id', 'count'), 
                    mean_duration=('actual_duration_days', 'mean')
                ).reset_index().sort_values('completion_month') # GARANTE A ORDEM
                if not monthly_kpis_calc.empty:
                    self.df_monthly_kpis_eda = monthly_kpis_calc
             
             # Tenta obter dos dados de PM, garantindo a ordenação
             if not self.df_projects_base.empty and 'completion_month' in self.df_projects_base.columns:
                kpi_temporal_calc = self.df_projects_base.groupby('completion_month').agg(
                    avg_lead_time=('actual_duration_days', 'mean'), 
                    throughput=('project_id', 'count')
                ).reset_index().sort_values('completion_month') # GARANTE A ORDEM
                if not kpi_temporal_calc.empty:
                    self.df_kpi_temporal = kpi_temporal_calc
             # --- FIM DA CORREÇÃO DE ORDENAÇÃO ---

             # Fallbacks (se existirem nos dados pré-processados)
             if self.df_monthly_kpis_eda.empty:
                 self.df_monthly_kpis_eda = self.tables_eda_orig.get('monthly_kpis_data', pd.DataFrame())
             if self.df_kpi_temporal.empty:
                self.df_kpi_temporal = self.metrics_orig.get('kpi_temporal_data', pd.DataFrame())

             self.df_monthly_fitness = self.metrics_orig.get('monthly_fitness_data', pd.DataFrame())
             self.df_milestone_transitions = self.metrics_orig.get('milestone_transition_data', pd.DataFrame())
             if self.df_wait_by_activity.empty: self.df_wait_by_activity = self.metrics_orig.get('wait_by_activity_data', pd.DataFrame())
        except Exception as e: print(f"Erro _prepare: Obter dados pré-calculados: {e}")


    def _add_insight(self, section_key, title, detail, card_key, level='problema', priority=10):
        """ Adiciona insight com referência mapeada e prioridade. """
        try:
            section_name = SECTION_MAP.get(section_key, section_key)
            card_title = CARD_TITLE_MAP.get(card_key, card_key)
            reference = f"Secção '{section_name}' > Cartão '{card_title}'" if card_title and section_name else card_key

            if section_key not in self.insights: self.insights[section_key] = []
            
            self.insights[section_key].append({
                'titulo': title, 'detalhe': detail, 'cartao_ref': reference, 'level': level, 'priority': priority
            })
        except Exception as e:
            print(f"Erro em _add_insight ({title}): {e}")

    def _get_trend(self, series_data, series_index=None):
        """ Calcula slope usando índice numérico sequencial para robustez. """
        if series_data is None: return 0
        if isinstance(series_data, pd.DataFrame):
            numeric_cols = series_data.select_dtypes(include=np.number).columns
            if not numeric_cols.empty: y_series = series_data[numeric_cols[0]]
            else: return 0
        elif isinstance(series_data, pd.Series): y_series = series_data
        else: return 0

        y = y_series.copy().dropna() # Remove NaNs de Y ANTES de criar X
        if y.empty or len(y) < 2: return 0

        x = np.arange(len(y)) # Usa índice numérico sequencial

        try:
            if np.all(x == x[0]) or np.all(y == y.iloc[0]): return 0 # Verifica variação
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope if not (math.isnan(slope) or math.isinf(slope)) else 0
        except ValueError: return 0
        except Exception as e: print(f"Erro inesperado em _get_trend: {e}"); return 0

    # --- NOVO MOTOR DE ANÁLISE (V8) ---

    def run(self):
        """ Orquestrador V8: Corre a análise contextual, constrói a narrativa e depois recolhe factos. """
        self._check_kpis() # 1. Preenche os KPIs de saúde geral
        self._run_contextual_analysis() # 2. O cérebro: define as flags narrativas
        self._build_executive_summary() # 3. A voz: escreve a história principal
        
        # 4. Recolhe factos de suporte (agora replicando a análise manual)
        self._check_custos_atrasos_facts()
        self._check_performance_prazos_facts()
        self._check_recursos_equipas_facts()
        self.
        _check_gargalos_esperas_facts()
        self._check_fluxo_conformidade_facts()
        
        # 5. Ordena os insights (corrigido para ignorar 'saude_geral')
        for section in self.insights:
            if section != 'saude_geral' and self.insights[section]:
                self.insights[section] = sorted(self.insights[section], key=lambda x: x['priority'])
                
        return self.insights

    def _run_contextual_analysis(self):
        """ (V8) PASSO 2: O CÉREBRO - Define as flags narrativas com base nos dados. """
        flags = {}
        
        # --- 1. Análise de Eficiência (Baseline) ---
        if not self.df_projects_base.empty and 'days_diff' in self.df_projects_base.columns:
            mean_days_diff = self.df_projects_base['days_diff'].mean()
            flags['mean_days_diff'] = mean_days_diff
            if not pd.isna(mean_days_diff) and mean_days_diff < -3:
                flags['baseline_efficient'] = True # Termina adiantado
            elif not pd.isna(mean_days_diff) and mean_days_diff > 3:
                flags['baseline_inefficient'] = True # Termina atrasado
            
        # --- 2. Análise de Tendência (Degradação) ---
        # (USA O DF_MONTHLY_KPIS_EDA QUE FOI CORRIGIDO E ORDENADO EM _prepare_aggregated_data)
        if not self.df_monthly_kpis_eda.empty and len(self.df_monthly_kpis_eda) > 3:
            trend_days_diff = self._get_trend(self.df_monthly_kpis_eda['mean_days_diff']) # dias/mês
            trend_duration = self._get_trend(self.df_monthly_kpis_eda['mean_duration']) # dias/mês
            flags['trend_days_diff'] = trend_days_diff
            flags['trend_duration'] = trend_duration
            
            # Se o adiantamento está a diminuir (slope > 0) ou o atraso a aumentar (slope > 0)
            if trend_days_diff > 0.2: flags['is_degrading_prazo'] = True
            if trend_duration > 0.2: flags['is_degrading_duracao'] = True
        
        # --- 3. Análise de Variabilidade (Caos vs. Padrão) ---
        precision = self.metrics_im.get('Precisão', 1.0)
        flags['precision'] = precision
        if precision < 0.7: flags['is_chaotic'] = True
        
        if not self.df_variants.empty and 'percentage' in self.df_variants.columns:
            # --- CORREÇÃO BUG DE FORMATAÇÃO ---
            # A coluna 'percentage' no seu V5 está como 0.5, 0.3 etc. (e não 50, 30)
            top_1_perc = self.df_variants.iloc[0]['percentage']
            top_10_perc = self.df_variants.head(10)['percentage'].sum()
            flags['top_1_variant_perc'] = top_1_perc
            flags['top_10_variant_perc'] = top_10_perc
            if top_1_perc < 20: flags['is_chaotic'] = True # Confirma o caos
            if top_1_perc > 70: flags['is_standardized'] = True
        
        # --- 4. Análise de Causa-Raiz (Rework vs. Filas) ---
        self.detected_rework_loops = self._find_rework_in_wait_matrix()
        if self.detected_rework_loops:
            flags['has_rework'] = True
            flags['rework_loops'] = self.detected_rework_loops
            
        if not self.df_activity_wait_stats.empty:
            df_wait = self.df_activity_wait_stats.copy()
            df_wait['wait_ratio'] = df_wait['waiting_time_days'] / (df_wait['service_time_days'] + 0.01)
            # Se alguma atividade tem mais tempo de espera do que de serviço
            if (df_wait['wait_ratio'] > 1.5).any():
                flags['has_high_queues'] = True
                flags['queue_bottlenecks'] = df_wait.nlargest(3, 'wait_ratio')['task_type'].tolist()
        
        if not self.df_activity_service_times.empty:
            df_serv = self.df_activity_service_times.copy()
            mean_service = df_serv['service_time_days'].mean()
            # Se alguma tarefa demora 3x mais que a média para ser executada
            if (df_serv['service_time_days'] > (mean_service * 3)).any():
                flags['has_service_bottlenecks'] = True
                flags['service_bottlenecks'] = df_serv.nlargest(3, 'service_time_days')['task_name'].tolist()

        # --- 5. Análise de Recursos ---
        if not self.df_workload.empty and not self.df_resources_base.empty:
            total_hours = self.df_workload['hours_worked'].sum()
            top_20_perc_count = int(len(self.df_resources_base) * 0.2)
            if top_20_perc_count > 0 and total_hours > 0:
                top_20_perc_hours = self.df_workload.head(top_20_perc_count)['hours_worked'].sum()
                if (top_20_perc_hours / total_hours) > 0.8:
                    flags['has_pareto_herois'] = True # Princípio de Pareto 80/20
                    
        if self.resource_handoffs_counts:
            top_handoff = self.resource_handoffs_counts.most_common(1)[0]
            if top_handoff[1] > (len(self.df_projects_base) * 0.1): # Se o handoff mais comum ocorre em > 10% dos casos
                flags['has_resource_hub'] = True
                flags['resource_hub_names'] = f"{top_handoff[0][0]} -> {top_handoff[0][1]}"

        self.narrative_flags = flags

    def _find_rework_in_wait_matrix(self):
        """ (V8) Deteta Rework usando a Matriz de Espera e a ordem lógica das tarefas. """
        rework_loops = []
        if self.df_wait_matrix.empty:
            return rework_loops
            
        # Transforma a matriz (índice, coluna, valor)
        wait_matrix_flat = self.df_wait_matrix.stack().reset_index()
        wait_matrix_flat.columns = ['previous_task_name', 'task_name', 'wait_days']
        wait_matrix_flat = wait_matrix_flat[wait_matrix_flat['wait_days'] > 0]
        
        if wait_matrix_flat.empty:
            return rework_loops
            
        def get_order(task_name_str):
            if not isinstance(task_name_str, str): return 99
            # Tenta encontrar correspondência exata primeiro
            for key, order in self.task_order_map.items():
                if key.lower() == task_name_str.lower():
                    return order
            # Tenta encontrar correspondência parcial
            for key, order in self.task_order_map.items():
                if key.lower() in task_name_str.lower():
                    return order
            return 99 # Ordem desconhecida

        wait_matrix_flat['from_order'] = wait_matrix_flat['previous_task_name'].apply(get_order)
        wait_matrix_flat['to_order'] = wait_matrix_flat['task_name'].apply(get_order)
        
        # Rework = ir de uma ordem alta para uma ordem baixa (ex: 5 -> 4) E não ser desconhecido
        df_rework = wait_matrix_flat[
            (wait_matrix_flat['from_order'] > wait_matrix_flat['to_order']) &
            (wait_matrix_flat['from_order'] != 99) &
            (wait_matrix_flat['to_order'] != 99)
        ]
        
        if not df_rework.empty:
            df_rework_sorted = df_rework.sort_values('wait_days', ascending=False)
            for _, row in df_rework_sorted.head(3).iterrows(): # Top 3
                loop_str = f"{row['previous_task_name']} -> {row['task_name']}"
                rework_loops.append({'loop': loop_str, 'wait_days': row['wait_days']})
                
        return rework_loops

    def _build_executive_summary(self):
        """ (V8) PASSO 3: A VOZ - Constrói a narrativa principal com base nas flags. """
        f = self.narrative_flags
        
        # --- Cenário 1: O Paradoxo da Eficiência (Dados do Utilizador) ---
        if f.get('baseline_efficient') and (f.get('is_degrading_prazo') or f.get('is_degrading_duracao')) and f.get('has_rework'):
            loop_ex = f['rework_loops'][0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: O Paradoxo da Eficiência e o Rework Oculto',
                              f"A sua operação vive um paradoxo. Embora os dados históricos mostrem alta eficiência (processos terminam em média {f['mean_days_diff']:.1f} dias *antes* do prazo), esta vantagem está a desaparecer. A performance está em degradação, com a duração dos processos a aumentar (tendência: +{f.get('trend_duration', 0):.1f} dias/mês). "
                              f"A causa raiz não é um gargalo de fila, mas sim **rework (retrabalho)**. O processo é caótico (Precisão: {f['precision']:.1%}) e fluxos de retorno, como '{loop_ex['loop']}', consomem dias ({loop_ex['wait_days']:.1f}d) e não são o 'caminho feliz'.",
                              "Cartão 1, 22, 58, 61", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "Investigue a causa raiz do rework. Foque-se em reduzir os fluxos de retorno (especialmente os identificados no Cartão 58 e 42), mesmo que isso signifique aumentar o tempo de execução da tarefa original para garantir a qualidade 'à primeira'.",
                              "Cartão 58", level='recomendacao', priority=2)

        # --- Cenário 2: Gargalo Clássico (Lento, mas Padronizado) ---
        elif f.get('baseline_inefficient') and (f.get('is_standardized') or not f.get('is_chaotic')) and f.get('has_high_queues') and not f.get('has_rework'):
            queue_ex = f['queue_bottlenecks'][0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Gargalo de Fila Clássico',
                              f"O seu processo é lento (atraso médio: {f['mean_days_diff']:.1f} dias) devido a um gargalo de fila clássico. O processo é padronizado e flui bem (Precisão: {f['precision']:.1%}), sem rework significativo. "
                              f"O problema é que as tarefas ficam paradas em fila, especialmente antes de '{queue_ex}'. O tempo de espera é superior ao tempo de execução nessas fases.",
                              "Cartão 23, 61, 44", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "Analise a capacidade de recursos na atividade '{queue_ex}'. O problema é de capacidade (poucos recursos) ou de batelada (trabalho acumulado)? Considere realocar recursos de fases com menos fila para este ponto.",
                              "Cartão 44, 46", level='recomendacao', priority=2)

        # --- Cenário 3: Processo Caótico (Rework é o Problema) ---
        elif f.get('baseline_inefficient') and f.get('is_chaotic') and f.get('has_rework'):
            loop_ex = f['rework_loops'][0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Processo Caótico Dominado por Rework',
                              f"O processo é ineficiente (atraso médio: {f['mean_days_diff']:.1f} dias) e a causa raiz é a falta de padronização. O processo é caótico (Precisão: {f['precision']:.1%}, Variante Top 1: {f.get('top_1_variant_perc', 0):.1%}), permitindo múltiplos fluxos. "
                              f"Isto resulta em ciclos de rework caros, como '{loop_ex['loop']}', que são os verdadeiros gargalos de tempo ({loop_ex['wait_days']:.1f}d).",
                              "Cartão 23, 61, 65, 58", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "Defina e imponha um 'caminho feliz' (processo padrão). O foco deve ser na qualidade e na passagem correta entre fases para eliminar o rework. Aumentar a conformidade (Fitness/Precisão) irá reduzir o tempo total.",
                              "Cartão 71", level='recomendacao', priority=2)

        # --- Cenário 4: Processo Lento (Tarefas Demoradas) ---
        elif f.get('baseline_inefficient') and f.get('has_service_bottlenecks') and not f.get('has_rework') and not f.get('has_high_queues'):
            service_ex = f['service_bottlenecks'][0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Processo Lento (Gargalo de Execução)',
                              f"O processo é ineficiente (atraso médio: {f['mean_days_diff']:.1f} dias), mas flui de forma padronizada, sem grandes filas ou rework. "
                              f"O problema reside no tempo de execução (tempo de serviço) de tarefas-chave. Atividades como '{service_ex}' são gargalos de execução, demorando significativamente mais que as restantes.",
                              "Cartão 23, 44, 49", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "Otimize a execução da tarefa '{service_ex}'. O problema é falta de automação, complexidade excessiva ou falta de formação? Melhorar a eficiência desta tarefa específica terá o maior impacto.",
                              "Cartão 49", level='recomendacao', priority=2)

        # --- Cenário 5: "O Herói" (Dependência de Recursos) ---
        elif f.get('has_pareto_herois') and f.get('has_resource_hub'):
            hub_ex = f['resource_hub_names']
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Risco de "Herói" e Hub de Comunicação',
                              "A sua operação apresenta um risco significativo de dependência de recursos. O Princípio de Pareto (80/20) aplica-se: cerca de 20% dos recursos estão a fazer 80% do trabalho. "
                              f"Além disso, existe um hub de comunicação claro ('{hub_ex}') que centraliza o fluxo, criando um ponto único de falha.",
                              "Cartão 31, 32, 40", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "Inicie um plano de cross-training e delegação de tarefas para reduzir a dependência dos 'heróis' e descentralizar o fluxo de trabalho do hub principal. A performance atual pode ser boa, mas é frágil.",
                              "Cartão 39", level='recomendacao', priority=2)
                              
        # --- Cenário 6: Processo Saudável (Catch-all Positivo) ---
        elif f.get('baseline_efficient') and not f.get('is_degrading_prazo') and not f.get('is_chaotic'):
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Processo Saudável e Padronizado',
                              f"Parabéns. A análise revela um processo saudável, eficiente (termina em média {f['mean_days_diff']:.1f} dias adiantado) e estável (sem degradação de performance). "
                              f"O fluxo é padronizado (Precisão: {f['precision']:.1%}) e os gargalos de espera ou rework não são significativos.",
                              "Cartão 1, 22, 61", level='facto', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "O foco deve ser a manutenção e monitorização contínua. Considere usar a Simulação de RL para testar cenários de 'what-if', como aumento de volume ou redução de equipa, para encontrar o ponto ótimo de eficiência.",
                              "Página de RL", level='recomendacao', priority=2)

        # --- Cenário 7: Genérico (Sem padrão claro) ---
        else:
             self._add_insight('resumo_executivo', 'Diagnóstico Principal: Múltiplos Pontos de Otimização',
                              "A análise não identifica um único arquétipo de problema, mas sim múltiplos pontos de otimização de menor impacto. O processo não está em crise, mas pode ser melhorado. "
                              "Recomendamos uma análise detalhada dos factos de suporte abaixo, focando-se nos maiores gargalos de tempo (Cartão 42) e nos recursos menos eficientes (Cartão 33).",
                              "Cartão 42, 33", level='problema', priority=1)

    # --- (V8) Funções de Recolha de Factos (Replicando a análise manual) ---
    
    def _check_kpis(self):
        try:
            dur_media = self.kpis.get('Duração Média Num', 0)
            espera_media = self.kpis.get('Espera Média (dias)', 0)
            perc_espera = (espera_media / max(dur_media, 1)) * 100 if dur_media > 0 else 0

            kpi_values = {
                'Duração Média': dur_media,
                'Custo Médio': self.kpis.get('Custo Médio', 0),
                'Atraso Médio': self.delay_kpis.get('Atraso Médio (dias)', 0),
                'Desvio Custo Médio': self.kpis.get('Desvio de Custo Médio', 0),
                'Espera Média (Total)': espera_media,
                '% Tempo em Espera': perc_espera
            }

            kpi_geral = [
                {'label': 'Duração Média', 'value': f"{kpi_values['Duração Média']:.1f} dias" if isinstance(kpi_values['Duração Média'], (int, float)) else 'N/A'},
                {'label': 'Custo Médio', 'value': f"€{kpi_values['Custo Médio']:,.0f}" if isinstance(kpi_values['Custo Médio'], (int, float)) else 'N/A'},
                {'label': 'Atraso Médio', 'value': f"{kpi_values['Atraso Médio']:.1f} dias" if isinstance(kpi_values['Atraso Médio'], (int, float)) else 'N/A'},
                {'label': 'Desvio Custo Médio', 'value': f"€{kpi_values['Desvio Custo Médio']:,.0f}" if isinstance(kpi_values['Desvio Custo Médio'], (int, float)) else 'N/A'},
                {'label': 'Espera Média (Total)', 'value': f"{kpi_values['Espera Média (Total)']:.1f} dias" if isinstance(kpi_values['Espera Média (Total)'], (int, float)) else 'N/A'},
                {'label': '% Tempo em Espera', 'value': f"{kpi_values['% Tempo em Espera']:.1f}%" if isinstance(kpi_values['% Tempo em Espera'], (int, float)) else 'N/A'}
            ]
            self.insights['saude_geral'] = kpi_geral
            
        except Exception as e: print(f"Erro em _check_kpis: {e}")

    def _check_custos_atrasos_facts(self):
        section = 'diagnostico_custos_atrasos'
        if self.df_projects_base.empty: return

        # Cartão 1 & 23: Matriz de Performance e Distribuição do Atraso
        try:
            if 'days_diff' in self.df_projects_base.columns and 'cost_diff' in self.df_projects_base.columns:
                df_valid = self.df_projects_base.dropna(subset=['days_diff', 'cost_diff'])
                if not df_valid.empty:
                    mean_days_diff = df_valid['days_diff'].mean()
                    q_bom = (df_valid['days_diff'] < 0) & (df_valid['cost_diff'] < 0)
                    perc_bom = q_bom.mean()
                    
                    if perc_bom > 0.6:
                        self._add_insight(section, 'Observação: Alta Eficiência (Baseline)', f"{perc_bom:.0%} dos processos terminam antes do prazo e abaixo do custo.", "Cartão 1", level='facto')
                        self._add_insight(section, 'Observação: Planeamento Conservador?', f"A maioria dos processos termina adiantada (média: {mean_days_diff:.1f} dias). Isto pode indicar planeamento pessimista ou alta eficiência.", "Cartão 23", level='facto')
                    else:
                        perc_mau = (df_valid['days_diff'] > 0).mean()
                        if perc_mau > 0.5:
                             self._add_insight(section, 'Observação: Baixa Pontualidade (Baseline)', f"{perc_mau:.0%} dos processos terminam atrasados (média: {mean_days_diff:.1f} dias).", "Cartão 1, 23", level='facto')
        except Exception as e: print(f"Erro Regra [1, 23]: {e}")

        # Cartão 2 & 8: Outliers de Custo e Duração
        try:
            if 'total_actual_cost' in self.df_projects_base.columns:
                df_sorted = self.df_projects_base.dropna(subset=['total_actual_cost']).sort_values('total_actual_cost', ascending=False)
                if not df_sorted.empty:
                    top_1 = df_sorted.iloc[0];
                    self._add_insight(section, 'Facto: Processo Mais Caro', f"'{top_1.get('project_name', 'N/A')}' foi o mais dispendioso (€{top_1['total_actual_cost']:,.0f}).", "Cartão 2", level='facto')
            
            if 'actual_duration_days' in self.df_projects_base.columns:
                df_sorted = self.df_projects_base.dropna(subset=['actual_duration_days']).sort_values('actual_duration_days', ascending=False)
                if not df_sorted.empty:
                    top_1 = df_sorted.iloc[0];
                    self._add_insight(section, 'Facto: Processo Mais Longo', f"'{top_1.get('project_name', 'N/A')}' foi o mais longo ({top_1['actual_duration_days']:.0f} dias).", "Cartão 8", level='facto')
        except Exception as e: print(f"Erro Regra [2, 8]: {e}")
        
        # Cartão 7: Custo por Tipo de Recurso
        try:
            if not self.df_cost_by_resource_type.empty:
                top_3 = self.df_cost_by_resource_type.nlargest(3, 'cost_of_work')['resource_type'].tolist()
                self._add_insight(section, 'Facto: Principais Centros de Custo (Recursos)', f"As funções com maior custo de mão-de-obra são: {', '.join(top_3)}.", "Cartão 7", level='facto')
        except Exception as e: print(f"Erro Regra [7]: {e}")
        
        # Cartão 22 & 3: Evolução da Performance
        try:
            if self.narrative_flags.get('is_degrading_prazo'):
                self._add_insight(section, 'Alerta: Performance de Prazo em Degradação', f"A "f"folga"f" de prazo está a diminuir. A tendência do atraso médio é de {self.narrative_flags['trend_days_diff']:.2f} dias/mês (a piorar).", "Cartão 22", level='problema', priority=5)
            elif self.narrative_flags.get('trend_days_diff', 0) < -0.2:
                self._add_insight(section, 'Destaque: Performance de Prazo em Melhoria', f"A pontualidade está a melhorar consistentemente (tendência: {self.narrative_flags['trend_days_diff']:.2f} dias/mês).", "Cartão 22", level='facto')

            if self.narrative_flags.get('is_degrading_duracao'):
                self._add_insight(section, 'Alerta: Duração Média a Aumentar', f"Os processos estão a demorar mais tempo a concluir (tendência: +{self.narrative_flags['trend_duration']:.2f} dias/mês).", "Cartão 3, 12", level='problema', priority=5)
        except Exception as e: print(f"Erro Regra [22, 3, 12]: {e}")

    def _check_performance_prazos_facts(self):
        section = 'diagnostico_custos_atrasos' # Adiciona a esta secção para consolidar
        
        # Cartão 18: Duração Média por Fase
        try:
            if not self.df_avg_cycle_time_phase.empty:
                top_2 = self.df_avg_cycle_time_phase.nlargest(2, 'cycle_time_days')
                fases_str = [f"'{r['phase']}' ({r['cycle_time_days']:.1f} dias)" for _, r in top_2.iterrows()]
                self._add_insight(section, 'Facto: Fases Mais Longas', f"As fases que consomem mais tempo são: {', '.join(fases_str)}.", "Cartão 18", level='facto')
        except Exception as e: print(f"Erro Regra [18]: {e}")

        # Cartão 14 & 15: Distribuição do Lead Time
        try:
            if 'actual_duration_days' in self.df_projects_base.columns:
                dur = self.df_projects_base['actual_duration_days'].dropna()
                if not dur.empty and dur.mean() > 0:
                    cv = dur.std() / dur.mean() # Coeficiente de Variação
                    if not pd.isna(cv) and cv > 0.5:
                        self._add_insight(section, 'Observação: Duração Imprevisível', f"A duração dos processos é muito variável (Coef. Variação: {cv:.1%}). Alguns processos demoram muito mais que outros.", "Cartão 14", level='facto')
                    elif not pd.isna(cv) and cv < 0.2:
                        self._add_insight(section, 'Destaque: Duração Altamente Previsível', f"A duração dos processos é muito consistente (Coef. Variação: {cv:.1%}).", "Cartão 14", level='facto')
        except Exception as e: print(f"Erro Regra [14, 15]: {e}")

    def _check_recursos_equipas_facts(self):
        section = 'diagnostico_recursos_equipas'
        
        # Cartão 26: Distribuição Recursos
        try:
            if not self.df_resources_base.empty and 'resource_type' in self.df_resources_base.columns:
                 resource_counts = self.df_resources_base['resource_type'].value_counts()
                 specialists = resource_counts[resource_counts <= 2]
                 if not specialists.empty:
                     critical_roles = ['Comité de Crédito', 'Diretor de Risco', 'ExCo']
                     critical_specialists = specialists[specialists.index.isin(critical_roles)]
                     if not critical_specialists.empty:
                         self._add_insight(section, 'Observação: Função Crítica com Poucos Recursos', f"Funções de decisão ({', '.join(critical_specialists.index)}) dependem de <= 2 pessoas.", "Cartão 26", level='facto', priority=11)
        except Exception as e: print(f"Erro Regra [26]: {e}")

        # Cartão 31 & 40: "Heróis" e "Hubs"
        try:
            if self.narrative_flags.get('has_pareto_herois'):
                top_1 = self.df_workload.iloc[0]
                self._add_insight(section, "Observação: Concentração de Esforço (Pareto)", f"Existe uma alta concentração de trabalho: o Top 20% dos recursos faz > 80% das horas. '{top_1['resource_name']}' é o recurso mais ativo.", "Cartão 31", level='facto')
            
            if self.narrative_flags.get('has_resource_hub'):
                hub_str = self.narrative_flags.get('resource_hub_names', 'N/A')
                self._add_insight(section, 'Observação: Hub de Comunicação', f"A interação '{hub_str}' é a mais frequente, funcionando como um hub (e potencial gargalo) de comunicação.", "Cartão 32, 40", level='facto', priority=11)
        except Exception as e: print(f"Erro Regra [31, 32, 40]: {e}")

        # Cartão 33: Eficiência
        try:
            if not self.df_efficiency_metrics.empty and 'avg_hours_per_task' in self.df_efficiency_metrics.columns:
                df_effic = self.df_efficiency_metrics.dropna(subset=['avg_hours_per_task']).sort_values(by='avg_hours_per_task')
                if len(df_effic) >= 2:
                    melhor = df_effic.iloc[0]; pior = df_effic.iloc[-1]
                    ratio = pior['avg_hours_per_task'] / max(melhor['avg_hours_per_task'], 0.1)
                    if not pd.isna(ratio) and ratio > 4:
                        self._add_insight(section, 'Observação: Disparidade de Performance', f"Recurso mais lento ({pior['resource_name']}) demora {ratio:.1f}x mais por tarefa que o mais rápido ({melhor['resource_name']}).", "Cartão 33", level='facto')
        except Exception as e: print(f"Erro Regra [33]: {e}")

        # Cartão 37: Skill vs Performance
        try:
            if not self.df_full_context_base.empty and 'skill_level' in self.df_full_context_base.columns and 'hours_worked' in self.df_full_context_base.columns:
                task_counts = self.df_full_context_base.groupby('resource_name')['task_id'].nunique()
                hours_sum = self.df_full_context_base.groupby('resource_name')['hours_worked'].sum()
                perf_df = pd.DataFrame({'hours_per_task': hours_sum / task_counts.replace(0, np.nan)}).dropna().reset_index()
                skill_map = self.df_full_context_base[['resource_name', 'skill_level']].drop_duplicates().set_index('resource_name')
                perf_df = perf_df.join(skill_map, on='resource_name').dropna(subset=['skill_level', 'hours_per_task'])
                if len(perf_df['skill_level'].unique()) > 1 and len(perf_df) > 1:
                    corr = perf_df['skill_level'].corr(perf_df['hours_per_task'])
                    if not pd.isna(corr) and abs(corr) < 0.2:
                        self._add_insight(section, 'Observação: Métrica de "Skill" Irrelevante?', f"O 'Nível de Competência' (skill) não tem correlação ({corr:.2f}) com a rapidez de execução das tarefas.", "Cartão 37", level='facto')
        except Exception as e: print(f"Erro Regra [37]: {e}")
        
        # Cartão 29: Lei de Brooks
        try:
            if not self.df_projects_base.empty and 'num_resources' in self.df_projects_base.columns and 'days_diff' in self.df_projects_base.columns:
                 df_corr = self.df_projects_base[['num_resources', 'days_diff']].apply(pd.to_numeric, errors='coerce').dropna()
                 if len(df_corr['num_resources'].unique()) > 2 and len(df_corr) > 5:
                     corr = df_corr['num_resources'].corr(df_corr['days_diff'])
                     if not pd.isna(corr) and corr > 0.3:
                         self._add_insight(section, 'Observação: Custo de Coordenação (Lei de Brooks?)', f"Equipas maiores tendem a ter MAIS atrasos (Corr: {corr:.2f}). Adicionar mais pessoas parece piorar a performance.", "Cartão 29", level='facto')
        except Exception as e: print(f"Erro Regra [29]: {e}")

    def _check_gargalos_esperas_facts(self):
        section = 'diagnostico_gargalos_esperas'
        
        # Alerta KPI de Espera (se for enganador)
        try:
            perc_espera = self.kpis.get('% Tempo em Espera', 0)
            if perc_espera < 5 and self.narrative_flags.get('has_rework'):
                 self._add_insight(section, 'Alerta: KPI de Espera Enganador', f"O KPI '% Tempo em Espera' ({perc_espera:.1f}%) é baixo, mas enganador. O verdadeiro tempo perdido está oculto em ciclos de rework (fluxos para trás), que não são medidos como 'espera'.", "KPIs Saúde Geral, Cartão 58", level='problema', priority=6)
            elif perc_espera > 40:
                 self._add_insight(section, "Alerta: Processo Dominado por Espera", f"Mais de {perc_espera:.0f}% da duração total é consumida por tempos de espera (filas).", "KPIs Saúde Geral", level='problema', priority=6)
        except Exception as e: print(f"Erro KPI Espera: {e}")

        # Factos sobre Rework (se detetado)
        if self.narrative_flags.get('has_rework'):
            loops = self.narrative_flags.get('rework_loops', [])
            if loops:
                top_loop = loops[0]
                self._add_insight(section, 'Alerta: Principal Gargalo é Rework', f"A maior espera detetada é um fluxo de retrabalho: '{top_loop['loop']}', com uma espera média de {top_loop['wait_days']*24:.1f} horas.", "Cartão 58, 42", level='problema', priority=7)
                
                # Adiciona os outros loops de rework
                for i, loop in enumerate(loops[1:]):
                     self._add_insight(section, f'Alerta: Rework Adicional #{i+1}', f"'{loop['loop']}' (espera de {loop['wait_days']*24:.1f} horas).", "Cartão 58", level='problema', priority=8)

        # Factos sobre Filas (se detetadas)
        if self.narrative_flags.get('has_high_queues'):
            queue_tasks = self.narrative_flags.get('queue_bottlenecks', [])
            if queue_tasks:
                 self._add_insight(section, 'Alerta: Principal Gargalo de Fila', f"A atividade '{queue_tasks[0]}' é um gargalo de fila: o tempo de espera é superior ao tempo de execução.", "Cartão 44", level='problema', priority=7)

        # Factos sobre Tarefas Lentas (se detetadas)
        if self.narrative_flags.get('has_service_bottlenecks'):
            service_tasks = self.narrative_flags.get('service_bottlenecks', [])
            if service_tasks:
                 self._add_insight(section, 'Alerta: Principal Gargalo de Execução', f"A atividade '{service_tasks[0]}' é um gargalo de execução (serviço), sendo das mais demoradas a realizar.", "Cartão 49", level='problema', priority=7)

        # Factos Gerais (Handoffs, Custo de Espera)
        try:
            if not self.df_handoffs.empty:
                # Top Handoff *que não seja rework*
                rework_loop_names = [l['loop'] for l in self.detected_rework_loops]
                df_forward_handoffs = self.df_handoffs[~self.df_handoffs['transition'].isin(rework_loop_names)]
                
                if not df_forward_handoffs.empty:
                    top_1_forward = df_forward_handoffs.nlargest(1, 'handoff_time_days').iloc[0]
                    self._add_insight(section, 'Facto: Principal Gargalo de Fila (Fluxo Normal)', f"No fluxo normal (sem rework), a maior espera é '{top_1_forward['transition']}' ({top_1_forward['handoff_time_days']:.1f} dias).", "Cartão 42, 52", level='facto')

                top_1_cost = self.df_handoffs.nlargest(1, 'estimated_cost_of_wait').iloc[0]
                self._add_insight(section, 'Facto: Gargalo Mais Caro', f"A espera mais cara é '{top_1_cost['transition']}' (custo estimado: €{top_1_cost['estimated_cost_of_wait']:,.0f}).", "Cartão 45", level='facto')
        except Exception as e: print(f"Erro Regra [42, 45, 52]: {e}")

        try:
            df_bn_res = self.tables_pre_orig.get('bottleneck_by_resource_data', pd.DataFrame())
            if not df_bn_res.empty:
                top_1_res = df_bn_res.nlargest(1, 'waiting_time_days').iloc[0]
                self._add_insight(section, 'Facto: Maior Gerador de Espera (Recurso)', f"'{top_1_res['resource_name']}' é o recurso que gera mais tempo de espera a jusante ({top_1_res['waiting_time_days']:.1f} dias).", "Cartão 46", level='facto')
        except Exception as e: print(f"Erro Regra [46]: {e}")

    def _check_fluxo_conformidade_facts(self):
        section = 'diagnostico_fluxo_conformidade'

        # Cartão 61 & 65: Variabilidade
        try:
            if self.narrative_flags.get('is_chaotic'):
                prec = self.narrative_flags.get('precision', 0)
                top_1_p = self.narrative_flags.get('top_1_variant_perc', 0)
                top_10_p = self.narrative_flags.get('top_10_variant_perc', 0)
                self._add_insight(section, 'Alerta: Baixa Padronização (Processo Caótico)', f"O processo é pouco padronizado e caótico. Precisão do Modelo: {prec:.1%}. Variante Top 1: {top_1_p:.1%}. Top 10 Variantes: {top_10_p:.1%}.", "Cartão 61, 65", level='problema', priority=8)
            elif self.narrative_flags.get('is_standardized'):
                prec = self.narrative_flags.get('precision', 0)
                top_1_p = self.narrative_flags.get('top_1_variant_perc', 0)
                self._add_insight(section, 'Destaque: Alta Padronização', f"O processo é altamente padronizado (Precisão: {prec:.1%}, Variante Top 1: {top_1_p:.1%}).", "Cartão 61, 65", level='facto')
            
            # Contextualizar a baixa precisão com o rework
            if self.narrative_flags.get('is_chaotic') and self.narrative_flags.get('has_rework'):
                 self._add_insight(section, 'Observação: Causa da Baixa Padronização', "A Baixa Precisão e o caos no fluxo são causados diretamente pelos ciclos de rework, que criam múltiplas 'variantes' não oficiais.", "Cartão 61, 58", level='facto')

        except Exception as e: print(f"Erro Regra [61, 65]: {e}")

        # Cartão 70 vs 58: Rework
        try:
            df_rework_table = self.tables_pre_orig.get('rework_loops_table', pd.DataFrame())
            if self.narrative_flags.get('has_rework') and (df_rework_table.empty or df_rework_table.iloc[0]['frequency'] < 5):
                 self._add_insight(section, 'Alerta: Rework Oculto', "A tabela 'Principais Loops de Rework' (Cartão 70) é enganadora e não deteta o retrabalho. A 'Matriz de Espera' (Cartão 58) prova que existem fluxos de retorno caros.", "Cartão 70, 58", level='problema', priority=9)
            elif not self.narrative_flags.get('has_rework'):
                 self._add_insight(section, 'Destaque: Ausência de Rework Significativo', "Não foram detetados ciclos de retrabalho significativos (fluxos para trás) na Matriz de Espera.", "Cartão 58, 70", level='facto')
        except Exception as e: print(f"Erro Regra [70, 58]: {e}")
        
        # Cartão 71: Conformidade ao Longo do Tempo
        try:
            if not self.df_monthly_fitness.empty and len(self.df_monthly_fitness) > 1:
                 slope = self._get_trend(self.df_monthly_fitness['fitness'])
                 if not pd.isna(slope):
                     if slope < -0.01: self._add_insight(section, 'Observação: Conformidade a Diminuir', f"A aderência ao processo padrão (fitness) está a diminuir (tendência: {slope:.3f}/mês).", "Cartão 71", level='facto')
                     elif slope > 0.01: self._add_insight(section, 'Destaque: Conformidade a Aumentar', f"A aderência ao processo padrão (fitness) está a aumentar (tendência: {slope:.3f}/mês).", "Cartão 71", level='facto')
        except Exception as e: print(f"Erro Regra [71]: {e}")

        # Cartão 76: Complexidade vs Atraso
        try:
            if not self.df_projects_base.empty and 'complexity_score' in self.df_projects_base.columns and 'days_diff' in self.df_projects_base.columns:
                df_corr = self.df_projects_base[['complexity_score', 'days_diff']].apply(pd.to_numeric, errors='coerce').dropna()
                if len(df_corr['complexity_score'].unique()) > 2 and len(df_corr) > 5:
                    corr = df_corr['complexity_score'].corr(df_corr['days_diff'])
                    if not pd.isna(corr) and abs(corr) < 0.2:
                        self._add_insight(section, 'Facto: Complexidade Bem Gerida (Prazo)', f"A 'complexidade' do processo não tem correlação significativa com atrasos (Corr: {corr:.2f}). O atraso não é explicado pela complexidade.", "Cartão 76", level='facto')
                    elif not pd.isna(corr) and corr > 0.5:
                        self._add_insight(section, 'Observação: Complexidade Impacta o Atraso', f"Processos mais complexos tendem a atrasar mais (Corr: {corr:.2f}).", "Cartão 76", level='facto')
        except Exception as e: print(f"Erro Regra [76]: {e}")


# --- FUNÇÃO DE RENDERIZAÇÃO DA PÁGINA DE DIAGNÓSTICO (V8 - NARRATIVA) ---
def render_diagnostics_page():
    """ Renderiza V8: Resumo Executivo primeiro, seguido de factos de suporte. """
    st.subheader("💡 Diagnóstico Automático e Insights")
    st.markdown("Análise automática dos principais indicadores, destacando pontos fortes, problemas e ineficiências para orientar a sua investigação.")

    tables_pre = st.session_state.get('tables_pre_mining', {})
    metrics = st.session_state.get('metrics', {})
    data_frames = st.session_state.get('data_frames_processed', {})
    tables_eda = st.session_state.get('tables_eda', {})

    if not data_frames or 'projects' not in data_frames or 'tasks' not in data_frames:
        st.error("Dados essenciais da análise (projetos, tarefas) não encontrados. Execute a análise em 'Configurações'.")
        return

    try:
        # Instancia o novo motor V8
        engine = DiagnosticEngineV5(tables_pre, metrics, data_frames, tables_eda)
        report = engine.run()
    except Exception as e:
        st.error(f"Erro ao executar o motor de diagnóstico: {e}")
        st.exception(e)
        return

    # --- Bloco 1: Saúde Geral (KPIs) ---
    st.markdown("---")
    st.markdown("<h4>1. Saúde Geral (KPIs de Baseline)</h4>", unsafe_allow_html=True)
    kpi_data = report.get('saude_geral', [])
    if kpi_data and len(kpi_data) == 6:
        cols1 = st.columns(3)
        cols2 = st.columns(3)
        for i, kpi in enumerate(kpi_data):
            target_col = cols1[i] if i < 3 else cols2[i-3]
            target_col.metric(label=kpi['label'], value=kpi['value'])
    else: st.warning("Não foi possível calcular os KPIs de Saúde Geral.")

    # --- Bloco 2: (NOVO) Resumo Executivo e Recomendações ---
    st.markdown("---")
    st.markdown("<h4>2. Resumo Executivo e Recomendações</h4>", unsafe_allow_html=True)
    st.markdown("A análise narrativa combina todos os pontos de dados para identificar a *história principal* e a *causa-raiz* dos problemas.")
    
    summary_items = report.get('resumo_executivo', [])
    if not summary_items:
        st.warning("Não foi possível gerar um resumo executivo. Verifique os dados de entrada.")
    else:
        for item in summary_items:
            if item['level'] == 'problema':
                st.error(icon="🚨", body=f"**{item['titulo']}**\n\n{item['detalhe']}\n\n*Evidência principal: {item.get('cartao_ref', 'N/A')}*")
            elif item['level'] == 'recomendacao':
                st.success(icon="💡", body=f"**{item['titulo']}**\n\n{item['detalhe']}\n\n*Ação sugerida com base em: {item.get('cartao_ref', 'N/A')}*")
            elif item['level'] == 'facto': # Caso de processo saudável
                st.success(icon="✅", body=f"**{item['titulo']}**\n\n{item['detalhe']}\n\n*Evidência principal: {item.get('cartao_ref', 'N/A')}*")


    # --- Bloco 3: Factos de Suporte e Observações Detalhadas ---
    st.markdown("---")
    st.markdown("<h4>3. Factos de Suporte e Observações por Área</h4>", unsafe_allow_html=True)
    st.markdown("Estes são os factos e observações detalhados que suportam o diagnóstico principal.")
    
    sections = [
        ('diagnostico_custos_atrasos', 'Análise: Custos e Prazos', 'bi-cash-coin'),
        ('diagnostico_gargalos_esperas', 'Análise: Gargalos e Rework', 'bi-traffic-light-fill'),
        ('diagnostico_recursos_equipas', 'Análise: Recursos e Equipas', 'bi-people-fill'),
        ('diagnostico_fluxo_conformidade', 'Análise: Fluxo e Variabilidade', 'bi-signpost-split-fill')
    ]
    
    found_any_fact = False
    for key, title, icon in sections:
        items = report.get(key, [])
        if items:
            found_any_fact = True
            with st.expander(f"{title}", expanded=False):
                cols = st.columns(2)
                for i, item in enumerate(items):
                    with cols[i % 2]:
                        # Definir a cor da borda com base no nível
                        if item['level'] == 'problema': border_color = "#dc3545" # Vermelho (Alerta Grave)
                        elif item['level'] == 'facto': border_color = "#0d6efd" # Azul (Facto)
                        elif item['level'] == 'recomendacao': border_color = "#198754" # Verde
                        else: border_color = "#6c757d" # Cinza (Observação)
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #dee2e6; border-left: 5px solid {border_color}; border-radius: 5px; padding: 12px 15px; margin-bottom: 12px; background-color: var(--card-background-color); box_shadow: 0 1px 3px rgba(0,0,0,0.04); height: 100%;">
                            <p style="font-weight: 600; margin-bottom: 5px; color: var(--text-color);">{item['titulo']}</p>
                            <p style="font-size: 0.95em; margin-bottom: 8px; color: var(--text-color);">{item['detalhe']}</p>
                            <p style="font-size: 0.8em; color: #0d6efd; margin-bottom: 0;"><i>Referência: {item.get('cartao_ref', 'N/A')}</i></p>
                        </div>
                        """, unsafe_allow_html=True)

    if not found_any_fact:
         st.info("Nenhum facto de suporte adicional foi gerado.")

# --- FIM DO BLOCO DE DIAGNÓSTICO V8 ---

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
                st.session_state.data_frames_processed = {
                    'projects': df_p,
                    'tasks': df_t,
                    'resources': df_r,
                    'full_context': df_fc
                }
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
        "fluxo": "5. Fluxo e Conformidade", "diagnostico": "💡 Diagnóstico preliminar"
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

    if st.session_state.current_section == "diagnostico":
        render_diagnostics_page()
    
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
        n = min(100, len(proj_ids))
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
            cost_impact_factor = st.number_input("Fator de Impacto do Custo", value=0.01)
            daily_time_penalty = st.number_input("Penalização Diária por Tempo", value=400.0)
            idle_penalty = st.number_input("Penalização por Inatividade", value=50.0)
        with rc2:
            per_day_early_bonus = st.number_input("Bónus por Dia de Adiantamento", value=2000.0)
            completion_base = st.number_input("Recompensa Base por Conclusão", value=10000.0)
            per_day_late_penalty = st.number_input("Penalização por Dia de Atraso", value=7500.0)
        with rc3:
            priority_task_bonus_factor = st.number_input("Bónus por Tarefa Prioritária", value=500)
            pending_task_penalty_factor = st.number_input("Penalização Base Diária por Tarefa Pendente", value=100)
        
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
                st.metric(label="Custo (€)", value=f"€{sim_cost:,.2f}", delta=f"€{sim_cost - real_cost:,.2f} vs Real")

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
