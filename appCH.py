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
    /* 👇 ADICIONE O BLOCO DE CÓDIGO ABAIXO AQUI 👇 */
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

def create_card(title, icon_html, chart_bytes=None, dataframe=None, use_container_width=False):
    if chart_bytes:
        b64_image = base64.b64encode(chart_bytes.getvalue()).decode()
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>{icon_html} {title}</h4></div>
            <div class="card-body">
                <img src="data:image/png;base64,{b64_image}" style="width: 100%; height: auto;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif dataframe is not None:
        df_html = dataframe.to_html(classes=['pandas-df-card'], index=False)
        st.markdown(f"""
        <div class="card">
            <div class="card-header"><h4>{icon_html} {title}</h4></div>
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
    
    fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(x='frequency', y='variant_str', data=variant_analysis.head(10), ax=ax, orient='h', hue='variant_str', legend=False, palette='coolwarm'); ax.set_title("Top 10 Variantes de Processo por Frequência")
    plots['variants_frequency'] = convert_fig_to_bytes(fig)
    
    rework_loops = Counter(f"{trace[i]} -> {trace[i+1]} -> {trace[i]}" for trace in variants_df['trace'] for i in range(len(trace) - 2) if trace[i] == trace[i+2] and trace[i] != trace[i+1])
    tables['rework_loops_table'] = pd.DataFrame(rework_loops.most_common(10), columns=['rework_loop', 'frequency'])

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
    variant_durations['variant_str'] = variant_durations['variant'].apply(lambda x: ' -> '.join([str(i) for i in x][:4]) + '...')
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='avg_duration_hours', y='variant_str', data=variant_durations.astype({'avg_duration_hours':'float'}), ax=ax, hue='variant_str', legend=False, palette='plasma'); ax.set_title('Duração Média das 10 Variantes Mais Comuns'); fig.tight_layout()
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

    dep_counts = df_dependencies.groupby('project_id').size().reset_index(name='dependency_count')
    task_counts = df_tasks.groupby('project_id').size().reset_index(name='task_count')
    project_complexity = pd.merge(dep_counts, task_counts, on='project_id', how='outer').fillna(0)
    project_complexity['complexity_ratio'] = (project_complexity['dependency_count'] / project_complexity['task_count']).fillna(0)

    project_aggregates = df_alloc_costs.groupby('project_id').agg(
        total_actual_cost=('cost_of_work', 'sum'),
        avg_hourly_rate=('cost_per_hour', 'mean'),
        num_resources=('resource_id', 'nunique')
    ).reset_index()

    df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    df_projects = df_projects.merge(project_complexity, on='project_id', how='left')
    df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
    df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)

    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
    df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']
    
    tables['stats_table'] = df_projects[['actual_duration_days', 'days_diff', 'budget_impact', 'total_actual_cost', 'cost_diff', 'num_resources', 'avg_hourly_rate']].describe().round(2)

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
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.histplot(data=df_projects, x='complexity_ratio', kde=True, color='darkslateblue', ax=ax); ax.set_title('Distribuição da Complexidade dos Processos')
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
        fig, ax = plt.subplots(figsize=(14, 9)); nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, ax=ax); ax.set_title(f'Grafo de Dependências: Processo {PROJECT_ID_EXAMPLE}')
        plots['plot_26'] = convert_fig_to_bytes(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6)); sns.regplot(data=df_projects, x='complexity_ratio', y='days_diff', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax); ax.set_title('Relação entre Complexidade e Atraso')
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
    
    # --- AMBIENTE E AGENTE (CLASSES) --- (Com alterações para maior realismo)
    class ProjectManagementEnv:
        def __init__(self, df_tasks, df_resources, df_dependencies, df_projects_info, df_resource_allocations=None, reward_config=None, min_progress_for_next_phase=0.7):
            self.rewards = reward_config
            self.df_tasks = df_tasks
            self.df_resources = df_resources
            self.df_dependencies = df_dependencies
            self.df_projects_info = df_projects_info
            self.df_resource_allocations = df_resource_allocations if df_resource_allocations is not None else pd.DataFrame()
            self.resource_types = sorted(self.df_resources['resource_type'].unique().tolist())
            self.task_types = sorted(self.df_tasks['task_type'].unique().tolist())
            self.resources_by_type = {rt: self.df_resources[self.df_resources['resource_type'] == rt] for rt in self.resource_types}
            self.all_actions = self._generate_all_actions()
            self.min_progress_for_next_phase = min_progress_for_next_phase
            self.TASK_TYPE_RESOURCE_MAP = {
            'Onboarding': ['Analista Comercial', 'Gerente Comercial'],
            'Validação KYC e Conformidade': ['Analista de Risco', 'Analista Operações/Legal'],
            'Análise Documental': ['Analista Comercial', 'Gerente Comercial'],
            'Análise de Risco e Proposta': ['Analista de Risco'],
            'Avaliação da Imóvel': ['Avaliador Externo'],
            # A tarefa 'Decisão de Crédito e Condições' é tratada pelo RISK_ESCALATION_MAP, 
            # por isso não precisa de estar aqui. Deixei-a comentada para referência.
            # 'Decisão de Crédito e Condições': [], 
            'Preparação Legal': ['Analista Operações/Legal'],
            'Fecho': ['Analista Operações/Legal']
        }
            self.RISK_ESCALATION_MAP = {'A': ['Analista de Risco'], 'B': ['Analista de Risco', 'Diretor de Risco'],'C': ['Analista de Risco', 'Diretor de Risco', 'Comité de Crédito'],'D': ['Analista de Risco', 'Diretor de Risco', 'Comité de Crédito', 'ExCo']}
        def _generate_all_actions(self):
            actions = set();
            for res_type in self.resource_types:
                actions.add((res_type, 'idle'));
                for task_type in self.task_types: actions.add((res_type, task_type))
            return tuple(sorted(list(actions)))

        # [CORREÇÃO PONTO 7] - Método reset robustecido para garantir reinicialização completa
        def reset(self, project_id):
            # Reinicialização completa de todas as variáveis de estado da simulação
            self.day_count = 0
            self.current_cost = 0.0
            self.total_estimated_effort = 0.0
            self.tasks_state = {}
            self.resources_used_today = set()
            self.daily_history = []
            self.episode_logs = []
            
            # garantir project_id com tipo consistente (string) e guardar contexto do projecto
            proj_id_str = str(project_id)
            self.current_project_id = proj_id_str
            
            # carregar info do projecto (filtro seguro por string)
            project_info = self.df_projects_info.loc[self.df_projects_info['project_id'].astype(str) == proj_id_str].iloc[0]
            self.current_risk_rating = project_info.get('risk_rating', None)
            self.current_date = project_info.get('start_date', None)
            
            # carregar tasks do projecto usando comparação por string para evitar mismatch de tipos
            project_tasks = self.df_tasks[self.df_tasks['project_id'].astype(str) == proj_id_str].sort_values('task_id')
            self.tasks_to_do_count = len(project_tasks)
            
            # tornar leitura de budget/custo robusta
            self.total_estimated_budget = project_info.get('total_actual_cost', None)
            
            # filtrar dependências por project_id usando comparação por string e garantir chaves string
            project_dependencies = self.df_dependencies[self.df_dependencies['project_id'].astype(str) == proj_id_str]
            self.task_dependencies = {str(row['task_id_successor']): str(row['task_id_predecessor']) for _, row in project_dependencies.iterrows()}
            
            # prefer observed allocation hours for this project if available
            alloc_df = getattr(self, 'df_resource_allocations', None)
            project_alloc_hours = {}
            task_alloc_hours = {}
            if alloc_df is not None and 'project_id' in alloc_df.columns and 'hours_worked' in alloc_df.columns:
                tmp_alloc = alloc_df.copy()
                tmp_alloc['project_id'] = tmp_alloc['project_id'].astype(str)
                tmp_alloc['task_id'] = tmp_alloc['task_id'].astype(str)
                tmp_alloc['hours_worked'] = pd.to_numeric(tmp_alloc['hours_worked'], errors='coerce').fillna(0)
                project_alloc_hours = tmp_alloc.groupby('project_id')['hours_worked'].sum().to_dict()
                task_alloc_hours = tmp_alloc.groupby('task_id')['hours_worked'].sum().to_dict()
                # garantir chaves de task_alloc_hours como strings
                task_alloc_hours = {str(k): v for k, v in task_alloc_hours.items()}
            
            # inferência de esforço total (HORAS)
            if project_alloc_hours.get(proj_id_str, 0) > 0:
                # usar horas observadas como esforço total
                self.total_estimated_effort = project_alloc_hours.get(proj_id_str, 0)
                HOURS_PER_DAY = 1
                inferred_unit = 'hours_from_allocations'
            else:
                # inferir unidade em project_tasks (dias ou horas)
                est_series = pd.to_numeric(project_tasks.get('estimated_effort', pd.Series([], dtype=float)), errors='coerce').dropna()
                if not est_series.empty and est_series.median() > 24:
                    HOURS_PER_DAY = 1
                    inferred_unit = 'hours_in_data'
                else:
                    HOURS_PER_DAY = 8
                    inferred_unit = 'days_in_data'
                self.total_estimated_effort = est_series.sum() * HOURS_PER_DAY if not est_series.empty else 0
            
            # build tasks_state with estimated_effort ALWAYS in HOURS (keys as strings)
            self.tasks_state = {}
            for _, task in project_tasks.iterrows():
                tid = str(task.get('task_id'))
                if task_alloc_hours and tid in task_alloc_hours:
                    est_hours = task_alloc_hours[tid]
                else:
                    try:
                        raw_eff = float(task.get('estimated_effort', 0) or 0)
                    except Exception:
                        raw_eff = 0.0
                    est_hours = raw_eff * HOURS_PER_DAY
                if est_hours <= 0:
                    est_hours = 1
                self.tasks_state[tid] = {
                    'status': 'Pendente',
                    'progress': 0.0,
                    'estimated_effort': float(est_hours),
                    'priority': int(task.get('priority', 0) or 0),
                    'task_type': task.get('task_type')
                }
            
            # save inference info for debug
            self._estimated_effort_inference = {'method': inferred_unit, 'total_est_hours': self.total_estimated_effort}
            
            return self.get_state()

        def get_state(self):
            progress_total = sum(d.get('progress', 0) for d in self.tasks_state.values()); progress_ratio = progress_total / self.total_estimated_effort if self.total_estimated_effort > 0 else 1.0
            budget_ratio = self.current_cost / self.total_estimated_budget if self.total_estimated_budget > 0 else 0.0
            project_info = self.df_projects_info.loc[self.df_projects_info['project_id'] == self.current_project_id].iloc[0]
            time_ratio = self.day_count / project_info['total_duration_days'] if project_info['total_duration_days'] > 0 else 0.0
            pending_tasks = sum(1 for t in self.tasks_state.values() if t['status'] != 'Concluída'); return (int(progress_ratio * 100), int(budget_ratio * 100), int(time_ratio * 100), pending_tasks)
        def get_possible_actions_for_state(self):
            possible_actions = set()
            for res_type in self.resource_types:
                has_eligible_task_for_type = False
                for task_type in self.task_types:
                    if any(t_data['task_type'] == task_type and self._is_task_eligible(t_id, res_type) for t_id, t_data in self.tasks_state.items()):
                        possible_actions.add((res_type, task_type)); has_eligible_task_for_type = True
                if not has_eligible_task_for_type: possible_actions.add((res_type, 'idle'))
            return list(possible_actions)
        def _is_task_eligible(self, task_id, res_type):
            task_key = str(task_id)
            task_data = self.tasks_state.get(task_key)
            if task_data is None:
                return False
            if task_data['status'] == 'Concluída': return False
            pred_id = self.task_dependencies.get(task_id)
            if pred_id:
                predecessor_data = self.tasks_state.get(pred_id, {})
                # Verifica se a tarefa anterior já foi concluída
                if predecessor_data.get('status') != 'Concluída':
                    return False
            
                # <<< INÍCIO DA NOVA LÓGICA DE ESPERA >>>
                completion_date = predecessor_data.get('completion_date')
                if completion_date:
                    # Calcula quantos dias úteis passaram desde a conclusão
                    days_passed = np.busday_count(completion_date.date(), self.current_date.date())
            
                    # Define um tempo de espera mínimo (ex: 2 dias úteis)
                    required_wait_days = 2
            
                    # Se ainda não passou tempo suficiente, a tarefa não é elegível
                    if days_passed < required_wait_days:
                        return False
                # <<< FIM DA NOVA LÓGICA DE ESPERA >>>
            task_type = task_data['task_type']
            if task_type == 'Decisão de Crédito e Condições':
                required_resources = self.RISK_ESCALATION_MAP.get(self.current_risk_rating, []); return res_type in required_resources
            else:
                allowed_resources = self.TASK_TYPE_RESOURCE_MAP.get(task_type, []); return res_type in allowed_resources
        
        # [CORREÇÕES PONTOS 1, 2, 3, 6, 9, 10, 11, 13] - Método step reescrito
        def step(self, action_list):
            self.resources_used_today = set() # Limpa recursos no início do dia
            
            if self.current_date.weekday() >= 5:
                self.current_date += timedelta(days=1)
                return 0, False # Dia não útil, sem recompensa, sem custo
            
            daily_cost = 0
            reward_from_tasks = 0
            
            # Itera sobre a lista de ações (mantém duplicados e ordem)
            for res_type, task_type in action_list:
                if task_type == "idle":
                    reward_from_tasks -= self.rewards['idle_penalty']
                    continue
                
                # Encontra um recurso disponível DESTE TIPO que ainda não trabalhou hoje
                available_resources = self.resources_by_type[res_type][~self.resources_by_type[res_type]['resource_id'].isin(self.resources_used_today)]
                if available_resources.empty:
                    continue # Não há mais recursos deste tipo disponíveis hoje
                
                # [CORREÇÃO PONTO 6] - Prioritiza a tarefa mais importante
                eligible_tasks = [
                    (tid, tdata) for tid, tdata in self.tasks_state.items() 
                    if tdata['task_type'] == task_type and self._is_task_eligible(tid, res_type)
                ]
                if not eligible_tasks:
                    continue
                
                # Ordena as tarefas elegíveis pela sua prioridade (valor mais alto primeiro)
                eligible_tasks.sort(key=lambda item: item[1]['priority'], reverse=True)
                task_id_to_work, task_data = eligible_tasks[0] # Escolhe a de maior prioridade
                
                # Escolhe um recurso específico aleatoriamente do pool disponível
                res_info = available_resources.sample(1).iloc[0]
                self.resources_used_today.add(res_info['resource_id'])

                # [CORREÇÃO PONTO 11] - Adiciona variabilidade no esforço ao iniciar a tarefa
                if task_data['status'] == 'Pendente':
                    task_data['status'] = 'Em Andamento'
                    # Simula incerteza: o esforço real só é "conhecido" quando se começa a trabalhar
                    uncertainty_factor = random.uniform(0.95, 1.10)
                    task_data['estimated_effort'] *= uncertainty_factor

                remaining_effort = task_data['estimated_effort'] - task_data['progress']
                
                daily_capacity = float(pd.to_numeric(res_info.get('daily_capacity', 0), errors='coerce') or 0.0)
                cost_per_hour = float(pd.to_numeric(res_info.get('cost_per_hour', 0), errors='coerce') or 0.0)
                # [CORREÇÃO PONTO 10] - Usa o fator de performance do recurso individual
                performance_factor = float(res_info.get('performance_factor', 1.0))

                # [CORREÇÃO PONTO 2 & 9] - Usa floats, sem arredondamentos forçados (int())
                hours_to_work = min(daily_capacity, max(0.0, remaining_effort / performance_factor))
                
                if hours_to_work <= 1e-9:
                    continue
                
                cost_today = hours_to_work * cost_per_hour
                daily_cost += cost_today
                
                progress_added = hours_to_work * performance_factor
                task_data['progress'] += progress_added

                self.episode_logs.append({
                    'day': self.day_count, 'resource_id': res_info['resource_id'], 'resource_type': res_type,
                    'task_id': task_id_to_work, 'hours_worked': hours_to_work, 'daily_cost': cost_today,
                    'action': f'Work on {task_type}'
                })
                
                # [CORREÇÃO PONTO 9] - Comparação robusta de floats para conclusão
                if task_data['progress'] >= task_data['estimated_effort'] - 1e-9:
                    task_data['status'] = 'Concluída'
                    task_data['completion_date'] = self.current_date  # <<< ADICIONAR ESTA LINHA
                    reward_from_tasks += task_data['priority'] * self.rewards['priority_task_bonus_factor']
            
            self.current_cost += daily_cost
            self.current_date += timedelta(days=1)
            self.day_count += 1 # Incrementa o contador de dias úteis trabalhados

            project_is_done = all(t['status'] == 'Concluída' for t in self.tasks_state.values())
            total_reward = reward_from_tasks - self.rewards['daily_time_penalty']
            if project_is_done:
                project_info = self.df_projects_info.loc[self.df_projects_info['project_id'] == self.current_project_id].iloc[0]
                time_diff = project_info['total_duration_days'] - self.day_count
                total_reward += self.rewards['completion_base']
                total_reward += time_diff * self.rewards['per_day_early_bonus'] if time_diff >= 0 else time_diff * self.rewards['per_day_late_penalty']
                total_reward -= self.current_cost * self.rewards['cost_impact_factor']
            
            return total_reward, project_is_done

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
    
    # --- O resto da função continua igual, usando as variáveis já preparadas ---
    #SEED = 123; random.seed(SEED); np.random.seed(SEED)
    df_projects_train = df_projects.sample(frac=0.8); df_projects_test = df_projects.drop(df_projects_train.index)
    env = ProjectManagementEnv(df_tasks, df_resources, df_dependencies, df_projects, df_resource_allocations=df_resource_allocations, reward_config=reward_config)
    env.reset(str(project_id_to_simulate))
    
    # --- Extrair parâmetros do agente (com defaults) ---
    lr = float(agent_params.get('lr', 0.1))
    gamma = float(agent_params.get('gamma', 0.9))
    epsilon = float(agent_params.get('epsilon', 1.0))
    epsilon_decay = float(agent_params.get('epsilon_decay', 0.9995))
    min_epsilon = float(agent_params.get('min_epsilon', 0.01))

    agent = QLearningAgent(actions=env.all_actions, lr=lr, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
    time_per_episode = 0.01

        
    for episode in range(num_episodes):
        project_id = str(df_projects_train.sample(1).iloc[0]['project_id'])
        state = env.reset(project_id)
        episode_reward, done = 0, False; calendar_day = 0
        while not done and calendar_day < 1000:
            possible_actions = env.get_possible_actions_for_state()
            # [CORREÇÃO PONTO 1] - Usa uma lista para permitir múltiplas ações do mesmo tipo
            action_list = []
            for res_type in env.resource_types:
                actions_for_res = [a for a in possible_actions if a[0] == res_type]
                if not actions_for_res:
                    continue
                # Permitir uma decisão por cada recurso disponível daquele tipo
                num_resources_of_type = len(env.resources_by_type.get(res_type, []))
                for _ in range(num_resources_of_type):
                    chosen_action = agent.choose_action(state, actions_for_res)
                    if chosen_action:
                        action_list.append(chosen_action)

            reward, done = env.step(action_list)
            next_state = env.get_state()
            # A atualização da Q-table continua a ser feita ação a ação para um aprendizado granular
            for action in action_list:
                agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            episode_reward += reward
            calendar_day += 1
            if env.day_count > 730: break # Limite de segurança de dias úteis
        agent.decay_epsilon(); agent.episode_rewards.append(episode_reward); agent.episode_durations.append(env.day_count); agent.episode_costs.append(env.current_cost)
        progress = (episode + 1) / num_episodes; progress_bar.progress(progress)
        remaining_time = (num_episodes - (episode + 1)) * time_per_episode
        status_text.info(f"A treinar... Episódio {episode + 1}/{num_episodes}. Tempo estimado restante: {remaining_time:.0f} segundos.")
    
    status_text.success("Treino e simulação concluídos!")
    status_text.info("A preparar os gráficos e análises finais. Por favor, aguarde...")
    
    # CÓDIGO CORRIGIDO (com média móvel corrigida e cor laranja)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    rewards, durations, costs, epsilon_history = agent.episode_rewards, agent.episode_durations, agent.episode_costs, agent.epsilon_history
    rolling_avg_window = 50
    
    # Gráfico de Recompensa
    axes[0, 0].plot(rewards, alpha=0.3, label='Recompensa do Episódio')
    # A alteração está aqui: .rolling(..., min_periods=1) e color='orange'
    axes[0, 0].plot(pd.Series(rewards).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)')
    axes[0, 0].set_title('Recompensa por Episódio')
    axes[0, 0].set_xlabel('Episódio')
    axes[0, 0].set_ylabel('Recompensa Total')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico de Duração
    axes[0, 1].plot(durations, alpha=0.3, label='Duração do Episódio')
    axes[0, 1].plot(pd.Series(durations).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)')
    axes[0, 1].set_title('Duração por Episódio')
    axes[0, 1].set_xlabel('Episódio')
    axes[0, 1].set_ylabel('Duração (dias úteis)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico de Epsilon
    axes[1, 0].plot(epsilon_history, color='green')
    axes[1, 0].set_title('Decaimento do Epsilon (Exploração)')
    axes[1, 0].set_xlabel('Episódio')
    axes[1, 0].set_ylabel('Valor de Epsilon')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico de Custo
    axes[1, 1].plot(costs, alpha=0.3, label='Custo do Episódio')
    axes[1, 1].plot(pd.Series(costs).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)')
    axes[1, 1].set_title('Custo por Episódio')
    axes[1, 1].set_xlabel('Episódio')
    axes[1, 1].set_ylabel('Custo (€)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    plots['training_metrics'] = convert_fig_to_bytes(fig)

    def evaluate_agent(agent, env, df_projects_to_evaluate):
        agent.epsilon = 0; results = []
        for _, prj_info in df_projects_to_evaluate.iterrows():
            proj_id_str = str(prj_info['project_id'])
            # [CORREÇÃO PONTO 7] - Confia-se exclusivamente no env.reset para limpar o estado
            state = env.reset(proj_id_str)
            
            done = False; 
            calendar_day = 0
            while not done and calendar_day < 1000:
                possible_actions = env.get_possible_actions_for_state()
                # [CORREÇÃO PONTO 1] - Usa lista também na avaliação
                action_list = []
                for res_type in env.resource_types:
                    actions_for_res = [a for a in possible_actions if a[0] == res_type]
                    if actions_for_res:
                        num_resources_of_type = len(env.resources_by_type.get(res_type, []))
                        for _ in range(num_resources_of_type):
                            chosen_action = agent.choose_action(state, actions_for_res)
                            if chosen_action: action_list.append(chosen_action)
                
                _, done = env.step(action_list)
                state = env.get_state()
                calendar_day += 1
                
                if done:
                    dbg_msg = f"EVAL_STEP_INTERNAL project_id={proj_id_str} ended_on_step={calendar_day} cause=done"
                    print(dbg_msg)
                    break

                if env.day_count >= 730:
                    dbg_msg = f"EVAL_STEP_INTERNAL project_id={proj_id_str} ended_on_step={calendar_day} cause=calendar_limit"
                    print(dbg_msg)
                    break

            proj_dbg_msg = (
                f"EVAL_DEBUG project_id={prj_info['project_id']} "
                f"simulated_day_count={getattr(env,'day_count',None)} "
                f"simulated_cost={getattr(env,'current_cost',None)} "
                f"total_est_hours={getattr(env,'total_estimated_effort',None)} "
                f"tasks_loaded={len(getattr(env,'tasks_state',{}))}"
            )
            print(proj_dbg_msg)

            results.append({
                'project_id': prj_info['project_id'],
                'simulated_duration': env.day_count, # Usa o contador de dias úteis do ambiente
                'simulated_cost': getattr(env, 'current_cost', None),
                'real_duration': prj_info.get('total_duration_days'),
                'real_cost': prj_info.get('total_actual_cost'),
            })           
        return pd.DataFrame(results)

    test_results_df = evaluate_agent(agent, env, df_projects_test)
    df_plot_test = test_results_df.sort_values(by='real_duration').reset_index(drop=True)
    # CÓDIGO CORRIGIDO
    fig, axes = plt.subplots(1, 2, figsize=(20, 8)); index_test = np.arange(len(df_plot_test)); bar_width = 0.35

    # Gráfico da Esquerda (Duração)
    axes[0].bar(index_test - bar_width/2, df_plot_test['real_duration'], bar_width, label='Real', color='orangered')
    axes[0].bar(index_test + bar_width/2, df_plot_test['simulated_duration'], bar_width, label='Simulado (RL)', color='dodgerblue')
    axes[0].set_title('Duração do Processo (Conjunto de Teste da Amostra)')
    axes[0].set_xlabel('ID do Processo')
    axes[0].set_ylabel('Duração (dias úteis)')
    axes[0].set_xticks(index_test)
    axes[0].set_xticklabels(df_plot_test['project_id'], rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Gráfico da Direita (Custo)
    axes[1].bar(index_test - bar_width/2, df_plot_test['real_cost'], bar_width, label='Real', color='orangered')
    axes[1].bar(index_test + bar_width/2, df_plot_test['simulated_cost'], bar_width, label='Simulado (RL)', color='dodgerblue')
    axes[1].set_title('Custo do Processo (Conjunto de Teste da Amostra)')
    axes[1].set_xlabel('ID do Processo')
    axes[1].set_ylabel('Custo (€)')
    axes[1].set_xticks(index_test)
    axes[1].set_xticklabels(df_plot_test['project_id'], rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plots['evaluation_comparison_test'] = convert_fig_to_bytes(fig)

    def get_global_performance_df(results_df):
        real_duration = results_df['real_duration'].sum(); sim_duration = results_df['simulated_duration'].sum(); real_cost = results_df['real_cost'].sum(); sim_cost = results_df['simulated_cost'].sum()
        dur_improv = real_duration - sim_duration; cost_improv = real_cost - sim_cost
        dur_improv_perc = (dur_improv / real_duration) * 100 if real_duration > 0 else 0; cost_improv_perc = (cost_improv / real_cost) * 100 if real_cost > 0 else 0
        perf_data = {'Métrica': ['Duração Total (dias úteis)', 'Custo Total (€)'], 'Real (Histórico)': [f"{real_duration:.0f}", f"€{real_cost:,.2f}"], 'Simulado (RL)': [f"{sim_duration:.0f}", f"€{sim_cost:,.2f}"], 'Melhoria': [f"{dur_improv:.0f} ({dur_improv_perc:.1f}%)", f"€{cost_improv:,.2f} ({cost_improv_perc:.1f}%)"]}
        return pd.DataFrame(perf_data)
        
    tables['global_performance_test'] = get_global_performance_df(test_results_df)
    
    agent.epsilon = 0; state = env.reset(project_id_to_simulate)
    done = False; calendar_day = 0
    while not done and calendar_day < 1000:
        possible_actions = env.get_possible_actions_for_state()
        action_list = []
        for res_type in env.resource_types:
            actions_for_res = [a for a in possible_actions if a[0] == res_type]
            if actions_for_res:
                num_resources_of_type = len(env.resources_by_type.get(res_type, []))
                for _ in range(num_resources_of_type):
                    action = agent.choose_action(state, actions_for_res);
                    if action: action_list.append(action)
        _, done = env.step(action_list); state = env.get_state(); calendar_day += 1
        if env.day_count > 730: break
    simulated_log = pd.DataFrame(env.episode_logs); sim_duration, sim_cost = env.day_count, env.current_cost
    
    # Usar os dataframes AMOSTRADOS (df_projects, df_resource_allocations, df_resources)

    project_info_full = df_projects.loc[df_projects['project_id'] == project_id_to_simulate].iloc[0]
    real_duration, real_cost = project_info_full.get('total_duration_days'), project_info_full.get('total_actual_cost')
    
    # Guardar o resumo do projeto usando valores reais da amostra
    tables['project_summary'] = pd.DataFrame({
        'Métrica': ['Duração (dias úteis)', 'Custo (€)'],
        'Real (Histórico)': [real_duration, real_cost],
        'Simulado (RL)': [sim_duration, sim_cost]
    })
    
    # Usar start_date e alocações da amostra
    project_start_date = df_projects.loc[df_projects['project_id'] == project_id_to_simulate, 'start_date'].iloc[0]
    real_allocations = df_resource_allocations[df_resource_allocations['project_id'] == project_id_to_simulate].copy()
    
    # garantir que allocation_date é datetime antes de calcular dias úteis
    if 'allocation_date' in real_allocations.columns:
        real_allocations['allocation_date'] = pd.to_datetime(real_allocations['allocation_date'], errors='coerce')
    
    real_allocations['day'] = real_allocations.apply(
        lambda row: np.busday_count(project_start_date.date(), row['allocation_date'].date())
        if pd.notna(row.get('allocation_date')) else 0,
        axis=1
    )

    # Prepara os dados de custo e progresso (horas) para o projeto simulado
    total_estimated_effort = env.total_estimated_effort
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    max_day_sim = simulated_log['day'].max() if not simulated_log.empty else 0
    max_day_plot = int(max(max_day_sim, real_duration))
    day_range = pd.RangeIndex(start=0, stop=max_day_plot + 1, name='day')

    # Dados para o Gráfico de Custo Acumulado (à esquerda)
    sim_daily_cost = simulated_log.groupby('day')['daily_cost'].sum()
    sim_cumulative_cost = sim_daily_cost.reindex(day_range, fill_value=0).cumsum()
    real_log_merged = real_allocations.merge(dfs['resources'][['resource_id', 'cost_per_hour']], on='resource_id', how='left')
    real_log_merged['daily_cost'] = real_log_merged['hours_worked'] * real_log_merged['cost_per_hour']
    real_daily_cost = real_log_merged.groupby('day')['daily_cost'].sum()
    real_cumulative_cost = real_daily_cost.reindex(day_range, fill_value=0).cumsum()

    # --- Gráfico de Custo Acumulado (Esquerda) ---
    axes[0].plot(sim_cumulative_cost.index, sim_cumulative_cost.values, label='Custo Simulado', marker='o', linestyle='--', color='b')
    axes[0].plot(real_cumulative_cost.index, real_cumulative_cost.values, label='Custo Real', marker='x', linestyle='-', color='r')
    axes[0].axvline(x=real_duration, color='k', linestyle=':', label=f'Fim Real ({real_duration} dias úteis)')
    axes[0].set_title('Custo Acumulado')
    axes[0].set_xlabel('Dias úteis')
    axes[0].set_ylabel('Custo Acumulado (€)')
    axes[0].legend()
    axes[0].grid(True)

    # Dados para o Gráfico de Horas Acumuladas (à direita)
    sim_daily_progress = simulated_log.groupby('day')['hours_worked'].sum()
    sim_cumulative_progress = sim_daily_progress.reindex(day_range, fill_value=0).cumsum()
    real_daily_progress = real_log_merged.groupby('day')['hours_worked'].sum()
    real_cumulative_progress = real_daily_progress.reindex(day_range, fill_value=0).cumsum()

    # --- Gráfico de Progresso Acumulado (Direita) ---
    axes[1].plot(sim_cumulative_progress.index, sim_cumulative_progress.values, label='Progresso Simulado', marker='o', linestyle='--', color='b')
    axes[1].plot(real_cumulative_progress.index, real_cumulative_progress.values, label='Progresso Real', marker='x', linestyle='-', color='r')
    axes[1].axhline(y=total_estimated_effort, color='g', linestyle='-.', label='Esforço Total Estimado (horas)')
    axes[1].set_title('Progresso Acumulado (Horas)')
    axes[1].set_xlabel('Dias úteis')
    axes[1].set_ylabel('Horas Acumuladas')
    axes[1].legend()
    axes[1].grid(True)

    # Finaliza e guarda a figura
    fig.tight_layout()
    plots['project_detailed_comparison'] = convert_fig_to_bytes(fig)
    
    return plots, tables, logs


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
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    
    upload_cols = st.columns(5)
    for i, name in enumerate(file_names):
        with upload_cols[i]:
            uploaded_file = st.file_uploader(f"Carregar `{name}.csv`", type="csv", key=f"upload_{name}")
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
            st.success("✅ Análise concluída! Navegue para o 'Dashboard Geral' ou para a página de 'Reinforcement Learning'.")
            st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Aguardando o carregamento de todos os ficheiros CSV para poder iniciar a análise.")


# --- PÁGINA DO DASHBOARD ---
def dashboard_page():
    # (O código desta função permanece exatamente o mesmo do ficheiro que forneceu)
    st.title("🏠 Dashboard Geral de Análise de Processos")

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
            create_card("Matriz de Performance (Custo vs Prazo) (PM)", '<i class="bi bi-bullseye"></i>', chart_bytes=plots_pre.get('performance_matrix'))
            create_card("Top 5 Processos Mais Caros", '<i class="bi bi-cash-coin"></i>', dataframe=tables_pre.get('outlier_cost'))
            create_card("Séries Temporais de KPIs de Performance", '<i class="bi bi-graph-up-arrow"></i>', chart_bytes=plots_post.get('kpi_time_series'))
            create_card("Distribuição do Status dos Processos", '<i class="bi bi-bar-chart-line-fill"></i>', chart_bytes=plots_eda.get('plot_01'))
            create_card("Custo Médio dos Processos por Trimestre", '<i class="bi bi-currency-euro"></i>', chart_bytes=plots_eda.get('plot_06'))
            create_card("Alocação de Custos por Orçamento e Recurso", '<i class="bi bi-pie-chart-fill"></i>', chart_bytes=plots_eda.get('plot_17'))
        with c2:
            create_card("Custo por Tipo de Recurso", '<i class="bi bi-tags-fill"></i>', chart_bytes=plots_pre.get('cost_by_resource_type'))
            create_card("Top 5 Processos Mais Longos", '<i class="bi bi-hourglass-split"></i>', dataframe=tables_pre.get('outlier_duration'))
            create_card("Custo Médio por Dia ao Longo do Tempo", '<i class="bi bi-wallet2"></i>', chart_bytes=plots_post.get('cost_per_day_time_series'))
            create_card("Custo Real vs. Orçamento por Processo", '<i class="bi bi-credit-card"></i>', chart_bytes=plots_eda.get('plot_04'))
            create_card("Distribuição do Custo por Dia (Eficiência)", '<i class="bi bi-lightbulb"></i>', chart_bytes=plots_eda.get('plot_16'))
            create_card("Evolução do Volume e Tamanho dos Processos", '<i class="bi bi-reception-4"></i>', chart_bytes=plots_eda.get('plot_31'))

    elif st.session_state.current_section == "performance":
        st.subheader("2. Performance e Prazos")
        c1, c2 = st.columns(2)
        with c1:
            create_card("Relação Lead Time vs Throughput", '<i class="bi bi-link-45deg"></i>', chart_bytes=plots_pre.get('lead_time_vs_throughput'))
            create_card("Distribuição do Lead Time", '<i class="bi bi-stopwatch"></i>', chart_bytes=plots_pre.get('lead_time_hist'))
            create_card("Distribuição da Duração dos Processos (PM)", '<i class="bi bi-distribute-vertical"></i>', chart_bytes=plots_pre.get('case_durations_boxplot'))
            create_card("Gráfico Acumulado de Throughput", '<i class="bi bi-graph-up"></i>', chart_bytes=plots_post.get('cumulative_throughput_plot'))
            create_card("Performance de Prazos por Trimestre", '<i class="bi bi-graph-down-arrow"></i>', chart_bytes=plots_eda.get('plot_05'))
        with c2:
            create_card("Duração Média por Fase do Processo", '<i class="bi bi-folder2-open"></i>', chart_bytes=plots_pre.get('cycle_time_breakdown'))
            create_card("Distribuição do Throughput (horas)", '<i class="bi bi-rocket-takeoff"></i>', chart_bytes=plots_pre.get('throughput_hist'))
            create_card("Boxplot do Throughput (horas)", '<i class="bi bi-box-seam"></i>', chart_bytes=plots_pre.get('throughput_boxplot'))
            create_card("Atividades por Dia da Semana", '<i class="bi bi-calendar-week"></i>', chart_bytes=plots_post.get('temporal_heatmap_fixed'))
            create_card("Evolução da Performance (Prazo e Custo)", '<i class="bi bi-activity"></i>', chart_bytes=plots_eda.get('plot_30'))
        c3, c4 = st.columns(2)
        with c3:
                create_card("Diferença entre Data Real e Planeada", '<i class="bi bi-calendar-range"></i>', chart_bytes=plots_eda.get('plot_03'))
        with c4:
            create_card("Estatísticas de Performance", '<i class="bi bi-table"></i>', dataframe=tables_pre.get('perf_stats'))
        create_card("Linha do Tempo de Todos os Processos (Gantt Chart)", '<i class="bi bi-kanban"></i>', chart_bytes=plots_post.get('gantt_chart_all_projects'))

    elif st.session_state.current_section == "recursos":
        st.subheader("3. Recursos e Equipa")
        
        # Primeira linha de cartões
        c1, c2 = st.columns(2)
        with c1:
            create_card("Distribuição de Recursos por Tipo", '<i class="bi bi-tools"></i>', chart_bytes=plots_eda.get('plot_12'))
            create_card("Recursos por Média de Tarefas/Processo", '<i class="bi bi-person-workspace"></i>', chart_bytes=plots_pre.get('resource_avg_events'))
            create_card("Eficiência Semanal (Horas Trabalhadas)", '<i class="bi bi-calendar3-week"></i>', chart_bytes=plots_pre.get('weekly_efficiency'))
            create_card("Impacto do Tamanho da Equipa no Atraso (PM)", '<i class="bi bi-people"></i>', chart_bytes=plots_pre.get('delay_by_teamsize'))
            create_card("Benchmark de Throughput por Equipa", '<i class="bi bi-trophy"></i>', chart_bytes=plots_pre.get('throughput_benchmark_by_teamsize'))
        with c2:
            create_card("Top 10 Recursos por Horas Trabalhadas (PM)", '<i class="bi bi-lightning-charge-fill"></i>', chart_bytes=plots_pre.get('resource_workload'))
            create_card("Top 10 Handoffs entre Recursos", '<i class="bi bi-arrow-repeat"></i>', chart_bytes=plots_pre.get('resource_handoffs'))
            create_card("Métricas de Eficiência: Top 10 Melhores e Piores Recursos", '<i class="bi bi-person-check"></i>', chart_bytes=plots_post.get('resource_efficiency_plot'))
            create_card("Duração Mediana por Tamanho da Equipa", '<i class="bi bi-speedometer"></i>', chart_bytes=plots_pre.get('median_duration_by_teamsize'))
            create_card("Nº Médio de Recursos por Processo a Cada Trimestre", '<i class="bi bi-person-plus"></i>', chart_bytes=plots_eda.get('plot_07'))
            create_card("Atraso Médio por Recurso", '<i class="bi bi-person-exclamation"></i>', chart_bytes=plots_eda.get('plot_14'))

        # Segunda linha de cartões, para os gráficos de análise de Skill
        c3, c4 = st.columns(2)
        with c3:
            if 'skill_vs_performance_adv' in plots_post:
                create_card("Relação entre Skill e Performance", '<i class="bi bi-graph-up-arrow"></i>', chart_bytes=plots_post.get('skill_vs_performance_adv'))
        with c4:
            create_card("Atraso por Nível de Competência", '<i class="bi bi-mortarboard"></i>', chart_bytes=plots_eda.get('plot_23'))

        # Gráficos complexos que ocupam a largura total
        if 'resource_network_bipartite' in plots_post:
            create_card("Rede de Recursos por Função", '<i class="bi bi-node-plus-fill"></i>', chart_bytes=plots_post.get('resource_network_bipartite'))

        create_card("Rede Social de Recursos (Handovers)", '<i class="bi bi-diagram-3-fill"></i>', chart_bytes=plots_post.get('resource_network_adv'))
        
        create_card("Heatmap de Esforço (Recurso vs Atividade)", '<i class="bi bi-map"></i>', chart_bytes=plots_pre.get('resource_activity_matrix'))
    
    elif st.session_state.current_section == "gargalos":
        st.subheader("4. Handoffs e Espera")
        create_card("Heatmap de Performance no Processo (Gargalos)", '<i class="bi bi-fire"></i>', chart_bytes=plots_post.get('performance_heatmap'))
        
        c1, c2 = st.columns(2)
        with c1:
            create_card("Atividades Mais Frequentes", '<i class="bi bi-speedometer2"></i>', chart_bytes=plots_pre.get('top_activities_plot'))
            create_card("Gargalos: Tempo de Serviço vs. Espera", '<i class="bi bi-traffic-light"></i>', chart_bytes=plots_pre.get('service_vs_wait_stacked'))
            create_card("Top 10 Handoffs por Custo de Espera", '<i class="bi bi-currency-exchange"></i>', chart_bytes=plots_pre.get('top_handoffs_cost'))
            create_card("Top Recursos por Tempo de Espera Gerado", '<i class="bi bi-sign-stop"></i>', chart_bytes=plots_pre.get('bottleneck_by_resource'))
            create_card("Custo Real vs. Atraso", '<i class="bi bi-cash-stack"></i>', chart_bytes=plots_eda.get('plot_18'))
            create_card("Nº de Recursos vs. Custo Total", '<i class="bi bi-people-fill"></i>', chart_bytes=plots_eda.get('plot_20'))
            
        with c2:
            create_card("Tempo Médio de Execução por Atividade", '<i class="bi bi-hammer"></i>', chart_bytes=plots_pre.get('activity_service_times'))
            create_card("Espera vs. Execução (Dispersão)", '<i class="bi bi-search"></i>', chart_bytes=plots_pre.get('wait_vs_service_scatter'))
            create_card("Evolução do Tempo Médio de Espera", '<i class="bi bi-clock-history"></i>', chart_bytes=plots_pre.get('wait_time_evolution'))
            create_card("Top 10 Handoffs por Tempo de Espera", '<i class="bi bi-pause-circle"></i>', chart_bytes=plots_pre.get('top_handoffs'))
            create_card("Rate Horário Médio vs. Atraso", '<i class="bi bi-alarm"></i>', chart_bytes=plots_eda.get('plot_19'))
            create_card("Atraso por Faixa de Orçamento", '<i class="bi bi-layers-half"></i>', chart_bytes=plots_eda.get('plot_22'))

        # --- INÍCIO DA ALTERAÇÃO ---
        # O cartão "Análise de Tempo entre Marcos" passa a ocupar uma linha inteira para melhor visualização
        if 'milestone_time_analysis_plot' in plots_post:
            create_card("Análise de Tempo entre Marcos do Processo", '<i class="bi bi-flag"></i>', chart_bytes=plots_post.get('milestone_time_analysis_plot'))
        
        # Nova linha de colunas para colocar os cartões lado a lado, como pedido
        c3, c4 = st.columns(2)
        with c3:
            create_card("Matriz de Correlação", '<i class="bi bi-bounding-box-circles"></i>', chart_bytes=plots_eda.get('plot_29'))
        with c4:
            create_card("Tempo Médio de Espera por Atividade", '<i class="bi bi-hourglass-bottom"></i>', chart_bytes=plots_post.get('avg_waiting_time_by_activity_plot'))
        # --- FIM DA ALTERAÇÃO ---
            
        create_card("Matriz de Tempo de Espera entre Atividades (horas)", '<i class="bi bi-grid-3x3-gap"></i>', chart_bytes=plots_post.get('waiting_time_matrix_plot'))

    elif st.session_state.current_section == "fluxo":
        st.subheader("5. Fluxo e Conformidade")

        create_card("Modelo - Inductive Miner", '<i class="bi bi-compass"></i>', chart_bytes=plots_post.get('model_inductive_petrinet'))
        create_card("Modelo - Heuristics Miner", '<i class="bi bi-gear"></i>', chart_bytes=plots_post.get('model_heuristic_petrinet'))

        c1, c2 = st.columns(2)
        with c1:
            create_card("Métricas (Inductive Miner)", '<i class="bi bi-clipboard-data"></i>', chart_bytes=plots_post.get('metrics_inductive'))
        with c2:
            create_card("Métricas (Heuristics Miner)", '<i class="bi bi-clipboard-check"></i>', chart_bytes=plots_post.get('metrics_heuristic'))
            create_card("Sequência de Atividades das 10 Variantes Mais Comuns", '<i class="bi bi-music-note-list"></i>', chart_bytes=plots_post.get('custom_variants_sequence_plot'))
        c3, c4 = st.columns(2)
        with c3:
            create_card("Duração Média das Variantes Mais Comuns", '<i class="bi bi-clock"></i>', chart_bytes=plots_post.get('variant_duration_plot'))
            create_card("Frequência das 10 Principais Variantes", '<i class="bi bi-masks"></i>', dataframe=tables_pre.get('variants_table'))
            create_card("Distribuição de Tarefas por Tipo", '<i class="bi bi-card-list"></i>', chart_bytes=plots_eda.get('plot_08'))
            create_card("Distribuição da Duração das Tarefas", '<i class="bi bi-hourglass"></i>', chart_bytes=plots_eda.get('plot_10'))
            create_card("Centralidade dos Tipos de Tarefa", '<i class="bi bi-arrows-angle-contract"></i>', chart_bytes=plots_eda.get('plot_25'))
            create_card("Relação entre Dependências e Desvio de Custo", '<i class="bi bi-journal-minus"></i>', chart_bytes=plots_eda.get('plot_28'))
        with c4:
            create_card("Score de Conformidade ao Longo do Tempo", '<i class="bi bi-check2-circle"></i>', chart_bytes=plots_post.get('conformance_over_time_plot'))
            create_card("Principais Loops de Rework", '<i class="bi bi-arrow-clockwise"></i>', dataframe=tables_pre.get('rework_loops_table'))
            create_card("Distribuição de Tarefas por Prioridade", '<i class="bi bi-award"></i>', chart_bytes=plots_eda.get('plot_09'))
            create_card("Top 10 Tarefas Específicas Mais Demoradas", '<i class="bi bi-sort-down"></i>', chart_bytes=plots_eda.get('plot_11'))
            create_card("Distribuição da Complexidade dos Processos", '<i class="bi bi-bezier"></i>', chart_bytes=plots_eda.get('plot_24'))
            create_card("Relação entre Complexidade e Atraso", '<i class="bi bi-arrows-collapse"></i>', chart_bytes=plots_eda.get('plot_27'))
        c5, c6 = st.columns(2)
        with c5:
                create_card("Top 10 Variantes de Processo por Frequência", '<i class="bi bi-sort-numeric-down"></i>', chart_bytes=plots_pre.get('variants_frequency'))
        with c6:
            create_card("Gráfico de Dependências: Processo 25", '<i class="bi bi-diagram-2"></i>', chart_bytes=plots_eda.get('plot_26'))

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
            num_episodes = st.number_input("Número de Episódios de Treino", min_value=20, max_value=10000, value=1000, step=100)

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
        create_card("Performance Global (Conjunto de Teste)", '<i class="bi bi-clipboard-data-fill"></i>', dataframe=tables_rl.get('global_performance_test'))

        st.markdown("<h4>Métricas de Treinamento do Agente</h4>", unsafe_allow_html=True)
        create_card("Evolução do Treino", '<i class="bi bi-robot"></i>', chart_bytes=plots_rl.get('training_metrics'))
        
        st.markdown("<h4>Comparação de Desempenho (Simulado vs. Real)</h4>", unsafe_allow_html=True)
        create_card("Comparação do Desempenho (Conjunto de Teste da Amostra)", '<i class="bi bi-bullseye"></i>', chart_bytes=plots_rl.get('evaluation_comparison_test'))
        
        st.markdown(f"<h4>Análise Detalhada da Simulação (Processo {st.session_state.project_id_simulated})</h4>", unsafe_allow_html=True)
        summary_df = tables_rl.get('project_summary')
        if summary_df is not None:
            metric_cols = st.columns(2)
            with metric_cols[0]:
                real_duration = summary_df.loc[summary_df['Métrica'] == 'Duração (dias úteis)', 'Real (Histórico)'].iloc[0]
                sim_duration = summary_df.loc[summary_df['Métrica'] == 'Duração (dias úteis)', 'Simulado (RL)'].iloc[0]
                st.metric(label="Duração (dias úteis)", value=f"{sim_duration:.0f}", delta=f"{sim_duration - real_duration:.0f} vs Real")
            with metric_cols[1]:
                real_cost = summary_df.loc[summary_df['Métrica'] == 'Custo (€)', 'Real (Histórico)'].iloc[0]
                sim_cost = summary_df.loc[summary_df['Métrica'] == 'Custo (€)', 'Simulado (RL)'].iloc[0]
                st.metric(label="Custo (€)", value=f"€{sim_cost:,.2f}", delta=f"€{sim_cost - real_cost:,.2f} vs Real")

        create_card(f"Comparação Detalhada (Processo {st.session_state.project_id_simulated})", '<i class="bi bi-search"></i>', chart_bytes=plots_rl.get('project_detailed_comparison'))

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

if __name__ == "__main__":
    main()
