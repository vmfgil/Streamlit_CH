import difflib
import json
import re
import streamlit as st
import streamlit.components.v1 as components
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
import html
from scipy import stats
from fpdf import FPDF
from io import BytesIO
import inspect
import base64
import tempfile
import os
from pathlib import Path
import google.generativeai as genai
import PIL.Image
from fpdf.enums import XPos, YPos
import sys 

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

# --- ESTILO CSS (tema claro, estilo SaaS moderno) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

    :root {
        --primary-color: #0f172a; --accent: #2563eb; --success-color: #16a34a; --danger: #dc2626; --warning-color: #d97706;
        --secondary-color: #64748b; --background-color: #f7f8fa; --sidebar-background: #ffffff; --card-background-color: #ffffff;
        --text-color: #0f172a; --text-muted-color: #64748b; --border-color: #e5e7eb;
    }
    html, body, [class*="st-"], .stMarkdown, button, input, textarea, select { font-family: 'Inter', sans-serif; }
    /* Ícones do Streamlit: repor a fonte de ícones (senão aparecem como texto, ex. "arrow_right") */
    [data-testid="stIconMaterial"], span[class*="material-symbols"], .material-icons, .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal; letter-spacing: normal;
        text-transform: none; white-space: nowrap; word-wrap: normal; direction: ltr; -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; }
    .stApp { background-color: var(--background-color); color: var(--text-color); }
    h1, h2, h3, h4, h5 { color: var(--text-color); font-weight: 700; letter-spacing: -0.01em; }
    .block-container { padding-top: 2.2rem; max-width: 1500px; }
    header[data-testid="stHeader"] { background: transparent; }


    button[kind="primary"]:disabled, button[data-testid="stBaseButton-primary"]:disabled { background:#e2e8f0 !important; color:#64748b !important; border-color:#e2e8f0 !important; }
    /* Leitura para decisão (Diagnóstico) */
    .pm-verdict { background:#fff; border:1px solid var(--border-color); border-left:6px solid #d97706; border-radius:16px; padding:20px 24px; margin:6px 0 18px; box-shadow:0 1px 3px rgba(15,23,42,.05); }
    .pm-verdict-top { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:8px; flex-wrap:wrap; }
    .pm-pill { border-radius:999px; padding:3px 12px; font-size:.8rem; font-weight:600; }
    .pm-src { font-size:.75rem; color:var(--text-muted-color); }
    .pm-verdict-title { font-size:1.35rem; font-weight:800; color:#0f172a; letter-spacing:-.01em; margin-bottom:6px; }
    .pm-verdict-text { font-size:.98rem; color:#334155; line-height:1.55; }
    .pm-h5 { font-weight:700; font-size:1.05rem; color:#0f172a; margin:8px 0 10px; }
    .pm-prio { display:flex; gap:14px; align-items:flex-start; }
    .pm-prio-num { flex:0 0 34px; height:34px; border-radius:50%; background:#eff6ff; color:#2563eb; font-weight:800; display:flex; align-items:center; justify-content:center; }
    .pm-prio-title { font-weight:700; font-size:1.02rem; color:#0f172a; margin-bottom:4px; }
    .pm-prio-title i { color:#2563eb; margin-right:4px; }
    .pm-prio-problem { color:#334155; margin-bottom:8px; }
    .pm-prio-row { display:flex; gap:10px; font-size:.9rem; margin:3px 0; }
    .pm-prio-row span { flex:0 0 110px; color:var(--text-muted-color); font-weight:500; }
    .pm-prio-row div { color:#0f172a; }
    .pm-tag { display:inline-block; font-size:.7rem; font-weight:600; background:#f1f5f9; color:#475569; border-radius:999px; padding:2px 8px; margin-left:6px; vertical-align:middle; }
    .pm-tag.risk { background:#fef3c7; color:#b45309; }
    .pm-gain { background:#f0fdf4; border-radius:12px; padding:10px 12px; margin-bottom:8px; }
    .pm-gain-lab { font-size:.72rem; color:#15803d; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
    .pm-gain-val { font-size:.95rem; font-weight:700; color:#14532d; }
    .pm-list-title { font-weight:700; margin-bottom:6px; }
    .pm-list-item { font-size:.9rem; color:#334155; padding:5px 0; border-top:1px solid #f1f5f9; }

    /* Barra lateral */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid var(--border-color); }
    [data-testid="stSidebar"] .stButton>button {
        justify-content: flex-start; background: transparent !important; color: #334155 !important;
        border: none !important; font-weight: 500; padding: 0.55rem 0.8rem; border-radius: 10px !important; box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover { background: #f1f5f9 !important; color: #0f172a !important; }
    [data-testid="stSidebar"] .stButton>button[kind="primary"] { background: #f1f5f9 !important; color: #0f172a !important; font-weight: 600; box-shadow: inset 3px 0 0 #0f172a !important; }
    .pm-brand { display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.05rem; color:#0f172a; padding: 0 0.3rem 0.6rem; }
    .pm-brand-badge { width:34px; height:34px; border-radius:10px; background:#0f172a; color:#fff; display:flex; align-items:center; justify-content:center; font-size:1rem; }
    .pm-user { display:flex; align-items:center; gap:10px; padding:0.5rem 0.3rem; }
    .pm-avatar { width:34px; height:34px; border-radius:50%; background:#0f172a; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:0.85rem; }
    .pm-user-name { font-weight:600; font-size:0.9rem; color:#0f172a; line-height:1.1; } .pm-user-role { font-size:0.75rem; color:#64748b; }
    .pm-side-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8; font-weight:600; margin:0.4rem 0 0.2rem; }

    /* Cabeçalho de página e secções */
    .pm-page-title { font-size: 2rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; line-height: 1.15; }
    .pm-page-sub { color: #64748b; font-size: 1rem; margin-top: 4px; }
    .pm-section-q { font-size: 1.35rem; font-weight: 700; color: #0f172a; margin-top: 0.6rem; }
    .pm-section-t { color: #64748b; margin-bottom: 0.9rem; }
    .pm-note { background:#eff6ff; border:1px solid #dbeafe; color:#1e3a8a; border-radius:12px; padding:0.6rem 0.9rem; font-size:0.88rem; margin:0.4rem 0 0.8rem; }

    /* Cartões (containers com borda) */
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) , div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important; border-color: var(--border-color) !important; background: #ffffff;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 16px rgba(15,23,42,0.04);
    }
    .pm-card-head { display:flex; align-items:flex-start; gap:12px; margin: 0.2rem 0 0.5rem; }
    .pm-tip { position:relative; display:inline-flex; align-items:center; justify-content:center; width:17px; height:17px; margin-left:8px; border-radius:50%;
        border:1.5px solid #cbd5e1; color:#94a3b8; font-size:.68rem; font-weight:700; cursor:help; vertical-align:middle; line-height:1; }
    .pm-tip:hover { color:#2563eb; border-color:#2563eb; }
    .pm-tip:hover::after { content: attr(data-tip); position:absolute; left:0; top:22px; z-index:1000; width:min(380px, 70vw); background:#0f172a; color:#fff;
        font-size:.8rem; font-weight:400; line-height:1.45; padding:10px 12px; border-radius:10px; box-shadow:0 8px 24px rgba(15,23,42,.25); white-space:normal; }
    .pm-card-icon { width:36px; height:36px; border-radius:10px; background:#f1f5f9; color:#0f172a; display:flex; align-items:center; justify-content:center; font-size:1.05rem; flex-shrink:0; }
    .pm-card-title { font-weight: 700; font-size: 1.02rem; color: #0f172a; }
    .pm-card-sub { color: #64748b; font-size: 0.84rem; margin-top: 1px; }

    /* KPIs */
    .pm-kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin: 0.3rem 0 1.1rem; }
    .pm-kpi { background:#ffffff; border:1px solid var(--border-color); border-radius:16px; padding:18px 20px;
              box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 16px rgba(15,23,42,0.04); }
    .pm-kpi-top { display:flex; justify-content:space-between; align-items:center; margin-bottom: 14px; }
    .pm-kpi-icon { width:40px; height:40px; border-radius:12px; background:#f1f5f9; display:flex; align-items:center; justify-content:center; font-size:1.15rem; color:#0f172a; }
    .pm-kpi-value.small { font-size: 1.05rem !important; line-height:1.3; }
    .pm-kpi-value { font-size: 1.75rem; font-weight: 800; color:#0f172a; letter-spacing:-0.02em; line-height:1.1; overflow-wrap:anywhere; }
    .pm-kpi-label { color:#64748b; font-size:0.88rem; margin-top:4px; }
    .pm-kpi-delta { font-size:0.8rem; font-weight:600; white-space:nowrap; } .pm-kpi-delta.up { color:#16a34a; } .pm-kpi-delta.down { color:#dc2626; } .pm-kpi-delta.neu { color:#64748b; }

    /* Tabela de variantes */
    .pm-table { width:100%; border-collapse:collapse; font-size:0.86rem; }
    .pm-table th { text-align:left; color:#64748b; font-weight:600; padding:8px 10px; border-bottom:1px solid var(--border-color); }
    .pm-table td { padding:10px; border-bottom:1px solid #f1f5f9; vertical-align:middle; }
    .pm-table tr:hover td { background:#f8fafc; }
    .pm-vr { color:#94a3b8; font-weight:600; width:36px; } .pm-num { text-align:right; white-space:nowrap; font-weight:600; color:#0f172a; }
    .pm-sub { color:#94a3b8; font-size:0.75rem; font-weight:400; }
    .pm-chips { display:flex; flex-wrap:wrap; align-items:center; gap:4px; }
    .pm-chip { background:#f1f5f9; color:#334155; border-radius:999px; padding:3px 10px; font-size:0.78rem; white-space:nowrap; }
    .pm-chip.dev { background:#fee2e2; color:#991b1b; font-weight:600; }
    .pm-sep { color:#cbd5e1; font-size:0.7rem; }

    /* Navegação por secções (segmented control) */
    [data-testid="stButtonGroup"] { margin: 0.6rem 0 0.4rem; }
    [data-testid="stButtonGroup"] button { border-radius: 10px !important; font-weight: 500; }

    /* Métricas nativas e botões */
    [data-testid="stMetric"] { background:#ffffff; border:1px solid var(--border-color); border-radius:16px; padding:18px 20px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
    [data-testid="stMetric"] label { color: var(--text-muted-color) !important; }
    [data-testid="stMetricValue"] { font-weight: 800; color:#0f172a !important; }
    .stButton>button, .stDownloadButton>button { border-radius: 10px !important; font-weight: 600; }
    .stButton>button[kind="primary"] { background:#0f172a; border-color:#0f172a; }
    [data-testid="stExpander"] { border-radius: 14px; background:#ffffff; border-color: var(--border-color); }
    [data-testid="stDataFrame"] { border-radius: 12px; }

    /* Cartões antigos (relatório RL, etc.) */
    .card { background:#ffffff; border-radius:16px; padding:18px 22px; border:1px solid var(--border-color); margin-bottom:20px; box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 16px rgba(15,23,42,0.04); }
    .card-header { padding-bottom:10px; border-bottom:1px solid #f1f5f9; position:relative; }
    .card .card-header h4 { font-size:1.02rem; margin:0; display:flex; align-items:center; gap:10px; }
    .card .card-header h4 i { font-size:1.1rem; color:#0f172a; }
    .card-body { padding-top: 14px; } .dataframe-card-body { max-height:320px; overflow:auto; padding:0; }
    .tooltip-icon { position:absolute; top:4px; right:0; color:#94a3b8; cursor:help; }
    .iniciar-analise-button .stButton>button { background:#0f172a !important; color:#fff !important; border-color:#0f172a !important; }

    /* Login */
    .pm-login-title { font-size:1.6rem; font-weight:800; color:#0f172a; text-align:center; margin-top:0.5rem; }
    .pm-login-sub { color:#64748b; text-align:center; margin-bottom:1rem; }
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
            ax.get_legend().get_frame().set_edgecolor('#dee2e6')
            
    fig.savefig(buf, format=format, bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def convert_gviz_to_bytes(gviz, format='png'):
    return io.BytesIO(gviz.pipe(format=format))

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

def make_unique_resource_names(df_resources):
    """Garante nomes de recurso únicos (há pessoas diferentes com o mesmo nome).
    Nomes repetidos passam a 'Nome (Função)' e, se ainda repetirem, 'Nome (Função #id)'."""
    df = df_resources.copy()
    if 'resource_name' not in df.columns or not df['resource_name'].duplicated(keep=False).any():
        return df
    dup = df['resource_name'].duplicated(keep=False)
    if 'resource_type' in df.columns:
        df.loc[dup, 'resource_name'] = df.loc[dup, 'resource_name'] + ' (' + df.loc[dup, 'resource_type'].astype(str) + ')'
    dup2 = df['resource_name'].duplicated(keep=False)
    df.loc[dup2, 'resource_name'] = df.loc[dup2, 'resource_name'].str.rstrip(')') + ' #' + df.loc[dup2, 'resource_id'].astype(str) + ')'
    return df

def build_task_event_log(df_tasks, df_resource_allocations, df_resources):
    """Event log com 1 evento por tarefa (time:timestamp = fim, start_timestamp = início)."""
    log = df_tasks[['project_id', 'task_id', 'task_name', 'start_date', 'end_date']].copy()
    if df_resource_allocations is not None and not df_resource_allocations.empty:
        main_res = (df_resource_allocations.groupby(['task_id', 'resource_id'])['hours_worked'].sum()
                    .reset_index().sort_values('hours_worked', ascending=False)
                    .drop_duplicates('task_id')[['task_id', 'resource_id']])
        main_res = main_res.merge(df_resources[['resource_id', 'resource_name']], on='resource_id', how='left')
        log = log.merge(main_res[['task_id', 'resource_name']], on='task_id', how='left')
    else:
        log['resource_name'] = 'Desconhecido'
    log['resource_name'] = log['resource_name'].fillna('Desconhecido')
    log['_ord'] = pd.to_numeric(log['task_id'], errors='coerce')
    log = log.sort_values(['project_id', 'start_date', 'end_date', '_ord']).drop(columns=['_ord', 'task_id'])
    log = log.rename(columns={'project_id': 'case:concept:name', 'task_name': 'concept:name',
                              'end_date': 'time:timestamp', 'start_date': 'start_timestamp',
                              'resource_name': 'org:resource'})
    log['lifecycle:transition'] = 'complete'
    return log.reset_index(drop=True)


# =====================================================================================
# --- CAMADA DE INGESTÃO: NORMALIZAÇÃO DOS DADOS E DETEÇÃO DE CAPACIDADES ---
# Aceita os 5 ficheiros completos ou subconjuntos (ex.: só tasks; tasks com pessoas mas
# sem horas). Preenche o que falta com valores neutros/estimados para que as análises
# corram, e regista que capacidades os dados REALMENTE suportam. Os gráficos e insights
# que dependem de dados em falta são escondidos (ver PLOT_REQUIREMENTS / CARD_REQUIREMENTS).
# =====================================================================================
CAPS_INFO = {
    'FLUXO':        ('Fluxo do processo', 'identificador do processo, atividade e data'),
    'TEMPOS':       ('Tempos de execução e espera', 'data de início e data de fim de cada tarefa'),
    'RECURSOS':     ('Recursos e equipas', 'pessoa (ou função) associada a cada tarefa'),
    'ESFORCO':      ('Esforço real (horas)', 'horas efetivamente trabalhadas por pessoa (resource_allocations)'),
    'CUSTO':        ('Custos', 'custo/hora dos recursos + horas trabalhadas'),
    'PLANO':        ('Prazos e atrasos', 'data planeada de fim do processo (ou um SLA em dias)'),
    'ORCAMENTO':    ('Orçamento', 'custo planeado por processo (budget_impact)'),
    'COMPETENCIA':  ('Competências', 'nível de competência dos recursos (skill_level)'),
    'PRIORIDADE':   ('Prioridades', 'prioridade de cada tarefa (priority)'),
    'ESTADO':       ('Estado dos processos', 'estado de cada processo (project_status)'),
}

# Que capacidades cada gráfico/tabela exige. Chave não listada = só precisa de FLUXO.
PLOT_REQUIREMENTS = {
    # pre-mining
    'performance_matrix': {'PLANO', 'CUSTO', 'ORCAMENTO'},
    'activity_service_times': {'ESFORCO'}, 'top_handoffs': {'TEMPOS'}, 'top_handoffs_cost': {'TEMPOS', 'CUSTO'},
    'resource_workload': {'ESFORCO'}, 'resource_avg_events': {'RECURSOS'}, 'resource_activity_matrix': {'ESFORCO'},
    'resource_handoffs': {'RECURSOS'}, 'cost_by_resource_type': {'CUSTO'}, 'delay_by_teamsize': {'PLANO', 'RECURSOS'},
    'median_duration_by_teamsize': {'RECURSOS'}, 'weekly_efficiency': {'ESFORCO'}, 'bottleneck_by_resource': {'TEMPOS', 'RECURSOS'},
    'service_vs_wait_stacked': {'TEMPOS'}, 'wait_vs_service_scatter': {'TEMPOS'}, 'wait_time_evolution': {'TEMPOS'},
    'throughput_benchmark_by_teamsize': {'RECURSOS'}, 'cycle_time_breakdown': {'TEMPOS'},
    'outlier_cost': {'CUSTO'}, 'cost_by_resource_type_data': {'CUSTO'}, 'resource_workload_data': {'ESFORCO'},
    'bottleneck_by_resource_data': {'TEMPOS', 'RECURSOS'}, 'bottleneck_by_activity_data': {'TEMPOS'},
    'handoff_stats_data': {'TEMPOS'}, 'resource_avg_events_data': {'RECURSOS'}, 'avg_cycle_time_by_task_type_data': {'TEMPOS'},
    # post-mining
    'performance_heatmap': {'TEMPOS'}, 'gantt_chart_all_projects': {'TEMPOS'}, 'resource_network_adv': {'RECURSOS'},
    'skill_vs_performance_adv': {'ESFORCO', 'COMPETENCIA'}, 'resource_network_bipartite': {'RECURSOS', 'COMPETENCIA'},
    'cost_per_day_time_series': {'CUSTO'}, 'milestone_time_analysis_plot': {'TEMPOS'}, 'waiting_time_matrix_plot': {'TEMPOS'},
    'resource_efficiency_plot': {'ESFORCO'}, 'avg_waiting_time_by_activity_plot': {'TEMPOS'},
    # EDA
    'plot_01': {'ESTADO'}, 'plot_03': {'PLANO'}, 'plot_04': {'CUSTO', 'ORCAMENTO'}, 'plot_05': {'PLANO'}, 'plot_06': {'CUSTO'},
    'plot_07': {'RECURSOS'}, 'plot_09': {'PRIORIDADE'}, 'plot_10': {'TEMPOS'}, 'plot_11': {'TEMPOS'}, 'plot_12': {'RECURSOS'},
    'plot_14': {'PLANO', 'RECURSOS'}, 'plot_16': {'CUSTO'}, 'plot_17': {'CUSTO', 'ORCAMENTO'}, 'plot_18': {'CUSTO', 'PLANO'},
    'plot_19': {'CUSTO', 'PLANO'}, 'plot_20': {'CUSTO', 'RECURSOS'}, 'plot_22': {'PLANO', 'ORCAMENTO'},
    'plot_23': {'PLANO', 'COMPETENCIA'}, 'plot_27': {'PLANO'}, 'plot_28': {'CUSTO', 'ORCAMENTO'},
    'plot_29': {'CUSTO', 'PLANO', 'ORCAMENTO', 'COMPETENCIA', 'PRIORIDADE'}, 'plot_30': {'PLANO', 'CUSTO', 'ORCAMENTO'},
    'stats_table': {'PLANO', 'CUSTO', 'ORCAMENTO'},
}

# Que capacidades cada "Cartão" referido nos insights do diagnóstico exige.
CARD_REQUIREMENTS = {
    1: {'PLANO', 'CUSTO', 'ORCAMENTO'}, 2: {'CUSTO'}, 4: {'ESTADO'}, 5: {'CUSTO'}, 6: {'CUSTO', 'ORCAMENTO'}, 7: {'CUSTO'},
    9: {'CUSTO'}, 10: {'CUSTO', 'ORCAMENTO'}, 11: {'CUSTO'}, 17: {'PLANO'}, 18: {'TEMPOS'}, 22: {'PLANO'}, 23: {'PLANO'},
    26: {'RECURSOS'}, 27: {'RECURSOS'}, 29: {'PLANO', 'RECURSOS'}, 31: {'ESFORCO'}, 32: {'RECURSOS'}, 33: {'ESFORCO'},
    36: {'PLANO', 'RECURSOS'}, 37: {'ESFORCO', 'COMPETENCIA'}, 39: {'RECURSOS', 'COMPETENCIA'}, 40: {'RECURSOS'},
    41: {'ESFORCO'}, 42: {'TEMPOS'}, 44: {'TEMPOS'}, 45: {'TEMPOS', 'CUSTO'}, 46: {'TEMPOS', 'RECURSOS'}, 49: {'ESFORCO'},
    51: {'TEMPOS'}, 52: {'TEMPOS'}, 55: {'TEMPOS'}, 57: {'TEMPOS'}, 58: {'TEMPOS'}, 72: {'PRIORIDADE'}, 76: {'PLANO'},
    77: {'CUSTO', 'ORCAMENTO'},
}

def get_visible_outputs():
    """Resultados guardados na sessão, já filtrados pelas capacidades dos dados carregados."""
    caps = st.session_state.get('caps')
    return (gate_outputs(st.session_state.plots_pre_mining, caps), gate_outputs(st.session_state.tables_pre_mining, caps),
            gate_outputs(st.session_state.plots_post_mining, caps), gate_outputs(st.session_state.plots_eda, caps),
            gate_outputs(st.session_state.tables_eda, caps))

def caps_ok(required, caps):
    """True se todas as capacidades exigidas estão disponíveis (caps=None significa 'tudo disponível')."""
    return caps is None or set(required).issubset(caps)

def gate_outputs(outputs, caps):
    """Esconde (põe a None) os gráficos/tabelas cujas capacidades não estão disponíveis."""
    if caps is None or outputs is None:
        return outputs
    return {k: (v if caps_ok(PLOT_REQUIREMENTS.get(k, {'FLUXO'}), caps) else None) for k, v in outputs.items()}

def _has_values(df, col):
    if df is None or col not in df.columns:
        return False
    s = df[col]
    if s.dtype == object:
        s = s.replace('', np.nan)
    return s.notna().mean() > 0.5

def _split_people(val):
    """'Ana; Rui, Rita' -> ['Ana', 'Rui', 'Rita'] (uma tarefa pode ter várias pessoas)."""
    if pd.isna(val):
        return []
    import re as _re
    return [p.strip() for p in _re.split(r'[;,|/]', str(val)) if p.strip()]

def normalize_inputs(raw, sla_days=None):
    """
    raw: dict com qualquer subconjunto de {'projects','tasks','resources','resource_allocations','dependencies'}.
         'tasks' é obrigatório. Em 'tasks' pode existir uma coluna 'resource_id' ou 'resource_name'
         (e opcionalmente 'resource_type') quando não há resource_allocations.
    sla_days: prazo-alvo em dias, usado como data planeada quando não existe planned_end_date.
    Devolve (dfs_completos, caps, notas).
    """
    notes, caps = [], set()
    t = raw.get('tasks')
    if t is None or t.empty:
        raise ValueError("É necessário pelo menos o ficheiro de tarefas (tasks) ou um log de eventos.")
    t = t.copy()
    for c in ['start_date', 'end_date']:
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], errors='coerce')
    if 'start_date' not in t.columns and 'end_date' not in t.columns:
        raise ValueError("O ficheiro de tarefas precisa de pelo menos uma coluna de data (start_date ou end_date).")
    if 'start_date' not in t.columns: t['start_date'] = t['end_date']
    if 'end_date' not in t.columns: t['end_date'] = t['start_date']
    t['start_date'] = t['start_date'].fillna(t['end_date']); t['end_date'] = t['end_date'].fillna(t['start_date'])
    t = t.dropna(subset=['project_id', 'task_name', 'start_date']).copy()
    if 'task_id' not in t.columns:
        t['task_id'] = range(1, len(t) + 1)
    for c in ['project_id', 'task_id']:
        t[c] = t[c].astype(str)
    caps.add('FLUXO')
    if (t['end_date'] > t['start_date']).mean() > 0.3:
        caps.add('TEMPOS')
    else:
        notes.append("As tarefas têm uma só data (ou início = fim): as análises de tempos de execução e espera ficam indisponíveis.")
    if 'task_type' not in t.columns or not _has_values(t, 'task_type'):
        t['task_type'] = t['task_name']
    if _has_values(t, 'priority'):
        caps.add('PRIORIDADE')
    t['priority'] = pd.to_numeric(t['priority'], errors='coerce').fillna(3) if 'priority' in t.columns else 3
    dur_days = ((t['end_date'] - t['start_date']).dt.days).clip(lower=1)
    if 'estimated_effort' not in t.columns:
        t['estimated_effort'] = dur_days
    t['estimated_effort'] = pd.to_numeric(t['estimated_effort'], errors='coerce').fillna(dur_days)

    # --- Recursos e alocações ---
    res = raw.get('resources'); alloc = raw.get('resource_allocations')
    res = res.copy() if res is not None and not res.empty else None
    task_res_col = next((c for c in ['resource_id', 'resource_name'] if c in t.columns and _has_values(t, c)), None)
    if alloc is not None and not alloc.empty and 'hours_worked' in alloc.columns:
        alloc = alloc.copy()
        alloc['task_id'] = alloc['task_id'].astype(str); alloc['resource_id'] = alloc['resource_id'].astype(str)
        alloc['allocation_date'] = pd.to_datetime(alloc['allocation_date'], errors='coerce')
        alloc['hours_worked'] = pd.to_numeric(alloc['hours_worked'], errors='coerce').fillna(0)
        caps.update({'RECURSOS', 'ESFORCO'})
        if res is None:
            ids = alloc['resource_id'].unique()
            res = pd.DataFrame({'resource_id': ids, 'resource_name': ids, 'resource_type': 'Desconhecido'})
            notes.append("Sem ficheiro de recursos: as pessoas aparecem pelo seu ID e sem função.")
    else:
        # Sem horas reais: gera alocações ESTIMADAS (8h por dia útil da tarefa) para que as análises corram.
        if task_res_col:
            caps.add('RECURSOS')
            cols = ['task_id', 'project_id', 'start_date', 'end_date', task_res_col] + (['resource_type'] if 'resource_type' in t.columns else [])
            rows = t[cols].copy()
            rows['_people'] = rows[task_res_col].apply(_split_people)
            rows = rows.explode('_people').dropna(subset=['_people'])
            if res is None:
                names = rows['_people'].unique()
                res = pd.DataFrame({'resource_id': names, 'resource_name': names})
                if 'resource_type' in rows.columns:
                    rtype = rows.groupby('_people')['resource_type'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else 'Desconhecido')
                    res['resource_type'] = res['resource_name'].map(rtype).fillna('Desconhecido')
                else:
                    res['resource_type'] = 'Desconhecido'
                rows['resource_id'] = rows['_people']
            elif task_res_col == 'resource_name':
                res['resource_id'] = res['resource_id'].astype(str)
                name2id = res.drop_duplicates('resource_name').set_index('resource_name')['resource_id'].to_dict()
                rows['resource_id'] = rows['_people'].map(name2id).fillna(rows['_people'])
            else:
                rows['resource_id'] = rows['_people'].astype(str)
            notes.append("Sem horas trabalhadas (resource_allocations): o esforço é ESTIMADO como 8h por dia útil da tarefa e só é usado na simulação; os gráficos de esforço, eficiência e custos ficam escondidos.")
        else:
            rows = t[['task_id', 'project_id', 'start_date', 'end_date']].copy()
            rows['resource_id'] = 'N/D'
            res = pd.DataFrame({'resource_id': ['N/D'], 'resource_name': ['Não identificado'], 'resource_type': ['Não identificado']})
            notes.append("Sem informação de quem executou as tarefas: as análises de recursos e equipas ficam indisponíveis.")
        rows['allocation_date'] = [list(pd.bdate_range(s, e)) or [s] for s, e in zip(rows['start_date'], rows['end_date'])]
        alloc = rows[['task_id', 'project_id', 'resource_id', 'allocation_date']].explode('allocation_date')
        alloc['allocation_date'] = pd.to_datetime(alloc['allocation_date'])
        alloc['hours_worked'] = 8.0
        alloc = alloc.reset_index(drop=True)
        alloc['allocation_id'] = range(1, len(alloc) + 1)
    res['resource_id'] = res['resource_id'].astype(str)
    if 'resource_name' not in res.columns: res['resource_name'] = res['resource_id']
    if 'resource_type' not in res.columns: res['resource_type'] = 'Desconhecido'
    res['resource_type'] = res['resource_type'].fillna('Desconhecido')
    if 'ESFORCO' in caps and 'cost_per_hour' in res.columns and pd.to_numeric(res['cost_per_hour'], errors='coerce').fillna(0).gt(0).any():
        caps.add('CUSTO')
    res['cost_per_hour'] = pd.to_numeric(res['cost_per_hour'], errors='coerce').fillna(0) if 'cost_per_hour' in res.columns else 0.0
    res['daily_capacity'] = pd.to_numeric(res['daily_capacity'], errors='coerce').fillna(8) if 'daily_capacity' in res.columns else 8.0
    if _has_values(res, 'skill_level') and 'RECURSOS' in caps:
        caps.add('COMPETENCIA')
    res['skill_level'] = pd.to_numeric(res['skill_level'], errors='coerce').fillna(1) if 'skill_level' in res.columns else 1
    if 'performance_factor' not in res.columns: res['performance_factor'] = 1.0
    alloc['task_id'] = alloc['task_id'].astype(str)
    if 'project_id' not in alloc.columns or alloc['project_id'].isna().any():
        alloc = alloc.drop(columns=['project_id'], errors='ignore').merge(t[['task_id', 'project_id']], on='task_id', how='left')
    alloc['project_id'] = alloc['project_id'].astype(str)
    if 'allocation_id' not in alloc.columns: alloc['allocation_id'] = range(1, len(alloc) + 1)

    # --- Processos (projects) ---
    derived = t.groupby('project_id').agg(_start=('start_date', 'min'), _end=('end_date', 'max')).reset_index()
    p = raw.get('projects')
    if p is not None and not p.empty:
        p = p.copy(); p['project_id'] = p['project_id'].astype(str)
        p = derived.merge(p, on='project_id', how='left')
    else:
        p = derived.copy()
        notes.append("Sem ficheiro de processos: início e fim de cada processo foram derivados das tarefas.")
    for c in ['start_date', 'end_date', 'planned_end_date']:
        if c in p.columns: p[c] = pd.to_datetime(p[c], errors='coerce')
    p['start_date'] = p['start_date'].fillna(p['_start']) if 'start_date' in p.columns else p['_start']
    p['end_date'] = p['end_date'].fillna(p['_end']) if 'end_date' in p.columns else p['_end']
    p = p.drop(columns=['_start', '_end'])
    if _has_values(p, 'planned_end_date'):
        caps.add('PLANO')
    elif sla_days:
        p['planned_end_date'] = p['start_date'] + pd.to_timedelta(int(sla_days), unit='D')
        caps.add('PLANO'); notes.append(f"Prazo planeado definido pelo SLA indicado: {int(sla_days)} dias após o início.")
    else:
        p['planned_end_date'] = p['end_date']
        notes.append("Sem data planeada nem SLA: as análises de atrasos ficam indisponíveis (pode indicar um SLA nas Configurações).")
    p['planned_end_date'] = p['planned_end_date'].fillna(p['end_date'])
    if _has_values(p, 'budget_impact'):
        caps.add('ORCAMENTO')
    p['budget_impact'] = pd.to_numeric(p['budget_impact'], errors='coerce').fillna(0) if 'budget_impact' in p.columns else 0.0
    if _has_values(p, 'project_status'): caps.add('ESTADO')
    p['project_status'] = p['project_status'].fillna('Concluído') if 'project_status' in p.columns else 'Concluído'
    if 'project_name' not in p.columns: p['project_name'] = 'Processo ' + p['project_id']
    p['project_name'] = p['project_name'].fillna('Processo ' + p['project_id'])
    p['path_name'] = p['path_name'].fillna('Geral') if 'path_name' in p.columns else 'Geral'
    if 'risk_rating' not in p.columns: p['risk_rating'] = 'N/D'

    # --- Dependências: derivadas da ordem das tarefas se não existirem ---
    # A tabela de dependências deixou de ser pedida: a sequência é sempre a observada nos dados
    d = None
    if True:
        ts = t.assign(_o=pd.to_numeric(t['task_id'], errors='coerce')).sort_values(['project_id', 'start_date', 'end_date', '_o'])
        ts['pred'] = ts.groupby('project_id')['task_id'].shift(1)
        d = ts.dropna(subset=['pred'])[['project_id', 'pred', 'task_id']].rename(columns={'pred': 'task_id_predecessor', 'task_id': 'task_id_successor'})
        d['dependency_id'] = range(1, len(d) + 1)
    else:
        d = d.copy()
    for c in ['project_id', 'task_id_predecessor', 'task_id_successor']:
        d[c] = d[c].astype(str)

    t = t.drop(columns=[c for c in ['resource_id', 'resource_name', 'resource_type'] if c in t.columns])
    dfs = {'projects': p, 'tasks': t, 'resources': res, 'resource_allocations': alloc, 'dependencies': d}
    return dfs, caps, notes


# =====================================================================================
# --- LOG ÚNICO: CONVERSÃO DE UM EVENT LOG (1 ficheiro) PARA O MODELO DE 5 TABELAS ---
# =====================================================================================
# Campos que o utilizador pode mapear. (chave, rótulo, obrigatório, palavras para sugestão automática)
LOG_FIELDS = [
    ('case',       'Identificador do processo (caso)', True,  ['case:concept:name', 'case_id', 'caseid', 'case', 'project_id', 'processo', 'process_id', 'dossier', 'id_processo', 'pedido', 'ticket']),
    ('activity',   'Atividade / tarefa',               True,  ['concept:name', 'activity', 'atividade', 'actividade', 'task_name', 'tarefa', 'task', 'etapa', 'step', 'evento', 'event']),
    ('task_key',   'Identificador da tarefa (agrupa várias linhas, ex.: registos de horas, numa só tarefa)', False, ['task_id', 'id_tarefa', 'task_key', 'tarefa_id']),
    ('activity_type', 'Tipo / fase da atividade',      False, ['task_type', 'tipo_tarefa', 'activity_type', 'fase', 'phase']),
    ('case_name',  'Nome do processo',                 False, ['project_name', 'case_name', 'nome_processo', 'descricao_processo']),
    ('start',      'Data de início',                   False, ['start_timestamp', 'start_date', 'start', 'inicio', 'início', 'data_inicio', 'data inicio', 'started']),
    ('end',        'Data de fim (ou data do evento)',  False, ['time:timestamp', 'end_date', 'end', 'fim', 'data_fim', 'data fim', 'timestamp', 'complete', 'completed', 'data', 'date', 'datetime']),
    ('lifecycle',  'Ciclo de vida (start/complete)',   False, ['lifecycle:transition', 'lifecycle', 'transition', 'event_type', 'ciclo']),
    ('resource',   'Pessoa / recurso',                 False, ['org:resource', 'resource_id', 'resource', 'resource_name', 'recurso', 'pessoa', 'user', 'utilizador', 'colaborador', 'owner', 'assignee', 'analista']),
    ('resource_label', 'Nome da pessoa (se o campo acima for um código)', False, ['resource_name', 'nome_recurso', 'nome_pessoa', 'nome_colaborador']),
    ('role',       'Função do recurso',                False, ['org:role', 'role', 'resource_type', 'funcao', 'função', 'perfil', 'departamento', 'team', 'equipa']),
    ('hours',      'Horas trabalhadas',                False, ['hours_worked', 'hours', 'horas', 'effort_hours', 'esforco_horas', 'duration_hours']),
    ('work_date',  'Data do registo de horas',         False, ['allocation_date', 'work_date', 'data_registo', 'data_horas']),
    ('cost_hour',  'Custo por hora',                   False, ['cost_per_hour', 'custo_hora', 'custo/hora', 'hourly_rate', 'rate']),
    ('capacity',   'Capacidade diária (horas)',        False, ['daily_capacity', 'capacidade', 'capacity']),
    ('skill',      'Nível de competência',             False, ['skill_level', 'skill', 'competencia', 'competência', 'nivel']),
    ('planned',    'Data planeada de fim do processo', False, ['planned_end_date', 'due_date', 'data_prevista', 'data_planeada', 'deadline', 'sla_date', 'prazo']),
    ('budget',     'Orçamento do processo',            False, ['budget_impact', 'budget', 'orcamento', 'orçamento', 'custo_planeado']),
    ('status',     'Estado do processo',               False, ['project_status', 'status', 'estado', 'case_status']),
    ('case_type',  'Tipo de processo',                 False, ['path_name', 'case_type', 'tipo', 'produto', 'product', 'segmento', 'channel', 'canal']),
    ('priority',   'Prioridade',                       False, ['priority', 'prioridade']),
]

def read_any_table(uploaded_file):
    """Lê CSV (separador detetado: , ; tab |), Excel (.xlsx/.xls) ou XES."""
    name = uploaded_file.name.lower()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    if name.endswith('.xes'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp:
            tmp.write(uploaded_file.getvalue()); path = tmp.name
        try:
            log = pm4py.read_xes(path)
            return pm4py.convert_to_dataframe(log) if not isinstance(log, pd.DataFrame) else log
        finally:
            os.remove(path)
    raw_bytes = uploaded_file.getvalue()
    for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python', encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python', encoding='latin-1')

def guess_log_mapping(columns):
    """Sugere, para cada campo, a coluna mais provável (por nome). Cada coluna só é usada uma vez."""
    def norm(x): return str(x).strip().lower().replace('-', '_')
    cols = list(columns); used = set(); mapping = {}
    for key, _, _, hints in LOG_FIELDS:
        found = None
        for h in hints:                      # 1º: nome exatamente igual
            for c in cols:
                if c not in used and norm(c) == norm(h): found = c; break
            if found: break
        if not found:                        # 2º: nome contém a pista (só pistas com 4+ letras)
            for h in hints:
                if len(h) < 4: continue
                for c in cols:
                    if c not in used and norm(h) in norm(c): found = c; break
                if found: break
        if found:
            mapping[key] = found; used.add(found)
    return mapping

def _to_dt(s):
    out = pd.to_datetime(s, errors='coerce')
    if out.notna().mean() < 0.5:   # tenta formato dia/mês/ano
        out2 = pd.to_datetime(s, errors='coerce', dayfirst=True)
        if out2.notna().mean() > out.notna().mean(): out = out2
    try:
        if getattr(out.dt, 'tz', None) is not None: out = out.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return out

def log_to_raw(df, mapping):
    """
    Converte um event log (1 linha por evento/tarefa) nas tabelas que a app usa.
    mapping: {campo: nome_da_coluna} (ver LOG_FIELDS). Devolve (raw_dict, avisos).
    """
    warns = []
    m = {k: v for k, v in mapping.items() if v}
    for req in ['case', 'activity']:
        if req not in m: raise ValueError(f"Falta mapear o campo obrigatório: {dict((k, l) for k, l, _, _ in LOG_FIELDS)[req]}.")
    if 'start' not in m and 'end' not in m:
        raise ValueError("É preciso mapear pelo menos uma data (início ou fim).")
    d = pd.DataFrame({k: df[c].values for k, c in m.items()})
    d['case'] = d['case'].astype(str); d['activity'] = d['activity'].astype(str)
    for c in ['start', 'end', 'planned', 'work_date']:
        if c in d.columns: d[c] = _to_dt(d[c])
    n0 = len(d)

    # Logs com eventos 'start' e 'complete' em linhas separadas: emparelhar numa tarefa com início e fim
    if 'lifecycle' in d.columns:
        lc = d['lifecycle'].astype(str).str.lower().str.strip()
        ts_col = 'end' if 'end' in d.columns else 'start'
        is_start = lc.isin(['start', 'started', 'inicio', 'início', 'begin', 'assign'])
        is_end = lc.isin(['complete', 'completed', 'end', 'fim', 'done', 'concluido', 'concluído', 'closed'])
        if is_start.any() and is_end.any():
            st_ev = d[is_start].copy(); en_ev = d[is_end].copy()
            for x in (st_ev, en_ev):
                x.sort_values(['case', ts_col], inplace=True)
                x['_n'] = x.groupby(['case', 'activity']).cumcount()
            st_ev = st_ev[['case', 'activity', '_n', ts_col]].rename(columns={ts_col: '_s'})
            d = en_ev.merge(st_ev, on=['case', 'activity', '_n'], how='left')
            d['start'] = d['_s'].fillna(d[ts_col]); d['end'] = d[ts_col]
            d = d.drop(columns=['_s', '_n', 'lifecycle'])
            warns.append(f"Eventos de início e fim emparelhados: {n0:,} eventos → {len(d):,} tarefas.")
        else:
            d = d[~lc.isin(['schedule', 'suspend', 'resume', 'withdraw', 'ate_abort', 'pi_abort'])].drop(columns=['lifecycle'])
    if 'start' not in d.columns: d['start'] = d['end']
    if 'end' not in d.columns: d['end'] = d['start']
    d['start'] = d['start'].fillna(d['end']); d['end'] = d['end'].fillna(d['start'])
    bad = d['start'].isna() | (d['case'].isin(['nan', 'None', '']))
    if bad.any():
        warns.append(f"{int(bad.sum()):,} linhas ignoradas por não terem data ou identificador de processo válidos.")
        d = d[~bad]
    swap = d['end'] < d['start']
    if swap.any():
        warns.append(f"{int(swap.sum()):,} tarefas com fim anterior ao início: datas trocadas.")
        d.loc[swap, ['start', 'end']] = d.loc[swap, ['end', 'start']].values
    if 'task_key' in d.columns:
        # várias linhas (ex.: registos de horas por pessoa/dia) pertencem à mesma tarefa
        d['task_id'] = d['task_key'].astype(str)
        g = d.groupby('task_id')
        tasks = pd.DataFrame({'project_id': g['case'].first(), 'task_name': g['activity'].first(),
                              'start_date': g['start'].min(), 'end_date': g['end'].max()}).reset_index()
        for src, dst in [('priority', 'priority'), ('activity_type', 'task_type')]:
            if src in d.columns: tasks[dst] = g[src].first().values
        if g.size().max() > 1:
            warns.append(f"{len(d):,} linhas agrupadas em {len(tasks):,} tarefas pelo identificador da tarefa.")
        tasks = tasks.sort_values(['project_id', 'start_date', 'end_date']).reset_index(drop=True)
    else:
        d = d.sort_values(['case', 'start', 'end']).reset_index(drop=True)
        d['task_id'] = (d.index + 1).astype(str)
        tasks = pd.DataFrame({'task_id': d['task_id'], 'project_id': d['case'], 'task_name': d['activity'],
                              'start_date': d['start'], 'end_date': d['end']})
        if 'priority' in d.columns: tasks['priority'] = d['priority']
        if 'activity_type' in d.columns: tasks['task_type'] = d['activity_type']
    raw = {'tasks': tasks}

    # Recursos: com horas -> resource_allocations reais; sem horas -> coluna de pessoa nas tarefas
    if 'resource' in d.columns:
        if 'hours' in d.columns:
            a = d[['task_id', 'case', 'resource', 'start', 'hours'] + (['work_date'] if 'work_date' in d.columns else [])].copy()
            if 'work_date' in a.columns: a['start'] = a['work_date'].fillna(a['start'])
            a['resource'] = a['resource'].apply(_split_people)
            a['hours'] = pd.to_numeric(a['hours'], errors='coerce').fillna(0) / a['resource'].apply(lambda x: max(len(x), 1))
            a = a.explode('resource').dropna(subset=['resource'])
            raw['resource_allocations'] = pd.DataFrame({'task_id': a['task_id'], 'project_id': a['case'], 'resource_id': a['resource'].astype(str),
                                                        'allocation_date': a['start'], 'hours_worked': a['hours']})
            raw['resource_allocations']['allocation_id'] = range(1, len(a) + 1)
        else:
            per_task = d.dropna(subset=['resource']).groupby('task_id')['resource'].agg(lambda x: '; '.join(dict.fromkeys(';'.join(map(str, x)).split(';'))))
            tasks['resource_id'] = tasks['task_id'].map(per_task)
            if 'role' in d.columns: tasks['resource_type'] = tasks['task_id'].map(d.groupby('task_id')['role'].first())
            raw['_resource_from_tasks'] = True
        extra = [c for c in ['role', 'cost_hour', 'resource_label', 'capacity', 'skill'] if c in d.columns]
        ppl = d[['resource'] + extra].copy()
        ppl['resource'] = ppl['resource'].apply(_split_people)
        ppl = ppl.explode('resource').dropna(subset=['resource'])
        mode_ = lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
        agg = {c: mode_ for c in ['role', 'resource_label'] if c in ppl.columns}
        for c in ['cost_hour', 'capacity', 'skill']:
            if c in ppl.columns:
                ppl[c] = pd.to_numeric(ppl[c], errors='coerce'); agg[c] = 'mean'
        r = ppl.groupby('resource').agg(agg).reset_index() if agg else ppl[['resource']].drop_duplicates()
        res = pd.DataFrame({'resource_id': r['resource'].astype(str)})
        res['resource_name'] = r['resource_label'].fillna(r['resource']).astype(str).values if 'resource_label' in r.columns else r['resource'].astype(str).values
        res['resource_type'] = r['role'].fillna('Desconhecido').values if 'role' in r.columns else 'Desconhecido'
        for src, dst in [('cost_hour', 'cost_per_hour'), ('capacity', 'daily_capacity'), ('skill', 'skill_level')]:
            if src in r.columns: res[dst] = r[src].values
        raw['resources'] = res
    elif 'hours' in d.columns:
        warns.append("Foram mapeadas horas mas não o recurso: as horas não podem ser atribuídas a ninguém e foram ignoradas.")

    # Atributos ao nível do processo
    proj_cols = {k: v for k, v in {'planned': 'planned_end_date', 'budget': 'budget_impact', 'status': 'project_status', 'case_type': 'path_name', 'case_name': 'project_name'}.items() if k in d.columns}
    if proj_cols:
        p = d.groupby('case')[list(proj_cols)].first().reset_index().rename(columns={'case': 'project_id', **proj_cols})
        raw['projects'] = p
    raw.pop('_resource_from_tasks', None)
    return raw, warns

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
if 'current_section' not in st.session_state: st.session_state.current_section = "visao_geral"
if 'dfs' not in st.session_state:
    st.session_state.dfs = {'projects': None, 'tasks': None, 'resources': None, 'resource_allocations': None, 'dependencies': None}
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'rl_analysis_run' not in st.session_state: st.session_state.rl_analysis_run = False
if 'project_id_simulated' not in st.session_state: st.session_state.project_id_simulated = None
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
if 'pdf_bytes_download' not in st.session_state: st.session_state.pdf_bytes_download = None
if 'pdf_bytes_for_download_v14' not in st.session_state: st.session_state.pdf_bytes_for_download_v14 = None
if 'pdf_ready_for_download' not in st.session_state: st.session_state.pdf_ready_for_download = False
if 'show_external_ai_modal' not in st.session_state: st.session_state.show_external_ai_modal = False
if 'show_ai_prep_area' not in st.session_state: st.session_state.show_ai_prep_area = False

# --- FUNÇÕES DE ANÁLISE (PROCESS MINING E EDA) ---
#@st.cache_data
def run_pre_mining_analysis(dfs):
    plots = {}
    tables = {}
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = make_unique_resource_names(dfs['resources'])
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

    # Event log ao nível da TAREFA (1 evento por tarefa), e não por dia de alocação.
    # time:timestamp = fim da tarefa; start_timestamp = início da tarefa; recurso = o que mais horas trabalhou na tarefa.
    log_df_final = build_task_event_log(df_tasks, df_resource_allocations, df_resources)
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
    
    lead_times = log_df_final.groupby("case:concept:name").agg(min=("start_timestamp", "min"), max=("time:timestamp", "max")).reset_index()
    lead_times["lead_time_days"] = (lead_times["max"] - lead_times["min"]).dt.total_seconds() / (24*60*60)
    def compute_avg_throughput(group):
        group = group.sort_values("time:timestamp"); deltas = group["time:timestamp"].diff().dropna()
        return deltas.mean().total_seconds() if not deltas.empty else 0
    throughput_per_case = log_df_final.groupby("case:concept:name").apply(compute_avg_throughput, include_groups=False).reset_index(name="avg_throughput_seconds")
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
    
    # horas por TAREFA (soma das alocações da tarefa) e só depois a média por atividade
    service_times = df_full_context.groupby(['task_id', 'task_name'])['hours_worked'].sum().reset_index().groupby('task_name')['hours_worked'].mean().reset_index()
    service_times['service_time_days'] = service_times['hours_worked'] / 8
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x='service_time_days', y='task_name', data=service_times.sort_values('service_time_days', ascending=False).head(10), ax=ax, hue='task_name', legend=False, palette='coolwarm'); ax.set_title("Tempo Médio de Execução por Atividade")
    plots['activity_service_times'] = convert_fig_to_bytes(fig)
    
    df_handoff = log_df_final.copy()  # já ordenado por caso e início
    df_handoff['previous_activity_end_time'] = df_handoff.groupby('case:concept:name')['time:timestamp'].shift(1)
    # espera = início desta tarefa - fim da anterior (nunca negativa)
    df_handoff['handoff_time_days'] = ((df_handoff['start_timestamp'] - df_handoff['previous_activity_end_time']).dt.total_seconds() / (24*3600)).clip(lower=0)
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
    
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=activity_counts.head(10).values, y=activity_counts.head(10).index, ax=ax, palette='YlGnBu', hue=activity_counts.head(10).index, legend=False); ax.set_title("Atividades Mais Frequentes")
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
    
    data_plot_freq = variant_analysis.head(10).copy()
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
    
    plots['variants_frequency'] = convert_fig_to_bytes(fig)
    
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
    
    tables['cost_of_delay_kpis'] = {
        'Custo Total Processos Atrasados': total_cost_delay,
        'Atraso Médio (dias)': mean_delay_days,
        'Custo Médio/Dia Atraso': mean_cost_per_day,
        '% Processos Atrasados': float((df_projects['days_diff'] > 0).mean() * 100),
        'Desvio Médio face ao Plano (dias)': float(df_projects['days_diff'].mean())
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
    tables['kpi_data']['Espera Média (dias)'] = df_tasks_analysis['waiting_time_days'].mean()  # por transição entre tarefas
    # Espera ao nível do PROCESSO: soma das esperas de cada processo e % da duração desse processo
    wait_per_case = df_tasks_analysis.groupby('project_id')['waiting_time_days'].sum()
    dur_per_case = df_projects.set_index('project_id')['actual_duration_days'].replace(0, np.nan)
    tables['kpi_data']['Espera Média por Processo (dias)'] = float(wait_per_case.mean())
    tables['kpi_data']['% Tempo em Espera'] = float((wait_per_case / dur_per_case.reindex(wait_per_case.index)).mean() * 100)

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
    
    # Duração da fase = soma das durações das suas tarefas (o intervalo min-max esticava com o rework)
    phase_times = df_tasks.assign(_d=(df_tasks['end_date'] - df_tasks['start_date']).dt.days).groupby(['project_id', 'task_type'])['_d'].sum().reset_index(name='cycle_time_days')
    avg_cycle_time_by_task_type = phase_times.groupby('task_type')['cycle_time_days'].mean()
    tables['avg_cycle_time_by_task_type_data'] = avg_cycle_time_by_task_type.reset_index() 
    fig, ax = plt.subplots(figsize=(8, 4)); 
    avg_cycle_time_by_task_type.plot(kind='bar', color=sns.color_palette('tab10'), ax=ax); 
    ax.set_title("Duração Média por Tipo de Tarefa"); # Altera o título
    plt.xticks(rotation=45,ha='right')
    plots['cycle_time_breakdown'] = convert_fig_to_bytes(fig)
        
    return plots, tables, event_log_pm4py, df_projects, df_tasks, df_resources, df_full_context

#@st.cache_data
def run_post_mining_analysis(_event_log_pm4py, _df_projects, _df_tasks_raw, _df_resources, _df_full_context):
    plots = {}
    metrics = {}
    
    # 1. Preparação de um log com ciclo de vida completo (start/complete) para certas análises
    # Log ao nível da tarefa: 1 evento por tarefa, com início (start_timestamp) e fim (time:timestamp).
    # (Antes juntava eventos 'start' e 'complete' como se fossem atividades distintas -> DFG com '0ns' e setas impossíveis.)
    log_df_full_lifecycle = pm4py.convert_to_dataframe(_event_log_pm4py)
    log_df_full_lifecycle['start_timestamp'] = pd.to_datetime(log_df_full_lifecycle['start_timestamp'])
    log_df_full_lifecycle['time:timestamp'] = pd.to_datetime(log_df_full_lifecycle['time:timestamp'])
    log_full_pm4py = _event_log_pm4py

    # 2. Descoberta de Modelos e Métricas (usando uma amostra das 10 variantes principais para performance)
    variants_dict = variants_filter.get_variants(_event_log_pm4py)
    top_variants_list = sorted(variants_dict.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    top_variant_names = [v[0] for v in top_variants_list]
    log_top_10_variants = variants_filter.apply(_event_log_pm4py, top_variant_names)
    
    pt_inductive = inductive_miner.apply(log_top_10_variants)
    net_im, im_im, fm_im = pt_converter.apply(pt_inductive)
    gviz_im = pn_visualizer.apply(net_im, im_im, fm_im)
    plots['model_inductive_petrinet'] = convert_gviz_to_bytes(gviz_im)
    
    def plot_metrics_chart(metrics_dict, title):
        df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Métrica', 'Valor'])
        fig, ax = plt.subplots(figsize=(8, 4)); barplot = sns.barplot(data=df_metrics, x='Métrica', y='Valor', ax=ax, hue='Métrica', legend=False, palette='coolwarm')
        for p in barplot.patches: ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points', color='#E5E7EB')
        ax.set_title(title); ax.set_ylim(0, 1.05); return fig
        
    metrics_im = {"Fitness": replay_fitness_evaluator.apply(log_top_10_variants, net_im, im_im, fm_im, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_10_variants, net_im, im_im, fm_im), "Generalização": generalization_evaluator.apply(log_top_10_variants, net_im, im_im, fm_im), "Simplicidade": simplicity_evaluator.apply(net_im)}
    plots['metrics_inductive'] = convert_fig_to_bytes(plot_metrics_chart(metrics_im, 'Métricas de Qualidade (Inductive Miner)'))
    metrics['inductive_miner'] = metrics_im

    net_hm, im_hm, fm_hm = heuristics_miner.apply(log_top_10_variants, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
    gviz_hm = pn_visualizer.apply(net_hm, im_hm, fm_hm)
    plots['model_heuristic_petrinet'] = convert_gviz_to_bytes(gviz_hm)
    
    metrics_hm = {"Fitness": replay_fitness_evaluator.apply(log_top_10_variants, net_hm, im_hm, fm_hm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED).get('average_trace_fitness', 0), "Precisão": precision_evaluator.apply(log_top_10_variants, net_hm, im_hm, fm_hm), "Generalização": generalization_evaluator.apply(log_top_10_variants, net_hm, im_hm, fm_hm), "Simplicidade": simplicity_evaluator.apply(net_hm)}
    plots['metrics_heuristic'] = convert_fig_to_bytes(plot_metrics_chart(metrics_hm, 'Métricas de Qualidade (Heuristics Miner)'))
    metrics['heuristics_miner'] = metrics_hm
    
    # 3. Análises de Performance 
    kpi_temporal = _df_projects.groupby('completion_month').agg(avg_lead_time=('actual_duration_days', 'mean'), throughput=('project_id', 'count')).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 6)); ax1.plot(kpi_temporal['completion_month'], kpi_temporal['avg_lead_time'], marker='o', color='#2563EB', label='Lead Time'); ax2 = ax1.twinx(); ax2.bar(kpi_temporal['completion_month'], kpi_temporal['throughput'], color='#06B6D4', alpha=0.6, label='Throughput'); fig.suptitle('Séries Temporais de KPIs de Performance')
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9)); ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.label.set_color('#2563EB'); ax2.yaxis.label.set_color('#06B6D4'); ax1.tick_params(axis='y', colors='#2563EB'); ax2.tick_params(axis='y', colors='#06B6D4')
    plots['kpi_time_series'] = convert_fig_to_bytes(fig)
    
    # DFG de ESPERA: para cada par A->B consecutivo, média de (início de B - fim de A), em segundos.
    # Só se mostram transições presentes em >= 1% dos casos (o resto é ruído raro).
    _dl = log_df_full_lifecycle.sort_values(['case:concept:name', 'start_timestamp', 'time:timestamp']).copy()
    _dl['prev_act'] = _dl.groupby('case:concept:name')['concept:name'].shift(1)
    _dl['prev_end'] = _dl.groupby('case:concept:name')['time:timestamp'].shift(1)
    _dl = _dl.dropna(subset=['prev_act'])
    _dl['wait_s'] = (_dl['start_timestamp'] - _dl['prev_end']).dt.total_seconds().clip(lower=0)
    _agg = _dl.groupby(['prev_act', 'concept:name']).agg(wait=('wait_s', 'mean'), n=('case:concept:name', 'nunique'))
    n_cases = log_df_full_lifecycle['case:concept:name'].nunique()
    _agg = _agg[_agg['n'] >= max(1, 0.01 * n_cases)]
    dfg_perf = {(a, b): float(r['wait']) for (a, b), r in _agg.iterrows()}
    gviz_dfg = dfg_visualizer.apply(dfg_perf, log=log_full_pm4py, variant=dfg_visualizer.Variants.PERFORMANCE)
    plots['performance_heatmap'] = convert_gviz_to_bytes(gviz_dfg)
    
    fig, ax = plt.subplots(figsize=(8, 4)); log_df_full_lifecycle['weekday'] = log_df_full_lifecycle['time:timestamp'].dt.day_name(); weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = log_df_full_lifecycle.groupby('weekday')['case:concept:name'].count().reindex(weekday_order).fillna(0); sns.barplot(x=heatmap_data.index, y=heatmap_data.values, ax=ax, hue=heatmap_data.index, legend=False, palette='coolwarm'); ax.set_title('Ocorrências de Atividades por Dia da Semana'); plt.xticks(rotation=45)
    plots['temporal_heatmap_fixed'] = convert_fig_to_bytes(fig)

    # 4. Gantt Chart (usa amostra de 50 se os dados forem grandes)
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
    
    # 5. Análises de Recursos
    log_df_complete = pm4py.convert_to_dataframe(_event_log_pm4py)
    handover_edges = Counter((log_df_complete.iloc[i]['org:resource'], log_df_complete.iloc[i+1]['org:resource']) for i in range(len(log_df_complete)-1) if log_df_complete.iloc[i]['case:concept:name'] == log_df_complete.iloc[i+1]['case:concept:name'] and log_df_complete.iloc[i]['org:resource'] != log_df_complete.iloc[i+1]['org:resource'])
        
    fig_net, ax_net = plt.subplots(figsize=(18, 12)); 
    G = nx.DiGraph();
    for (source, target), weight in handover_edges.items(): G.add_edge(str(source), str(target), weight=weight)

    # Filtrar o gráfico para mostrar apenas os nós mais relevantes
    node_degrees = dict(G.degree())
    recursos_ordenados = sorted(node_degrees, key=node_degrees.get, reverse=True)
    top_recursos = set(recursos_ordenados[:30])
    
    G_filtrado = G.subgraph(top_recursos).copy() # Lógica simplificada
    
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
    
        ax_net.set_title('Rede Social de Recursos (Top 30 por Conexões)')
        plots['resource_network_adv'] = convert_fig_to_bytes(fig_net)
    

    
    if 'skill_level' in _df_resources.columns:
        perf_recursos = _df_full_context.groupby('resource_id').agg(total_hours=('hours_worked', 'sum'), total_tasks=('task_id', 'nunique')).reset_index()
        perf_recursos['avg_hours_per_task'] = perf_recursos['total_hours'] / perf_recursos['total_tasks']
        perf_recursos = perf_recursos.merge(_df_resources[['resource_id', 'skill_level', 'resource_name']], on='resource_id')
        fig, ax = plt.subplots(figsize=(8, 5)); sns.regplot(data=perf_recursos, x='skill_level', y='avg_hours_per_task', ax=ax, scatter_kws={'color': '#06B6D4'}, line_kws={'color': '#FBBF24'}); ax.set_title("Relação entre Skill e Performance")
        plots['skill_vs_performance_adv'] = convert_fig_to_bytes(fig)
        
        resource_role_counts = _df_full_context.groupby(['resource_name', 'skill_level']).size().reset_index(name='count')

        # Filtrar o dataframe antes de construir o gráfico
        recursos_ordenados_df = resource_role_counts.sort_values('count', ascending=False)
        top_30_recursos = set(recursos_ordenados_df['resource_name'].head(30))
        df_filtrado = resource_role_counts[resource_role_counts['resource_name'].isin(top_30_recursos)] # Lógica simplificada
        
        # Construir o gráfico bipartido a partir dos dados já filtrados
        G_bipartite = nx.Graph()
        resources_nodes = df_filtrado['resource_name'].unique()
        roles_nodes = df_filtrado['skill_level'].unique()
        
        G_bipartite.add_nodes_from(resources_nodes, bipartite=0)
        G_bipartite.add_nodes_from(roles_nodes, bipartite=1)
        for _, row in df_filtrado.iterrows():
            G_bipartite.add_edge(row['resource_name'], row['skill_level'], weight=row['count'])
        
        # Desenhar o gráfico filtrado
        fig, ax = plt.subplots(figsize=(12, max(8, len(resources_nodes) * 0.3))) # Altura dinâmica
        pos = nx.bipartite_layout(G_bipartite, resources_nodes)
        
        nx.draw(G_bipartite, pos, with_labels=True,
                node_color=['#0d6efd' if node in resources_nodes else '#ffc107' for node in G_bipartite.nodes()],
                node_size=2500, ax=ax, font_size=9, edge_color='#dee2e6', font_color='black',
                labels={node: str(node).replace(" ", "\n") for node in G_bipartite.nodes()})
        
        edge_labels = nx.get_edge_attributes(G_bipartite, 'weight')
        nx.draw_networkx_edge_labels(G_bipartite, pos, edge_labels=edge_labels, ax=ax, font_color='#dc3545', font_size=8)
        ax.set_title('Rede de Top 30 Recursos por Função (Atividade)')
        plots['resource_network_bipartite'] = convert_fig_to_bytes(fig)

    variants_df = log_df_full_lifecycle.groupby('case:concept:name').agg(variant=('concept:name', lambda x: tuple(x)), start_timestamp=('start_timestamp', 'min'), end_timestamp=('time:timestamp', 'max')).reset_index()
    variants_df['duration_hours'] = (variants_df['end_timestamp'] - variants_df['start_timestamp']).dt.total_seconds() / 3600
    variant_durations = variants_df.groupby('variant').agg(count=('case:concept:name', 'count'), avg_duration_hours=('duration_hours', 'mean')).reset_index().sort_values(by='count', ascending=False).head(10)
    
    variant_durations['variant_str_wrapped'] = variant_durations['variant'].apply(
        lambda x: textwrap.fill(' -> '.join(map(str, x)), width=90) # Aumentado para 90
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.barplot(x='avg_duration_hours', y='variant_str_wrapped',
                data=variant_durations.astype({'avg_duration_hours':'float'}),
                ax=ax, hue='variant_str_wrapped', legend=False, palette='plasma')
    
    ax.tick_params(axis='y', labelsize=5) 
    
    ax.set_title('Duração Média das 10 Variantes Mais Comuns')
    ax.set_ylabel('Variante do Processo')
    ax.set_xlabel('Duração Média (horas)')
    fig.tight_layout()
    plots['variant_duration_plot'] = convert_fig_to_bytes(fig)
    
    ax.set_title('Duração Média das 10 Variantes Mais Comuns')
    ax.set_ylabel('Variante do Processo')
    ax.set_xlabel('Duração Média (horas)')
    fig.tight_layout()
    plots['variant_duration_plot'] = convert_fig_to_bytes(fig)
    
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

    # 7. Análises Finais
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
            colors = plt.get_cmap('tab10', len(variant_sequences.keys()))
            for i, (variant_name, sequence) in enumerate(variant_sequences.items()):
                ax.plot(range(len(sequence)), [activity_to_y[activity] for activity in sequence], marker='o', linestyle='-', label=variant_name, color=colors(i))
            ax.set_yticks(list(activity_to_y.values()))
            ax.set_yticklabels(list(activity_to_y.keys()))
            ax.set_title('Sequência de Atividades das 10 Variantes Mais Comuns')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
        return fig
    plots['custom_variants_sequence_plot'] = convert_fig_to_bytes(generate_custom_variants_plot(log_full_pm4py))
    
    # Análise de Tempo entre Tipos de Tarefa (Milestones Dinâmicos) ---
    # 1. Obter o início/fim de cada 'task_type' dentro de cada projeto
    df_task_type_times = _df_tasks_raw.groupby(['project_id', 'task_type']).agg(
        type_start=('start_date', 'min'),
        type_end=('end_date', 'max')
    ).reset_index()
    
    # 2. Ordenar por projeto e data de início para obter a sequência
    df_task_type_times.sort_values(['project_id', 'type_start'], inplace=True)
    
    milestone_pairs = []
    if not df_task_type_times.empty:
        # 3. Iterar sobre cada projeto para encontrar as transições
        for project_id, group in df_task_type_times.groupby('project_id'):
            for i in range(len(group) - 1):
                current_task_type = group.iloc[i]
                next_task_type = group.iloc[i+1]
    
                # Calcular a duração entre o fim de um tipo e o início do seguinte
                duration_sec = (next_task_type['type_start'] - current_task_type['type_end']).total_seconds()
                duration_hours = duration_sec / 3600
    
                # Apenas contam tempos de espera positivos
                if duration_hours >= 0: 
                    milestone_pairs.append({
                        'transition': f"{current_task_type['task_type']} -> {next_task_type['task_type']}", 
                        'duration_hours': duration_hours
                    })
    
    df_milestone_pairs = pd.DataFrame(milestone_pairs)
    
    if not df_milestone_pairs.empty:
        # 4. Calcular a média de espera para as N piores transições
        top_transitions_by_wait = df_milestone_pairs.groupby('transition')['duration_hours'].mean().nlargest(15).reset_index()
    
        fig, ax = plt.subplots(figsize=(10, 8)); # Aumentar tamanho para legibilidade
        # Usar barplot da média em vez de boxplot
        sns.barplot(data=top_transitions_by_wait, x='duration_hours', y='transition', ax=ax, orient='h', hue='transition', legend=False, palette='coolwarm')
        ax.set_title('Análise de Tempo entre Tipos de Tarefa (Top 15 Esperas Médias em Horas)'); 
        fig.tight_layout()
        plots['milestone_time_analysis_plot'] = convert_fig_to_bytes(fig)
    else:
        plots['milestone_time_analysis_plot'] = None # Define como None se não houver dados
    
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
    
    # --- TOP 10 MELHORES E PIORES ---
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

# ---  FUNÇÃO DE ANÁLISE (EDA) ---
#@st.cache_data
def run_eda_analysis(dfs):
    plots = {}
    tables = {}
    
    # --- Pré-processamento e Feature Engineering ---
    df_projects = dfs['projects'].copy()
    df_tasks = dfs['tasks'].copy()
    df_resources = make_unique_resource_names(dfs['resources'])
    df_resource_allocations = dfs['resource_allocations'].copy()
    df_dependencies = dfs['dependencies'].copy()

    for df in [df_projects, df_tasks, df_resource_allocations]:
        for col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    
    if 'project_id' in df_projects.columns:
        # força project_id para string em todos os dataframes relevantes
        df_projects['project_id'] = df_projects['project_id'].astype(str)
    for tmp_df in [df_tasks, df_resource_allocations, df_dependencies]:
        if 'project_id' in tmp_df.columns:
            tmp_df['project_id'] = tmp_df['project_id'].astype(str)
    
    
    df_projects = df_projects.drop_duplicates(subset=['project_id']).reset_index(drop=True)
    
    # --- Calcular days_diff de forma robusta (mantém NaN para inspeção) ---
    
    df_projects['end_date'] = pd.to_datetime(df_projects['end_date'], errors='coerce')
    df_projects['planned_end_date'] = pd.to_datetime(df_projects['planned_end_date'], errors='coerce')
    
    # calcula diferença em dias; preserva NaN quando faltar data
    df_projects['days_diff'] = (df_projects['end_date'] - df_projects['planned_end_date']).dt.days
    
    # coluna indicadora de validade (para usar nas visualizações e evitar mascarar NaNs)
    df_projects['days_diff_valid'] = df_projects['days_diff'].notna()
    
    
    df_projects['days_diff_abs'] = df_projects['days_diff'].abs()
    df_projects['actual_duration_days'] = (df_projects['end_date'] - df_projects['start_date']).dt.days
    df_projects['project_type'] = df_projects['path_name']
    df_tasks['task_duration_days'] = (df_tasks['end_date'] - df_tasks['start_date']).dt.days
    df_projects['completion_quarter'] = df_projects['end_date'].dt.to_period('Q')
    df_projects['completion_month'] = df_projects['end_date'].dt.to_period('M')

    df_alloc_costs = df_resource_allocations.merge(df_resources, on='resource_id')
    df_alloc_costs['cost_of_work'] = df_alloc_costs['hours_worked'] * df_alloc_costs['cost_per_hour']

    project_aggregates = df_alloc_costs.groupby('project_id').agg(
        total_actual_cost=('cost_of_work', 'sum'),
        avg_hourly_rate=('cost_per_hour', 'mean'),
        num_resources=('resource_id', 'nunique')
    ).reset_index()
    
    df_projects = df_projects.merge(project_aggregates, on='project_id', how='left')
    df_projects['total_actual_cost'] = df_projects['total_actual_cost'].fillna(0)

    dep_counts = df_dependencies.groupby('project_id').size().reset_index(name='dependency_count')
    df_projects = df_projects.merge(dep_counts, on='project_id', how='left')
    df_projects['dependency_count'] = df_projects['dependency_count'].fillna(0)
    
    df_projects['cost_diff'] = df_projects['total_actual_cost'] - df_projects['budget_impact']
    df_projects['cost_per_day'] = df_projects['total_actual_cost'] / df_projects['actual_duration_days'].replace(0, np.nan)

    df_full_context = df_tasks.merge(df_projects, on='project_id', suffixes=('_task', '_project'))
    df_full_context = df_full_context.merge(df_resource_allocations.drop(columns=['project_id'], errors='ignore'), on='task_id')
    df_full_context = df_full_context.merge(df_resources, on='resource_id')
    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']

    df_full_context['cost_of_work'] = df_full_context['hours_worked'] * df_full_context['cost_per_hour']
    
    # --- INÍCIO DA LÓGICA DE COMPLEXIDADE DINÂMICA ---
    if 'complexity_score' not in df_projects.columns:
        # 1. Contar tarefas por projeto
        tasks_per_project = df_tasks.groupby('project_id').size().reset_index(name='calc_num_tasks')
    
        # 2. Contar tipos de recursos únicos por projeto (requer merge)
        alloc_with_types = df_resource_allocations.merge(df_resources[['resource_id', 'resource_type']], on='resource_id', how='left')
        roles_per_project = alloc_with_types.groupby('project_id')['resource_type'].nunique().reset_index(name='calc_num_roles')
    
        df_projects = df_projects.merge(tasks_per_project, on='project_id', how='left')
        df_projects = df_projects.merge(roles_per_project, on='project_id', how='left')
    
        # Preencher NaNs com 0 para o cálculo
        df_projects['calc_num_tasks'] = df_projects['calc_num_tasks'].fillna(0)
        df_projects['calc_num_roles'] = df_projects['calc_num_roles'].fillna(0)
        df_projects['dependency_count'] = df_projects['dependency_count'].fillna(0)
    
        # Função de normalização Min-Max (escala 0-1)
        def min_max_scale(series):
            if series.max() == series.min(): return pd.Series(0.0, index=series.index)
            return (series - series.min()) / (series.max() - series.min())
    
        if not df_projects.empty:
            # Calcular scores normalizados
            score_tasks = min_max_scale(df_projects['calc_num_tasks'])
            score_roles = min_max_scale(df_projects['calc_num_roles'])
            score_deps = min_max_scale(df_projects['dependency_count'])
    
            # Score final (média ponderada ou soma) e escalar para 0-100
            # Damos peso igual a cada componente
            df_projects['complexity_score'] = min_max_scale(score_tasks + score_roles + score_deps) * 100
        else:
            df_projects['complexity_score'] = 0.0
    
    else:
        
        df_projects['complexity_score'] = pd.to_numeric(df_projects['complexity_score'], errors='coerce').fillna(0)
    
    tables['stats_table'] = df_projects[['actual_duration_days', 'days_diff', 'budget_impact', 'total_actual_cost', 'cost_diff', 'num_resources', 'avg_hourly_rate', 'complexity_score']].describe().round(2)
    
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
    fig.tight_layout() 
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
    
    if 'task_id' in df_tasks.columns:
        df_tasks['task_id'] = df_tasks['task_id'].astype(str)
    if 'task_id_predecessor' in df_dependencies.columns:
        df_dependencies['task_id_predecessor'] = df_dependencies['task_id_predecessor'].astype(str)
    if 'task_id_successor' in df_dependencies.columns:
        df_dependencies['task_id_successor'] = df_dependencies['task_id_successor'].astype(str)
    
    df_dependencies = df_dependencies.dropna(subset=[c for c in ['task_id_predecessor', 'task_id_successor'] if c in df_dependencies.columns])
    
    predecessor_counts = df_dependencies.merge(df_tasks, left_on='task_id_predecessor', right_on='task_id', how='left')['task_type'].value_counts()
    successor_counts = df_dependencies.merge(df_tasks, left_on='task_id_successor', right_on='task_id', how='left')['task_type'].value_counts()
    # ====================================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)); predecessor_counts.plot(kind='bar', color=sns.color_palette("Paired")[1], title='Mais Comuns como Predecessoras', ax=ax1); successor_counts.plot(kind='bar', color=sns.color_palette("Paired")[3], title='Mais Comuns como Sucessoras', ax=ax2);
    plots['plot_25'] = convert_fig_to_bytes(fig)

    if not df_dependencies.empty:
        # Pega o primeiro project_id que tem pelo menos uma dependência
        PROJECT_ID_EXAMPLE = df_dependencies['project_id'].iloc[0] 
        project_deps = df_dependencies[df_dependencies['project_id'] == PROJECT_ID_EXAMPLE]
        if not project_deps.empty:
            G = nx.from_pandas_edgelist(project_deps, 'task_id_predecessor', 'task_id_successor', create_using=nx.DiGraph()); pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(14, 9)); nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, ax=ax); 
            ax.set_title(f'Gráfico de Dependências (Exemplo: Proc {PROJECT_ID_EXAMPLE})') # Título dinâmico
            plots['plot_26'] = convert_fig_to_bytes(fig)
    else:
        plots['plot_26'] = None 
    
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

# --- MOTOR DE DIAGNÓSTICO ---
from scipy import stats
import matplotlib.dates as mdates
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import math # Para verificar isnan
import re # Para extrair IDs de cartão

# --- MAPEAMENTO DE REFERÊNCIAS ---
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
    """ Lógica de investigação flexível"""

    def __init__(self, tables_pre, metrics, data_frames, tables_eda, caps=None):
        self.caps = caps  # None = todas as capacidades disponíveis (comportamento original)
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

        # --- INÍCIO: Descoberta Dinâmica da Ordem das Tarefas ---
        self.task_order_map = {}
        if not self.df_tasks_base.empty and not self.df_projects_base.empty and 'start_date' in self.df_projects_base.columns and 'start_date' in self.df_tasks_base.columns:
            try:
                # Juntar tarefas com data de início do projeto
                df_tasks_merged = self.df_tasks_base.merge(
                    self.df_projects_base[['project_id', 'start_date']], 
                    on='project_id', 
                    suffixes=('_task', '_project')
                )
        
                # Calcular dias desde o início do projeto
                df_tasks_merged['days_from_start'] = (
                    df_tasks_merged['start_date_task'] - df_tasks_merged['start_date_project']
                ).dt.total_seconds() / (24 * 3600)
        
                # Calcular a média de 'dias_from_start' para cada task_type
                avg_start_days = df_tasks_merged.groupby('task_type')['days_from_start'].mean().sort_values()
        
                # Criar o mapa de ordem
                self.task_order_map = {task_type: i for i, task_type in enumerate(avg_start_days.index)}
                print("--- ORDEM DE TAREFAS GERADA DINAMICAMENTE ---")
                print(self.task_order_map)
        
            except Exception as e:
                print(f"Erro ao gerar task_order_map dinâmico: {e}")
                # Fallback: cria um mapa vazio se falhar
                self.task_order_map = {} 
        
        self.insights = { k: [] for k in ['saude_geral'] + list(SECTION_MAP.keys()) }
        self.narrative_flags = {} # O cérebro da V10

        self._preprocess_data()
        self._prepare_aggregated_data()

    def _initialize_aggregated_dfs(self):
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
                 self.df_activity_service_times = self.df_full_context_base.groupby(['task_id', 'task_name'])['hours_worked'].sum().reset_index().groupby('task_name')['hours_worked'].mean().reset_index()
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
                          self.df_monthly_wait_time = df_wait_evo.groupby('completion_month')['waiting_time_days'].mean().reset_index().sort_values('completion_month')


                df_tasks_analysis['previous_task_name'] = df_tasks_analysis.groupby('project_id')['task_name'].shift(1)
                if 'previous_task_name' in df_tasks_analysis.columns: self.df_wait_matrix = df_tasks_analysis.pivot_table(index='previous_task_name', columns='task_name', values='waiting_time_days', aggfunc='mean')

            except Exception as e: print(f"Erro _prepare: Cálculos Tempo Espera: {e}")

            # --- Duração Média por Fase ---
            try:
                 def get_phase(task_type):
                     if not isinstance(task_type, str): return 'Fase Desconhecida'
                     # Simplificado para maior robustez
                     if 'onboarding' in task_type.lower() or 'kyc' in task_type.lower() or 'documental' in task_type.lower(): return '1. Onboarding'
                     elif 'risco' in task_type.lower() or 'proposta' in task_type.lower(): return '2. Análise Risco'
                     elif 'avaliação' in task_type.lower() or 'imóvel' in task_type.lower(): return '3. Avaliação'
                     elif 'decisão' in task_type.lower() or 'condições' in task_type.lower(): return '4. Decisão'
                     elif 'fecho' in task_type.lower() or 'legal' in task_type.lower() or 'desembolso' in task_type.lower(): return '5. Contratação'
                     return 'Outra Fase'

                 if 'task_type' in self.df_tasks_base.columns:
                     df_ph = self.df_tasks_base.dropna(subset=['project_id', 'start_date', 'end_date', 'task_type']).copy()
                     df_ph['_d'] = (df_ph['end_date'] - df_ph['start_date']).dt.days
                     ph_times = df_ph.groupby(['project_id', 'task_type'])['_d'].sum().reset_index(name='cycle_time_days')
                     ph_times = ph_times[ph_times['cycle_time_days'] >= 0]
                     if not ph_times.empty: 
                         # A variável de classe é preenchida com os dados de task_type
                         self.df_avg_cycle_time_phase = ph_times.groupby('task_type')['cycle_time_days'].mean().reset_index()
            except Exception as e: print(f"Erro _prepare: Duração Fase: {e}")

        # --- Cálculos de Handoffs e Rede Social (log reconstruído) ---
        try:
             if not self.df_full_context_base.empty and 'project_id' in self.df_full_context_base.columns:
                log_cols = ['project_id', 'task_id', 'task_name', 'start_date_task', 'end_date_task', 'resource_name', 'hours_worked']
                if all(c in self.df_full_context_base.columns for c in log_cols):
                    # Log ao nível da TAREFA (antes: 1 linha por dia de alocação -> falsos handoffs A->A)
                    fc = self.df_full_context_base[log_cols].dropna(subset=['project_id', 'task_id']).copy()
                    main_res = fc.groupby(['task_id', 'resource_name'])['hours_worked'].sum().reset_index().sort_values('hours_worked', ascending=False).drop_duplicates('task_id')
                    log_df = fc.drop_duplicates('task_id').drop(columns=['resource_name', 'hours_worked']).merge(main_res[['task_id', 'resource_name']], on='task_id', how='left')
                    log_df.rename(columns={'project_id': 'case', 'task_name': 'act', 'start_date_task': 'start', 'end_date_task': 'ts', 'resource_name': 'res'}, inplace=True)
                    log_df['ts'] = pd.to_datetime(log_df['ts'], errors='coerce'); log_df['start'] = pd.to_datetime(log_df['start'], errors='coerce')
                    log_df.dropna(subset=['ts', 'case', 'res'], inplace=True)
                    log_df['_ord'] = pd.to_numeric(log_df['task_id'], errors='coerce')
                    log_df.sort_values(['case', 'start', 'ts', '_ord'], inplace=True)

                    if not log_df.empty:
                        # Handoffs Atividade: espera = início desta tarefa - fim da anterior
                        log_df['prev_end'] = log_df.groupby('case')['ts'].shift(1)
                        log_df['wait_sec']=(log_df['start'] - log_df['prev_end']).dt.total_seconds()
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
                        # Ignorar passagens que envolvem funções com 1 só pessoa: são estrutura do processo
                        # (ex.: Diretor de Risco -> Comité), não um 'hub' emergente. Esse risco é tratado no insight de funções especializadas.
                        if not self.df_resources_base.empty and 'resource_type' in self.df_resources_base.columns:
                            role_size = self.df_resources_base['resource_type'].value_counts()
                            name_role = self.df_resources_base.set_index('resource_name')['resource_type'].to_dict()
                            single = {n for n, r in name_role.items() if role_size.get(r, 0) <= 1}
                            res_handoffs = Counter({k: v for k, v in res_handoffs.items() if k[0] not in single and k[1] not in single})
                        self.resource_handoffs_counts = res_handoffs
                else: print("Aviso _prepare: Colunas essenciais em falta em df_full_context.")
             else: print("Aviso _prepare: df_full_context vazio ou sem project_id.")
        except Exception as e: print(f"Erro _prepare: Cálculos Log: {e}")

        # --- Obter dados pré-calculados que não são recalculados  ---
        try:
             # Tenta obter dos dados processados pela EDA, garantindo a ordenação
             if not self.df_projects_base.empty and 'completion_month' in self.df_projects_base.columns:
                monthly_kpis_calc = self.df_projects_base.groupby('completion_month').agg(
                    mean_days_diff=('days_diff', 'mean'),
                    mean_cost_diff=('cost_diff', 'mean'),
                    completed_projects=('project_id', 'count'),
                    mean_duration=('actual_duration_days', 'mean')
                ).reset_index()
                # Tenta converter para período para ordenar corretamente
                try:
                    monthly_kpis_calc['month_period'] = pd.to_datetime(monthly_kpis_calc['completion_month'], format='%Y-%m').dt.to_period('M')
                    monthly_kpis_calc = monthly_kpis_calc.sort_values('month_period').drop(columns=['month_period'])
                except ValueError: # Se o formato não for YYYY-MM, ordena como string
                    monthly_kpis_calc = monthly_kpis_calc.sort_values('completion_month')

                if not monthly_kpis_calc.empty:
                    self.df_monthly_kpis_eda = monthly_kpis_calc

             # Tenta obter dos dados de PM, garantindo a ordenação
             if not self.df_projects_base.empty and 'completion_month' in self.df_projects_base.columns:
                kpi_temporal_calc = self.df_projects_base.groupby('completion_month').agg(
                    avg_lead_time=('actual_duration_days', 'mean'),
                    throughput=('project_id', 'count')
                ).reset_index()
                try:
                    kpi_temporal_calc['month_period'] = pd.to_datetime(kpi_temporal_calc['completion_month'], format='%Y-%m').dt.to_period('M')
                    kpi_temporal_calc = kpi_temporal_calc.sort_values('month_period').drop(columns=['month_period'])
                except ValueError:
                    kpi_temporal_calc = kpi_temporal_calc.sort_values('completion_month')

                if not kpi_temporal_calc.empty:
                    self.df_kpi_temporal = kpi_temporal_calc

             # Fallbacks (se existirem nos dados pré-processados)
             if self.df_monthly_kpis_eda.empty:
                 self.df_monthly_kpis_eda = self.tables_eda_orig.get('monthly_kpis_data', pd.DataFrame())
             if self.df_kpi_temporal.empty:
                self.df_kpi_temporal = self.metrics_orig.get('kpi_temporal_data', pd.DataFrame())

             self.df_monthly_fitness = self.metrics_orig.get('monthly_fitness_data', pd.DataFrame())
             self.df_milestone_transitions = self.metrics_orig.get('milestone_transition_data', pd.DataFrame())
             if self.df_wait_by_activity.empty: self.df_wait_by_activity = self.metrics_orig.get('wait_by_activity_data', pd.DataFrame())
        except Exception as e: print(f"Erro _prepare: Obter dados pré-calculados: {e}")

    # ---  MAPEAR REFERÊNCIAS ---
    def _map_card_keys_to_names(self, card_keys_str):
        """ Mapeia uma string de chaves de cartão (ex: 'Cartão 1, Cartão 22') para nomes. """
        if not isinstance(card_keys_str, str):
            return "Referência Inválida"

        keys = [key.strip() for key in card_keys_str.split(',')]
        # Não citar como evidência cartões que estão escondidos por falta de dados
        def _visible(key):
            if key.startswith('Cartão'):
                try: return caps_ok(CARD_REQUIREMENTS.get(int(key.split()[-1]), set()), self.caps)
                except ValueError: return True
            return True
        keys = [k for k in keys if _visible(k)] or keys[:0]
        if not keys:
            return "Análise de fluxo do processo"
        names = [CARD_TITLE_MAP.get(key, key) for key in keys] # Usa a chave se o nome não for encontrado

        # Remove duplicados mantendo a ordem (caso haja referência repetida)
        unique_names = []
        for name in names:
            if name not in unique_names:
                unique_names.append(name)

        return ", ".join(unique_names)

    def _add_insight(self, section_key, title, detail, card_keys_str, level='problema', priority=10):
        """ Adiciona insight com referência mapeada (NOMES) e prioridade. """
        try:
            # Esconde insights que dependem de dados que o utilizador não tem (exceto o resumo executivo,
            # cujos cenários já são decididos com base nas capacidades disponíveis).
            if self.caps is not None and section_key != 'resumo_executivo' and isinstance(card_keys_str, str):
                needed = set()
                for key in card_keys_str.split(','):
                    key = key.strip()
                    if key.startswith('Cartão'):
                        try: needed |= CARD_REQUIREMENTS.get(int(key.split()[-1]), set())
                        except ValueError: pass
                    elif key == 'KPIs Saúde Geral':
                        needed.add('TEMPOS')
                if not caps_ok(needed, self.caps):
                    return
            section_name = SECTION_MAP.get(section_key, section_key)
            card_names = self._map_card_keys_to_names(card_keys_str) # USA A NOVA FUNÇÃO
            reference = f"{card_names}" # Referência simplificada para apenas os nomes

            if section_key not in self.insights: self.insights[section_key] = []

            self.insights[section_key].append({
                'titulo': title, 'detalhe': detail, 'cartao_ref': reference, 'level': level, 'priority': priority
            })
        except Exception as e:
            print(f"Erro em _add_insight ({title}): {e}")

    def _get_trend(self, series_data, series_index=None):
        # (Mantém-se igual)
        if series_data is None: return 0
        if isinstance(series_data, pd.DataFrame):
            numeric_cols = series_data.select_dtypes(include=np.number).columns
            if not numeric_cols.empty: y_series = series_data[numeric_cols[0]]
            else: return 0
        elif isinstance(series_data, pd.Series): y_series = series_data
        else: return 0

        y = y_series.copy().dropna()
        if y.empty or len(y) < 2: return 0

        x = np.arange(len(y))

        try:
            if np.all(x == x[0]) or np.all(y == y.iloc[0]): return 0
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope if not (math.isnan(slope) or math.isinf(slope)) else 0
        except ValueError: return 0
        except Exception as e: print(f"Erro inesperado em _get_trend: {e}"); return 0

    def run(self):
        self._check_kpis()
        self._run_contextual_analysis()
        self._build_executive_summary()
        self._check_custos_atrasos_facts()
        self._check_performance_prazos_facts()
        self._check_recursos_equipas_facts()
        self._check_gargalos_esperas_facts()
        self._check_fluxo_conformidade_facts()

        for section in self.insights:
            if section != 'saude_geral' and self.insights[section]:
                self.insights[section] = sorted(self.insights[section], key=lambda x: x['priority'])

        return self.insights

    def _run_contextual_analysis(self):
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
        if not self.df_monthly_kpis_eda.empty and len(self.df_monthly_kpis_eda) > 3:
            trend_days_diff = self._get_trend(self.df_monthly_kpis_eda['mean_days_diff']) # dias/mês
            trend_duration = self._get_trend(self.df_monthly_kpis_eda['mean_duration']) # dias/mês
            flags['trend_days_diff'] = trend_days_diff
            flags['trend_duration'] = trend_duration

            if trend_days_diff > 0.2: flags['is_degrading_prazo'] = True
            if trend_duration > 0.2: flags['is_degrading_duracao'] = True

        # --- 3. Análise de Variabilidade (Caos vs. Padrão) ---
        precision = self.metrics_im.get('Precisão', 1.0)
        flags['precision'] = precision
        if precision < 0.7: flags['is_chaotic'] = True

        if not self.df_variants.empty and 'percentage' in self.df_variants.columns:
            top_1_perc = self.df_variants.iloc[0]['percentage']
            top_10_perc = self.df_variants.head(10)['percentage'].sum()
            flags['top_1_variant_perc'] = top_1_perc
            flags['top_10_variant_perc'] = top_10_perc
            if top_1_perc < 20: flags['is_chaotic'] = True
            if top_1_perc > 70:
                # Uma variante dominante (>70% dos casos) é padronização, mesmo que a precisão do modelo seja baixa
                flags['is_standardized'] = True
                flags['is_chaotic'] = False

        # --- 4. Análise de Causa-Raiz (Rework vs. Filas) ---
        self.detected_rework_loops = self._find_rework_in_wait_matrix()
        # Prevalência: % de processos com pelo menos uma atividade repetida (rework)
        rework_pct = 0.0
        if not self.df_tasks_base.empty and {'project_id', 'task_name'}.issubset(self.df_tasks_base.columns):
            rework_pct = float(self.df_tasks_base.groupby('project_id')['task_name'].apply(lambda x: x.duplicated().any()).mean() * 100)
        flags['rework_case_pct'] = rework_pct
        if self.detected_rework_loops:
            flags['rework_loops'] = self.detected_rework_loops
            # Só é tratado como problema estrutural se afetar >= 5% dos processos
            if rework_pct >= 5:
                flags['has_rework'] = True
            else:
                flags['has_rework_pontual'] = True

        if not self.df_activity_wait_stats.empty:
            df_wait = self.df_activity_wait_stats.copy()
            # Adiciona tratamento para service_time_days inexistente ou nulo
            if 'service_time_days' in df_wait.columns:
                 df_wait['wait_ratio'] = df_wait['waiting_time_days'] / (df_wait['service_time_days'].fillna(0) + 0.01)
                 if (df_wait['wait_ratio'] > 1.5).any():
                     flags['has_high_queues'] = True
                     flags['queue_bottlenecks'] = df_wait.nlargest(3, 'wait_ratio')['task_type'].tolist()
            else: # Se service_time_days não existe, usa apenas a espera absoluta
                 if (df_wait['waiting_time_days'] > 1.0).any(): # Ex: > 1 dia de espera média
                      flags['has_high_queues'] = True
                      flags['queue_bottlenecks'] = df_wait.nlargest(3, 'waiting_time_days')['task_type'].tolist()


        if not self.df_activity_service_times.empty and 'service_time_days' in self.df_activity_service_times.columns:
            df_serv = self.df_activity_service_times.dropna(subset=['service_time_days'])
            if not df_serv.empty:
                 mean_service = df_serv['service_time_days'].mean()
                 if mean_service > 0 and (df_serv['service_time_days'] > (mean_service * 3)).any():
                     flags['has_service_bottlenecks'] = True
                     flags['service_bottlenecks'] = df_serv.nlargest(3, 'service_time_days')['task_name'].tolist()

        # --- 5. Análise de Recursos ---
        if not self.df_workload.empty and not self.df_resources_base.empty:
            total_hours = self.df_workload['hours_worked'].sum()
            top_20_perc_count = int(len(self.df_resources_base) * 0.2)
            if top_20_perc_count > 0 and total_hours > 0:
                top_20_perc_hours = self.df_workload.head(top_20_perc_count)['hours_worked'].sum()
                if (top_20_perc_hours / total_hours) > 0.8:
                    flags['has_pareto_herois'] = True

        if self.resource_handoffs_counts:
             if not self.df_projects_base.empty:
                 top_handoff = self.resource_handoffs_counts.most_common(1)
                 if top_handoff:
                     top_handoff = top_handoff[0]
                     # Verifica se a contagem é significativa (ex: > 10% do número de processos)
                     if top_handoff[1] > (len(self.df_projects_base) * 0.1):
                         flags['has_resource_hub'] = True
                         flags['resource_hub_names'] = f"{top_handoff[0][0]} -> {top_handoff[0][1]}"


        # Só manter as conclusões que os dados disponíveis suportam
        c = self.caps
        if not caps_ok({'PLANO'}, c):
            for k in ['baseline_efficient', 'baseline_inefficient', 'is_degrading_prazo', 'mean_days_diff', 'trend_days_diff']: flags.pop(k, None)
        if not caps_ok({'TEMPOS'}, c):
            for k in ['has_high_queues', 'queue_bottlenecks', 'has_rework', 'rework_loops']: flags.pop(k, None)
            if flags.get('rework_case_pct', 0) > 0: flags['has_rework_pontual'] = flags.get('rework_case_pct', 0) < 5
            if flags.get('rework_case_pct', 0) >= 5: flags['has_rework_freq_sem_tempos'] = True
        if not caps_ok({'ESFORCO'}, c):
            for k in ['has_service_bottlenecks', 'service_bottlenecks', 'has_pareto_herois']: flags.pop(k, None)
        if not caps_ok({'RECURSOS'}, c):
            for k in ['has_resource_hub', 'resource_hub_names']: flags.pop(k, None)
        self.narrative_flags = flags

    def _find_rework_in_wait_matrix(self):
        rework_loops = []
        # Verifica se o mapa dinâmico foi criado
        if self.df_wait_matrix.empty or not self.task_order_map:
            return rework_loops
    
        wait_matrix_flat = self.df_wait_matrix.stack().reset_index()
        wait_matrix_flat.columns = ['previous_task_name', 'task_name', 'wait_days']
        wait_matrix_flat = wait_matrix_flat[wait_matrix_flat['wait_days'] > 0]
    
        if wait_matrix_flat.empty:
            return rework_loops
    
        # Mapear 'task_name' (das colunas da matriz) para 'task_type'
        task_name_to_type_map = self.df_tasks_base[['task_name', 'task_type']].drop_duplicates().set_index('task_name')['task_type']
    
        wait_matrix_flat['from_type'] = wait_matrix_flat['previous_task_name'].map(task_name_to_type_map)
        wait_matrix_flat['to_type'] = wait_matrix_flat['task_name'].map(task_name_to_type_map)
    
        # Usar o mapa de ordem dinâmico (self.task_order_map) que é baseado em task_type
        wait_matrix_flat['from_order'] = wait_matrix_flat['from_type'].map(self.task_order_map).fillna(99)
        wait_matrix_flat['to_order'] = wait_matrix_flat['to_type'].map(self.task_order_map).fillna(99)
    
        # Rework é quando a ordem 'from' é maior que a 'to' (fluxo para trás)
        df_rework = wait_matrix_flat[
            (wait_matrix_flat['from_order'] > wait_matrix_flat['to_order']) &
            (wait_matrix_flat['from_order'] != 99) &
            (wait_matrix_flat['to_order'] != 99)
        ]
    
        if not df_rework.empty:
            # Agrupa por tipo de transição para consolidar
            df_rework_agg = df_rework.groupby(['from_type', 'to_type'])['wait_days'].mean().reset_index().sort_values('wait_days', ascending=False)
    
            for _, row in df_rework_agg.head(3).iterrows():
                # O loop agora é entre *tipos* de tarefa
                loop_str = f"{row['from_type']} -> {row['to_type']}"
                rework_loops.append({'loop': loop_str, 'wait_days': row['wait_days']})
    
        return rework_loops

    def _build_executive_summary(self):
        f = self.narrative_flags

        # --- Cenário 1: O Paradoxo da Eficiência ---
        if f.get('baseline_efficient') and (f.get('is_degrading_prazo') or f.get('is_degrading_duracao')) and f.get('has_rework'):
            loop_ex = f['rework_loops'][0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: O Paradoxo da Eficiência e o Rework Oculto',
                              f"A operação vive um paradoxo: embora os processos terminem adiantados face ao plano (média: {f.get('mean_days_diff', 0):.1f} dias), a performance está a degradar-se (duração a aumentar ~{f.get('trend_duration', 0):.1f} dias/mês). "
                              f"A causa raiz é **rework (retrabalho)**. Fluxos de retorno, como '{loop_ex['loop']}', consomem tempo ({loop_ex['wait_days']:.1f}d) e indicam falta de padronização (Precisão do Modelo: {f.get('precision', 0):.1%}).",
                              "Cartão 1, Cartão 22, Cartão 58, Cartão 61", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              f"Investigue a causa raiz do rework, especialmente no fluxo '{loop_ex['loop']}'. Reduzir estes ciclos de retorno é crucial para travar a degradação da performance.",
                              "Cartão 58, Cartão 42", level='recomendacao', priority=2)

        # --- Cenário 2: Gargalo Clássico (Lento, mas Padronizado) ---
        elif f.get('baseline_inefficient') and (f.get('is_standardized') or not f.get('is_chaotic')) and f.get('has_high_queues') and not f.get('has_rework'):
            queue_ex = f.get('queue_bottlenecks', ['N/A'])[0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Gargalo de Fila Clássico',
                              f"O processo é lento (atraso médio: {f.get('mean_days_diff', 0):.1f} dias) devido a um gargalo de fila. O fluxo é relativamente padronizado (Precisão: {f.get('precision', 0):.1%}), sem rework significativo. "
                              f"O problema são tarefas paradas em fila, especialmente antes de '{queue_ex}', onde o tempo de espera excede o tempo de execução.",
                              "Cartão 23, Cartão 61, Cartão 44", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              f"Analise a capacidade de recursos alocados à atividade '{queue_ex}'. O problema é falta de capacidade ou processamento em lote? Considere realocar recursos.",
                              "Cartão 44, Cartão 46", level='recomendacao', priority=2)

        # --- Cenário 3: Processo Caótico (Rework é o Problema) ---
        elif (f.get('baseline_inefficient') or f.get('is_degrading_prazo')) and f.get('is_chaotic') and f.get('has_rework'):
            loop_ex = f['rework_loops'][0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Processo Caótico Dominado por Rework',
                              f"A performance é baixa ou está a degradar-se devido à falta de padronização. O processo é caótico (Precisão: {f.get('precision', 0):.1%}, Variante Top 1: {f.get('top_1_variant_perc', 0):.1f}%). "
                              f"Isto resulta em ciclos de rework caros, como '{loop_ex['loop']}', que são os principais gargalos de tempo ({loop_ex['wait_days']:.1f}d).",
                              "Cartão 23, Cartão 22, Cartão 61, Cartão 65, Cartão 58", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "Defina e reforce um 'caminho feliz' (processo padrão). O foco deve ser na qualidade 'à primeira' para eliminar o rework. Aumentar a conformidade (ver 'Score de Conformidade') irá reduzir o tempo total.",
                              "Cartão 71", level='recomendacao', priority=2)

        # --- Cenário 4: Processo Lento (Tarefas Demoradas) ---
        elif f.get('baseline_inefficient') and f.get('has_service_bottlenecks') and not f.get('has_rework') and not f.get('has_high_queues'):
            service_ex = f.get('service_bottlenecks', ['N/A'])[0]
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Processo Lento (Gargalo de Execução)',
                              f"O processo é ineficiente (atraso médio: {f.get('mean_days_diff', 0):.1f} dias), mas flui de forma relativamente padronizada. O problema principal é o tempo de execução (serviço) de tarefas-chave. "
                              f"Atividades como '{service_ex}' são gargalos de execução, demorando significativamente mais que as restantes.",
                              "Cartão 23, Cartão 44, Cartão 49", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              f"Otimize a execução da tarefa '{service_ex}'. Avalie automação, complexidade ou necessidade de formação. Melhorar a eficiência desta tarefa específica terá o maior impacto.",
                              "Cartão 49", level='recomendacao', priority=2)

        # --- Cenário 5: "O Herói" (Dependência de Recursos) ---
        elif f.get('has_pareto_herois') and f.get('has_resource_hub'):
            hub_ex = f.get('resource_hub_names', 'N/A')
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Risco de "Herói" e Hub de Comunicação',
                              "A operação apresenta risco de dependência de recursos: ~20% dos recursos concentram >80% do trabalho. "
                              f"Adicionalmente, a interação '{hub_ex}' funciona como um hub central, criando um potencial ponto único de falha.",
                              "Cartão 31, Cartão 32, Cartão 40", level='problema', priority=1)
            self._add_insight('resumo_executivo', 'Recomendação Imediata',
                              "Inicie um plano de cross-training e delegação para reduzir a dependência dos 'heróis' e descentralizar o fluxo do hub principal. A performance atual pode ser boa, mas é frágil.",
                              "Cartão 39", level='recomendacao', priority=2)

        # --- Cenário 6: Processo Saudável ---
        elif f.get('baseline_efficient') and not f.get('is_degrading_prazo') and not f.get('is_chaotic'):
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Processo Saudável e Padronizado',
                              f"A análise revela um processo globalmente saudável: eficiente (termina em média {abs(f.get('mean_days_diff', 0)):.1f} dias antes do planeado) e padronizado ({f.get('top_1_variant_perc', 0):.1f}% dos processos seguem a mesma sequência). "
                              + (f"Ponto de atenção: a duração média está a aumentar (+{f.get('trend_duration', 0):.2f} dias/mês). " if f.get('is_degrading_duracao') else "A performance está estável. ")
                              + (f"O rework existe mas é pontual ({f.get('rework_case_pct', 0):.1f}% dos processos)." if f.get('has_rework_pontual') else "Não há rework significativo."),
                              "Cartão 1, Cartão 22, Cartão 61", level='facto', priority=1)
            self._add_insight('resumo_executivo', 'Próximos Passos Sugeridos',
                              "O foco deve ser a manutenção e monitorização contínua. Use a Simulação de RL para testar cenários de otimização (aumento de volume, redução de equipa) e identificar oportunidades de melhoria incremental.",
                              "Página de RL", level='recomendacao', priority=2)

        # --- Cenário 6b: Padronizado mas sem dados de prazo (sem PLANO) ---
        elif f.get('is_standardized') and not caps_ok({'PLANO'}, self.caps):
            partes = [f"O processo é padronizado: {f.get('top_1_variant_perc', 0):.1f}% dos casos seguem a mesma sequência de atividades."]
            if f.get('is_degrading_duracao'):
                partes.append(f"Ponto de atenção: a duração média está a aumentar (+{f.get('trend_duration', 0):.2f} dias/mês).")
            if f.get('rework_case_pct', 0) > 0:
                partes.append(f"Há retrabalho em {f.get('rework_case_pct', 0):.1f}% dos processos.")
            partes.append("Sem data planeada (ou SLA) não é possível avaliar atrasos; indique um SLA nas Configurações para ativar essa análise.")
            self._add_insight('resumo_executivo', 'Diagnóstico Principal: Processo Padronizado', " ".join(partes), "Cartão 65", level='facto', priority=1)
            if caps_ok({'TEMPOS'}, self.caps):
                self._add_insight('resumo_executivo', 'Próximos Passos Sugeridos',
                                  "Analise as maiores esperas entre atividades (secção Handoffs e Espera): num processo padronizado, reduzir a espera nos handoffs mais lentos é a alavanca com maior impacto na duração.",
                                  "Cartão 42, Cartão 52", level='recomendacao', priority=2)

        # --- Cenário 7: Genérico ---
        else:
             self._add_insight('resumo_executivo', 'Diagnóstico Principal: Múltiplos Pontos de Otimização',
                              "A análise não identifica um único arquétipo de problema dominante. Existem múltiplos pontos de otimização de menor impacto. O processo não está em crise, mas pode ser melhorado. "
                              "Recomendamos focar nos maiores gargalos de tempo ('Análise: Gargalos e Rework') e nos recursos menos eficientes ('Análise: Recursos e Equipas').",
                              "Cartão 42, Cartão 33", level='problema', priority=1)

    # --- Funções de Recolha de Factos ---

    def _check_kpis(self):
        try:
            def fmt(v, f):
                return f.format(v) if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else 'N/A'
            dur_media = self.kpis.get('Duração Média Num', np.nan)
            # % espera calculada por processo (soma das esperas do processo / duração do processo)
            perc_espera = self.kpis.get('% Tempo em Espera', np.nan)
            espera_proc = self.kpis.get('Espera Média por Processo (dias)', np.nan)
            kpi_geral = [
                {'label': 'Duração Média', 'value': fmt(dur_media, "{:.1f} dias")},
                {'label': 'Custo Médio', 'value': fmt(self.kpis.get('Custo Médio'), "€{:,.0f}")},
                {'label': 'Desvio Custo Médio', 'value': fmt(self.kpis.get('Desvio de Custo Médio'), "€{:,.0f}")},
                {'label': '% Processos Atrasados', 'value': fmt(self.delay_kpis.get('% Processos Atrasados'), "{:.1f}%")},
                {'label': 'Atraso Médio dos Atrasados', 'value': fmt(self.delay_kpis.get('Atraso Médio (dias)'), "{:.1f} dias")},
                {'label': 'Desvio Médio face ao Plano', 'value': fmt(self.delay_kpis.get('Desvio Médio face ao Plano (dias)'), "{:+.1f} dias")},
                {'label': 'Espera Média por Processo', 'value': fmt(espera_proc, "{:.1f} dias")},
                {'label': '% Tempo em Espera', 'value': fmt(perc_espera, "{:.1f}%")},
            ]
            kpi_req = {'Custo Médio': {'CUSTO'}, 'Desvio Custo Médio': {'CUSTO', 'ORCAMENTO'},
                       '% Processos Atrasados': {'PLANO'}, 'Atraso Médio dos Atrasados': {'PLANO'},
                       'Desvio Médio face ao Plano': {'PLANO'}, 'Espera Média por Processo': {'TEMPOS'}, '% Tempo em Espera': {'TEMPOS'}}
            self.insights['saude_geral'] = [k for k in kpi_geral if caps_ok(kpi_req.get(k['label'], set()), self.caps)]
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
                             self._add_insight(section, 'Observação: Baixa Pontualidade (Baseline)', f"{perc_mau:.0%} dos processos terminam atrasados (média: {mean_days_diff:.1f} dias).", "Cartão 1, Cartão 23", level='facto')
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
                self._add_insight(section, 'Alerta: Performance de Prazo em Degradação', f"A "f"folga"f" de prazo está a diminuir. A tendência do atraso médio é de {self.narrative_flags.get('trend_days_diff', 0):.2f} dias/mês (a piorar).", "Cartão 22", level='problema', priority=5)
            elif self.narrative_flags.get('trend_days_diff', 0) < -0.2:
                self._add_insight(section, 'Destaque: Performance de Prazo em Melhoria', f"A pontualidade está a melhorar consistentemente (tendência: {self.narrative_flags.get('trend_days_diff', 0):.2f} dias/mês).", "Cartão 22", level='facto')

            if self.narrative_flags.get('is_degrading_duracao'):
                self._add_insight(section, 'Alerta: Duração Média a Aumentar', f"Os processos estão a demorar mais tempo a concluir (tendência: +{self.narrative_flags.get('trend_duration', 0):.2f} dias/mês).", "Cartão 3, Cartão 12", level='problema', priority=5)
        except Exception as e: print(f"Erro Regra [22, 3, 12]: {e}")

    def _check_performance_prazos_facts(self):
        section = 'diagnostico_custos_atrasos' # Adiciona a esta secção para consolidar

        # Cartão 18: Duração Média por Fase
        try:
            if not self.df_avg_cycle_time_phase.empty:
                top_2 = self.df_avg_cycle_time_phase.nlargest(2, 'cycle_time_days')
                fases_str = [f"'{r['task_type']}' ({r['cycle_time_days']:.1f} dias)" for _, r in top_2.iterrows()]
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
                 specialists = resource_counts[resource_counts <= 2] # Encontra funções com 1 ou 2 pessoas
                 if not specialists.empty:
                     # Remove a lógica de 'critical_roles'
                     # O insight agora reporta sobre os especialistas encontrados
                     specialist_names = specialists.index.tolist()
                     # Limita a 3 nomes para não poluir o insight
                     display_names = ', '.join(specialist_names[:3]) 
                     if len(specialist_names) > 3: display_names += "..."
            
                     self._add_insight(section, 'Observação: Funções Especializadas (Potencial Risco)', f"Funções como '{display_names}' dependem de <= 2 pessoas, criando potenciais pontos de falha ou gargalos.", "Cartão 26", level='facto', priority=11)
        except Exception as e: print(f"Erro Regra [26]: {e}")

        # Cartão 31 & 40: "Heróis" e "Hubs"
        try:
            if self.narrative_flags.get('has_pareto_herois'):
                top_1_workload = self.df_workload.iloc[0] if not self.df_workload.empty else None
                if top_1_workload is not None:
                     self._add_insight(section, "Observação: Concentração de Esforço (Pareto)", f"Existe uma alta concentração de trabalho: o Top 20% dos recursos faz > 80% das horas. '{top_1_workload['resource_name']}' é o recurso mais ativo.", "Cartão 31", level='facto')

            if self.narrative_flags.get('has_resource_hub'):
                hub_str = self.narrative_flags.get('resource_hub_names', 'N/A')
                self._add_insight(section, 'Observação: Hub de Comunicação', f"A interação '{hub_str}' é a mais frequente, funcionando como um hub (e potencial gargalo) de comunicação.", "Cartão 32, Cartão 40", level='facto', priority=11)
        except Exception as e: print(f"Erro Regra [31, 32, 40]: {e}")

        # Cartão 33: Eficiência
        try:
            if not self.df_efficiency_metrics.empty and 'avg_hours_per_task' in self.df_efficiency_metrics.columns:
                df_effic = self.df_efficiency_metrics.dropna(subset=['avg_hours_per_task']).sort_values(by='avg_hours_per_task')
                if len(df_effic) >= 2:
                    melhor = df_effic.iloc[0]; pior = df_effic.iloc[-1]
                    ratio = pior['avg_hours_per_task'] / max(melhor['avg_hours_per_task'], 0.1)
                    if not pd.isna(ratio) and ratio > 4:
                        self._add_insight(section, 'Observação: Disparidade de Performance', f"Recurso mais lento ('{pior['resource_name']}') demora {ratio:.1f}x mais por tarefa que o mais rápido ('{melhor['resource_name']}').", "Cartão 33", level='facto')
        except Exception as e: print(f"Erro Regra [33]: {e}")

        # Cartão 37: Skill vs Performance
        try:
            if not self.df_full_context_base.empty and 'skill_level' in self.df_full_context_base.columns and 'hours_worked' in self.df_full_context_base.columns:
                # (Lógica da V8 para calcular corr skill/performance)
                task_counts = self.df_full_context_base.groupby('resource_name')['task_id'].nunique()
                hours_sum = self.df_full_context_base.groupby('resource_name')['hours_worked'].sum()
                perf_df = pd.DataFrame({'hours_per_task': hours_sum / task_counts.replace(0, np.nan)}).dropna().reset_index()
                skill_map = self.df_full_context_base[['resource_name', 'skill_level']].drop_duplicates().set_index('resource_name')
                perf_df = perf_df.join(skill_map, on='resource_name').dropna(subset=['skill_level', 'hours_per_task'])
                if len(perf_df['skill_level'].unique()) > 1 and len(perf_df) > 1:
                    corr = perf_df['skill_level'].corr(perf_df['hours_per_task'])
                    if not pd.isna(corr) and abs(corr) < 0.2:
                        self._add_insight(section, 'Observação: Métrica de "Skill" Pouco Correlacionada', f"O 'Nível de Competência' (skill) não tem correlação ({corr:.2f}) com a rapidez de execução das tarefas.", "Cartão 37", level='facto') # Removido "Irrelevante?"
        except Exception as e: print(f"Erro Regra [37]: {e}")

        # Cartão 29: Lei de Brooks
        try:
            if not self.df_projects_base.empty and 'num_resources' in self.df_projects_base.columns and 'days_diff' in self.df_projects_base.columns:
                 df_corr = self.df_projects_base[['num_resources', 'days_diff']].apply(pd.to_numeric, errors='coerce').dropna()
                 if len(df_corr['num_resources'].unique()) > 2 and len(df_corr) > 5: # Aumentado para >5 para mais robustez
                     corr = df_corr['num_resources'].corr(df_corr['days_diff'])
                     if not pd.isna(corr) and corr > 0.3:
                         self._add_insight(section, 'Observação: Custo de Coordenação (Lei de Brooks?)', f"Equipas maiores tendem a ter MAIS atrasos (Corr: {corr:.2f}). Adicionar mais pessoas parece piorar a performance.", "Cartão 29", level='facto')
        except Exception as e: print(f"Erro Regra [29]: {e}")

    def _check_gargalos_esperas_facts(self):
        section = 'diagnostico_gargalos_esperas'

        # Insight sobre KPI de Espera
        try:
            # Usa o valor calculado e guardado em _check_kpis
            perc_espera = self.kpis.get('% Tempo em Espera', None)
            if perc_espera is not None: # Verifica se o KPI foi calculado
                if perc_espera > 40:
                    self._add_insight(section, "Alerta: Processo Dominado por Espera em Fila",
                                      f"Mais de {perc_espera:.0f}% da duração total é consumida por tempos de espera *entre tarefas sequenciais* (filas).",
                                      "KPIs Saúde Geral", level='problema', priority=6)
        except Exception as e: print(f"Erro KPI Espera Insight: {e}")


        # Factos sobre Rework (se detetado)
        if self.narrative_flags.get('has_rework'):
            loops = self.narrative_flags.get('rework_loops', [])
            if loops:
                top_loop = loops[0]
                self._add_insight(section, 'Alerta: Principal Gargalo é Rework', f"A maior espera detetada é um fluxo de retrabalho: '{top_loop['loop']}', com uma espera média de {top_loop['wait_days']*24:.1f} horas.", "Cartão 58, Cartão 42", level='problema', priority=7)

                for i, loop in enumerate(loops[1:]):
                     self._add_insight(section, f'Alerta: Rework Adicional #{i+1}', f"'{loop['loop']}' (espera de {loop['wait_days']*24:.1f} horas).", "Cartão 58", level='problema', priority=8)

        # Rework pontual (existe mas afeta < 5% dos processos)
        if self.narrative_flags.get('has_rework_pontual'):
            loops = self.narrative_flags.get('rework_loops', [])
            pct = self.narrative_flags.get('rework_case_pct', 0)
            if loops:
                self._add_insight(section, 'Observação: Rework Pontual',
                                  f"Existe retrabalho, mas só em {pct:.1f}% dos processos. Quando acontece, o retorno mais demorado é '{loops[0]['loop']}' (espera média de {loops[0]['wait_days']*24:.1f} horas). Não é a causa principal da duração.",
                                  "Cartão 58, Cartão 70", level='facto', priority=9)
            else:
                self._add_insight(section, 'Observação: Rework Pontual',
                                  f"Existe retrabalho (atividades repetidas), mas só em {pct:.1f}% dos processos.",
                                  "Cartão 70", level='facto', priority=9)
        if self.narrative_flags.get('has_rework_freq_sem_tempos'):
            self._add_insight(section, 'Alerta: Rework Frequente',
                              f"{self.narrative_flags.get('rework_case_pct', 0):.1f}% dos processos repetem pelo menos uma atividade. Sem datas de início/fim não é possível medir o tempo perdido.",
                              "Cartão 70", level='problema', priority=7)

        # Factos sobre Filas (se detetadas)
        if self.narrative_flags.get('has_high_queues'):
            queue_tasks = self.narrative_flags.get('queue_bottlenecks', [])
            if queue_tasks:
                 self._add_insight(section, 'Alerta: Principal Gargalo de Fila', f"A atividade '{queue_tasks[0]}' é um gargalo de fila: o tempo de espera antes de iniciar é significativamente maior que o tempo de execução.", "Cartão 44, Cartão 57", level='problema', priority=7) # Adicionado Cartão 57

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
                    self._add_insight(section, 'Facto: Principal Gargalo de Fila (Fluxo Normal)', f"No fluxo normal (sem rework), a maior espera é '{top_1_forward['transition']}' ({top_1_forward['handoff_time_days']:.1f} dias).", "Cartão 42, Cartão 52", level='facto')

                top_1_cost = self.df_handoffs.nlargest(1, 'estimated_cost_of_wait').iloc[0]
                self._add_insight(section, 'Facto: Gargalo Mais Caro (Tempo de Espera)', f"A espera entre '{top_1_cost['transition']}' tem o maior custo estimado (€{top_1_cost['estimated_cost_of_wait']:,.0f}).", "Cartão 45", level='facto') # Clarificado que é custo da espera
        except Exception as e: print(f"Erro Regra [42, 45, 52]: {e}")

        try:
            # Usando df_bottleneck_res diretamente se existir
            df_bn_res = self.tables_pre_orig.get('bottleneck_by_resource_data', pd.DataFrame())
            if not df_bn_res.empty and 'resource_name' in df_bn_res.columns and 'waiting_time_days' in df_bn_res.columns:
                 df_bn_res_valid = df_bn_res.dropna(subset=['resource_name', 'waiting_time_days'])
                 if not df_bn_res_valid.empty:
                     top_1_res = df_bn_res_valid.nlargest(1, 'waiting_time_days').iloc[0]
                     self._add_insight(section, 'Facto: Maior Gerador de Espera (Recurso)', f"'{top_1_res['resource_name']}' é o recurso que gera mais tempo de espera a jusante ({top_1_res['waiting_time_days']:.1f} dias).", "Cartão 46", level='facto')
        except Exception as e: print(f"Erro Regra [46]: {e}")


    def _check_fluxo_conformidade_facts(self):
        section = 'diagnostico_fluxo_conformidade'

        # Cartão 61 & 65: Variabilidade
        try:
            if self.narrative_flags.get('is_chaotic'):
                prec = self.narrative_flags.get('precision', 0)
                # --- CORREÇÃO FORMAT % V10 ---
                top_1_p = self.narrative_flags.get('top_1_variant_perc', 0)  # já está em %
                top_10_p = self.narrative_flags.get('top_10_variant_perc', 0)  # já está em %
                self._add_insight(section, 'Alerta: Baixa Padronização (Processo Caótico)', f"O processo é pouco padronizado e caótico. Precisão do Modelo: {prec:.1%}. Variante Top 1: {top_1_p:.1f}%. Top 10 Variantes: {top_10_p:.1f}%.", "Cartão 61, Cartão 65", level='problema', priority=8)
            elif self.narrative_flags.get('is_standardized'):
                prec = self.narrative_flags.get('precision', 0)
                top_1_p = self.narrative_flags.get('top_1_variant_perc', 0)  # já está em %
                self._add_insight(section, 'Destaque: Alta Padronização', f"O processo é altamente padronizado: {top_1_p:.1f}% dos processos seguem exatamente a mesma sequência de atividades.", "Cartão 61, Cartão 65", level='facto')

            # Contextualizar a baixa precisão com o rework (sem criticar)
            if self.narrative_flags.get('is_chaotic') and self.narrative_flags.get('has_rework'):
                 self._add_insight(section, 'Observação: Causa da Baixa Padronização', "A Baixa Precisão do modelo e a alta variabilidade (muitas variantes) são consistentes com a existência de ciclos de rework, que criam múltiplos caminhos no processo.", "Cartão 61, Cartão 58, Cartão 65", level='facto') # Adicionado Cartão 65

        except Exception as e: print(f"Erro Regra [61, 65]: {e}")

        # Cartão 70 vs 58: Rework
        try:
            df_rework_table = self.tables_pre_orig.get('rework_loops_table', pd.DataFrame())
            rework_detected_in_matrix = self.narrative_flags.get('has_rework', False)

            if rework_detected_in_matrix:
                # Se a Matriz mostra rework, mas a tabela não (ou pouco)
                if df_rework_table.empty or df_rework_table['frequency'].sum() < (len(self.df_projects_base) * 0.05): # Menos de 5% dos casos
                     self._add_insight(section, 'Observação: Rework Melhor Detetado pela Matriz de Espera', "A Matriz de Espera (Cartão 58) revela fluxos de retorno (rework) que não são proeminentes na Tabela de Loops (Cartão 70), sugerindo que a Matriz oferece uma visão mais completa do retrabalho no processo.", "Cartão 58, Cartão 70", level='facto', priority=9)
                # Se ambos mostram rework, apenas confirma
                else:
                     self._add_insight(section, 'Facto: Rework Confirmado por Múltiplas Análises', "Tanto a Matriz de Espera (Cartão 58) como a Tabela de Loops (Cartão 70) indicam a presença de ciclos de retrabalho.", "Cartão 58, Cartão 70", level='facto')
            # Se a Matriz NÃO mostra rework
            elif self.narrative_flags.get('has_rework_pontual'):
                 pass  # já reportado como 'Rework Pontual' na secção de gargalos
            elif not df_rework_table.empty:
                 # Mas a Tabela mostra (improvável, mas possível dependendo das definições)
                 self._add_insight(section, 'Observação: Loops Específicos Detetados (Tabela)', "A Tabela de Loops (Cartão 70) deteta sequências A->...->A, embora a Matriz de Espera (Cartão 58) não mostre fluxos de retorno significativos entre fases.", "Cartão 70, Cartão 58", level='facto')
            else:
                 # Nenhum mostra rework
                 self._add_insight(section, 'Destaque: Ausência de Rework Significativo', "Nenhuma das análises (Matriz de Espera ou Tabela de Loops) detetou ciclos de retrabalho significativos.", "Cartão 58, Cartão 70", level='facto')
        except Exception as e: print(f"Erro Regra [70, 58]: {e}")

        # Cartão 71: Conformidade ao Longo do Tempo
        try:
            if not self.df_monthly_fitness.empty and len(self.df_monthly_fitness) > 1:
                 # Garante ordenação por data (assumindo que o índice ou uma coluna 'month' existe)
                 df_fit_sorted = self.df_monthly_fitness.copy()
                 if 'end_month' in df_fit_sorted.columns:
                      try:
                           df_fit_sorted['month_period'] = pd.to_datetime(df_fit_sorted['end_month'], format='%Y-%m').dt.to_period('M')
                           df_fit_sorted = df_fit_sorted.sort_values('month_period')
                      except ValueError: # Se formato for diferente, tenta ordenar como string
                           df_fit_sorted = df_fit_sorted.sort_values('end_month')

                 if 'fitness' in df_fit_sorted.columns:
                     slope = self._get_trend(df_fit_sorted['fitness'])
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
                        self._add_insight(section, 'Facto: Complexidade Não Explica Atraso', f"A 'complexidade' do processo não tem correlação significativa com atrasos (Corr: {corr:.2f}).", "Cartão 76", level='facto') # Removido "Bem Gerida"
                    elif not pd.isna(corr) and corr > 0.5:
                        self._add_insight(section, 'Observação: Complexidade Correlacionada com Atraso', f"Processos mais complexos tendem a atrasar mais (Corr: {corr:.2f}).", "Cartão 76", level='facto')
        except Exception as e: print(f"Erro Regra [76]: {e}")


# --- FUNÇÃO DE RENDERIZAÇÃO DA PÁGINA DE DIAGNÓSTICO ---
def render_diagnostics_page():
    st.subheader("💡 Diagnóstico Automático e Insights")
    st.markdown("Análise automática dos principais indicadores, destacando pontos fortes, problemas e ineficiências para orientar a sua investigação.")

    # --- Obtenção de Dados ---
    tables_pre = st.session_state.get('tables_pre_mining', {})
    metrics = st.session_state.get('metrics', {})
    data_frames = st.session_state.get('data_frames_processed', {})
    tables_eda = st.session_state.get('tables_eda', {})

    if not data_frames or 'projects' not in data_frames or 'tasks' not in data_frames:
        st.error("Dados essenciais da análise (projetos, tarefas) não encontrados. Execute a análise em 'Configurações'.")
        return

    # --- Execução do Motor ---
    try:
        engine = DiagnosticEngineV5(tables_pre, metrics, data_frames, tables_eda, caps=st.session_state.get('caps'))
        report = engine.run()
    except Exception as e:
        st.error(f"Erro ao executar o motor de diagnóstico: {e}")
        st.exception(e)
        return

    # --- Bloco 1: Saúde Geral (KPIs) (igual) ---
    st.markdown("---")
    st.markdown("<h4>1. Saúde Geral (KPIs de Baseline)</h4>", unsafe_allow_html=True)
    kpi_data = report.get('saude_geral', [])
    if kpi_data:
        for start in range(0, len(kpi_data), 4):
            row = st.columns(4)
            for col, kpi in zip(row, kpi_data[start:start+4]):
                col.metric(label=kpi['label'], value=kpi['value'])
    else: st.warning("Não foi possível calcular os KPIs de Saúde Geral.")

    # --- Bloco 2: Resumo Executivo e Recomendações (igual) ---
    st.markdown("---")
    st.markdown("<h4>2. Resumo Executivo e Recomendações</h4>", unsafe_allow_html=True)
    st.markdown("A análise combina os dados para identificar a *história principal* e a *causa-raiz* dos problemas ou destacar os pontos fortes.")

    summary_items = report.get('resumo_executivo', [])
    if not summary_items:
        st.warning("Não foi possível gerar um resumo executivo. Verifique os dados de entrada.")
    else:
        for item in summary_items:
            # Usa as cores e ícones definidos
            if item['level'] == 'problema':
                st.error(icon="🚨", body=f"**{item['titulo']}**\n\n{item['detalhe']}\n\n*Evidência principal: {item.get('cartao_ref', 'N/A')}*")
            elif item['level'] == 'recomendacao':
                st.info(icon="💡", body=f"**{item['titulo']}**\n\n{item['detalhe']}\n\n*Ação sugerida com base em: {item.get('cartao_ref', 'N/A')}*") # Mudado para st.info
            elif item['level'] == 'facto': # Caso de processo saudável ou destaque positivo
                st.success(icon="✅", body=f"**{item['titulo']}**\n\n{item['detalhe']}\n\n*Evidência principal: {item.get('cartao_ref', 'N/A')}*")

    # --- Bloco 3: Factos de Suporte e Observações Detalhadas ---
    st.markdown("---")
    st.markdown("<h4>3. Análise Detalhada por Área</h4>", unsafe_allow_html=True)
    st.markdown("Factos, alertas e observações detalhadas que suportam o diagnóstico principal.")

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
            st.markdown(f"<h5 style='margin-top: 25px; margin-bottom: 15px; border-bottom: 1px solid var(--border-color); padding-bottom: 5px;'><i class='bi {icon}'></i> {title}</h5>", unsafe_allow_html=True)
            
            cols = st.columns(2)
            for i, item in enumerate(items):
                with cols[i % 2]:
                    # Definir a cor da borda com base no nível
                    if item['level'] == 'problema': border_color = "#dc3545" # Vermelho (Alerta Grave)
                    elif item['level'] == 'facto': border_color = "#0d6efd" # Azul (Facto/Observação Neutra)
                    elif item['level'] == 'recomendacao': border_color = "#198754" # Verde (Recomendação/Destaque)
                    else: border_color = "#6c757d" # Cinza (Default)

                    st.markdown(f"""
                    <div style="border: 1px solid #dee2e6; border-left: 5px solid {border_color}; border-radius: 5px; padding: 12px 15px; margin-bottom: 12px; background-color: var(--card-background-color); box_shadow: 0 1px 3px rgba(0,0,0,0.04); height: 100%;">
                        <p style="font-weight: 600; margin-bottom: 5px; color: var(--text-color);">{item['titulo']}</p>
                        <p style="font-size: 0.95em; margin-bottom: 8px; color: var(--text-color);">{item['detalhe']}</p>
                        <p style="font-size: 0.8em; color: #0d6efd; margin-bottom: 0;"><i>Referência: {item.get('cartao_ref', 'N/A')}</i></p>
                    </div>
                    """, unsafe_allow_html=True)

    if not found_any_fact:
         st.info("Nenhum facto de suporte detalhado foi gerado.")


# --- PÁGINA DE LOGIN ---

def login_page():
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div style='display:flex;justify-content:center;margin-top:10px'><span class='pm-brand-badge' style='width:48px;height:48px;font-size:1.4rem'><i class='bi bi-diagram-3-fill'></i></span></div>"
                        "<div class='pm-login-title'>Process Intelligence</div><div class='pm-login-sub'>Transformação inteligente de processos</div>", unsafe_allow_html=True)
            username = st.text_input("Utilizador", placeholder="Vasco", value="Vasco")
            password = st.text_input("Senha", type="password", placeholder="1234", value="1234")
            entrar = st.button("Entrar", use_container_width=True, type="primary")
    if entrar:
        if username == "Vasco" and password == "1234":
            st.session_state.authenticated = True
            st.session_state.user_name = "Vasco"
            st.session_state.show_welcome_message = True
            st.rerun()
        else:
            with mid: st.error("Utilizador ou senha inválidos.")


# --- PÁGINA DE CONFIGURAÇÕES / UPLOAD ---

def _upload_multi_files():
    """Modo 'vários ficheiros': devolve o dict raw com as tabelas carregadas, ou None se faltar tasks."""
    st.info("Carregue os ficheiros que tiver. Só o ficheiro de **tarefas** é obrigatório; com os 4 ficheiros a análise fica completa. "
            "A app deteta automaticamente que análises os seus dados permitem e esconde as restantes.")
    # Dicionário com os textos para as tooltips
    tooltips = {
        'projects': "OPCIONAL. 1 linha por processo. Colunas: project_id; recomendadas: project_name, start_date, end_date, planned_end_date (ativa a análise de atrasos), budget_impact (ativa a análise de orçamento), project_status, path_name. Se não existir, início/fim de cada processo são derivados das tarefas.",
        'tasks': "OBRIGATÓRIO. 1 linha por tarefa. Colunas mínimas: project_id, task_name e uma data (start_date ou end_date). Com start_date E end_date ativam-se os tempos de execução e espera. Opcionais: task_id, task_type, priority, estimated_effort. Se não tiver resource_allocations, pode indicar aqui quem fez a tarefa com uma coluna resource_name (ou resource_id) e opcionalmente resource_type (função); várias pessoas separadas por ';'.",
        'resources': "OPCIONAL. 1 linha por recurso: resource_id, resource_name, resource_type (função), cost_per_hour (ativa custos), daily_capacity, skill_level (ativa competências).",
        'resource_allocations': "OPCIONAL. Horas efetivamente trabalhadas: task_id, resource_id, allocation_date, hours_worked. Ativa as análises de esforço, eficiência e (com cost_per_hour) custos.",
    }
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations']
    labels = {'projects': 'projects.csv', 'tasks': 'tasks.csv ★', 'resources': 'resources.csv',
              'resource_allocations': 'resource_allocations.csv'}

    upload_cols = st.columns(4)
    for i, name in enumerate(file_names):
        with upload_cols[i]:
            uploaded_file = st.file_uploader(f"Carregar `{labels[name]}`", type="csv", key=f"upload_{name}", help=tooltips[name])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                # ids como string e datas em datetime; o resto é tratado na normalização (normalize_inputs)
                for id_col in ['project_id', 'task_id', 'resource_id']:
                    if id_col in df.columns:
                        df[id_col] = df[id_col].astype(str)
                for date_col in ['start_date', 'end_date', 'planned_end_date', 'allocation_date']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                st.session_state.dfs[name] = df
                st.markdown(f'<p style="font-size: small; color: #198754;">✓ {len(df):,} linhas</p>', unsafe_allow_html=True)
            else:
                st.session_state.dfs[name] = None

    raw = {k: v for k, v in st.session_state.dfs.items() if v is not None and k != 'dependencies'}
    if 'tasks' not in raw:
        st.warning("Carregue pelo menos o ficheiro de tarefas (tasks.csv) para poder iniciar a análise.")
        return None
    return raw

def _upload_single_log():
    """Modo 'log único': upload de 1 ficheiro + mapeamento de colunas. Devolve raw ou None."""
    up = st.file_uploader("Carregar event log (CSV, Excel ou XES)", type=['csv', 'txt', 'xlsx', 'xls', 'xes'], key="upload_single_log",
                          help="Uma linha por tarefa (com início e fim) ou uma linha por evento. O separador do CSV é detetado automaticamente.")
    if not up:
        st.info("Carregue um ficheiro. Colunas mínimas: identificador do processo, atividade e uma data.")
        return None
    sig = (up.name, up.size)
    if st.session_state.get('_log_sig') != sig:
        try:
            st.session_state['_log_df'] = read_any_table(up)
        except Exception as e:
            st.error(f"Não foi possível ler o ficheiro: {e}")
            return None
        st.session_state['_log_sig'] = sig
        st.session_state['_log_map'] = guess_log_mapping(st.session_state['_log_df'].columns)
    df = st.session_state['_log_df']
    st.caption(f"✓ {len(df):,} linhas · {len(df.columns)} colunas")
    with st.expander("Pré-visualizar o ficheiro", expanded=False):
        st.dataframe(df.head(10))

    st.markdown("**Mapeamento de colunas** — confirme ou corrija o que a app sugeriu. Campos com * são obrigatórios; "
                "para as datas basta uma (início ou fim). Os restantes são opcionais e desbloqueiam análises adicionais.")
    options = ['— (não existe)'] + [str(c) for c in df.columns]
    guess = st.session_state.get('_log_map', {})
    mapping = {}
    cols = st.columns(3)
    for i, (key, label, required, _) in enumerate(LOG_FIELDS):
        default = guess.get(key)
        idx = options.index(str(default)) if default is not None and str(default) in options else 0
        choice = cols[i % 3].selectbox(label + (" *" if required else ""), options, index=idx, key=f"map_{key}")
        if choice != options[0]:
            mapping[key] = df.columns[[str(c) for c in df.columns].index(choice)]
    dup = [c for c in set(mapping.values()) if list(mapping.values()).count(c) > 1]
    if dup:
        st.warning(f"A mesma coluna está mapeada para mais de um campo: {', '.join(map(str, dup))}.")
    try:
        raw, warns = log_to_raw(df, mapping)
    except ValueError as e:
        st.warning(str(e))
        return None
    for w in warns:
        st.caption(f"ℹ️ {w}")
    return raw

def settings_page():
    ui_page_header("Dados e Configurações", "Carregue o histórico de processos e veja que análises os dados permitem")
    st.warning("Se carregou novos ficheiros CSV, clique primeiro neste botão para limpar a memória da aplicação antes de iniciar a nova análise.")
    if st.button("🔴 Limpar Cache e Recomeçar Análise"):
        st.cache_data.clear()
        st.success("Cache limpa com sucesso! A página será recarregada. Por favor, carregue os seus ficheiros novamente.")
        st.rerun()
    ############################################

    st.markdown("---") # Para separar visualmente
    st.markdown("---")
    st.subheader("Upload dos Dados")
    modo = st.radio("Como estão os seus dados?",
                    ["Vários ficheiros (até 4 tabelas)", "Um único ficheiro (event log)"],
                    horizontal=True, key="modo_upload",
                    help="Vários ficheiros: projects, tasks, resources, resource_allocations (só tasks é obrigatório). A sequência das tarefas é a observada nas datas. "
                         "Event log: um ficheiro CSV, Excel ou XES com uma linha por tarefa ou por evento.")
    if modo.startswith("Um único"):
        raw = _upload_single_log()
    else:
        raw = _upload_multi_files()
    if raw is None:
        return

    sla_days = st.number_input("SLA / prazo-alvo por processo (dias) — opcional, usado só se não houver planned_end_date",
                               min_value=0, max_value=3650, value=int(st.session_state.get('sla_days', 0) or 0), step=1,
                               help="Permite analisar atrasos quando os dados não têm data planeada. 0 = não usar.")
    st.session_state.sla_days = sla_days


    # --- Painel de capacidades: o que estes dados permitem analisar ---
    try:
        dfs_norm, caps, notes = normalize_inputs(raw, sla_days=sla_days or None)
    except Exception as e:
        st.error(f"Não foi possível interpretar os ficheiros: {e}")
        return
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Dados detetados: o que é possível analisar")
    cap_cols = st.columns(2)
    for idx, (cap, (titulo, requisito)) in enumerate(CAPS_INFO.items()):
        ok = cap in caps
        icon = "✅" if ok else "❌"
        txt = f"{icon} **{titulo}**" + ("" if ok else f" — falta: {requisito}")
        cap_cols[idx % 2].markdown(txt)
    for n in notes:
        st.caption(f"ℹ️ {n}")

    if st.toggle("Visualizar as primeiras 5 linhas dos ficheiros", value=False):
        for name, df in raw.items():
            st.markdown(f"**Ficheiro: `{name}.csv`**")
            st.dataframe(df.head())

    st.subheader("Execução da Análise")
    st.markdown('<div class="iniciar-analise-button">', unsafe_allow_html=True)
    if st.button("🚀 Iniciar Process Mining Analysis", use_container_width=True):
        with st.spinner("A executar a análise... Este processo pode demorar alguns minutos."):
            st.session_state.dfs_norm = dfs_norm
            st.session_state.caps = caps
            st.session_state.caps_notes = notes
            plots_pre, tables_pre, event_log, df_p, df_t, df_r, df_fc = run_pre_mining_analysis(dfs_norm)
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
            plots_eda, tables_eda = run_eda_analysis(dfs_norm)
            st.session_state.plots_eda = plots_eda
            st.session_state.tables_eda = tables_eda
            st.session_state.pop('rl_sample_ids', None)
            st.session_state.view = None
            st.session_state.diag_report = None; st.session_state.ai_reading = None; st.session_state.ai_chat = []
            st.session_state.rl_analysis_run = False

        st.session_state.analysis_run = True
        st.success("✅ Análise concluída! Navegue para o 'Process Mining' ou para a página de 'Reinforcement Learning'.")
        st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)

# --- FUNÇÃO AUXILIAR PARA GERAR O PDF ---
def generate_pdf_report(plots_pre, tables_pre, plots_post, plots_eda, tables_eda):
    print("\n--- Iniciando Geração do PDF V15 ---")
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Relatório de Análise de Processos", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)

        # --- ADICIONADO: Incluir KPIs na primeira secção ---
        sections_data = {
            "1. Visão Geral e Custos": {
                'tables': [ # KPIs primeiro
                    (tables_pre.get('kpi_data'), "KPIs de Baseline"),
                    (tables_pre.get('cost_of_delay_kpis'), "KPIs de Custo de Atraso"),
                    (tables_pre.get('outlier_cost'), "Top 5 Processos Mais Caros"),
                    (tables_pre.get('outlier_duration'), "Top 5 Processos Mais Longos"),
                ],
                'plots': [ # Gráficos depois
                    (plots_pre.get('performance_matrix'), "Matriz de Performance (Custo vs Prazo) (PM)"),
                    (plots_pre.get('cost_by_resource_type'), "Custo por Tipo de Recurso"),
                    (plots_post.get('kpi_time_series'), "Séries Temporais de KPIs de Performance"),
                    (plots_post.get('cost_per_day_time_series'), "Custo Médio por Dia ao Longo do Tempo"),
                    (plots_eda.get('plot_01'), "Distribuição do Status dos Processos"),
                    (plots_eda.get('plot_06'), "Custo Médio dos Processos por Trimestre"),
                    (plots_eda.get('plot_16'), "Distribuição do Custo por Dia (Eficiência)"),
                    (plots_eda.get('plot_04'), "Custo Real vs. Orçamento por Processo"),
                    (plots_eda.get('plot_17'), "Alocação de Custos por Orçamento e Recurso"),
                    (plots_eda.get('plot_31'), "Evolução do Volume e Tamanho dos Processos"),
                ]
            },
            "2. Performance e Prazos": {
                'plots': [
                     (plots_pre.get('lead_time_vs_throughput'), "Relação Lead Time vs Throughput"),
                     (plots_pre.get('lead_time_hist'), "Distribuição do Lead Time"),
                     (plots_pre.get('case_durations_boxplot'), "Distribuição da Duração dos Processos (PM)"),
                     (plots_post.get('cumulative_throughput_plot'), "Gráfico Acumulado de Throughput"),
                     (plots_eda.get('plot_05'), "Performance de Prazos por Trimestre"),
                     (plots_pre.get('cycle_time_breakdown'), "Duração Média por Fase do Processo"),
                     (plots_pre.get('throughput_hist'), "Distribuição do Throughput (horas)"),
                     (plots_pre.get('throughput_boxplot'), "Boxplot do Throughput (horas)"),
                     (plots_post.get('temporal_heatmap_fixed'), "Atividades por Dia da Semana"),
                     (plots_eda.get('plot_30'), "Evolução da Performance (Prazo e Custo)"),
                     (plots_eda.get('plot_03'), "Diferença entre Data Real e Planeada"),
                     (plots_post.get('gantt_chart_all_projects'), "Linha do Tempo (Gantt Chart) - Amostra"),
                ], 'tables': [(tables_pre.get('perf_stats'), "Estatísticas de Performance"),]
            },
            "3. Recursos e Equipa": {
                'plots': [
                    (plots_eda.get('plot_12'), "Distribuição de Recursos por Tipo"),
                    (plots_pre.get('resource_avg_events'), "Recursos por Média de Tarefas/Processo"),
                    (plots_pre.get('weekly_efficiency'), "Eficiência Semanal (Horas Trabalhadas)"),
                    (plots_pre.get('delay_by_teamsize'), "Impacto do Tamanho da Equipa no Atraso (PM)"),
                    (plots_pre.get('throughput_benchmark_by_teamsize'), "Benchmark de Throughput por Equipa"),
                    (plots_pre.get('resource_workload'), "Top 10 Recursos por Horas Trabalhadas (PM)"),
                    (plots_pre.get('resource_handoffs'), "Top 10 Handoffs entre Recursos"),
                    (plots_post.get('resource_efficiency_plot'), "Métricas de Eficiência: Top Recursos"),
                    (plots_pre.get('median_duration_by_teamsize'), "Duração Mediana por Tamanho da Equipa"),
                    (plots_eda.get('plot_07'), "Nº Médio de Recursos por Processo a Cada Trimestre"),
                    (plots_eda.get('plot_14'), "Atraso Médio por Recurso"),
                    (plots_post.get('skill_vs_performance_adv'), "Relação entre Skill e Performance"),
                    (plots_eda.get('plot_23'), "Atraso por Nível de Competência"),
                    (plots_post.get('resource_network_bipartite'), "Rede de Recursos por Função"),
                    (plots_post.get('resource_network_adv'), "Rede Social de Recursos (Handovers)"),
                    (plots_pre.get('resource_activity_matrix'), "Heatmap de Esforço (Recurso vs Atividade)"),
                ], 'tables': []
            },
            "4. Gargalos e Espera": {
                 'plots': [
                    (plots_post.get('performance_heatmap'), "Heatmap de Performance no Processo (Gargalos)"),
                    (plots_pre.get('top_activities_plot'), "Atividades Mais Frequentes"),
                    (plots_pre.get('service_vs_wait_stacked'), "Gargalos: Tempo de Serviço vs. Espera"),
                    (plots_pre.get('top_handoffs_cost'), "Top 10 Handoffs por Custo de Espera"),
                    (plots_pre.get('bottleneck_by_resource'), "Top Recursos por Tempo de Espera Gerado"),
                    (plots_eda.get('plot_18'), "Custo Real vs. Atraso"),
                    (plots_eda.get('plot_20'), "Nº de Recursos vs. Custo Total"),
                    (plots_pre.get('activity_service_times'), "Tempo Médio de Execução por Atividade"),
                    (plots_pre.get('wait_vs_service_scatter'), "Espera vs. Execução (Dispersão)"),
                    (plots_pre.get('wait_time_evolution'), "Evolução do Tempo Médio de Espera"),
                    (plots_pre.get('top_handoffs'), "Top 10 Handoffs por Tempo de Espera"),
                    (plots_eda.get('plot_19'), "Rate Horário Médio vs. Atraso"),
                    (plots_eda.get('plot_22'), "Atraso por Faixa de Orçamento"),
                    (plots_post.get('milestone_time_analysis_plot'), "Análise de Tempo entre Marcos do Processo"),
                    (plots_eda.get('plot_29'), "Matriz de Correlação"),
                    (plots_post.get('avg_waiting_time_by_activity_plot'), "Tempo Médio de Espera por Atividade"),
                    (plots_post.get('waiting_time_matrix_plot'), "Matriz de Tempo de Espera entre Atividades (horas)"),
                ], 'tables': []
            },
             "5. Fluxo e Conformidade": {
                'plots': [
                    (plots_post.get('model_inductive_petrinet'), "Modelo - Inductive Miner"),
                    (plots_post.get('model_heuristic_petrinet'), "Modelo - Heuristics Miner"),
                    (plots_post.get('metrics_inductive'), "Métricas (Inductive Miner)"),
                    (plots_post.get('metrics_heuristic'), "Métricas (Heuristics Miner)"),
                    (plots_post.get('custom_variants_sequence_plot'), "Sequência de Atividades das 10 Variantes Mais Comuns"),
                    (plots_post.get('variant_duration_plot'), "Duração Média das Variantes Mais Comuns"),
                    (plots_pre.get('variants_frequency'), "Top 10 Variantes de Processo por Frequência"),
                    (plots_eda.get('plot_08'), "Distribuição de Tarefas por Tipo"),
                    (plots_eda.get('plot_10'), "Distribuição da Duração das Tarefas"),
                    (plots_eda.get('plot_25'), "Centralidade dos Tipos de Tarefa"),
                    (plots_post.get('conformance_over_time_plot'), "Score de Conformidade ao Longo do Tempo"),
                    (plots_eda.get('plot_09'), "Distribuição de Tarefas por Prioridade"),
                    (plots_eda.get('plot_11'), "Top 10 Tarefas Específicas Mais Demoradas"),
                    (plots_eda.get('plot_24'), "Distribuição da Complexidade dos Processos"),
                    (plots_eda.get('plot_27'), "Relação entre Complexidade e Atraso"),
                ],
                'tables': [
                    (tables_pre.get('variants_table'), "Frequência das 10 Principais Variantes"),
                    (tables_pre.get('rework_loops_table'), "Principais Loops de Rework (Tabela)"),
                ]
            },
        }

        page_width = pdf.w - 2 * pdf.l_margin 
        col_width = page_width / 2 - 5 
        title_height = 10 
        table_line_height = 5 
        img_est_height = col_width / 1.6 

        is_first_section = True
        for section_title, data in sections_data.items():
            print(f"Adding section: {section_title}")
            # Adiciona linha e página antes das secções (exceto a primeira)
            if not is_first_section:
                pdf.add_page()
            else:
                 is_first_section = False

            pdf.set_draw_color(200, 200, 200) 
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y()) 
            pdf.ln(2) 
            pdf.set_font("Helvetica", "B", 14); pdf.cell(0, title_height, section_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(5)
            pdf.set_font("Helvetica", "", 10)

            plot_pair_buffer = [] # Buffer para guardar plots para layout lado a lado

            for table_df, table_title in data['tables']:
                print(f"  Attempting to add table: {table_title}")
                start_y = pdf.get_y() # Posição Y antes de adicionar
                border_color = (220, 220, 220) # Cinza claro para a borda

                # --- Lógica para desenhar uma caixa ---
                pdf.set_draw_color(border_color[0], border_color[1], border_color[2])
                pdf.rect(pdf.l_margin - 2 , start_y - 2, page_width + 4, 1) # Placeholder height

                pdf.set_font("Helvetica", "B", 11); pdf.cell(0, title_height, table_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", "", 8)

                if table_df is not None and isinstance(table_df, pd.DataFrame) and not table_df.empty:
                    try:
                        table_string = table_df.astype(str).to_string(index=False, justify='left', line_width=100) # Ajuste line_width
                        pdf.write(table_line_height, table_string); pdf.ln(2)
                        print(f"    Success adding table: {table_title}")
                    except Exception as table_err:
                        print(f"    ERROR adding table '{table_title}': {table_err}")
                        pdf.set_text_color(255, 0, 0); pdf.write(table_line_height, f"[Erro ao adicionar tabela '{table_title}']"); pdf.set_text_color(0, 0, 0); pdf.ln(2)
                elif table_df is not None and isinstance(table_df, dict):
                     try:
                         for k, v in table_df.items(): pdf.write(table_line_height, f"{k}: {v}\n")
                         pdf.ln(2)
                         print(f"    Success adding dict table: {table_title}")
                     except Exception as dict_err:
                        print(f"    ERROR adding dict table '{table_title}': {dict_err}")
                        pdf.set_text_color(255, 0, 0); pdf.write(table_line_height, f"[Erro ao adicionar tabela dict '{table_title}']"); pdf.set_text_color(0, 0, 0); pdf.ln(2)
                else:
                     pdf.write(table_line_height, "[Dados não disponíveis]")
                     pdf.ln(2)
                     print(f"    Skipping table '{table_title}': None, not DataFrame/Dict, or empty.")

                end_y = pdf.get_y()
                box_height = end_y - start_y + 4 # Adiciona padding
                pdf.rect(pdf.l_margin - 2 , start_y - 2, page_width + 4, box_height)
                pdf.set_y(end_y + 3) # Define posição Y após a caixa + margem


            # --- Processa Gráficos (tentativa de layout lado a lado) ---
            valid_plots = [(p, t) for p, t in data['plots'] if p and isinstance(p, BytesIO)]

            for i, (plot_bytes, plot_title) in enumerate(valid_plots):
                print(f"  Attempting to add plot: {plot_title}")
                plot_pair_buffer.append((plot_bytes, plot_title))

                # Se existe um par ou é o último gráfico da lista
                if len(plot_pair_buffer) == 2 or i == len(valid_plots) - 1:
                    start_y_plots = pdf.get_y()
                    current_x = pdf.l_margin
                    max_h_in_pair = 0

                    # --- Verifica page break ANTES de adicionar o par/single ---
                    estimated_pair_height = title_height + img_est_height + 10 # Title + img + padding
                    if pdf.will_page_break(estimated_pair_height):
                        print(f"    Page break predicted before plots: {[p[1] for p in plot_pair_buffer]}. Adding new page.")
                        pdf.add_page()
                        start_y_plots = pdf.get_y() # Atualiza Y inicial
                    # ---------------------------------------------------------

                    # Desenha a caixa (placeholder)
                    box_start_y = start_y_plots - 2
                    pdf.set_draw_color(220, 220, 220)
                    pdf.rect(pdf.l_margin - 2 , box_start_y, page_width + 4, 1) # Placeholder height


                    for pb, pt in plot_pair_buffer:
                        pdf.set_xy(current_x, start_y_plots) # Define posição X,Y
                        pdf.set_font("Helvetica", "B", 11); pdf.cell(col_width, title_height, pt, new_x=XPos.LEFT, new_y=YPos.NEXT) # Título dentro da coluna
                        pdf.set_font("Helvetica", "", 10)
                        temp_img_path = None
                        try:
                            pb.seek(0)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                                temp_img.write(pb.getvalue())
                                temp_img_path = temp_img.name
    
                            pdf.image(temp_img_path, x=current_x, w=col_width)
                            plot_end_y = pdf.get_y() # Captura Y após a imagem (pode não ser útil se a imagem for alta)
                            max_h_in_pair = max(max_h_in_pair, plot_end_y - start_y_plots) # Tenta rastrear altura máxima na linha
                            print(f"    Success adding plot: {pt}")
                        except Exception as img_err:
                            print(f"    ERROR adding plot '{pt}': {img_err}")
                            pdf.set_xy(current_x, pdf.get_y() + 5) # Move para baixo para erro
                            pdf.set_text_color(255, 0, 0); pdf.write(5, f"[Erro gráfico '{pt}']"); pdf.set_text_color(0, 0, 0)
                            max_h_in_pair = max(max_h_in_pair, 20) # Assume altura mínima para erro
                        finally:
                            if temp_img_path and os.path.exists(temp_img_path):
                                try: os.remove(temp_img_path); print(f"    Removed temp file: {temp_img_path}")
                                except Exception as rm_err: print(f"    ERROR removing temp file '{temp_img_path}': {rm_err}")

                        current_x += col_width + 10 # Move para a próxima coluna (largura + margem)

                    # Move para a linha abaixo após adicionar o par ou o último single
                    final_y_plots = start_y_plots + max_h_in_pair + 5 # Usa a altura máxima estimada + margem
                    if final_y_plots < pdf.get_y(): # Se o write do erro foi mais baixo
                        final_y_plots = pdf.get_y() + 5

                    # Desenha a caixa completa
                    box_height_plots = final_y_plots - start_y_plots + 4 # Adiciona padding
                    pdf.rect(pdf.l_margin - 2 , box_start_y, page_width + 4, box_height_plots)

                    pdf.set_y(final_y_plots + 3) # Define Y após a caixa + margem

                    plot_pair_buffer = [] # Limpa o buffer

            # Limpa buffer no final da secção (caso haja um número ímpar de gráficos)
            if plot_pair_buffer:
                 # Processa o último gráfico sozinho (código similar ao loop acima, mas só para um)
                 pb, pt = plot_pair_buffer[0]
                 start_y_plots = pdf.get_y()
                 estimated_single_height = title_height + img_est_height + 10
                 if pdf.will_page_break(estimated_single_height):
                     print(f"    Page break predicted before last single plot: {pt}. Adding new page.")
                     pdf.add_page()
                     start_y_plots = pdf.get_y()

                 box_start_y = start_y_plots - 2
                 pdf.set_draw_color(220, 220, 220)
                 pdf.rect(pdf.l_margin - 2 , box_start_y, page_width + 4, 1)

                 pdf.set_xy(pdf.l_margin, start_y_plots)
                 pdf.set_font("Helvetica", "B", 11); pdf.cell(page_width, title_height, pt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                 pdf.set_font("Helvetica", "", 10)
                 temp_img_path = None
                 plot_end_y = start_y_plots + title_height # Y inicial para cálculo da altura
                 try:
                     pb.seek(0)
                     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                         temp_img.write(pb.getvalue())
                         temp_img_path = temp_img.name
                     pdf.image(temp_img_path, x=pdf.l_margin, w=col_width) # Usa largura de coluna mesmo para single
                     plot_end_y = pdf.get_y() # Atualiza Y após imagem
                     print(f"    Success adding last single plot: {pt}")
                 except Exception as img_err:
                     print(f"    ERROR adding last single plot '{pt}': {img_err}")
                     pdf.set_text_color(255, 0, 0); pdf.write(5, f"[Erro gráfico '{pt}']"); pdf.set_text_color(0, 0, 0)
                     plot_end_y = pdf.get_y() + 5 # Atualiza Y após erro
                 finally:
                     if temp_img_path and os.path.exists(temp_img_path):
                         try: os.remove(temp_img_path); print(f"    Removed temp file: {temp_img_path}")
                         except Exception as rm_err: print(f"    ERROR removing temp file '{temp_img_path}': {rm_err}")

                 box_height_plots = plot_end_y - start_y_plots + 4
                 pdf.rect(pdf.l_margin - 2 , box_start_y, page_width + 4, box_height_plots)
                 pdf.set_y(plot_end_y + 3) # Move para baixo
                 plot_pair_buffer = []


        print(f"--- Finalizing PDF Generation Attempt ---")
        pdf_output_bytes = b""
        try:
            pdf_output = pdf.output(dest='S')
            print(f"    pdf.output(dest='S') returned type: {type(pdf_output)}")

            if isinstance(pdf_output, str):
                print("    Output is string, attempting to encode to latin-1...")
                pdf_output_bytes = pdf_output.encode('latin-1', errors='replace')
                print(f"    Successfully encoded string PDF. Size: {len(pdf_output_bytes)} bytes.")
            elif isinstance(pdf_output, (bytes, bytearray)):
                print("    Output is already bytes/bytearray. Using directly.")
                pdf_output_bytes = bytes(pdf_output)
                print(f"    PDF bytes size: {len(pdf_output_bytes)} bytes.")
            else:
                print(f"    ERROR: pdf.output(dest='S') returned unexpected type: {type(pdf_output)}")
                st.error(f"Tipo inesperado retornado pela geração do PDF: {type(pdf_output)}")
                return None

        except Exception as pdf_err:
            print(f"    ERROR during final pdf.output() call: {pdf_err}")
            st.error(f"Erro crítico ao finalizar o PDF: {pdf_err}")
            return None

        if not pdf_output_bytes or len(pdf_output_bytes) < 1000:
             print(f"    WARNING: PDF output size ({len(pdf_output_bytes)} bytes) too small. Returning None.")
             st.warning("O PDF gerado parece estar vazio ou incompleto.")
             return None

        print("--- PDF Generation Function Completed Successfully ---")
        return pdf_output_bytes

    except Exception as general_err:
         print(f"--- UNEXPECTED ERROR during PDF generation: {general_err} ---")
         st.error(f"Erro inesperado ao gerar PDF: {general_err}")
         import traceback
         traceback.print_exc()
         return None


# --- PÁGINA DO DASHBOARD ---

# =====================================================================================
# --- CAMADA VISUAL V2: gráficos interativos (Plotly), cartões e secções por área ---
# Os cálculos partem das tabelas já normalizadas/processadas (não das imagens matplotlib).
# Os gráficos matplotlib continuam a ser gerados para o relatório PDF.
# =====================================================================================
import plotly.graph_objects as go
import plotly.express as px

UI = dict(ink='#0f172a', text='#334155', muted='#64748b', line='#e5e7eb', grid='#f1f5f9', bg='#f7f8fa',
          blue='#2563eb', green='#16a34a', red='#dc2626', amber='#d97706', purple='#7c3aed', teal='#0d9488')
UI_SEQ = ['#2563eb', '#0d9488', '#d97706', '#7c3aed', '#db2777', '#16a34a', '#64748b', '#ea580c', '#4f46e5', '#0891b2']


def _style(fig, height=340, legend=True):
    fig.update_layout(template='plotly_white', height=height, margin=dict(l=8, r=8, t=12, b=8),
                      font=dict(family='Inter, Poppins, sans-serif', size=12, color=UI['text']),
                      colorway=UI_SEQ, showlegend=legend, plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, title_text=''),
                      hoverlabel=dict(bgcolor='white', bordercolor=UI['line'], font=dict(color=UI['ink'])))
    fig.update_xaxes(showgrid=False, linecolor=UI['line'], ticks='', title_font=dict(color=UI['muted']))
    fig.update_yaxes(gridcolor=UI['grid'], zeroline=False, linecolor=UI['line'], title_font=dict(color=UI['muted']))
    return fig


def ui_page_header(title, subtitle):
    st.markdown(f"<div class='pm-page-title'>{title}</div><div class='pm-page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def ui_section_intro(question, text):
    st.markdown(f"<div class='pm-section-q'>{question}</div><div class='pm-section-t'>{text}</div>", unsafe_allow_html=True)


def ui_kpis(items):
    """items: lista de dicts {icon, value, label, delta (texto), good (True/False/None), help}"""
    if not items:
        return
    cards = []
    for it in items:
        delta = it.get('delta')
        if delta:
            good = it.get('good')
            cls = 'up' if good is True else ('down' if good is False else 'neu')
            arrow = 'bi-arrow-up-right' if good is True else ('bi-arrow-down-right' if good is False else 'bi-dash')
            delta_html = f"<span class='pm-kpi-delta {cls}'><i class='bi {arrow}'></i> {delta}</span>"
        else:
            delta_html = ""
        title_attr = f" title='{html.escape(it.get('help', ''), quote=True)}'" if it.get('help') else ""
        cards.append(f"<div class='pm-kpi'{title_attr}><div class='pm-kpi-top'><span class='pm-kpi-icon'><i class='bi {it.get('icon', 'bi-graph-up')}'></i></span>{delta_html}</div>"
                     f"<div class='pm-kpi-value{' small' if len(str(it['value'])) > 12 else ''}'>{it['value']}</div><div class='pm-kpi-label'>{it['label']}</div></div>")
    st.markdown(f"<div class='pm-kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


CHART_TIPS = {
 # Visão geral
 'Volume e duração ao longo do tempo': "Barras: processos concluídos por mês. Linhas: duração mediana (metade dos processos demora menos) e P90 (só 10% demoram mais). Se as linhas sobem, o processo está a ficar mais lento. Meses incompletos nas pontas não entram nas linhas.",
 'Entradas, conclusões e processos em curso': "Compara quantos processos entram e quantos terminam por mês, e quantos estão em curso. Se entram mais do que saem, acumula-se trabalho pendente e as esperas vão aumentar.",
 'Distribuição da duração dos processos': "Quantos processos demoraram cada número de dias. Uma cauda longa à direita indica casos problemáticos que merecem análise individual.",
 'Duração por tipo de processo': "Compara a duração entre tipos de processo (caixa = metade central dos casos; linha = mediana). Ajuda a perceber se algum tipo é sistematicamente mais lento.",
 'Estado dos processos': "Quantos processos existem em cada estado (ex.: concluído, em curso, cancelado).",
 'Processos mais longos': "Os 10 processos que mais demoraram. Útil para investigar caso a caso o que correu mal.",
 # Fluxo
 'Mapa do processo': "Mapa real do processo, descoberto a partir dos dados. Caixas = atividades (nº de vezes executadas). Setas = passagens entre atividades; a espessura mostra a frequência. Em 'Espera média', a cor vai de azul (pouca espera) a vermelho (muita). Use o filtro para esconder passagens raras.",
 'Variantes do processo': "Uma variante é uma sequência completa de atividades. A #1 é o caminho principal. Os passos a vermelho são os que diferem do caminho principal (atividades a mais, a menos ou fora de ordem).",
 'Ciclos de retrabalho': "Retornos a uma atividade que já tinha sido feita (ex.: voltar à Análise de Risco depois da Decisão). Cada retorno custa tempo e esforço.",
 'Conformidade ao longo do tempo': "Percentagem de processos concluídos em cada mês que seguiram o caminho principal, e percentagem com retrabalho. Uma descida indica que o processo está a ficar menos padronizado.",
 'Frequência das atividades': "Quantas vezes cada atividade foi executada e em que percentagem dos processos aparece. Mais de 100% significa que a atividade se repete em alguns processos.",
 # Tempos
 'Execução vs. espera por atividade': "Para cada atividade: tempo médio a trabalhar (azul) e tempo médio parado à espera antes de começar (laranja). Muita espera indica fila, falta de capacidade ou dependência de terceiros.",
 'Passagens onde os processos ficam mais tempo parados': "As passagens entre atividades com mais dias de espera acumulados (espera média × nº de processos). São os maiores pontos de perda de tempo do processo. Retrabalho excluído.",
 'Matriz de espera entre atividades': "Espera média (dias) entre o fim de uma atividade (linha) e o início da seguinte (coluna). Células vermelhas são as passagens mais lentas. Inclui retornos.",
 'Execução e espera ao longo do tempo': "Dias de trabalho e dias de espera por processo, por mês de conclusão. Mostra se a perda de tempo está a crescer ou a diminuir.",
 'Atividade por dia da semana': "Tarefas iniciadas e concluídas por dia da semana. Ajuda a ver picos de carga e o efeito dos fins de semana.",
 'Variabilidade da duração de cada atividade': "Dispersão da duração de cada atividade: caixa = metade central das execuções, pontos = casos extremos. Caixas largas indicam trabalho pouco normalizado e imprevisível.",
 'Linha do tempo de um processo': "As tarefas de um processo ao longo do tempo, com quem as executou. Os intervalos vazios entre barras são esperas.",
 # Recursos
 'Carga de trabalho por pessoa': "Horas trabalhadas (ou nº de tarefas, se não houver horas) por pessoa, por função. Mostra concentração de trabalho em poucas pessoas.",
 'Utilização da capacidade por função': "Horas trabalhadas ÷ horas disponíveis das pessoas ativas em cada mês (média móvel de 3 meses). Acima de ~85% as filas crescem rapidamente; perto de 100% a função está no limite.",
 'Passagens de trabalho entre funções': "Quantas vezes o trabalho passa de uma função (linha) para outra (coluna). Cada passagem é um ponto de coordenação onde costuma haver espera (ver no hover).",
 'Rede de colaboração entre pessoas': "Cada círculo é uma pessoa (cor = função, tamanho = nº de tarefas). Cada linha liga duas pessoas que passam trabalho uma à outra; quanto mais grossa, mais passagens. Pessoas no centro, com muitas ligações, são pontos críticos: se faltarem, o fluxo é afetado.",
 'Eficiência comparada dentro de cada função': "Cada ponto é uma pessoa: horas que gasta por tarefa face à mediana da sua função. À direita = mais horas que os colegas. Diferenças podem refletir tarefas mais difíceis, não só eficiência.",
 'Espera gerada a seguir a cada pessoa': "Tempo médio até a tarefa seguinte do processo começar, depois de cada pessoa terminar a sua. Valores altos podem indicar entregas incompletas ou passagens mal coordenadas.",
 'Especialização: horas por pessoa e atividade': "Horas de cada pessoa em cada atividade. Mostra quem é especialista em quê e onde há pouca polivalência.",
 'Duração vs. nº de pessoas envolvidas': "Duração dos processos consoante o número de pessoas que participaram. Mais pessoas costuma significar mais passagens e mais espera.",
 'Pessoas por processo ao longo do tempo': "Número médio de pessoas envolvidas por processo, por trimestre de início.",
 'Competência vs. horas por tarefa': "Horas por tarefa consoante o nível de competência de quem a executou. Se não houver diferença, o nível de competência registado pode não refletir o desempenho real.",
 'Atraso vs. competência da equipa': "Desvio face ao prazo consoante a competência média da equipa do processo.",
 'Desvio de prazo dos processos de cada pessoa': "Desvio médio face ao prazo dos processos em que cada pessoa participou. É uma correlação: não prova que a pessoa causou o atraso.",
 # Custos
 'Matriz de desempenho: prazo vs. orçamento': "Cada ponto é um processo. Direita = terminou depois do prazo; acima = custou mais do que o orçamento. O canto superior direito concentra os casos mais problemáticos.",
 'Pontualidade ao longo do tempo': "Barras: percentagem de processos atrasados por mês. Linha: desvio médio face ao prazo (negativo = termina antes do prazo).",
 'Distribuição do desvio face ao prazo': "Quantos processos terminaram cada número de dias antes (à esquerda da linha) ou depois (à direita) do prazo.",
 'Custo por função': "Custo total de mão-de-obra (horas × custo/hora) por função e percentagem do total.",
 'Custo por atividade': "Custo total de mão-de-obra por atividade e percentagem do total. Mostra onde se gasta o dinheiro.",
 'Custo médio por processo ao longo do tempo': "Custo médio real por processo e orçamento médio, por mês de conclusão.",
 'Custo real vs. orçamento': "Cada ponto é um processo. Acima da diagonal = custou mais do que o orçamentado.",
 'Processos mais caros': "Os 10 processos com maior custo real, para análise individual.",
 'Matriz de correlação': "Mede a relação linear entre pares de indicadores, de −1 a 1. Perto de 1: sobem juntos; perto de −1: um sobe quando o outro desce; perto de 0: sem relação. Correlação não é causa.",
 'Custo real vs. desvio de prazo': "Relação entre o custo de cada processo e o seu desvio face ao prazo.",
 'Custo vs. nº de pessoas envolvidas': "Custo dos processos consoante o número de pessoas envolvidas.",
 'Desvio de prazo por faixa de orçamento': "Desvio face ao prazo para processos agrupados em quatro faixas de orçamento (Q1 = 25% mais baratos).",
 'Custo por faixa de orçamento e função': "Custo médio por processo em cada faixa de orçamento, repartido por função.",
 # Simulação
 'Real vs. simulado': "O simulador repete os mesmos processos, nas mesmas datas, 5 vezes. Se os números simulados ficam perto dos reais, o modelo é fiável para testar cenários.",
 'Duração ao longo do tempo: real vs. simulado': "Duração média por mês, real (preto) e simulada (azul, com intervalo entre réplicas). Se as linhas acompanham os mesmos altos e baixos, o modelo capta o efeito do volume e da capacidade.",
 'Espera antes de cada atividade': "Espera média antes de cada atividade, real vs. simulada (ou base vs. cenário). Serve para confirmar que o modelo reproduz cada etapa, não só o total.",
 'Duração de cada atividade': "Tempo médio de execução de cada atividade, real vs. simulado.",
 'De onde vem a espera?': "Divide a espera de cada passagem em duas partes. Vermelho: fila por falta de pessoas (desce com mais capacidade ou melhor alocação). Laranja: espera estrutural, como agendamentos, aprovações e terceiros (não desce com mais pessoas; exige mudar regras ou prazos).",
 'Ocupação da capacidade por função': "Percentagem do tempo em que a capacidade de cada função está ocupada na simulação. Acima de 85% qualquer aumento de volume cria filas rapidamente.",
 'Distribuição da duração': "Quantos processos demoram cada número de dias, na base e no cenário. Se a nuvem azul se desloca para a esquerda, o cenário torna os processos mais rápidos.",
 'Ocupação por função': "Ocupação da capacidade de cada função, na base e no cenário. Barras vermelhas: acima de 85%, risco de filas.",
 'Comparação de cenários': "Todos os cenários testados nesta sessão lado a lado com a base simulada (situação atual).",
 'Prática atual vs. regra simples vs. agente': "Compara três formas de decidir o que atender primeiro, nos mesmos processos: por ordem de chegada (prática atual), pelo prazo mais próximo (regra simples) e a política aprendida pelo agente. Menor = melhor.",
 'Efeito por tipo de processo': "Duração média por tipo com cada política. Serve para garantir que melhorar a média não prejudica um tipo de processo em particular.",
 'Evolução do treino': "Em cada iteração o agente testa várias políticas no simulador e fica com as melhores. A linha azul deve descer e ficar abaixo da linha vermelha (prática atual).",
 'Regras de decisão recomendadas': "O que o agente faz de diferente da prática atual, traduzido em regras que uma equipa pode aplicar.",
 'Definir cenário': "Escolha o que quer alterar. Tudo o que não alterar fica igual à situação atual.",
 'Contexto de treino': "Define em que situação o agente aprende. A forma de priorizar só faz diferença quando há filas; por isso o treino usa, por defeito, mais volume do que o atual.",
 'Resumo para o diretor': "Conclusão automática dos testes feitos nesta página, em linguagem de negócio, pronta a copiar para um email ou apresentação.",
}


def ui_card_header(title, subtitle=None, icon=None, help=None):
    ic = f"<span class='pm-card-icon'><i class='bi {icon}'></i></span>" if icon else ""
    sub = f"<div class='pm-card-sub'>{subtitle}</div>" if subtitle else ""
    tip = help or CHART_TIPS.get(title)
    tip_html = f"<span class='pm-tip' data-tip='{html.escape(tip, quote=True)}'>?</span>" if tip else ""
    st.markdown(f"<div class='pm-card-head'>{ic}<div style='flex:1'><div class='pm-card-title'>{title}{tip_html}</div>{sub}</div></div>", unsafe_allow_html=True)


def ui_chart(fig, title, subtitle=None, icon=None, key=None):
    if fig is None:
        return
    with st.container(border=True):
        ui_card_header(title, subtitle, icon)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=key)


def ui_note(text):
    st.markdown(f"<div class='pm-note'><i class='bi bi-info-circle'></i> {text}</div>", unsafe_allow_html=True)


def _fmt_days(x):
    return "—" if x is None or pd.isna(x) else f"{x:,.1f} dias".replace(',', ' ')


def _fmt_eur(x):
    return "—" if x is None or pd.isna(x) else f"€{x:,.0f}".replace(',', ' ')


def _pct_change(new, old):
    if old is None or new is None or pd.isna(old) or pd.isna(new) or old == 0:
        return None
    return (new - old) / abs(old) * 100


# ---------------------------------------------------------------------------------
# Preparação da "vista": tabelas derivadas, calculadas uma vez e depois filtradas
# ---------------------------------------------------------------------------------
def prepare_view(dp, dt, dr, alloc):
    P = dp.copy()
    for c in ['start_date', 'end_date', 'planned_end_date']:
        if c in P.columns: P[c] = pd.to_datetime(P[c], errors='coerce')
    P['project_id'] = P['project_id'].astype(str)
    P['lead_time'] = (P['end_date'] - P['start_date']).dt.days
    P['end_month'] = P['end_date'].dt.to_period('M').dt.to_timestamp()
    P['start_month'] = P['start_date'].dt.to_period('M').dt.to_timestamp()

    T = dt.copy()
    T['project_id'] = T['project_id'].astype(str); T['task_id'] = T['task_id'].astype(str)
    for c in ['start_date', 'end_date']: T[c] = pd.to_datetime(T[c], errors='coerce')
    T['_o'] = pd.to_numeric(T['task_id'], errors='coerce')
    T = T.sort_values(['project_id', 'start_date', 'end_date', '_o']).drop(columns='_o')
    T['dur_days'] = (T['end_date'] - T['start_date']).dt.total_seconds() / 86400
    T['prev_task'] = T.groupby('project_id')['task_name'].shift(1)
    T['prev_end'] = T.groupby('project_id')['end_date'].shift(1)
    T['wait_days'] = ((T['start_date'] - T['prev_end']).dt.total_seconds() / 86400).clip(lower=0)
    T['seq'] = T.groupby('project_id').cumcount()
    T['repeat'] = T.groupby(['project_id', 'task_name']).cumcount() > 0

    var = T.groupby('project_id')['task_name'].agg(tuple)
    vc = var.value_counts()
    rank = {v: i + 1 for i, v in enumerate(vc.index)}
    P['variant'] = P['project_id'].map(var)
    P['variant_rank'] = P['variant'].map(rank)
    P['happy_path'] = P['variant_rank'] == 1
    P['has_rework'] = P['project_id'].map(T.groupby('project_id')['repeat'].any()).fillna(False)
    P['n_tasks'] = P['project_id'].map(T.groupby('project_id').size())
    P['wait_total'] = P['project_id'].map(T.groupby('project_id')['wait_days'].sum())

    A = alloc.copy()
    A['task_id'] = A['task_id'].astype(str); A['resource_id'] = A['resource_id'].astype(str); A['project_id'] = A['project_id'].astype(str)
    A['allocation_date'] = pd.to_datetime(A['allocation_date'], errors='coerce')
    R = dr.copy(); R['resource_id'] = R['resource_id'].astype(str)
    A = A.merge(R[['resource_id', 'resource_name', 'resource_type', 'cost_per_hour']], on='resource_id', how='left')
    A['cost'] = A['hours_worked'] * A['cost_per_hour'].fillna(0)
    A = A.merge(T[['task_id', 'task_name']], on='task_id', how='left')
    # recurso principal de cada tarefa
    main = A.groupby(['task_id', 'resource_id', 'resource_name', 'resource_type'])['hours_worked'].sum().reset_index() \
            .sort_values('hours_worked', ascending=False).drop_duplicates('task_id')
    T = T.merge(main[['task_id', 'resource_name', 'resource_type']], on='task_id', how='left')
    T['prev_role'] = T.groupby('project_id')['resource_type'].shift(1)
    return {'P': P, 'T': T, 'A': A, 'R': R}


def apply_view_filters(V, period=None, types=None, variant_mode='Todos'):
    P = V['P']
    m = pd.Series(True, index=P.index)
    if period is not None:
        m &= (P['end_date'] >= pd.Timestamp(period[0])) & (P['end_date'] <= pd.Timestamp(period[1]) + pd.Timedelta(days=1))
    if types:
        m &= P['path_name'].isin(types)
    if variant_mode == 'Só caminho principal':
        m &= P['happy_path']
    elif variant_mode == 'Só desvios ao caminho principal':
        m &= ~P['happy_path']
    ids = set(P.loc[m, 'project_id'])
    return {'P': P[m], 'T': V['T'][V['T']['project_id'].isin(ids)], 'A': V['A'][V['A']['project_id'].isin(ids)], 'R': V['R']}


def _full_months(P):
    """Meses com volume representativo (>= 30% da mediana mensal). Os meses das pontas costumam estar incompletos."""
    c = P.groupby('end_month').size()
    return set(c[c >= 0.3 * c.median()].index) if len(c) else set()


def _monthly(P):
    g = P.groupby('end_month').agg(concluidos=('project_id', 'count'), lead_mean=('lead_time', 'mean'),
                                   lead_p50=('lead_time', 'median'), lead_p90=('lead_time', lambda x: x.quantile(0.9))).reset_index()
    full = _full_months(P)
    g.loc[~g['end_month'].isin(full), ['lead_mean', 'lead_p50', 'lead_p90']] = np.nan  # não desenhar médias de meses incompletos
    return g


def _recent_vs_prev(P, col, agg='mean', months=3):
    """Compara os últimos N meses (por mês de conclusão) com os N anteriores. Devolve (atual, anterior)."""
    mm = sorted(_full_months(P))
    if len(mm) < 2 * months:
        return None, None
    cur = P[P['end_month'].isin(mm[-months:])][col]; prev = P[P['end_month'].isin(mm[-2 * months:-months])][col]
    f = (lambda s: s.mean()) if agg == 'mean' else (lambda s: s.median()) if agg == 'median' else agg
    return f(cur), f(prev)


def _delta_kpi(P, col, agg='mean', lower_is_better=True, unit='%'):
    """Variação dos últimos 3 meses completos face aos 3 anteriores. unit='pp' para taxas (diferença em pontos percentuais)."""
    cur, prev = _recent_vs_prev(P, col, agg)
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
        return None, None
    if unit == 'pp':
        ch = (cur - prev) * 100
        if abs(ch) < 0.5: return f"{ch:+.1f} p.p. (estável)", None
        return f"{ch:+.1f} p.p. vs 3 meses anteriores", (ch < 0) if lower_is_better else (ch > 0)
    ch = _pct_change(cur, prev)
    if ch is None:
        return None, None
    if abs(ch) < 1: return f"{ch:+.1f}% (estável)", None
    return f"{ch:+.1f}% vs 3 meses anteriores", (ch < 0) if lower_is_better else (ch > 0)


# ---------------------------------------------------------------------------------
# Gráficos (cada função devolve uma figura Plotly ou None)
# ---------------------------------------------------------------------------------
def fig_volume_leadtime(P):
    m = _monthly(P)
    if m.empty: return None
    fig = go.Figure()
    fig.add_bar(x=m['end_month'], y=m['concluidos'], name='Processos concluídos', marker_color='#cbd5e1', yaxis='y2',
                hovertemplate='%{x|%b %Y}<br>%{y} concluídos<extra></extra>')
    fig.add_scatter(x=m['end_month'], y=m['lead_p50'], name='Duração mediana (P50)', mode='lines+markers', line=dict(color=UI['blue'], width=3),
                    hovertemplate='%{x|%b %Y}<br>P50: %{y:.1f} dias<extra></extra>')
    fig.add_scatter(x=m['end_month'], y=m['lead_p90'], name='Duração P90 (10% mais lentos)', mode='lines', line=dict(color=UI['amber'], width=2, dash='dot'),
                    hovertemplate='%{x|%b %Y}<br>P90: %{y:.1f} dias<extra></extra>')
    fig.update_layout(yaxis=dict(title='Duração (dias)'), yaxis2=dict(title='Concluídos', overlaying='y', side='right', showgrid=False))
    return _style(fig, 360)


def fig_wip(P):
    """NOVO: entradas vs saídas por mês e processos em curso (backlog)."""
    if P.empty: return None
    months = pd.period_range(P['start_date'].min(), P['end_date'].max(), freq='M').to_timestamp()
    ent = P.groupby('start_month').size().reindex(months, fill_value=0)
    sai = P.groupby('end_month').size().reindex(months, fill_value=0)
    wip = [((P['start_date'] <= m + pd.offsets.MonthEnd(0)) & (P['end_date'] > m + pd.offsets.MonthEnd(0))).sum() for m in months]
    fig = go.Figure()
    fig.add_bar(x=months, y=ent.values, name='Entradas', marker_color=UI['blue'], opacity=0.85)
    fig.add_bar(x=months, y=sai.values, name='Conclusões', marker_color=UI['teal'], opacity=0.85)
    fig.add_scatter(x=months, y=wip, name='Em curso no fim do mês', mode='lines', line=dict(color=UI['ink'], width=2.5))
    fig.update_layout(barmode='group', yaxis_title='Nº de processos')
    return _style(fig, 340)


def fig_leadtime_dist(P):
    s = P['lead_time'].dropna()
    if s.empty: return None
    fig = px.histogram(P, x='lead_time', nbins=40, color_discrete_sequence=[UI['blue']])
    fig.update_traces(hovertemplate='%{x} dias<br>%{y} processos<extra></extra>', marker_line_width=0)
    for q, lbl, col in [(0.5, 'P50', UI['ink']), (0.9, 'P90', UI['amber'])]:
        v = s.quantile(q)
        fig.add_vline(x=v, line=dict(color=col, dash='dash', width=2))
        fig.add_annotation(x=v, y=1, yref='paper', text=f"{lbl}: {v:.0f} d", showarrow=False, xanchor='left', yanchor='top', font=dict(color=col))
    fig.update_layout(xaxis_title='Duração do processo (dias)', yaxis_title='Nº de processos', bargap=0.05)
    return _style(fig, 320, legend=False)


def fig_leadtime_by_type(P):
    """NOVO: comparação da duração por tipo de processo."""
    if P['path_name'].nunique() < 2: return None
    order = P.groupby('path_name')['lead_time'].median().sort_values().index.tolist()
    fig = px.box(P, x='lead_time', y='path_name', category_orders={'path_name': order}, points=False, color_discrete_sequence=[UI['blue']])
    fig.update_layout(xaxis_title='Duração (dias)', yaxis_title='')
    return _style(fig, max(260, 46 * len(order)), legend=False)


def fig_status(P):
    s = P['project_status'].value_counts()
    fig = go.Figure(go.Pie(labels=s.index, values=s.values, hole=0.62, marker=dict(colors=UI_SEQ), textinfo='percent',
                           hovertemplate='%{label}: %{value}<extra></extra>'))
    return _style(fig, 300)


def _elab(t):
    """Rótulo de seta afastado da linha (célula vazia à esquerda) para não ficar por cima da seta."""
    return f'<<table border="0" cellpadding="0" cellspacing="0"><tr><td width="16"></td><td>{html.escape(t)}</td></tr></table>>'


def build_dfg_dot(T, metric='freq', min_share=0.02, total_cases=None):
    """DFG em Graphviz (nativo no Streamlit, nítido). metric: 'freq' (nº de casos) ou 'wait' (espera média)."""
    if T.empty: return None
    total_cases = total_cases or T['project_id'].nunique()
    starts = T.groupby('project_id').first()['task_name'].value_counts()
    ends = T.groupby('project_id').last()['task_name'].value_counts()
    e = T.dropna(subset=['prev_task']).groupby(['prev_task', 'task_name']).agg(n=('project_id', 'nunique'), wait=('wait_days', 'mean')).reset_index()
    e = e[e['n'] >= max(1, min_share * total_cases)]
    acts = T['task_name'].value_counts()
    keep = set(e['prev_task']) | set(e['task_name']) | set(starts.index[:3])
    wmax = max(e['wait'].max(), 0.01) if not e.empty else 1
    def wcol(w):
        r = min(w / wmax, 1.0)
        c1, c2 = (37, 99, 235), (220, 38, 38)
        return '#%02x%02x%02x' % tuple(int(c1[i] + (c2[i] - c1[i]) * r) for i in range(3))
    lines = ['digraph G {', 'rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.35; size="9,11";',
             'node [shape=box style="rounded,filled" fillcolor="#ffffff" color="#cbd5e1" fontname="Helvetica" fontsize=11 fontcolor="#0f172a" margin="0.18,0.08"];',
             'edge [fontname="Helvetica" fontsize=9 fontcolor="#475569" arrowsize=0.6];',
             '"__start" [label="" shape=circle width=0.22 style=filled fillcolor="#16a34a" color="#16a34a"];',
             '"__end" [label="" shape=doublecircle width=0.18 style=filled fillcolor="#0f172a" color="#0f172a"];']
    for a in keep:
        lines.append(f'"{a}" [label="{a}\\n{acts.get(a, 0):,} ×"];')
    for a, n in starts.items():
        if a in keep and n >= min_share * total_cases:
            lines.append(f'"__start" -> "{a}" [color="#94a3b8" penwidth={1 + 4 * n / total_cases:.2f} label={_elab(f"{n / total_cases:.0%}")}];')
    for a, n in ends.items():
        if a in keep and n >= min_share * total_cases:
            lines.append(f'"{a}" -> "__end" [color="#94a3b8" penwidth={1 + 4 * n / total_cases:.2f}];')
    for _, r in e.iterrows():
        pw = 1 + 5 * r['n'] / total_cases
        if metric == 'wait':
            lab = f"{r['wait']:.1f} d" if r['wait'] >= 0.05 else "0 d"; col = wcol(r['wait'])
        else:
            lab = f"{r['n'] / total_cases:.0%}"; col = '#475569'
        lines.append(f'"{r["prev_task"]}" -> "{r["task_name"]}" [label={_elab(lab)} color="{col}" penwidth={pw:.2f}];')
    lines.append('}')
    return '\n'.join(lines)


def variants_table_html(P, T, top=8):
    vc = P['variant'].value_counts()
    if vc.empty: return ""
    happy = vc.index[0]
    stats = P.groupby('variant').agg(n=('project_id', 'count'), dur=('lead_time', 'median'))
    rows = []
    for i, (v, n) in enumerate(vc.head(top).items()):
        dur = stats.loc[[v], 'dur'].iloc[0]
        chips = []
        okpos = set()
        for blk in difflib.SequenceMatcher(None, list(happy), list(v), autojunk=False).get_matching_blocks():
            okpos.update(range(blk.b, blk.b + blk.size))
        for k, a in enumerate(v):
            dev = k not in okpos
            chips.append(f"<span class='pm-chip{' dev' if dev and i > 0 else ''}'>{html.escape(str(a))}</span>")
        sep = "<i class='bi bi-chevron-right pm-sep'></i>"
        rows.append(f"<tr><td class='pm-vr'>#{i + 1}</td><td><div class='pm-chips'>{sep.join(chips)}</div></td>"
                    f"<td class='pm-num'>{n:,}<div class='pm-sub'>{n / len(P):.1%}</div></td><td class='pm-num'>{dur:.0f} d</td></tr>")
    return ("<table class='pm-table'><thead><tr><th></th><th>Sequência de atividades</th><th>Processos</th><th>Duração mediana</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table><div class='pm-sub' style='margin-top:6px'>A vermelho: passos que diferem do caminho principal (#1). "
            f"Total de variantes: {len(vc):,}.</div>")


def fig_rework_impact(P):
    """NOVO: impacto do rework na duração."""
    if not P['has_rework'].any() or P['has_rework'].all(): return None
    d = P.assign(grupo=np.where(P['has_rework'], 'Com rework', 'Sem rework'))
    fig = px.box(d, x='lead_time', y='grupo', color='grupo', points=False,
                 color_discrete_map={'Com rework': UI['red'], 'Sem rework': UI['blue']})
    fig.update_layout(xaxis_title='Duração do processo (dias)', yaxis_title='')
    return _style(fig, 220, legend=False)


def rework_loops_df(T):
    r = T[T['repeat']]
    if r.empty: return pd.DataFrame()
    back = r.groupby(['prev_task', 'task_name']).agg(ocorrencias=('project_id', 'size'), processos=('project_id', 'nunique'), espera_media=('wait_days', 'mean')).reset_index()
    back['Retorno'] = back['prev_task'] + ' → ' + back['task_name']
    return back.sort_values('processos', ascending=False)[['Retorno', 'processos', 'ocorrencias', 'espera_media']].rename(
        columns={'processos': 'Processos', 'ocorrencias': 'Ocorrências', 'espera_media': 'Espera (d)'}).head(10)


def fig_activity_freq(T, P):
    n = P['project_id'].nunique()
    a = T.groupby('task_name').agg(ocorr=('task_id', 'size'), casos=('project_id', 'nunique')).reset_index()
    a['pct'] = a['casos'] / max(n, 1) * 100
    a = a.sort_values('ocorr')
    fig = go.Figure(go.Bar(x=a['ocorr'], y=a['task_name'], orientation='h', marker_color=UI['blue'],
                           customdata=a[['pct', 'casos']], text=[f"{p:.0f}% dos processos" for p in a['pct']], textposition='inside', insidetextanchor='end', textfont=dict(color='white'),
                           hovertemplate='%{y}<br>%{x:,} execuções · %{customdata[1]:,} processos (%{customdata[0]:.0f}%)<extra></extra>'))
    fig.update_layout(xaxis_title='Nº de execuções', yaxis_title='')
    return _style(fig, max(260, 30 * len(a)), legend=False)


def fig_conformance_month(P):
    m = P.groupby('end_month').agg(happy=('happy_path', 'mean'), rework=('has_rework', 'mean')).reset_index()
    m = m[m['end_month'].isin(_full_months(P))]
    if len(m) < 2: return None
    fig = go.Figure()
    fig.add_scatter(x=m['end_month'], y=m['happy'] * 100, name='% no caminho principal', mode='lines+markers', line=dict(color=UI['blue'], width=3))
    fig.add_scatter(x=m['end_month'], y=m['rework'] * 100, name='% com rework', mode='lines+markers', line=dict(color=UI['red'], width=2))
    fig.update_layout(yaxis_title='% dos processos concluídos', yaxis_range=[0, 105])
    return _style(fig, 300)


def fig_service_vs_wait(T):
    a = T.groupby('task_name').agg(exec=('dur_days', 'mean'), wait=('wait_days', 'mean'), n=('task_id', 'size')).reset_index()
    a['total'] = a['exec'] + a['wait']; a = a.sort_values('total')
    fig = go.Figure()
    fig.add_bar(y=a['task_name'], x=a['exec'], name='Execução', orientation='h', marker_color=UI['blue'],
                hovertemplate='%{y}<br>Execução: %{x:.1f} dias<extra></extra>')
    fig.add_bar(y=a['task_name'], x=a['wait'], name='Espera antes de começar', orientation='h', marker_color=UI['amber'],
                hovertemplate='%{y}<br>Espera: %{x:.1f} dias<extra></extra>')
    fig.update_layout(barmode='stack', xaxis_title='Dias (média por execução)', yaxis_title='')
    return _style(fig, max(280, 32 * len(a)))


def fig_top_handoffs(T, cost_per_day=None):
    e = T.dropna(subset=['prev_task'])
    e = e[~e['repeat']]
    h = e.groupby(['prev_task', 'task_name']).agg(wait=('wait_days', 'mean'), n=('project_id', 'nunique'), tot=('wait_days', 'sum')).reset_index()
    h = h[h['n'] >= max(3, 0.01 * T['project_id'].nunique())]
    if h.empty: return None
    h['lbl'] = h['prev_task'] + ' → ' + h['task_name']
    h = h.sort_values('tot').tail(10)
    txt = [f"{w:.1f} d · {n:,} proc." for w, n in zip(h['wait'], h['n'])]
    hov = '%{y}<br>Espera média: %{customdata[0]:.1f} dias<br>Processos: %{customdata[1]:,}<br>Total de dias parados: %{x:,.0f}'
    cd = h[['wait', 'n']].values
    if cost_per_day:
        h['custo'] = h['tot'] * cost_per_day; cd = h[['wait', 'n', 'custo']].values; hov += '<br>Custo estimado: €%{customdata[2]:,.0f}'
    fig = go.Figure(go.Bar(x=h['tot'], y=h['lbl'], orientation='h', marker_color=UI['amber'], text=txt, textposition='inside', insidetextanchor='end',
                           customdata=cd, hovertemplate=hov + '<extra></extra>'))
    fig.update_layout(xaxis_title='Total de dias de espera (soma em todos os processos)', yaxis_title='')
    return _style(fig, max(320, 38 * len(h)), legend=False)


def fig_wait_matrix(T):
    e = T.dropna(subset=['prev_task'])
    mtx = e.pivot_table(index='prev_task', columns='task_name', values='wait_days', aggfunc='mean')
    cnt = e.pivot_table(index='prev_task', columns='task_name', values='wait_days', aggfunc='size')
    if mtx.empty: return None
    order = T.groupby('task_name')['seq'].median().sort_values().index.tolist()
    mtx = mtx.reindex(index=[o for o in order if o in mtx.index], columns=[o for o in order if o in mtx.columns])
    cnt = cnt.reindex_like(mtx)
    txt = np.where(mtx.notna(), mtx.round(1).astype(str), '')
    fig = go.Figure(go.Heatmap(z=mtx.values, x=mtx.columns, y=mtx.index, colorscale=[[0, '#eff6ff'], [0.5, '#93c5fd'], [1, '#b91c1c']],
                               text=txt, texttemplate='%{text}', customdata=cnt.values, xgap=2, ygap=2,
                               hovertemplate='De: %{y}<br>Para: %{x}<br>Espera média: %{z:.1f} dias<br>Ocorrências: %{customdata}<extra></extra>',
                               colorbar=dict(title='dias', thickness=10)))
    fig.update_layout(xaxis_title='Atividade seguinte', yaxis_title='Atividade anterior', yaxis_autorange='reversed')
    fig.update_xaxes(tickangle=-30)
    return _style(fig, 460, legend=False)


def fig_wait_month(T, P):
    d = T.merge(P[['project_id', 'end_month']], on='project_id')
    m = d.groupby('end_month').agg(exec=('dur_days', 'sum'), wait=('wait_days', 'sum'), n=('project_id', 'nunique')).reset_index()
    m = m[m['end_month'].isin(_full_months(P))]
    if len(m) < 2: return None
    m['exec'] /= m['n']; m['wait'] /= m['n']
    fig = go.Figure()
    fig.add_scatter(x=m['end_month'], y=m['wait'], name='Espera por processo', stackgroup='a', line=dict(color=UI['amber']))
    fig.add_scatter(x=m['end_month'], y=m['exec'], name='Execução por processo', stackgroup='a', line=dict(color=UI['blue']))
    fig.update_layout(yaxis_title='Dias por processo')
    return _style(fig, 300)


def fig_task_duration_box(T):
    order = T.groupby('task_name')['dur_days'].median().sort_values().index.tolist()
    fig = px.box(T, x='dur_days', y='task_name', category_orders={'task_name': order}, points='outliers', color_discrete_sequence=[UI['blue']])
    fig.update_traces(marker=dict(size=3, opacity=0.4))
    fig.update_layout(xaxis_title='Duração da tarefa (dias)', yaxis_title='')
    return _style(fig, max(280, 32 * len(order)), legend=False)


def fig_weekday(T):
    wd = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    s = T['start_date'].dt.weekday.value_counts().reindex(range(7), fill_value=0)
    e = T['end_date'].dt.weekday.value_counts().reindex(range(7), fill_value=0)
    fig = go.Figure()
    fig.add_bar(x=wd, y=s.values, name='Tarefas iniciadas', marker_color=UI['blue'])
    fig.add_bar(x=wd, y=e.values, name='Tarefas concluídas', marker_color=UI['teal'])
    fig.update_layout(barmode='group', yaxis_title='Nº de tarefas')
    return _style(fig, 280)


def fig_gantt_case(T, pid):
    d = T[T['project_id'] == str(pid)].copy()
    if d.empty: return None
    d['fim'] = d['end_date'].where(d['end_date'] > d['start_date'], d['start_date'] + pd.Timedelta(hours=12))
    d['quem'] = d['resource_name'].fillna('—')
    order = d.sort_values('start_date')['task_name'].drop_duplicates().tolist()
    fig = px.timeline(d, x_start='start_date', x_end='fim', y='task_name', color='resource_type' if d['resource_type'].notna().any() else None,
                      hover_data={'quem': True, 'wait_days': ':.1f', 'dur_days': ':.1f'}, color_discrete_sequence=UI_SEQ, category_orders={'task_name': order})
    fig.update_yaxes(autorange='reversed', title='')
    return _style(fig, max(260, 34 * len(d)))


def fig_workload(T, A, R, has_hours):
    if has_hours:
        w = A.groupby(['resource_name', 'resource_type'])['hours_worked'].sum().reset_index(name='v')
        xt = 'Horas trabalhadas'
    else:
        w = T.groupby(['resource_name', 'resource_type']).size().reset_index(name='v')
        xt = 'Nº de tarefas (principal responsável)'
    w = w.sort_values('v').tail(25)
    fig = px.bar(w, x='v', y='resource_name', color='resource_type', orientation='h', color_discrete_sequence=UI_SEQ)
    fig.update_layout(xaxis_title=xt, yaxis_title='', legend_title_text='')
    return _style(fig, max(320, 22 * len(w)))


def fig_utilization(A, R):
    """NOVO: utilização da capacidade por função (horas trabalhadas / horas disponíveis), por mês."""
    if A.empty or 'daily_capacity' not in R.columns: return None
    a = A.assign(m=A['allocation_date'].dt.to_period('M').dt.to_timestamp()).groupby(['m', 'resource_type'])['hours_worked'].sum().reset_index()
    act = A.assign(m=A['allocation_date'].dt.to_period('M').dt.to_timestamp()).groupby(['m', 'resource_type'])['resource_id'].nunique()
    capd = R.set_index('resource_id')['daily_capacity']
    cap_role = A.drop_duplicates('resource_id').assign(c=lambda x: x['resource_id'].map(capd)).groupby('resource_type')['c'].mean()
    a['ativos'] = [act.get((m, r), 0) for m, r in zip(a['m'], a['resource_type'])]
    a['dias_uteis'] = [np.busday_count(m.date(), (m + pd.offsets.MonthBegin(1)).date()) for m in a['m']]
    a['util'] = a['hours_worked'] / (a['ativos'] * a['dias_uteis'] * a['resource_type'].map(cap_role)) * 100
    vol = a.groupby('m')['hours_worked'].sum()
    a = a[a['m'].isin(vol[vol >= 0.3 * vol.median()].index)]
    if a.empty: return None
    a['util'] = a.sort_values('m').groupby('resource_type')['util'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    fig = px.line(a, x='m', y='util', color='resource_type', markers=False, color_discrete_sequence=UI_SEQ)
    fig.add_hline(y=100, line=dict(color=UI['red'], dash='dash', width=1))
    fig.update_layout(yaxis_title='% da capacidade (média móvel 3 meses)', xaxis_title='')
    return _style(fig, 340)


def fig_efficiency(T, A):
    h = A.groupby(['task_id', 'resource_name', 'resource_type'])['hours_worked'].sum().reset_index()
    e = h.groupby(['resource_name', 'resource_type'])['hours_worked'].agg(['mean', 'count']).reset_index()
    e = e[e['count'] >= 5]
    if e.empty: return None
    e['vs'] = e['mean'] / e.groupby('resource_type')['mean'].transform('median') * 100 - 100
    fig = px.strip(e, x='vs', y='resource_type', color='resource_type', hover_name='resource_name',
                   hover_data={'mean': ':.1f', 'count': True, 'vs': ':.0f', 'resource_type': False}, color_discrete_sequence=UI_SEQ)
    fig.add_vline(x=0, line=dict(color=UI['muted'], dash='dash'))
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(xaxis_title='% face à mediana da função (direita = mais horas)', yaxis_title='')
    return _style(fig, max(260, 50 * e['resource_type'].nunique()), legend=False)


def fig_social_network(T, top=35):
    """Rede de colaboração: pessoas ligadas quando passam trabalho uma à outra no mesmo processo."""
    d = T.dropna(subset=['resource_name']).sort_values(['project_id', 'start_date'])
    d = d.assign(prev_res=d.groupby('project_id')['resource_name'].shift(1))
    e = d.dropna(subset=['prev_res']); e = e[e['prev_res'] != e['resource_name']]
    if e.empty: return None
    w = e.groupby(['prev_res', 'resource_name']).size().reset_index(name='n')
    w['a'] = w[['prev_res', 'resource_name']].min(axis=1); w['b'] = w[['prev_res', 'resource_name']].max(axis=1)
    w = w.groupby(['a', 'b'])['n'].sum().reset_index()
    deg = pd.concat([w.groupby('a')['n'].sum(), w.groupby('b')['n'].sum()]).groupby(level=0).sum().sort_values(ascending=False)
    role0 = d.drop_duplicates('resource_name').set_index('resource_name')['resource_type'].fillna('—')
    nr = max(role0.reindex(deg.index).nunique(), 1); per = int(np.ceil(top / nr))
    keep = set(deg.groupby(role0.reindex(deg.index).fillna('—').values).head(per).index)   # as pessoas mais ligadas de cada função
    w = w[w['a'].isin(keep) & w['b'].isin(keep)]
    if len(w) > 2.5 * len(keep): w = w[w['n'] >= w['n'].quantile(0.7)]                       # só as ligações mais fortes
    if w.empty: return None
    G = nx.Graph()
    for r in w.itertuples(): G.add_edge(r.a, r.b, weight=r.n)
    pos = nx.spring_layout(G, k=1.6 / np.sqrt(max(G.number_of_nodes(), 1)), seed=42, weight='weight', iterations=200)
    role = d.drop_duplicates('resource_name').set_index('resource_name')['resource_type'].fillna('—')
    ntask = d.groupby('resource_name').size()
    fig = go.Figure()
    wmax = w['n'].max()
    for r in w.itertuples():
        (x0, y0), (x1, y1) = pos[r.a], pos[r.b]
        fig.add_scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=0.5 + 5 * r.n / wmax, color='rgba(100,116,139,0.35)'),
                        hoverinfo='skip', showlegend=False)
    roles = sorted({role.get(n, '—') for n in G.nodes()})
    for i, rl in enumerate(roles):
        ns = [n for n in G.nodes() if role.get(n, '—') == rl]
        fig.add_scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode='markers', name=rl,
                        marker=dict(size=[10 + 26 * np.sqrt(ntask.get(n, 1) / ntask.max()) for n in ns], color=UI_SEQ[i % len(UI_SEQ)], line=dict(color='white', width=1.5)),
                        text=[f"{n}<br>{rl}<br>{ntask.get(n, 0):,} tarefas · {G.degree(n)} ligações" for n in ns], hovertemplate='%{text}<extra></extra>')
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _style(fig, 520)


def fig_role_handoffs(T):
    e = T.dropna(subset=['prev_role', 'resource_type'])
    e = e[e['prev_role'] != e['resource_type']]
    if e.empty: return None
    m = e.groupby(['prev_role', 'resource_type']).agg(n=('task_id', 'size'), w=('wait_days', 'mean')).reset_index()
    pv = m.pivot(index='prev_role', columns='resource_type', values='n')
    wv = m.pivot(index='prev_role', columns='resource_type', values='w').reindex_like(pv)
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, customdata=wv.values, colorscale='Blues', xgap=2, ygap=2,
                               text=np.where(pv.notna(), pv.fillna(0).astype(int).astype(str), ''), texttemplate='%{text}',
                               hovertemplate='De: %{y}<br>Para: %{x}<br>Passagens: %{z:,}<br>Espera média: %{customdata:.1f} dias<extra></extra>',
                               colorbar=dict(title='passagens', thickness=10)))
    fig.update_layout(xaxis_title='Função que recebe', yaxis_title='Função que entrega')
    fig.update_xaxes(tickangle=-25)
    return _style(fig, 460, legend=False)


def fig_resource_activity(A, has_hours):
    if A.empty: return None
    top = A.groupby('resource_name')['hours_worked'].sum().sort_values(ascending=False).head(25).index
    d = A[A['resource_name'].isin(top)]
    pv = d.pivot_table(index='resource_name', columns='task_name', values='hours_worked', aggfunc='sum' if has_hours else 'count').reindex(top)
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale='Blues', xgap=1, ygap=1,
                               hovertemplate='%{y}<br>%{x}<br>%{z:,.0f}' + (' h' if has_hours else ' registos') + '<extra></extra>', colorbar=dict(thickness=10)))
    fig.update_xaxes(tickangle=-30)
    return _style(fig, max(380, 20 * len(top)), legend=False)


def fig_teamsize(P):
    if 'num_resources' not in P.columns or P['num_resources'].nunique() < 2: return None
    d = P.dropna(subset=['num_resources']).copy()
    d['equipa'] = d['num_resources'].astype(int).astype(str) + ' pessoas'
    order = sorted(d['equipa'].unique(), key=lambda x: int(x.split()[0]))
    fig = px.box(d, x='equipa', y='lead_time', category_orders={'equipa': order}, points=False, color_discrete_sequence=[UI['blue']])
    fig.update_layout(xaxis_title='Pessoas envolvidas no processo', yaxis_title='Duração (dias)')
    return _style(fig, 300, legend=False)


def fig_resource_wait(T):
    d = T.dropna(subset=['resource_name']).copy()
    d['next_wait'] = d.groupby('project_id')['wait_days'].shift(-1)
    g = d.groupby(['resource_name', 'resource_type']).agg(w=('next_wait', 'mean'), n=('task_id', 'size')).reset_index()
    g = g[g['n'] >= 10].sort_values('w').tail(15)
    if g.empty: return None
    fig = px.bar(g, x='w', y='resource_name', color='resource_type', orientation='h', color_discrete_sequence=UI_SEQ,
                 hover_data={'n': True})
    fig.update_layout(xaxis_title='Espera média até a tarefa seguinte começar (dias)', yaxis_title='')
    return _style(fig, 380)


def fig_perf_matrix(P):
    d = P.dropna(subset=['days_diff', 'cost_diff'])
    if d.empty: return None
    fig = px.scatter(d, x='days_diff', y='cost_diff', color='path_name', hover_name='project_name', opacity=0.7,
                     hover_data={'lead_time': True, 'total_actual_cost': ':,.0f', 'path_name': False}, color_discrete_sequence=UI_SEQ)
    fig.add_hline(y=0, line=dict(color=UI['muted'], dash='dash')); fig.add_vline(x=0, line=dict(color=UI['muted'], dash='dash'))
    for x, y, t in [(1, 1, 'Atrasado e acima do orçamento'), (0, 0, 'Adiantado e abaixo do orçamento')]:
        fig.add_annotation(x=x, y=y, xref='paper', yref='paper', text=t, showarrow=False, xanchor='right' if x else 'left',
                           yanchor='top' if y else 'bottom', font=dict(color=UI['muted'], size=11))
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(xaxis_title='Desvio face ao prazo (dias) · negativo = antes do prazo', yaxis_title='Desvio face ao orçamento (€)')
    return _style(fig, 400)


def fig_delay_month(P):
    m = P.groupby('end_month').agg(late=('days_diff', lambda x: (x > 0).mean() * 100), dev=('days_diff', 'mean')).reset_index()
    m = m[m['end_month'].isin(_full_months(P))]
    if len(m) < 2: return None
    fig = go.Figure()
    fig.add_bar(x=m['end_month'], y=m['late'], name='% processos atrasados', marker_color='#fecaca', yaxis='y2')
    fig.add_scatter(x=m['end_month'], y=m['dev'], name='Desvio médio face ao prazo (dias)', mode='lines+markers', line=dict(color=UI['ink'], width=2.5))
    fig.add_hline(y=0, line=dict(color=UI['muted'], dash='dot'))
    fig.update_layout(yaxis_title='Dias', yaxis2=dict(title='% atrasados', overlaying='y', side='right', showgrid=False, rangemode='tozero'))
    return _style(fig, 320)


def fig_delay_dist(P):
    fig = px.histogram(P, x='days_diff', nbins=40, color_discrete_sequence=[UI['blue']])
    fig.add_vline(x=0, line=dict(color=UI['red'], width=2))
    fig.update_layout(xaxis_title='Dias face ao prazo (à direita da linha = atrasado)', yaxis_title='Nº de processos', bargap=0.05)
    return _style(fig, 300, legend=False)


def fig_cost_role(A):
    c = A.groupby('resource_type').agg(cost=('cost', 'sum'), h=('hours_worked', 'sum')).reset_index().sort_values('cost')
    fig = go.Figure(go.Bar(x=c['cost'], y=c['resource_type'], orientation='h', marker_color=UI['blue'], customdata=c[['h']].values,
                           text=[f"{v / c['cost'].sum():.0%}" for v in c['cost']], textposition='inside', insidetextanchor='end', textfont=dict(color='white'),
                           hovertemplate='%{y}<br>€%{x:,.0f}<br>%{customdata[0]:,.0f} horas<extra></extra>'))
    fig.update_layout(xaxis_title='Custo total de mão-de-obra (€)', yaxis_title='')
    return _style(fig, max(260, 36 * len(c)), legend=False)


def fig_cost_activity(A):
    """NOVO: onde se gasta o dinheiro, por atividade."""
    c = A.groupby('task_name')['cost'].sum().sort_values()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation='h', marker_color=UI['teal'],
                           text=[f"{v / c.sum():.0%}" for v in c.values], textposition='inside', insidetextanchor='end', textfont=dict(color='white'), hovertemplate='%{y}<br>€%{x:,.0f}<extra></extra>'))
    fig.update_layout(xaxis_title='Custo total (€)', yaxis_title='')
    return _style(fig, max(260, 32 * len(c)), legend=False)


def fig_cost_vs_budget(P):
    d = P[P['budget_impact'] > 0]
    if d.empty: return None
    fig = px.scatter(d, x='budget_impact', y='total_actual_cost', color='path_name', hover_name='project_name', opacity=0.7, color_discrete_sequence=UI_SEQ)
    mx = max(d['budget_impact'].max(), d['total_actual_cost'].max())
    fig.add_scatter(x=[0, mx], y=[0, mx], mode='lines', line=dict(color=UI['muted'], dash='dash'), name='Custo = orçamento')
    fig.update_layout(xaxis_title='Orçamento (€)', yaxis_title='Custo real (€)')
    return _style(fig, 360)


def fig_cost_month(P):
    m = P.groupby('end_month').agg(c=('total_actual_cost', 'mean'), b=('budget_impact', 'mean')).reset_index()
    if len(m) < 2: return None
    fig = go.Figure()
    fig.add_scatter(x=m['end_month'], y=m['c'], name='Custo médio real', mode='lines+markers', line=dict(color=UI['blue'], width=3))
    if (m['b'] > 0).any():
        fig.add_scatter(x=m['end_month'], y=m['b'], name='Orçamento médio', mode='lines', line=dict(color=UI['muted'], dash='dash'))
    fig.update_layout(yaxis_title='€ por processo')
    return _style(fig, 300)


def outliers_df(P, by, n=10, caps=None):
    cols = {'project_name': 'Processo', 'path_name': 'Tipo', 'lead_time': 'Duração (dias)', 'n_tasks': 'Nº tarefas', 'has_rework': 'Rework'}
    if caps_ok({'CUSTO'}, caps): cols['total_actual_cost'] = 'Custo (€)'
    if caps_ok({'PLANO'}, caps): cols['days_diff'] = 'Desvio prazo (dias)'
    d = P.sort_values(by, ascending=False).head(n)[list(cols)].rename(columns=cols)
    if 'Custo (€)' in d.columns: d['Custo (€)'] = d['Custo (€)'].round(0)
    d['Rework'] = d['Rework'].map({True: 'Sim', False: ''})
    return d


# ---------------------------------------------------------------------------------
# Secções do dashboard
# ---------------------------------------------------------------------------------
def _caps():
    return st.session_state.get('caps')


def section_overview(V, caps):
    P, T = V['P'], V['T']
    ui_section_intro("Como está o processo, em geral?", "Volume, duração e principais indicadores no período e filtros selecionados.")
    d_lead, g_lead = _delta_kpi(P, 'lead_time', 'median')
    k = [dict(icon='bi-collection', value=f"{len(P):,}".replace(',', ' '), label='Processos concluídos'),
         dict(icon='bi-stopwatch', value=_fmt_days(P['lead_time'].median()), label='Duração mediana (P50)', delta=d_lead, good=g_lead,
              help='Metade dos processos demora menos do que isto.'),
         dict(icon='bi-hourglass-split', value=_fmt_days(P['lead_time'].quantile(0.9)), label='P90 — 10% mais lentos demoram mais de',
              help='Indicador de previsibilidade: o prazo que se consegue garantir a 90% dos clientes.'),
         dict(icon='bi-signpost-split', value=f"{P['happy_path'].mean():.0%}", label='Seguem o caminho principal')]
    if caps_ok({'PLANO'}, caps):
        d, g = _delta_kpi(P.assign(_l=(P['days_diff'] > 0).astype(float)), '_l', 'mean', unit='pp')
        k.append(dict(icon='bi-alarm', value=f"{(P['days_diff'] > 0).mean():.1%}", label='Processos atrasados', delta=d, good=g))
    if caps_ok({'TEMPOS'}, caps):
        pw = (P['wait_total'] / P['lead_time'].replace(0, np.nan)).mean()
        k.append(dict(icon='bi-pause-circle', value=f"{pw:.0%}", label='Do tempo é espera entre tarefas'))
    if caps_ok({'CUSTO'}, caps):
        d, g = _delta_kpi(P, 'total_actual_cost', 'mean')
        k.append(dict(icon='bi-currency-euro', value=_fmt_eur(P['total_actual_cost'].mean()), label='Custo médio por processo', delta=d, good=g))
    ui_kpis(k[:8])
    ui_chart(fig_volume_leadtime(P), 'Volume e duração ao longo do tempo', 'Barras: processos concluídos por mês · Linhas: duração mediana e P90 (meses incompletos sem linha)', 'bi-graph-up-arrow', key='ov_vol')
    c1, c2 = st.columns(2)
    with c1: ui_chart(fig_wip(P), 'Entradas, conclusões e processos em curso', 'Se as entradas superam as conclusões, o backlog cresce', 'bi-inboxes', key='ov_wip')
    with c2: ui_chart(fig_leadtime_dist(P), 'Distribuição da duração dos processos', 'Com as marcas de P50 e P90', 'bi-bar-chart', key='ov_dist')
    f_type = fig_leadtime_by_type(P)
    show_status = caps_ok({'ESTADO'}, caps) and P['project_status'].nunique() > 1
    if f_type is not None or show_status:
        c1, c2 = st.columns([3, 2]) if f_type is not None and show_status else (st.container(), None)
        with c1: ui_chart(f_type, 'Duração por tipo de processo', 'Comparação entre segmentos (mediana, quartis e extremos)', 'bi-diagram-2', key='ov_type')
        if c2 is not None:
            with c2: ui_chart(fig_status(P), 'Estado dos processos', None, 'bi-pie-chart', key='ov_status')
        elif f_type is None and show_status:
            ui_chart(fig_status(P), 'Estado dos processos', None, 'bi-pie-chart', key='ov_status2')
    with st.container(border=True):
        ui_card_header('Processos mais longos', 'Os 10 casos com maior duração, para investigação individual', 'bi-list-ol')
        st.dataframe(outliers_df(P, 'lead_time', caps=caps), hide_index=True, use_container_width=True)


# --- Análise detalhada (antigos gráficos matplotlib, agora interativos e responsivos) ---
def _proc_skill(A, R):
    if 'skill_level' not in R.columns: return None
    sk = A.merge(R[['resource_id', 'skill_level']], on='resource_id', how='left')
    sk['skill_level'] = pd.to_numeric(sk['skill_level'], errors='coerce')
    return sk.dropna(subset=['skill_level'])


def fig_skill_hours(A, R):
    sk = _proc_skill(A, R)
    if sk is None or sk.empty or sk['skill_level'].nunique() < 2: return None
    h = sk.groupby(['task_id', 'resource_id', 'skill_level', 'resource_type'])['hours_worked'].sum().reset_index()
    h['nível'] = 'Nível ' + h['skill_level'].round().astype(int).astype(str)
    order = sorted(h['nível'].unique())
    fig = px.box(h, x='nível', y='hours_worked', color='nível', points=False, category_orders={'nível': order}, color_discrete_sequence=UI_SEQ)
    fig.update_layout(xaxis_title='Nível de competência', yaxis_title='Horas por tarefa')
    return _style(fig, 300, legend=False)


def fig_delay_by_skill(P, A, R):
    sk = _proc_skill(A, R)
    if sk is None or sk.empty or 'days_diff' not in P.columns: return None
    m = sk.groupby('project_id')['skill_level'].mean().reset_index().merge(P[['project_id', 'days_diff']], on='project_id').dropna()
    if m['skill_level'].nunique() < 3: return None
    m['faixa'] = pd.qcut(m['skill_level'], q=min(4, m['skill_level'].nunique()), duplicates='drop').astype(str)
    order = sorted(m['faixa'].unique(), key=lambda x: float(x.strip('([').split(',')[0]))
    fig = px.box(m, x='faixa', y='days_diff', points=False, category_orders={'faixa': order}, color_discrete_sequence=[UI['blue']])
    fig.add_hline(y=0, line=dict(color=UI['muted'], dash='dash'))
    fig.update_layout(xaxis_title='Competência média da equipa (faixas)', yaxis_title='Dias face ao prazo')
    return _style(fig, 300, legend=False)


def fig_delay_by_resource(P, A):
    if 'days_diff' not in P.columns or A.empty: return None
    m = A[['project_id', 'resource_name', 'resource_type']].drop_duplicates().merge(P[['project_id', 'days_diff']], on='project_id')
    g = m.groupby(['resource_name', 'resource_type']).agg(d=('days_diff', 'mean'), n=('project_id', 'nunique')).reset_index()
    g = g[g['n'] >= 10].sort_values('d').tail(15)
    if g.empty: return None
    fig = px.bar(g, x='d', y='resource_name', color='resource_type', orientation='h', hover_data={'n': True}, color_discrete_sequence=UI_SEQ)
    fig.update_layout(xaxis_title='Desvio médio face ao prazo dos processos em que participou (dias)', yaxis_title='')
    return _style(fig, 380)


def fig_resources_quarter(P):
    if 'num_resources' not in P.columns or P['num_resources'].nunique() < 2: return None
    q = P.assign(q=P['start_date'].dt.to_period('Q').astype(str)).groupby('q')['num_resources'].mean().reset_index()
    fig = go.Figure(go.Bar(x=q['q'], y=q['num_resources'], marker_color=UI['blue'], hovertemplate='%{x}<br>%{y:.1f} pessoas<extra></extra>'))
    fig.update_layout(xaxis_title='Trimestre de início', yaxis_title='Pessoas por processo (média)')
    return _style(fig, 280, legend=False)


def fig_corr(P):
    cols = {'lead_time': 'Duração', 'days_diff': 'Desvio prazo', 'total_actual_cost': 'Custo', 'cost_diff': 'Desvio custo',
            'budget_impact': 'Orçamento', 'num_resources': 'Nº pessoas', 'n_tasks': 'Nº tarefas', 'wait_total': 'Espera', 'loan_amount_eur': 'Montante'}
    c = [k for k in cols if k in P.columns and pd.to_numeric(P[k], errors='coerce').nunique() > 1]
    if len(c) < 3: return None
    m = P[c].apply(pd.to_numeric, errors='coerce').corr().rename(index=cols, columns=cols)
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, zmin=-1, zmax=1, colorscale='RdBu_r', xgap=2, ygap=2,
                               text=m.round(2).values, texttemplate='%{text}', hovertemplate='%{y} × %{x}: %{z:.2f}<extra></extra>',
                               colorbar=dict(thickness=10)))
    fig.update_yaxes(autorange='reversed')
    return _style(fig, 420, legend=False)


def fig_cost_vs_delay(P):
    d = P.dropna(subset=['total_actual_cost', 'days_diff'])
    if d.empty: return None
    fig = px.scatter(d, x='days_diff', y='total_actual_cost', color='path_name', hover_name='project_name', opacity=0.6, color_discrete_sequence=UI_SEQ)
    fig.add_vline(x=0, line=dict(color=UI['muted'], dash='dash'))
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(xaxis_title='Dias face ao prazo', yaxis_title='Custo real (€)')
    return _style(fig, 320)


def fig_teamsize_cost(P):
    if 'num_resources' not in P.columns or P['num_resources'].nunique() < 2: return None
    d = P.dropna(subset=['num_resources', 'total_actual_cost']).copy()
    d['equipa'] = d['num_resources'].astype(int)
    fig = px.box(d.sort_values('equipa'), x=d.sort_values('equipa')['equipa'].astype(str), y='total_actual_cost', points=False, color_discrete_sequence=[UI['teal']])
    fig.update_layout(xaxis_title='Pessoas envolvidas no processo', yaxis_title='Custo real (€)')
    return _style(fig, 320, legend=False)


def _budget_band(P):
    b = pd.to_numeric(P['budget_impact'], errors='coerce')
    if b.nunique() < 4: return None
    return pd.qcut(b, 4, labels=['Q1 (mais baixo)', 'Q2', 'Q3', 'Q4 (mais alto)'], duplicates='drop')


def fig_delay_by_budget(P):
    if 'budget_impact' not in P.columns or 'days_diff' not in P.columns: return None
    band = _budget_band(P)
    if band is None: return None
    d = P.assign(faixa=band.astype(str)).dropna(subset=['faixa'])
    fig = px.box(d, x='faixa', y='days_diff', points=False, color_discrete_sequence=[UI['blue']],
                 category_orders={'faixa': ['Q1 (mais baixo)', 'Q2', 'Q3', 'Q4 (mais alto)']})
    fig.add_hline(y=0, line=dict(color=UI['muted'], dash='dash'))
    fig.update_layout(xaxis_title='Faixa de orçamento (quartis)', yaxis_title='Dias face ao prazo')
    return _style(fig, 320, legend=False)


def fig_cost_budget_role(P, A):
    if 'budget_impact' not in P.columns or A.empty: return None
    band = _budget_band(P)
    if band is None: return None
    m = A.merge(P[['project_id']].assign(faixa=band.astype(str).values), on='project_id')
    g = m.groupby(['faixa', 'resource_type'])['cost'].sum().reset_index()
    g['cost'] = g['cost'] / g['faixa'].map(P.assign(faixa=band.astype(str)).groupby('faixa').size())
    fig = px.bar(g, x='faixa', y='cost', color='resource_type', color_discrete_sequence=UI_SEQ,
                 category_orders={'faixa': ['Q1 (mais baixo)', 'Q2', 'Q3', 'Q4 (mais alto)']})
    fig.update_layout(barmode='stack', xaxis_title='Faixa de orçamento (quartis)', yaxis_title='Custo médio por processo (€)')
    return _style(fig, 340)


def section_flow(V, caps):
    P, T = V['P'], V['T']
    ui_section_intro("Que caminhos seguem os processos?", "Mapa do processo real, variantes, retrabalho e conformidade com o caminho principal.")
    n_var = P['variant'].nunique()
    k = [dict(icon='bi-signpost-split', value=f"{n_var:,}", label='Variantes diferentes'),
         dict(icon='bi-check2-circle', value=f"{P['happy_path'].mean():.1%}", label='Seguem o caminho principal'),
         dict(icon='bi-arrow-repeat', value=f"{P['has_rework'].mean():.1%}", label='Processos com rework'),
         dict(icon='bi-list-task', value=f"{P['n_tasks'].median():.0f}", label='Tarefas por processo (mediana)')]
    if P['has_rework'].any() and not P['has_rework'].all():
        extra = P.loc[P['has_rework'], 'lead_time'].median() - P.loc[~P['has_rework'], 'lead_time'].median()
        k.append(dict(icon='bi-clock-history', value=f"{extra:+.1f} dias", label='Duração extra quando há rework'))
    ui_kpis(k)
    with st.container(border=True):
        ui_card_header('Mapa do processo', 'Caixas: atividades (nº de execuções) · Setas: passagens mais frequentes', 'bi-diagram-3')
        c1, c2 = st.columns([2, 3])
        metric = c1.segmented_control('Mostrar nas setas', ['% de processos', 'Espera média'], default='% de processos', key='dfg_metric') \
            if caps_ok({'TEMPOS'}, caps) else '% de processos'
        share = c2.slider('Esconder passagens com menos de (% dos processos)', 0, 20, 2, key='dfg_share')
        dot = build_dfg_dot(T, 'wait' if metric == 'Espera média' else 'freq', share / 100, P['project_id'].nunique())
        if dot: st.graphviz_chart(dot, use_container_width=False)
        if metric == 'Espera média': ui_note("Cor das setas: azul = espera curta, vermelho = espera longa antes de a atividade seguinte começar.")
    with st.container(border=True):
        ui_card_header('Variantes do processo', 'As sequências de atividades mais frequentes', 'bi-list-columns')
        st.markdown(variants_table_html(P, T), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            ui_card_header('Ciclos de retrabalho', 'Retornos a uma atividade já feita', 'bi-arrow-counterclockwise')
            rl = rework_loops_df(T)
            if rl.empty: st.caption('Não foram detetados ciclos de retrabalho.')
            else:
                if not caps_ok({'TEMPOS'}, caps): rl = rl.drop(columns=['Espera média (dias)'])
                st.dataframe(rl.round(1), hide_index=True, use_container_width=True)
            f = fig_rework_impact(P)
            if f is not None:
                st.caption('Duração dos processos com e sem retrabalho')
                st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False}, key='fl_rwimp')
    with c2: ui_chart(fig_conformance_month(P), 'Conformidade ao longo do tempo', '% de processos no caminho principal e com retrabalho, por mês', 'bi-check2-square', key='fl_conf')
    ui_chart(fig_activity_freq(T, P), 'Frequência das atividades', 'Execuções e % de processos em que cada atividade aparece', 'bi-bar-chart-steps', key='fl_freq')
    with st.expander("Análise detalhada: modelos formais de processo (Petri nets) e métricas de qualidade"):
        po = gate_outputs(st.session_state.plots_post_mining, caps)
        me = st.session_state.get('metrics', {})
        if me.get('inductive_miner'):
            mt = pd.DataFrame({'Inductive Miner': me['inductive_miner'], 'Heuristics Miner': me.get('heuristics_miner', {})}).T.round(2)
            st.dataframe(mt, use_container_width=True)
            st.caption("Fitness: o modelo reproduz o que aconteceu · Precisão: o modelo não permite caminhos que nunca aconteceram · "
                       "Generalização e Simplicidade: robustez e legibilidade do modelo.")
        for k_, t_ in [('model_inductive_petrinet', 'Modelo — Inductive Miner'), ('model_heuristic_petrinet', 'Modelo — Heuristics Miner')]:
            if po.get(k_) is not None:
                st.markdown(f"**{t_}**"); st.image(po[k_].getvalue(), use_container_width=True)


def section_times(V, caps):
    P, T = V['P'], V['T']
    if not caps_ok({'TEMPOS'}, caps):
        ui_note("Esta secção precisa de datas de início e de fim por tarefa."); return
    ui_section_intro("Onde se perde tempo?", "Separação entre tempo de execução e tempo de espera, e as passagens onde os processos ficam parados.")
    ex = T.groupby('project_id')['dur_days'].sum().mean(); wt = P['wait_total'].mean()
    top = T.dropna(subset=['prev_task'])[~T.dropna(subset=['prev_task'])['repeat']].groupby(['prev_task', 'task_name'])['wait_days'].sum().sort_values()
    k = [dict(icon='bi-play-circle', value=_fmt_days(ex), label='Execução por processo (média)'),
         dict(icon='bi-pause-circle', value=_fmt_days(wt), label='Espera por processo (média)'),
         dict(icon='bi-percent', value=f"{(P['wait_total'] / P['lead_time'].replace(0, np.nan)).mean():.0%}", label='Do tempo total é espera')]
    if not top.empty:
        k.append(dict(icon='bi-sign-stop', value=top.index[-1][1], label=f"Maior espera acumulada: antes desta atividade"))
    ui_kpis(k)
    ui_chart(fig_service_vs_wait(T), 'Execução vs. espera por atividade', 'Tempo médio a trabalhar e tempo médio parado antes de começar', 'bi-hourglass-split', key='tm_sw')
    cpd = None
    if caps_ok({'CUSTO'}, caps) and 'cost_per_day' in P.columns:
        cpd = P['cost_per_day'].mean()
    ui_chart(fig_top_handoffs(T, cpd), 'Passagens onde os processos ficam mais tempo parados', 'Top 10 por total de dias de espera (frequência × espera média)', 'bi-sign-stop', key='tm_ho')
    ui_chart(fig_wait_matrix(T), 'Matriz de espera entre atividades', 'Espera média (dias) entre o fim de uma atividade e o início da seguinte — inclui retornos', 'bi-grid-3x3-gap', key='tm_mx')
    c1, c2 = st.columns(2)
    with c1: ui_chart(fig_wait_month(T, P), 'Execução e espera ao longo do tempo', 'Dias por processo, por mês de conclusão', 'bi-graph-up', key='tm_wm')
    with c2: ui_chart(fig_weekday(T), 'Atividade por dia da semana', 'Tarefas iniciadas e concluídas', 'bi-calendar-week', key='tm_wd')
    ui_chart(fig_task_duration_box(T), 'Variabilidade da duração de cada atividade', 'Mediana, quartis e casos extremos', 'bi-distribute-horizontal', key='tm_box')
    with st.container(border=True):
        ui_card_header('Linha do tempo de um processo', 'Escolha um processo para ver as suas tarefas, quem as fez e onde esperou', 'bi-kanban')
        opts = P.sort_values('lead_time', ascending=False)
        lab = {r.project_id: f"{r.project_name} · {r.lead_time:.0f} dias" for r in opts.itertuples()}
        if lab:
            pid = st.selectbox('Processo', list(lab), format_func=lambda x: lab[x], key='gantt_pid')
            f = fig_gantt_case(T, pid)
            if f is not None: st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False}, key='tm_gantt')


def section_resources(V, caps):
    P, T, A, R = V['P'], V['T'], V['A'], V['R']
    if not caps_ok({'RECURSOS'}, caps):
        ui_note("Esta secção precisa de saber quem executou cada tarefa."); return
    has_h = caps_ok({'ESFORCO'}, caps)
    ui_section_intro("Quem faz o trabalho e onde há pressão?", "Carga, capacidade, eficiência e passagens de trabalho entre funções.")
    roles = R[R['resource_id'].isin(A['resource_id'].unique())]['resource_type'].value_counts()
    k = [dict(icon='bi-people', value=f"{A['resource_id'].nunique():,}", label='Pessoas envolvidas'),
         dict(icon='bi-diagram-2', value=f"{len(roles)}", label='Funções')]
    single = roles[roles <= 1]
    if len(single): k.append(dict(icon='bi-exclamation-triangle', value=f"{len(single)}", label='Funções com 1 só pessoa',
                                  help='Ponto único de falha: ' + ', '.join(single.index[:5])))
    if has_h:
        hw = A.groupby('resource_id')['hours_worked'].sum().sort_values(ascending=False)
        top20 = hw.head(max(1, int(len(hw) * 0.2))).sum() / hw.sum()
        k.append(dict(icon='bi-bar-chart', value=f"{top20:.0%}", label='Das horas feitas por 20% das pessoas'))
    ui_kpis(k)
    c1, c2 = st.columns(2)
    with c1: ui_chart(fig_workload(T, A, R, has_h), 'Carga de trabalho por pessoa', 'Top 25, por função', 'bi-person-workspace', key='rs_wl')
    with c2:
        if has_h: ui_chart(fig_utilization(A, R), 'Utilização da capacidade por função', 'Horas trabalhadas ÷ horas disponíveis das pessoas ativas no mês (linha vermelha = 100%)', 'bi-speedometer2', key='rs_ut')
        else: ui_chart(fig_resource_wait(T), 'Espera gerada a seguir a cada pessoa', 'Tempo médio até a tarefa seguinte começar', 'bi-sign-stop', key='rs_rw0')
    ui_chart(fig_social_network(T), 'Rede de colaboração entre pessoas', 'Quem passa trabalho a quem · as pessoas mais ligadas de cada função e as ligações mais fortes', 'bi-share', key='rs_net')
    c1, c2 = st.columns(2)
    with c1: ui_chart(fig_role_handoffs(T), 'Passagens de trabalho entre funções', 'Nº de passagens; no hover, a espera média em cada passagem', 'bi-arrow-left-right', key='rs_rh')
    with c2:
        if has_h: ui_chart(fig_efficiency(T, A), 'Eficiência comparada dentro de cada função', 'Cada ponto é uma pessoa (≥5 tarefas)', 'bi-person-check', key='rs_ef')
        else: ui_chart(fig_teamsize(P), 'Duração vs. nº de pessoas envolvidas', None, 'bi-people', key='rs_ts0')
    if has_h:
        ui_chart(fig_resource_wait(T), 'Espera gerada a seguir a cada pessoa', 'Tempo médio até a tarefa seguinte começar (pessoas com ≥10 tarefas)', 'bi-sign-stop', key='rs_rw')
        ui_chart(fig_resource_activity(A, has_h), 'Especialização: horas por pessoa e atividade', 'Top 25 pessoas por horas', 'bi-grid', key='rs_ra')
    with st.expander("Análise detalhada: tamanho das equipas, competências e composição"):
        has_sk, has_p = caps_ok({'COMPETENCIA'}, caps), caps_ok({'PLANO'}, caps)
        c1, c2 = st.columns(2)
        with c1:
            if has_h: ui_chart(fig_teamsize(P), 'Duração vs. nº de pessoas envolvidas', None, 'bi-people', key='rs_ts')
        with c2: ui_chart(fig_resources_quarter(P), 'Pessoas por processo ao longo do tempo', 'Média por trimestre de início', 'bi-people-fill', key='rs_rq')
        c1, c2 = st.columns(2)
        with c1:
            if has_sk and has_h: ui_chart(fig_skill_hours(A, R), 'Competência vs. horas por tarefa', 'Distribuição das horas por tarefa, por nível de competência', 'bi-mortarboard', key='rs_sk')
        with c2:
            if has_sk and has_p: ui_chart(fig_delay_by_skill(P, A, R), 'Atraso vs. competência da equipa', None, 'bi-mortarboard-fill', key='rs_sd')
        if has_p: ui_chart(fig_delay_by_resource(P, A), 'Desvio de prazo dos processos de cada pessoa', 'Pessoas com ≥10 processos (correlação, não causa)', 'bi-person-exclamation', key='rs_dr')


def section_costs(V, caps):
    P, A = V['P'], V['A']
    has_c, has_p, has_b = caps_ok({'CUSTO'}, caps), caps_ok({'PLANO'}, caps), caps_ok({'ORCAMENTO'}, caps)
    if not (has_c or has_p):
        ui_note("Esta secção precisa de custos (custo/hora + horas) ou de prazos (data planeada ou SLA)."); return
    ui_section_intro("Cumprimos prazos e orçamentos? Onde se gasta?", "Desvios face ao planeado e distribuição dos custos.")
    k = []
    if has_p:
        late = P[P['days_diff'] > 0]
        k += [dict(icon='bi-alarm', value=f"{(P['days_diff'] > 0).mean():.1%}", label='Processos atrasados'),
              dict(icon='bi-calendar-x', value=_fmt_days(late['days_diff'].mean()) if len(late) else '—', label='Atraso médio dos atrasados'),
              dict(icon='bi-calendar-check', value=f"{P['days_diff'].mean():+.1f} dias", label='Desvio médio face ao prazo', help='Negativo = termina antes do prazo')]
    if has_c:
        k += [dict(icon='bi-currency-euro', value=_fmt_eur(P['total_actual_cost'].mean()), label='Custo médio por processo'),
              dict(icon='bi-cash-stack', value=_fmt_eur(A['cost'].sum()), label='Custo total no período')]
    if has_c and caps_ok({'TEMPOS'}, caps) and 'cost_per_day' in P.columns:
        k.append(dict(icon='bi-hourglass-bottom', value=_fmt_eur(P['wait_total'].sum() * P['cost_per_day'].mean()), label='Custo estimado da espera',
                      help='Dias de espera × custo médio por dia de processo. É uma estimativa de custo de oportunidade.'))
    ui_kpis(k)
    if has_p and has_c and has_b:
        ui_chart(fig_perf_matrix(P), 'Matriz de desempenho: prazo vs. orçamento', 'Cada ponto é um processo', 'bi-bullseye', key='cs_pm')
    if has_p:
        c1, c2 = st.columns(2)
        with c1: ui_chart(fig_delay_month(P), 'Pontualidade ao longo do tempo', '% de atrasados e desvio médio por mês', 'bi-calendar-range', key='cs_dm')
        with c2: ui_chart(fig_delay_dist(P), 'Distribuição do desvio face ao prazo', None, 'bi-bar-chart', key='cs_dd')
    if has_c:
        c1, c2 = st.columns(2)
        with c1: ui_chart(fig_cost_role(A), 'Custo por função', None, 'bi-tags', key='cs_cr')
        with c2: ui_chart(fig_cost_activity(A), 'Custo por atividade', None, 'bi-list-check', key='cs_ca')
        c1, c2 = st.columns(2)
        with c1: ui_chart(fig_cost_month(P), 'Custo médio por processo ao longo do tempo', None, 'bi-graph-up', key='cs_cm')
        with c2:
            if has_b: ui_chart(fig_cost_vs_budget(P), 'Custo real vs. orçamento', 'Acima da diagonal = acima do orçamento', 'bi-credit-card', key='cs_cb')
        with st.container(border=True):
            ui_card_header('Processos mais caros', None, 'bi-list-ol')
            st.dataframe(outliers_df(P, 'total_actual_cost', caps=caps), hide_index=True, use_container_width=True)
    with st.expander("Análise detalhada: correlações e relações entre variáveis"):
        ui_chart(fig_corr(P), 'Matriz de correlação', 'Relação linear entre indicadores do processo (−1 a 1)', 'bi-grid-3x3', key='cs_corr')
        c1, c2 = st.columns(2)
        with c1:
            if has_c and has_p: ui_chart(fig_cost_vs_delay(P), 'Custo real vs. desvio de prazo', None, 'bi-currency-euro', key='cs_cvd')
        with c2:
            if has_c: ui_chart(fig_teamsize_cost(P), 'Custo vs. nº de pessoas envolvidas', None, 'bi-people', key='cs_tsc')
        if has_b:
            c1, c2 = st.columns(2)
            with c1:
                if has_p: ui_chart(fig_delay_by_budget(P), 'Desvio de prazo por faixa de orçamento', None, 'bi-calendar-range', key='cs_dbb')
            with c2:
                if has_c: ui_chart(fig_cost_budget_role(P, A), 'Custo por faixa de orçamento e função', None, 'bi-tags', key='cs_cbr')


def render_filters(V):
    """Filtros globais na barra lateral. Devolve a vista filtrada."""
    P = V['P']
    with st.sidebar:
        st.markdown("<div class='pm-side-label'>Filtros</div>", unsafe_allow_html=True)
        dmin, dmax = P['end_date'].min().date(), P['end_date'].max().date()
        period = st.date_input('Período (data de conclusão)', value=(dmin, dmax), min_value=dmin, max_value=dmax, key='flt_period', format='DD/MM/YYYY')
        if not isinstance(period, (list, tuple)) or len(period) != 2: period = (dmin, dmax)
        types = None
        if P['path_name'].nunique() > 1:
            types = st.multiselect('Tipo de processo', sorted(P['path_name'].dropna().unique()), key='flt_types', placeholder='Todos')
        vmode = st.selectbox('Variantes', ['Todos', 'Só caminho principal', 'Só desvios ao caminho principal'], key='flt_var')
    FV = apply_view_filters(V, period, types or None, vmode)
    return FV, (period != (dmin, dmax)) or bool(types) or vmode != 'Todos'


# =====================================================================================
# --- BLOCO IA: FACTOS -> PRIORIDADES -> LEITURA PARA DECISÃO (com ou sem LLM) ---
# =====================================================================================
AREA_SECTION = {'fluxo': ('fluxo', 'Fluxo e Variantes'), 'tempos': ('tempos', 'Tempos e Esperas'),
                'recursos': ('recursos', 'Recursos e Equipas'), 'custos': ('custos', 'Custos e Prazos')}
AREA_ICON = {'fluxo': 'bi-signpost-split', 'tempos': 'bi-hourglass-split', 'recursos': 'bi-people', 'custos': 'bi-currency-euro'}


def _r(x, n=1):
    try:
        return None if x is None or pd.isna(x) else round(float(x), n)
    except (TypeError, ValueError):
        return None


def _anon_map(V):
    """Nomes de pessoas -> 'Função #n' (para não enviar nomes a um serviço externo)."""
    A, T = V['A'], V['T']
    src = pd.concat([A[['resource_name', 'resource_type']] if 'resource_name' in A.columns else pd.DataFrame(columns=['resource_name', 'resource_type']),
                     T[['resource_name', 'resource_type']] if 'resource_name' in T.columns else pd.DataFrame(columns=['resource_name', 'resource_type'])]).dropna(subset=['resource_name']).drop_duplicates('resource_name')
    out, cnt = {}, {}
    for n, r in zip(src['resource_name'], src['resource_type'].fillna('Recurso')):
        cnt[r] = cnt.get(r, 0) + 1; out[n] = f"{r} #{cnt[r]}"
    return out


def _role_utilization_recent(A, R, months=3):
    """Utilização média por função nos últimos N meses completos (horas / capacidade das pessoas ativas)."""
    if A.empty or 'daily_capacity' not in R.columns: return {}
    a = A.assign(m=A['allocation_date'].dt.to_period('M').dt.to_timestamp())
    vol = a.groupby('m')['hours_worked'].sum(); full = sorted(vol[vol >= 0.3 * vol.median()].index)[-months:]
    a = a[a['m'].isin(full)]
    if a.empty: return {}
    capd = R.set_index('resource_id')['daily_capacity']
    out = {}
    for role, g in a.groupby('resource_type'):
        cap = 0
        for m, gm in g.groupby('m'):
            bd = np.busday_count(m.date(), (m + pd.offsets.MonthBegin(1)).date())
            cap += sum(capd.get(rid, 8) for rid in gm['resource_id'].unique()) * bd
        out[role] = g['hours_worked'].sum() / cap * 100 if cap else None
    return out


def compute_priorities(V, caps, max_items=5):
    """Oportunidades ordenadas por impacto (em € se houver custos, senão em dias). Calculado por código: não depende de IA."""
    P, T, A, R = V['P'], V['T'], V['A'], V['R']
    has_t, has_c, has_p, has_b = caps_ok({'TEMPOS'}, caps), caps_ok({'CUSTO'}, caps), caps_ok({'PLANO'}, caps), caps_ok({'ORCAMENTO'}, caps)
    cpd = P['cost_per_day'].mean() if has_c and 'cost_per_day' in P.columns else None
    eur = lambda d: d * cpd if cpd else None
    items = []

    def add(area, titulo, problema, evidencia, acao, dias=None, euros=None, peso=None, nivel=None, sim=None):
        items.append(dict(area=area, titulo=titulo, problema=problema, evidencia=evidencia, acao=acao, sim=sim,
                          impacto_dias=_r(dias, 0), impacto_eur=_r(euros, 0), _peso=peso if peso is not None else (euros if euros else (dias or 0) * (cpd or 1)), nivel=nivel))

    try:
        if has_t:
            e = T.dropna(subset=['prev_task']); e = e[~e['repeat']]
            h = e.groupby(['prev_task', 'task_name']).agg(w=('wait_days', 'mean'), n=('project_id', 'nunique'), tot=('wait_days', 'sum')).reset_index()
            h = h[h['n'] >= max(3, 0.01 * P['project_id'].nunique())].sort_values('tot', ascending=False)
            tot_wait = max(T['wait_days'].sum(), 1e-9)
            for i, r in enumerate(h.head(2).itertuples()):
                if r.tot / tot_wait < 0.08 and i > 0: break
                add('tempos', f"Espera entre «{r.prev_task}» e «{r.task_name}»",
                    f"Os processos ficam em média {r.w:.1f} dias parados nesta passagem.",
                    f"{r.n:,} processos · {r.tot:,.0f} dias parados no total ({r.tot / tot_wait:.0%} de toda a espera)".replace(',', ' '),
                    "Perceber porque o trabalho fica em fila: quem recebe tem capacidade? Há critérios de entrada ou aprovações em lote? Definir um SLA interno para esta passagem.",
                    dias=r.tot, euros=eur(r.tot), sim={'name': f"Metade da espera {r.prev_task} → {r.task_name}", 'exo': [[r.prev_task, r.task_name]], 'exo_pct': 50})
    except Exception: pass

    try:
        if P['has_rework'].any() and not P['has_rework'].all():
            rw, nrw = P[P['has_rework']], P[~P['has_rework']]
            extra = rw['lead_time'].mean() - nrw['lead_time'].mean()
            ex_eur = (rw['total_actual_cost'].mean() - nrw['total_actual_cost'].mean()) * len(rw) if has_c and 'total_actual_cost' in P.columns else None
            loops = rework_loops_df(T) if has_t else pd.DataFrame()
            top_loop = loops.iloc[0]['Retorno'] if not loops.empty else None
            if extra > 0:
                add('fluxo', "Retrabalho", f"{len(rw) / len(P):.1%} dos processos voltam atrás; cada um demora mais {extra:.1f} dias.",
                    f"{len(rw):,} processos com retrabalho".replace(',', ' ') + (f" · retorno mais comum: {top_loop}" if top_loop else ""),
                    ("Atacar a causa do retorno mais frequente" + (f" ({top_loop})" if top_loop else "") + ": validar na origem o que hoje é detetado mais à frente (checklist, documentação completa à entrada)."),
                    dias=extra * len(rw), euros=ex_eur if ex_eur and ex_eur > 0 else eur(extra * len(rw)), sim={'name': 'Retrabalho −50%', 'rework': 50})
    except Exception: pass

    try:
        if has_p:
            late = P[P['days_diff'] > 0]; lr = len(late) / len(P)
            cur, prev = _recent_vs_prev(P.assign(_l=(P['days_diff'] > 0).astype(float)), '_l')
            worsening = cur is not None and prev is not None and cur - prev > 0.03
            if lr > 0.05 or worsening:
                add('custos', "Atrasos face ao prazo", f"{lr:.1%} dos processos terminam depois do prazo (atraso médio {late['days_diff'].mean():.1f} dias)." if len(late) else "Taxa de atrasos a subir.",
                    f"{late['days_diff'].sum():,.0f} dias de atraso acumulados".replace(',', ' ') + (f" · últimos 3 meses {cur:.0%} vs {prev:.0%} antes" if cur is not None else ""),
                    "Identificar o que os processos atrasados têm em comum (tipo, variante, equipa) e criar um alerta precoce para processos em risco.",
                    dias=late['days_diff'].sum(), euros=eur(late['days_diff'].sum()), sim={'tab': 'rl'})
    except Exception: pass

    try:
        cur, prev = _recent_vs_prev(P, 'lead_time')
        if cur and prev and (cur - prev) / prev > 0.05:
            nm = P['end_month'].value_counts().median()
            add('tempos', "Duração dos processos a aumentar", f"Os processos estão a demorar mais: {cur:.1f} dias nos últimos 3 meses vs {prev:.1f} nos 3 anteriores.",
                f"+{(cur - prev) / prev:.0%} em 3 meses · ~{nm:.0f} processos concluídos por mês",
                "Verificar se o aumento vem de mais volume, menos pessoas ou de uma atividade específica (ver Tempos e Esperas por mês).",
                dias=(cur - prev) * nm * 3, euros=eur((cur - prev) * nm * 3))
    except Exception: pass

    try:
        if has_t:
            g = T.groupby('task_name')['dur_days']
            v = pd.DataFrame({'p50': g.median(), 'p90': g.quantile(0.9), 'n': g.size()})
            v['gap'] = (v['p90'] - v['p50']) * v['n'] * 0.1
            v = v[v['p50'] > 0.5].sort_values('gap', ascending=False)
            if not v.empty and v.iloc[0]['p90'] > 2 * v.iloc[0]['p50']:
                a, r = v.index[0], v.iloc[0]
                add('tempos', f"Duração imprevisível em «{a}»", f"Metade das execuções demora até {r['p50']:.1f} dias, mas 10% demoram mais de {r['p90']:.1f}.",
                    f"{int(r['n']):,} execuções".replace(',', ' '),
                    "Analisar os casos mais longos desta atividade (o que têm em comum?) e normalizar o trabalho (modelos, critérios, escalonamento).",
                    dias=r['gap'], euros=eur(r['gap']), sim={'name': f"{a} 20% mais rápida", 'dur': [a], 'dur_pct': 20})
    except Exception: pass

    try:
        if has_b and has_c:
            over = P[P['cost_diff'] > 0]
            if len(over) / len(P) > 0.15:
                add('custos', "Processos acima do orçamento", f"{len(over) / len(P):.0%} dos processos custam mais do que o orçamentado.",
                    f"{over['cost_diff'].sum():,.0f} € acima do orçamento no total".replace(',', ' '),
                    "Rever os pressupostos do orçamento por tipo de processo e identificar as atividades que mais derrapam.",
                    euros=over['cost_diff'].sum())
    except Exception: pass

    ref = max([i['_peso'] for i in items] + [1])
    try:
        if caps_ok({'ESFORCO'}, caps):
            util = _role_utilization_recent(A, R)
            hot = {k: v for k, v in util.items() if v and v > 90}
            if hot:
                k = max(hot, key=hot.get)
                add('recursos', f"Função «{k}» no limite da capacidade", f"A função trabalhou {hot[k]:.0f}% da capacidade disponível nos últimos 3 meses.",
                    ", ".join(f"{a}: {b:.0f}%" for a, b in sorted(hot.items(), key=lambda x: -x[1])),
                    "Reforçar ou redistribuir trabalho desta função antes de aumentar volume; verificar se a espera antes das suas tarefas está a crescer.",
                    peso=0.6 * ref, nivel='risco', sim={'name': f"+1 {k}", 'people': {k: 1}})
        if caps_ok({'RECURSOS'}, caps) and not A.empty:
            roles = R[R['resource_id'].isin(A['resource_id'].unique())]['resource_type'].value_counts()
            single = list(roles[roles <= 1].index)
            if single:
                add('recursos', "Funções dependentes de uma só pessoa", f"{len(single)} funções têm apenas uma pessoa: ausências param o processo.",
                    ", ".join(map(str, single[:5])),
                    "Formar uma segunda pessoa (backup) ou definir delegação formal para estas funções.", peso=0.35 * ref, nivel='risco')
    except Exception: pass

    items = sorted(items, key=lambda x: -(x['_peso'] or 0))[:max_items]
    for i in items: i.pop('_peso', None)
    return items


def build_process_facts(V, caps, report=None, anonymize=True):
    """Resumo estruturado (JSON) de tudo o que a app calculou. É isto que vai para o modelo de IA — números, não imagens."""
    P, T, A, R = V['P'], V['T'], V['A'], V['R']
    amap = _anon_map(V) if anonymize else {}
    nm = lambda x: amap.get(x, x)
    has = lambda *c: caps_ok(set(c), caps)
    F = {'ambito': {'processos': int(len(P)), 'periodo': f"{P['start_date'].min():%Y-%m} a {P['end_date'].max():%Y-%m}",
                    'tipos_de_processo': {str(k): int(v) for k, v in P['path_name'].value_counts().head(8).items()} if 'path_name' in P.columns else None,
                    'atividades': int(T['task_name'].nunique()),
                    'dados_disponiveis': [CAPS_INFO[c][0] for c in CAPS_INFO if caps is None or c in caps],
                    'dados_em_falta': [CAPS_INFO[c][0] for c in CAPS_INFO if caps is not None and c not in caps]}}
    k = {}
    if has('TEMPOS') or True:
        k['duracao_media_dias'] = _r(P['lead_time'].mean()); k['duracao_mediana_dias'] = _r(P['lead_time'].median()); k['duracao_p90_dias'] = _r(P['lead_time'].quantile(0.9))
        c, p = _recent_vs_prev(P, 'lead_time'); k['duracao_ultimos_3m_vs_3m_anteriores'] = [_r(c), _r(p)] if c is not None else None
    k['variantes'] = int(P['variant'].nunique()); k['pct_caminho_principal'] = _r(P['happy_path'].mean() * 100); k['pct_com_retrabalho'] = _r(P['has_rework'].mean() * 100)
    if has('TEMPOS'):
        k['espera_media_por_processo_dias'] = _r(P['wait_total'].mean()); k['pct_tempo_em_espera'] = _r((P['wait_total'] / P['lead_time'].replace(0, np.nan)).mean() * 100)
    if has('PLANO'):
        k['pct_atrasados'] = _r((P['days_diff'] > 0).mean() * 100); k['desvio_medio_prazo_dias'] = _r(P['days_diff'].mean())
        c, p = _recent_vs_prev(P.assign(_l=(P['days_diff'] > 0).astype(float) * 100), '_l'); k['pct_atrasados_ultimos_3m_vs_anteriores'] = [_r(c), _r(p)] if c is not None else None
    if has('CUSTO'):
        k['custo_medio_processo_eur'] = _r(P['total_actual_cost'].mean(), 0); k['custo_total_eur'] = _r(A['cost'].sum(), 0)
        c, p = _recent_vs_prev(P, 'total_actual_cost'); k['custo_medio_ultimos_3m_vs_anteriores'] = [_r(c, 0), _r(p, 0)] if c is not None else None
        if has('TEMPOS') and 'cost_per_day' in P.columns: k['custo_estimado_da_espera_eur'] = _r(P['wait_total'].sum() * P['cost_per_day'].mean(), 0)
    if has('ORCAMENTO') and has('CUSTO'):
        k['pct_acima_orcamento'] = _r((P['cost_diff'] > 0).mean() * 100); k['desvio_medio_orcamento_eur'] = _r(P['cost_diff'].mean(), 0)
    F['indicadores'] = k
    try:
        if 'path_name' in P.columns and P['path_name'].nunique() > 1:
            agg = {'processos': ('project_id', 'count'), 'duracao_media': ('lead_time', 'mean'), 'retrabalho_pct': ('has_rework', 'mean')}
            if has('PLANO'): agg['atraso_pct'] = ('days_diff', lambda x: (x > 0).mean())
            if has('CUSTO'): agg['custo_medio'] = ('total_actual_cost', 'mean')
            g = P.groupby('path_name').agg(**agg)
            for c in ['retrabalho_pct', 'atraso_pct']:
                if c in g: g[c] *= 100
            F['por_tipo'] = {str(i): {c: _r(v) for c, v in r.items()} for i, r in g.round(1).iterrows()}
    except Exception: pass
    try:
        vc = P['variant'].value_counts().head(5); med = P.groupby('variant')['lead_time'].median()
        F['variantes_top'] = [{'sequencia': ' > '.join(map(str, v)), 'pct': _r(n / len(P) * 100), 'duracao_mediana': _r(med[v])} for v, n in vc.items()]
        if has('TEMPOS'):
            lp = rework_loops_df(T)
            if not lp.empty: F['retornos_top'] = lp.head(5).round(1).to_dict('records')
            rw = P[P['has_rework']]
            if len(rw) and len(rw) < len(P): F['dias_extra_quando_ha_retrabalho'] = _r(rw['lead_time'].mean() - P[~P['has_rework']]['lead_time'].mean())
    except Exception: pass
    if has('TEMPOS'):
        try:
            a = T.groupby('task_name').agg(execucao=('dur_days', 'mean'), espera_antes=('wait_days', 'mean'), p90_execucao=('dur_days', lambda x: x.quantile(0.9)), execucoes=('task_id', 'size'))
            F['atividades'] = {str(i): {c: _r(v) for c, v in r.items()} for i, r in a.sort_values('espera_antes', ascending=False).iterrows()}
            e = T.dropna(subset=['prev_task']); e = e[~e['repeat']]
            h = e.groupby(['prev_task', 'task_name']).agg(espera_media=('wait_days', 'mean'), processos=('project_id', 'nunique'), dias_totais=('wait_days', 'sum')).reset_index().sort_values('dias_totais', ascending=False).head(5)
            F['passagens_com_mais_espera'] = [{'de': r.prev_task, 'para': r.task_name, 'espera_media_dias': _r(r.espera_media), 'processos': int(r.processos), 'dias_totais': _r(r.dias_totais, 0)} for r in h.itertuples()]
        except Exception: pass
    if has('RECURSOS'):
        try:
            rec = {}
            roles = A.groupby('resource_type').agg(pessoas=('resource_id', 'nunique'), horas=('hours_worked', 'sum'))
            if has('ESFORCO'): roles['pct_horas'] = roles['horas'] / roles['horas'].sum() * 100
            util = _role_utilization_recent(A, R) if has('ESFORCO') else {}
            rec['por_funcao'] = {str(i): {'pessoas': int(r['pessoas']), **({'pct_horas': _r(r['pct_horas'])} if 'pct_horas' in r else {}),
                                         **({'utilizacao_ultimos_3m_pct': _r(util.get(i))} if util else {})} for i, r in roles.iterrows()}
            if has('ESFORCO'):
                hw = A.groupby('resource_id')['hours_worked'].sum().sort_values(ascending=False)
                rec['pct_horas_feitas_por_20pct_das_pessoas'] = _r(hw.head(max(1, int(len(hw) * 0.2))).sum() / hw.sum() * 100)
            if has('TEMPOS'):
                d = T.dropna(subset=['resource_name']).copy(); d['nw'] = d.groupby('project_id')['wait_days'].shift(-1)
                g = d.groupby('resource_name').agg(w=('nw', 'mean'), n=('task_id', 'size')); g = g[g['n'] >= 10].sort_values('w', ascending=False).head(5)
                rec['pessoas_apos_as_quais_ha_mais_espera'] = [{'pessoa': nm(i), 'espera_seguinte_dias': _r(r['w']), 'tarefas': int(r['n'])} for i, r in g.iterrows()]
            F['recursos'] = rec
        except Exception: pass
    if has('CUSTO'):
        try:
            F['custos'] = {'por_funcao_pct': {str(i): _r(v) for i, v in (A.groupby('resource_type')['cost'].sum() / A['cost'].sum() * 100).sort_values(ascending=False).items()},
                           'por_atividade_pct': {str(i): _r(v) for i, v in (A.groupby('task_name')['cost'].sum() / A['cost'].sum() * 100).sort_values(ascending=False).items()}}
        except Exception: pass
    F['prioridades_calculadas'] = compute_priorities(V, caps)
    if report:
        eng = []
        for key, area in [('resumo_executivo', 'resumo'), ('diagnostico_custos_atrasos', 'custos'), ('diagnostico_gargalos_esperas', 'tempos'),
                          ('diagnostico_recursos_equipas', 'recursos'), ('diagnostico_fluxo_conformidade', 'fluxo')]:
            for it in report.get(key, []) or []:
                if isinstance(it, dict) and 'titulo' in it:
                    det = it['detalhe']
                    for a, b in amap.items(): det = det.replace(a, b)
                    eng.append({'area': area, 'tipo': it.get('level'), 'titulo': it['titulo'], 'detalhe': det})
        F['observacoes_motor_regras'] = eng
    return F


AI_SYSTEM = """És um consultor sénior de melhoria de processos (process mining) a escrever para um diretor que NÃO vai ver os gráficos.
Recebes FACTOS em JSON, calculados a partir do histórico real de processos. Regras:
- Usa APENAS números que estejam nos factos. Não inventes valores, nomes nem causas como se fossem certezas.
- Uma causa que não esteja nos dados é uma hipótese: escreve-a como tal ("provavelmente", "a confirmar").
- Português de Portugal, linguagem de negócio, frases curtas, sem jargão técnico (não uses "DFG", "fitness", "variante" sem explicar).
- Cada prioridade tem de ser concreta e acionável (quem faz o quê). Ordena por impacto. Usa 'prioridades_calculadas' como ponto de partida, mas podes juntar, reordenar ou acrescentar se os factos o justificarem.
- O ganho estimado só pode usar números dos factos e deve dizer o pressuposto (ex.: "reduzir esta espera para metade ≈ 1 000 dias/ano").
- Se os dados não permitem uma conclusão, diz isso em 'limitacoes_dados'.
Responde APENAS com JSON válido, sem texto à volta, neste esquema:
{"veredicto": "1 frase, máx. 20 palavras",
 "estado": "saudavel" | "atencao" | "critico",
 "resumo": "3 a 5 frases: o que se passa, o que mais pesa, o que fazer primeiro",
 "prioridades": [{"titulo": "...", "area": "fluxo|tempos|recursos|custos", "problema": "1 frase", "evidencia": "números dos factos",
                  "causa_provavel": "hipótese", "acao": "o que fazer", "ganho_estimado": "...", "esforco": "baixo|médio|alto", "confianca": "alta|média|baixa"}],
 "pontos_fortes": ["..."], "a_vigiar": ["..."], "limitacoes_dados": ["..."], "perguntas_para_o_negocio": ["3 perguntas que os dados não respondem e que ajudariam a decidir"]}"""

AI_CHAT_SYSTEM = """És um analista de processos. Respondes a perguntas sobre um processo com base APENAS nos FACTOS em JSON abaixo.
Português de Portugal, resposta curta (máx. 6 frases ou uma lista curta), cita os números. Se os factos não chegam para responder, diz o que faltaria e em que secção da app procurar
(Visão Geral, Fluxo e Variantes, Tempos e Esperas, Recursos e Equipas, Custos e Prazos). Não inventes."""


def _llm_settings():
    sec = {}
    try:
        sec = {k: st.secrets[k] for k in st.secrets}
    except Exception:
        pass
    get = lambda k: st.session_state.get(f'llm_{k}') or sec.get(k.upper()) or os.environ.get(k.upper())
    keys = {'Anthropic (Claude)': st.session_state.get('llm_key_anthropic') or sec.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'),
            'Google (Gemini)': st.session_state.get('llm_key_gemini') or sec.get('GEMINI_API_KEY') or sec.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')}
    prov = get('provider') or next((p for p in ['Google (Gemini)', 'Anthropic (Claude)'] if keys[p]), 'Google (Gemini)')
    if prov not in keys: prov = 'Anthropic (Claude)' if 'anthropic' in str(prov).lower() or 'claude' in str(prov).lower() else 'Google (Gemini)'
    default_model = st.session_state.get(f'auto_model_{prov}') or {'Anthropic (Claude)': 'claude-sonnet-5-5', 'Google (Gemini)': 'gemini-2.5-flash'}[prov]
    user_model = st.session_state.get('llm_model') or sec.get('ANTHROPIC_MODEL' if prov.startswith('Anthropic') else 'GEMINI_MODEL')
    model = user_model or default_model
    key = keys[prov]
    bad = bool(key) and st.session_state.get('llm_bad_key') == key
    return {'provider': prov, 'key': None if bad else key, 'model': model, 'bad_key': bad, 'auto': not user_model}


class LLMError(Exception):
    pass


def _llm_friendly(status, txt, provider):
    t = (txt or '').lower()
    if status in (401, 403) or 'api key not valid' in t or 'api_key_invalid' in t or 'api key' in t or 'api_key' in t or 'authentication' in t or 'permission' in t:
        return 'chave', f"A chave de API do {provider} não é válida (ou foi revogada). Confirme a chave em Configurar IA ou nos Secrets."
    if any(w in t for w in ['max retries', 'connection', 'proxy', 'timed out', 'name resolution']) and not status:
        return 'rede', "Não foi possível ligar ao serviço de IA (sem acesso à internet a partir do servidor, ou serviço em baixo). Tente novamente dentro de instantes."
    if status == 429 or 'quota' in t or 'rate limit' in t or 'resource_exhausted' in t or 'resource has been exhausted' in t:
        return 'limite', "Limite de utilização atingido (quota ou demasiados pedidos). Aguarde um minuto ou verifique o saldo/plano da conta."
    if status == 404 or ('model' in t and 'not found' in t):
        return 'modelo', "O modelo de IA indicado já não está disponível. Apague o que estiver escrito no campo Modelo e carregue em Testar ligação: a app escolhe automaticamente um modelo disponível."
    if status in (400,) and ('credit' in t or 'billing' in t or 'balance' in t):
        return 'saldo', "A conta não tem saldo/crédito disponível. Carregue saldo na consola do fornecedor."
    if (status and status >= 500) or 'overloaded' in t:
        return 'servico', "O serviço de IA está temporariamente indisponível. Tente novamente dentro de instantes."
    return 'outro', "Não foi possível contactar o serviço de IA. Tente novamente; se persistir, verifique a configuração."


def call_llm(system, messages, json_mode=False, max_tokens=3000):
    """messages: [{'role': 'user'|'assistant', 'content': str}]. Devolve texto. Erros -> LLMError com mensagem clara."""
    cfg = _llm_settings()
    if not cfg['key']:
        raise LLMError("Não há chave de API configurada.")
    try:
        return _call_llm_raw(cfg, system, messages, json_mode, max_tokens)
    except LLMError:
        raise
    except Exception as e:
        status = getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None) or getattr(e, 'code', None)
        try: status = int(status)
        except (TypeError, ValueError): status = None
        kind, msg = _llm_friendly(status, str(e), cfg['provider'])
        if kind == 'chave': st.session_state['llm_bad_key'] = cfg['key']
        print(f"[IA] erro {status}: {e}")
        raise LLMError(msg)


def _show_llm_error(e):
    st.warning(str(e))
    d = st.session_state.get('llm_last_detail')
    if d: st.caption(f"Detalhe técnico (para suporte): {d}")


def _ranked_models(cfg):
    """Modelos disponíveis na conta, do mais adequado para o menos (os nomes mudam com frequência)."""
    import requests, re
    ver = lambda m: [float(x) for x in re.findall(r'\d+(?:\.\d+)?', m.replace('-', ' ').replace('_', ' '))[:2]] + [0 if 'preview' in m else 1]
    try:
        if cfg['provider'].startswith('Google'):
            r = requests.get('https://generativelanguage.googleapis.com/v1beta/models?pageSize=200', headers={'x-goog-api-key': cfg['key']}, timeout=30)
            ms = [m['name'].split('/')[-1] for m in r.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
            bad = ('image', 'tts', 'live', 'audio', 'embedding', 'learnlm', 'gemma', 'robotics', 'computer', 'thinking')
            ms = [m for m in ms if m.startswith('gemini') and not any(b in m for b in bad)]
            flash = sorted([m for m in ms if 'flash' in m and 'lite' not in m], key=ver, reverse=True)
            lite = sorted([m for m in ms if 'flash' in m and 'lite' in m], key=ver, reverse=True)
            other = sorted([m for m in ms if 'flash' not in m], key=ver, reverse=True)
            return flash + lite + other
        r = requests.get('https://api.anthropic.com/v1/models?limit=100', headers={'x-api-key': cfg['key'], 'anthropic-version': '2023-06-01'}, timeout=30)
        ms = [m['id'] for m in r.json().get('data', [])]
        return sorted([m for m in ms if 'sonnet' in m], key=ver, reverse=True) + sorted([m for m in ms if 'haiku' in m], key=ver, reverse=True)
    except Exception as e:
        print(f"[IA] não foi possível listar modelos: {e}")
        return []


def _call_llm_raw(cfg, system, messages, json_mode, max_tokens):
    """Tenta o modelo configurado; se estiver indisponível, sobrecarregado ou sem quota, tenta outros modelos da conta."""
    tried, last = [], None
    queue = [cfg['model']]
    listed = False
    while queue and len(tried) < 5:
        m = queue.pop(0)
        if m in tried: continue
        tried.append(m)
        for attempt in range(2):
            try:
                out = _call_llm_once({**cfg, 'model': m}, system, messages, json_mode, max_tokens)
                if m != cfg['model']: st.session_state[f"auto_model_{cfg['provider']}"] = m
                st.session_state.pop('llm_last_detail', None)
                return out
            except LLMError as e:
                last = e; kind = getattr(e, 'kind', None)
                if kind in ('servico', 'limite') and attempt == 0:
                    time.sleep(2); continue
                break
        if getattr(last, 'kind', None) in ('chave', 'saldo'): break
        if cfg.get('auto') and not listed:
            queue += [x for x in _ranked_models(cfg) if x not in tried]; listed = True
    raise last


def _call_llm_once(cfg, system, messages, json_mode, max_tokens):
    if cfg['provider'].startswith('Anthropic'):
        import requests
        r = requests.post('https://api.anthropic.com/v1/messages', timeout=180,
                          headers={'x-api-key': cfg['key'], 'anthropic-version': '2023-06-01', 'content-type': 'application/json'},
                          json={'model': cfg['model'], 'max_tokens': max_tokens, 'system': system, 'messages': messages})
        if r.status_code != 200:
            kind, msg = _llm_friendly(r.status_code, r.text, cfg['provider'])
            if kind == 'chave': st.session_state['llm_bad_key'] = cfg['key']
            print(f"[IA] erro {r.status_code}: {r.text[:300]}")
            err = LLMError(msg); err.kind = kind; st.session_state['llm_last_detail'] = f"{cfg['model']} · HTTP {r.status_code} · {r.text[:400]}"; raise err
        return ''.join(b.get('text', '') for b in r.json().get('content', []) if b.get('type') == 'text')
    # Gemini por REST (com timeout e erros HTTP claros; a biblioteca google-generativeai está descontinuada)
    import requests
    body = {'systemInstruction': {'parts': [{'text': system}]},
            'contents': [{'role': 'model' if m['role'] == 'assistant' else 'user', 'parts': [{'text': m['content']}]} for m in messages],
            'generationConfig': {'maxOutputTokens': max_tokens, **({'responseMimeType': 'application/json'} if json_mode else {})}}
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent", timeout=120,
                      headers={'x-goog-api-key': cfg['key'], 'content-type': 'application/json'}, json=body)
    if r.status_code != 200:
        kind, msg = _llm_friendly(r.status_code, r.text, cfg['provider'])
        if kind == 'chave' or (r.status_code == 400 and 'api key' in r.text.lower()): st.session_state['llm_bad_key'] = cfg['key']
        print(f"[IA] erro {r.status_code}: {r.text[:300]}")
        err = LLMError(msg); err.kind = kind; st.session_state['llm_last_detail'] = f"{cfg['model']} · HTTP {r.status_code} · {r.text[:400]}"; raise err
    cand = (r.json().get('candidates') or [{}])[0]
    txt = ''.join(pt.get('text', '') for pt in cand.get('content', {}).get('parts', []))
    if not txt:
        raise LLMError("A IA não devolveu texto (possível bloqueio de conteúdo ou limite de tamanho). Tente novamente.")
    return txt


def _parse_json(txt):
    t = txt.strip()
    if t.startswith('```'): t = t.split('\n', 1)[1].rsplit('```', 1)[0]
    i, j = t.find('{'), t.rfind('}')
    return json.loads(t[i:j + 1])


def deterministic_reading(report, priorities):
    """Leitura sem IA, no mesmo formato da resposta do modelo (para a app ser útil mesmo sem chave)."""
    res = (report or {}).get('resumo_executivo', []) or []
    main = next((x for x in res if isinstance(x, dict) and x.get('level') in ('problema', 'facto')), None)
    estado = 'saudavel' if main and main.get('level') == 'facto' else ('critico' if len([p for p in priorities if p.get('nivel') != 'risco']) >= 3 else 'atencao')
    if main:
        verdict = main['titulo'].replace('Diagnóstico Principal:', '').strip(); resumo = main['detalhe']
    else:
        verdict = 'Leitura automática dos indicadores'; resumo = ''
    prios = [dict(titulo=p['titulo'], area=p['area'], problema=p['problema'], evidencia=p['evidencia'], acao=p['acao'],
                  ganho_estimado=(_fmt_eur(p['impacto_eur']) + ' de impacto estimado' if p.get('impacto_eur') else
                                  (f"{p['impacto_dias']:,.0f} dias acumulados".replace(',', ' ') if p.get('impacto_dias') else None)),
                  nivel=p.get('nivel'), sim=p.get('sim')) for p in priorities]
    fortes = [x['titulo'] for k in ['diagnostico_custos_atrasos', 'diagnostico_fluxo_conformidade', 'diagnostico_gargalos_esperas', 'diagnostico_recursos_equipas']
              for x in (report or {}).get(k, []) or [] if isinstance(x, dict) and x.get('level') == 'recomendacao'][:4]
    return {'veredicto': verdict, 'estado': estado, 'resumo': resumo, 'prioridades': prios, 'pontos_fortes': fortes, 'a_vigiar': [], 'limitacoes_dados': [], 'perguntas_para_o_negocio': []}


def _go_simulate(sim):
    """Abre a Simulação com o cenário da prioridade já preenchido e corre-o automaticamente."""
    st.session_state.current_page = 'RL'
    if sim.get('tab') == 'rl':
        st.session_state.sim_tab = 'rl'; st.session_state.sim_tab_ctl = 'rl'; return
    st.session_state.sim_tab = 'cenarios'; st.session_state.sim_tab_ctl = 'cenarios'
    for k in list(st.session_state.keys()):
        if k.startswith('sc_p_'): st.session_state[k] = 0
    st.session_state.update({'sc_vol': 0, 'sc_rw': sim.get('rework', 0), 'sc_exo_sel': [f"{a} → {b}" for a, b in sim.get('exo', [])],
                             'sc_exo_pct': sim.get('exo_pct', 50), 'sc_dur_sel': list(sim.get('dur', [])), 'sc_dur_pct': sim.get('dur_pct', 20),
                             'sc_name': sim.get('name', 'Cenário do diagnóstico'), 'sc_autorun': True})
    for r, d in (sim.get('people') or {}).items(): st.session_state[f'sc_p_{r}'] = d


def _attach_sim(ai_prios, calc):
    """Liga as prioridades redigidas pela IA às prioridades calculadas (para o botão Simular)."""
    tok = lambda x: set(re.findall(r'\w{4,}', str(x).lower()))
    for p in ai_prios or []:
        if p.get('sim'): continue
        best, sc = None, 0.0
        for c in calc:
            if not c.get('sim') or str(c.get('area')) != str(p.get('area', '')).lower(): continue
            a, b = tok(p.get('titulo', '') + ' ' + p.get('problema', '')), tok(c['titulo'] + ' ' + c['problema'])
            j = len(a & b) / max(len(a | b), 1)
            if j > sc: best, sc = c, j
        if best and sc >= 0.12: p['sim'] = best['sim']
    return ai_prios


def sim_facts():
    """Resultados da Simulação (se já foram corridos) em formato compacto, para a IA e para o resumo executivo."""
    out = {}
    SV = st.session_state.get('sim_validation')
    if SV:
        ws = SV['wait_split']; tot = ((ws['fila'] + ws['estrutural']) * ws['n']).sum()
        out['modelo'] = {'erro_medio_pct': _r(SV['score']), 'pct_espera_por_falta_capacidade': _r((ws['fila'] * ws['n']).sum() / tot * 100) if tot else None,
                         'ocupacao_por_funcao_pct': {k: _r(v, 0) for k, v in SV['util'].items()}}
    B = st.session_state.get('sim_base_summary'); lst = st.session_state.get('sim_scen_list') or []
    if B and lst:
        def kp(S): return {k: _r(S.get(k)) for k in ['lead', 'p90', 'wait', 'late', 'cost', 'umax'] if S.get(k) is not None}
        def _clean(pr):
            o = {k: v for k, v in pr.items() if v and k not in ('people', 'exo')}
            if pr.get('exo'): o['reduzir_espera_pct'] = {f"{a} → {b}": v for (a, b), v in pr['exo'].items()}
            return o
        out['cenarios'] = {'base': kp(B), 'testados': [{'nome': n, 'parametros': _clean(pr) |
                                                        ({'pessoas': {r: d for r, d in pr['people'].items() if d}} if any((pr.get('people') or {}).values()) else {}),
                                                        'resultado': kp(S_)} for n, pr, S_ in lst]}
        out['cenarios']['legenda'] = 'lead=duração média (dias), p90=duração P90, wait=espera por processo, late=% atrasados, cost=custo mão-de-obra/processo (€), umax=ocupação máxima de uma função (%)'
    RR = st.session_state.get('rl_result')
    if RR:
        R_, ev, rules = RR; nm = list(ev)
        out['otimizacao_rl'] = {'objetivo': RL_OBJECTIVES[R_['kind']], 'volume_pct': R_['volume'], 'politica_escolhida': R_.get('chosen'),
                                'resultados': {n: {'objetivo': _r(ev[n]['obj']), 'duracao': _r(ev[n]['lead']), 'atrasados_pct': _r(ev[n].get('late'))} for n in nm},
                                'regras': [f"{t}: {d}" for t, d, _ in rules]}
    return out


DIRECTOR_SYSTEM = """És um consultor de operações. Com base APENAS nos resultados de simulação em JSON, escreve para um diretor um resumo executivo em português de Portugal:
- 1 frase de conclusão; depois 3 a 5 bullets com a recomendação, o impacto esperado (números do JSON) e o que não compensa fazer;
- termina com 1 linha de riscos/pressupostos. Máx. 170 palavras. Sem jargão técnico. Não inventes números."""


def render_director_summary(key, payload, fallback_lines):
    """Resumo para decisão: por regras (sempre) e redigido por IA (se houver chave)."""
    with st.container(border=True):
        ui_card_header('Resumo para o diretor', 'Conclusão dos testes feitos, pronta a copiar', 'bi-briefcase')
        for l in fallback_lines: st.markdown(f"- {l}")
        cfg = _llm_settings()
        if cfg['key']:
            if st.button("Redigir com IA", icon=":material/auto_awesome:", key=f'dir_{key}'):
                with st.spinner("A redigir..."):
                    try:
                        st.session_state[f'dir_txt_{key}'] = call_llm(DIRECTOR_SYSTEM, [{'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, default=str)}], max_tokens=900)
                    except LLMError as e:
                        _show_llm_error(e)
            if st.session_state.get(f'dir_txt_{key}'):
                st.markdown(st.session_state[f'dir_txt_{key}'])
        else:
            st.caption("Com uma chave de IA (Diagnóstico → Configurar IA) este resumo pode ser redigido automaticamente.")


def _go_section(sec):
    st.session_state.current_section = sec
    st.session_state.nav_sections = sec


def render_reading(rd, source):
    est = rd.get('estado', 'atencao')
    color, icon, lab = {'saudavel': ('#16a34a', 'bi-check-circle-fill', 'Saudável'), 'atencao': ('#d97706', 'bi-exclamation-circle-fill', 'Requer atenção'),
                        'critico': ('#dc2626', 'bi-x-octagon-fill', 'Crítico')}.get(est, ('#d97706', 'bi-exclamation-circle-fill', 'Requer atenção'))
    st.markdown(f"""<div class='pm-verdict' style='border-left-color:{color}'>
        <div class='pm-verdict-top'><span class='pm-pill' style='background:{color}1a;color:{color}'><i class='bi {icon}'></i> {lab}</span>
        <span class='pm-src'>{html.escape(source)}</span></div>
        <div class='pm-verdict-title'>{html.escape(str(rd.get('veredicto', '')))}</div>
        <div class='pm-verdict-text'>{html.escape(str(rd.get('resumo', '')))}</div></div>""", unsafe_allow_html=True)
    pr = rd.get('prioridades') or []
    if pr:
        st.markdown("<div class='pm-h5'>O que fazer primeiro</div>", unsafe_allow_html=True)
    for i, p in enumerate(pr):
        area = str(p.get('area', '')).lower()
        sec = AREA_SECTION.get(area)
        with st.container(border=True):
            c1, c2 = st.columns([5, 1.5], vertical_alignment='top')
            meta = []
            if p.get('nivel') == 'risco': meta.append("<span class='pm-tag risk'>Risco</span>")
            if p.get('esforco'): meta.append(f"<span class='pm-tag'>Esforço {html.escape(str(p['esforco']))}</span>")
            if p.get('confianca'): meta.append(f"<span class='pm-tag'>Confiança {html.escape(str(p['confianca']))}</span>")
            rows = [('Evidência', p.get('evidencia')), ('Causa provável', p.get('causa_provavel')), ('Ação', p.get('acao'))]
            body = ''.join(f"<div class='pm-prio-row'><span>{k}</span><div>{html.escape(str(v))}</div></div>" for k, v in rows if v)
            with c1:
                st.markdown(f"""<div class='pm-prio'><div class='pm-prio-num'>{i + 1}</div><div style='flex:1'>
                    <div class='pm-prio-title'><i class='bi {AREA_ICON.get(area, "bi-flag")}'></i> {html.escape(str(p.get('titulo', '')))} {''.join(meta)}</div>
                    <div class='pm-prio-problem'>{html.escape(str(p.get('problema', '')))}</div>{body}</div></div>""", unsafe_allow_html=True)
            with c2:
                if p.get('ganho_estimado'):
                    st.markdown(f"<div class='pm-gain'><div class='pm-gain-lab'>Impacto / ganho</div><div class='pm-gain-val'>{html.escape(str(p['ganho_estimado']))}</div></div>", unsafe_allow_html=True)
                if p.get('sim'):
                    st.button("Simular" if p['sim'].get('tab') != 'rl' else "Otimizar prioridades", key=f"sim_{source[:3]}_{i}", on_click=_go_simulate, args=(p['sim'],),
                              use_container_width=True, type='primary', icon=":material/science:",
                              help="Abre o simulador com este cenário já preenchido e mostra o impacto esperado.")
                if sec:
                    st.button(f"Ver em {sec[1]}", key=f"go_{source[:3]}_{i}", on_click=_go_section, args=(sec[0],), use_container_width=True, icon=":material/arrow_forward:")
    cols_data = [('Pontos fortes', rd.get('pontos_fortes'), 'bi-hand-thumbs-up', '#16a34a'), ('A vigiar', rd.get('a_vigiar'), 'bi-eye', '#d97706'),
                 ('Perguntas para o negócio', rd.get('perguntas_para_o_negocio'), 'bi-question-circle', '#2563eb')]
    cols_data = [c for c in cols_data if c[1]]
    if cols_data:
        cols = st.columns(len(cols_data))
        for col, (t, lst, ic, cl) in zip(cols, cols_data):
            with col, st.container(border=True):
                st.markdown(f"<div class='pm-list-title' style='color:{cl}'><i class='bi {ic}'></i> {t}</div>"
                            + ''.join(f"<div class='pm-list-item'>{html.escape(str(x))}</div>" for x in lst), unsafe_allow_html=True)
    if rd.get('limitacoes_dados'):
        ui_note("Limitações dos dados: " + " · ".join(map(str, rd['limitacoes_dados'])))


def get_diag_report():
    if st.session_state.get('diag_report') is None:
        try:
            eng = DiagnosticEngineV5(st.session_state.get('tables_pre_mining', {}), st.session_state.get('metrics', {}),
                                     st.session_state.get('data_frames_processed', {}), st.session_state.get('tables_eda', {}), caps=st.session_state.get('caps'))
            st.session_state.diag_report = eng.run()
        except Exception:
            st.session_state.diag_report = {}
    return st.session_state.diag_report


def render_ai_section(caps):
    """Separador Diagnóstico: leitura para decisão (IA ou regras), perguntas à IA e detalhe do motor de regras."""
    V = st.session_state.view
    report = get_diag_report()
    ui_section_intro("O que se passa e o que fazer?", "Leitura para decisão: o veredicto, as prioridades ordenadas por impacto e onde ver a evidência. Usa sempre todos os dados (ignora filtros).")
    cfg = _llm_settings()
    anon = True  # nomes de pessoas são sempre substituídos pela função (ex.: "Analista de Risco #3")
    facts = build_process_facts(V, caps, report, anonymize=anon)
    sf = sim_facts()
    if sf: facts['simulacao'] = sf
    facts_json = json.dumps({**facts, 'prioridades_calculadas': [{k: v for k, v in p.items() if k != 'sim'} for p in facts['prioridades_calculadas']]},
                            ensure_ascii=False, default=str)
    sig = hash(facts_json + cfg['provider'] + cfg['model'])

    c1, c2, c3 = st.columns([3.2, 1.4, 1.4], vertical_alignment='center')
    with c1:
        if cfg['key']:
            st.caption(f"IA: {cfg['provider']} · {cfg['model']} · envia apenas números agregados, sem nomes de pessoas")
        else:
            if cfg.get('bad_key'):
                st.markdown("<div class='pm-note' style='border-color:#fecaca;background:#fef2f2;color:#991b1b'><i class='bi bi-key'></i> A chave de IA configurada foi recusada pelo fornecedor. "
                            "A leitura abaixo é feita por regras. Corrija a chave em <b>Configurar IA</b> (fundo da página) ou use <b>Exportar para IA externa</b>.</div>", unsafe_allow_html=True)
            else:
                st.caption("Sem IA configurada: a leitura abaixo é feita por regras (gratuita). Para texto redigido e perguntas, configure uma chave ou use «Exportar para IA externa».")
    with c2:
        gen = st.button("Gerar leitura com IA" if st.session_state.get('ai_reading_sig') != sig else "Gerar novamente", icon=":material/auto_awesome:",
                        type="primary", use_container_width=True, disabled=not cfg['key'], key='ai_gen')
    with c3:
        st.download_button("Exportar para IA externa", data=(AI_SYSTEM + "\n\nFACTOS:\n" + json.dumps(facts, ensure_ascii=False, indent=1, default=str)).encode('utf-8'),
                           file_name="factos_processo_para_IA.txt", mime="text/plain", use_container_width=True, icon=":material/download:", key='ai_export',
                           help="Prompt + factos num ficheiro de texto, para colar em qualquer assistente de IA (ChatGPT, Gemini, Claude).")
    if gen:
        with st.spinner("A IA está a ler os factos do processo..."):
            try:
                txt = call_llm(AI_SYSTEM, [{'role': 'user', 'content': "FACTOS:\n" + facts_json}], json_mode=True, max_tokens=4000)
                rd_ = _parse_json(txt); rd_['prioridades'] = _attach_sim(rd_.get('prioridades'), facts['prioridades_calculadas'])
                st.session_state.ai_reading = rd_; st.session_state.ai_reading_sig = sig
            except LLMError as e:
                _show_llm_error(e)
                if st.session_state.get('llm_bad_key'): st.rerun()
            except (ValueError, KeyError):
                st.warning("A IA respondeu num formato inesperado. Carregue em «Gerar novamente».")

    if st.session_state.get('ai_reading') and st.session_state.get('ai_reading_sig') == sig:
        render_reading(st.session_state.ai_reading, f"Redigido por IA ({cfg['provider']}) a partir dos factos calculados")
    else:
        render_reading(deterministic_reading(report, facts['prioridades_calculadas']), "Leitura automática por regras (sem IA)")

    with st.container(border=True):
        ui_card_header('Perguntar sobre o processo', 'Ex.: "Porque é que os processos CH_Reforço são mais caros?" · "Que função devo reforçar primeiro?"', 'bi-chat-dots')
        if not cfg['key']:
            st.caption("Disponível com uma chave de IA válida (Configurar IA, abaixo). Sem chave: use «Exportar para IA externa» e faça as perguntas no seu assistente habitual.")
        else:
            hist = st.session_state.setdefault('ai_chat', [])
            for m in hist:
                with st.chat_message(m['role']): st.markdown(m['content'])
            q = st.chat_input("Escreva a sua pergunta", key='ai_chat_input')
            if q:
                hist.append({'role': 'user', 'content': q})
                with st.chat_message('user'): st.markdown(q)
                try:
                    with st.spinner("A pensar..."):
                        ans = call_llm(AI_CHAT_SYSTEM + "\n\nFACTOS:\n" + facts_json, hist[-10:], max_tokens=1200)
                    with st.chat_message('assistant'): st.markdown(ans)
                    hist.append({'role': 'assistant', 'content': ans})
                except LLMError as e:
                    hist.pop()  # a pergunta não fica no histórico sem resposta
                    _show_llm_error(e)
                    if st.session_state.get('llm_bad_key'): time.sleep(2.5); st.rerun()
            if hist and st.button("Limpar conversa", key='ai_chat_clear'):
                st.session_state.ai_chat = []; st.rerun()

    with st.expander("Configurar IA"):
        st.markdown("**Opção gratuita (Gemini):** crie uma chave grátis em [aistudio.google.com/apikey](https://aistudio.google.com/apikey) com a sua conta Google "
                    "e cole-a abaixo. Para ficar permanente, guarde-a nos *Secrets* da app no Streamlit Cloud (Settings → Secrets):\n"
                    "```toml\nGEMINI_API_KEY = \"a-sua-chave\"\n```")
        a, b = st.columns(2)
        prov = a.selectbox("Fornecedor", ['Google (Gemini)', 'Anthropic (Claude)'], index=0 if cfg['provider'].startswith('Google') else 1, key='llm_provider')
        b.text_input("Modelo (vazio = predefinido)", key='llm_model',
                     placeholder='claude-sonnet-5-5' if prov.startswith('Anthropic') else 'gemini-2.5-flash')
        st.text_input("Chave de API (só esta sessão)", type="password", key='llm_key_anthropic' if prov.startswith('Anthropic') else 'llm_key_gemini')
        if st.button("Testar ligação", icon=":material/wifi_tethering:", key='llm_test'):
            st.session_state.pop('llm_bad_key', None)
            try:
                call_llm("Responde apenas: OK", [{'role': 'user', 'content': 'teste'}], max_tokens=10)
                st.success(f"Ligação OK ({_llm_settings()['provider']} · {_llm_settings()['model']}).")
            except LLMError as e:
                _show_llm_error(e)

    with st.expander("Diagnóstico detalhado (motor de regras): todos os factos e alertas por área"):
        render_diagnostics_page()


def _scroll_top_if_moved():
    """Ao mudar de página ou de secção, volta ao topo (o Streamlit mantém o scroll anterior)."""
    loc = (st.session_state.get('current_page'), st.session_state.get('current_section'))
    if st.session_state.get('_last_loc') != loc:
        st.session_state['_last_loc'] = loc
        _embed = getattr(st, 'iframe', None)
        _js = ("""<script>
        const go = () => { const d = window.parent.document;
          for (const sel of ['[data-testid="stMain"]', 'section.main', '[data-testid="stAppViewContainer"]']) {
            const el = d.querySelector(sel); if (el) el.scrollTo({top: 0, behavior: 'instant'}); }
          window.parent.scrollTo(0, 0); };
        go(); setTimeout(go, 150); setTimeout(go, 600);
        </script>""")
        if _embed: _embed(_js, height=1)
        else: components.html(_js, height=0)


def dashboard_page():

    ### START PDF/AI STATE INIT (if not global) ###
    if 'show_ai_modal' not in st.session_state: st.session_state.show_ai_modal = False
    if 'gemini_analysis' not in st.session_state: st.session_state.gemini_analysis = None
    if 'pdf_bytes_ready' not in st.session_state: st.session_state.pdf_bytes_ready = None
    ### END PDF/AI STATE INIT ###

    ### START PDF/AI BUTTONS & MODAL ###
    if st.session_state.analysis_run:
        col_buffer, col_pdf = st.columns([6, 1.4], vertical_alignment='center')
        with col_buffer:
            _p = st.session_state.data_frames_processed.get('projects', pd.DataFrame())
            _sub = (f"{len(_p):,} processos · {pd.to_datetime(_p['start_date']).min():%m/%Y} a {pd.to_datetime(_p['end_date']).max():%m/%Y}"
                    if len(_p) else "Análise do histórico de processos")
            ui_page_header("Análise de Processos", _sub.replace(',', ' '))
    
        with col_pdf:
            # Flag para controlar se o botão de download deve ser mostrado
            show_download = st.session_state.get('pdf_ready_for_download', False)

            # Botão para INICIAR a geração e PREPARAR o download
            if st.button("📄 Gerar PDF", help="Preparar relatório completo para download", use_container_width=True, key="generate_pdf_button_v14"):
                with st.spinner("Gerando PDF... Este processo pode demorar."):
                    try:
                        # Reúne os dados necessários DENTRO do clique
                        plots_pre, tables_pre, plots_post, plots_eda, tables_eda = get_visible_outputs()
                        pdf_bytes_output = generate_pdf_report(plots_pre, tables_pre, plots_post, plots_eda, tables_eda)

                        # Verifica se a geração foi bem-sucedida
                        if pdf_bytes_output and len(pdf_bytes_output) > 100:
                             # Guarda os bytes para o botão de download usar
                             st.session_state.pdf_bytes_for_download_v14 = pdf_bytes_output
                             st.session_state.pdf_ready_for_download = True # Flag para mostrar o botão
                             st.info("PDF pronto. Clique em 'Download PDF'.") # Feedback
                        else:
                             st.error("Falha ao gerar o conteúdo do PDF. Verifique os logs.")
                             st.session_state.pdf_bytes_for_download_v14 = None
                             st.session_state.pdf_ready_for_download = False

                        # Rerun para atualizar a UI e mostrar/esconder o botão de download
                        time.sleep(0.1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Erro durante a geração do PDF: {e}")
                        st.session_state.pdf_bytes_for_download_v14 = None
                        st.session_state.pdf_ready_for_download = False
                        st.rerun()

            # Botão de download condicional - usa os bytes guardados
            if show_download and 'pdf_bytes_for_download_v14' in st.session_state and st.session_state.pdf_bytes_for_download_v14:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_bytes_for_download_v14, # Usa os bytes guardados
                    file_name="relatorio_process_mining.pdf",
                    mime="application/pdf",
                    key="pdf_download_button_conditional_v14",
                    use_container_width=True,
                    # Limpa a flag e os bytes após o clique para esconder o botão
                    on_click=lambda: st.session_state.update({
                        'pdf_bytes_for_download_v14': None,
                        'pdf_ready_for_download': False
                    })
                )
        
    
    if st.session_state.get('show_welcome_message', False):
        st.success(f"Bem-vindo, {st.session_state.user_name}!")
        st.session_state.show_welcome_message = False

    if not st.session_state.analysis_run:
        st.warning("A análise ainda não foi executada. Vá à página de 'Configurações' para carregar os dados e iniciar.")
        return
        
    caps = st.session_state.get('caps')
    # Vista derivada (calculada uma vez por análise) e filtros globais
    if st.session_state.get('view') is None:
        _dfp = st.session_state.data_frames_processed
        _alloc = st.session_state.get('dfs_norm', st.session_state.dfs)['resource_allocations']
        st.session_state.view = prepare_view(_dfp['projects'], _dfp['tasks'], _dfp['resources'], _alloc)
    V, filtered = render_filters(st.session_state.view)

    sections = {"visao_geral": ("Visão Geral", ":material/dashboard:"), "diagnostico": ("Diagnóstico", ":material/lightbulb:"),
                "fluxo": ("Fluxo e Variantes", ":material/account_tree:")}
    if caps_ok({'TEMPOS'}, caps): sections["tempos"] = ("Tempos e Esperas", ":material/schedule:")
    if caps_ok({'RECURSOS'}, caps): sections["recursos"] = ("Recursos e Equipas", ":material/groups:")
    if caps_ok({'CUSTO'}, caps) or caps_ok({'PLANO'}, caps): sections["custos"] = ("Custos e Prazos", ":material/payments:")
    if st.session_state.current_section not in sections: st.session_state.current_section = "visao_geral"
    choice = st.segmented_control("Secção", list(sections), format_func=lambda k: sections[k][0], selection_mode='single',
                                  default=st.session_state.current_section, key='nav_sections', label_visibility='collapsed')
    if choice: st.session_state.current_section = choice

    if caps is not None:
        missing = [CAPS_INFO[c][0] for c in CAPS_INFO if c not in caps]
        if missing:
            ui_note("Algumas análises não aparecem porque os dados não as suportam: " + ", ".join(missing) + ". O detalhe está em Configurações.")
    if filtered:
        ui_note(f"Filtros ativos: {len(V['P']):,} de {len(st.session_state.view['P']):,} processos. O Diagnóstico usa sempre todos os dados.".replace(',', ' '))
    if len(V['P']) == 0:
        st.warning("Nenhum processo corresponde aos filtros escolhidos."); return

    sec = st.session_state.current_section
    if sec == "diagnostico": render_ai_section(caps)
    elif sec == "visao_geral": section_overview(V, caps)
    elif sec == "fluxo": section_flow(V, caps)
    elif sec == "tempos": section_times(V, caps)
    elif sec == "recursos": section_resources(V, caps)
    elif sec == "custos": section_costs(V, caps)


# --- (REINFORCEMENT LEARNING) ---
# =====================================================================================
# --- SIMULADOR DO PROCESSO (eventos discretos) CALIBRADO COM O HISTÓRICO ---
# Base para cenários "e se" e para treinar o agente de RL num mundo realista.
# =====================================================================================
import heapq as _heapq
import itertools as _itertools
from collections import deque as _deque

_START, _END = '__INICIO__', '__FIM__'


def _person_concurrency(T):
    """Nº de tarefas em simultâneo que cada pessoa costuma ter (percentil 90 dos dias em que trabalhou)."""
    d = T.dropna(subset=['resource_name', 'start_date', 'end_date'])[['resource_name', 'resource_type', 'start_date', 'end_date']].copy()
    if d.empty: return pd.DataFrame(columns=['resource_name', 'resource_type', 'conc'])
    d['s'] = d['start_date'].dt.normalize(); d['e'] = d['end_date'].dt.normalize()
    d['n'] = ((d['e'] - d['s']).dt.days + 1).clip(lower=1, upper=120)
    rep = d.loc[d.index.repeat(d['n'])].copy()
    rep['day'] = rep['s'] + pd.to_timedelta(rep.groupby(level=0).cumcount(), unit='D')
    daily = rep.groupby(['resource_name', 'resource_type', 'day']).size().reset_index(name='k')
    out = daily.groupby(['resource_name', 'resource_type'])['k'].quantile(0.9).reset_index(name='conc')
    out['conc'] = np.ceil(out['conc']).clip(lower=1)
    return out


def _bpos(ts, origin):
    """Posição em dias úteis desde a origem (as durações reais dependem dos fins de semana; o relógio do simulador é em dias úteis)."""
    ts = pd.to_datetime(ts)
    d = ts.dt.normalize()
    ok = d.notna()
    out = pd.Series(np.nan, index=ts.index, dtype=float)
    out[ok] = np.busday_count(np.datetime64(origin.date()), d[ok].values.astype('datetime64[D]')) + \
              ((ts[ok] - d[ok]).dt.total_seconds() / 86400).values
    return out


def _bday_to_cal(b, origin):
    """Converte posições em dias úteis para dias de calendário desde a origem."""
    b = np.asarray(b, dtype=float); out = np.full(b.shape, np.nan)
    ok = ~np.isnan(b)
    whole = np.floor(b[ok]).astype(int)
    dates = np.busday_offset(np.datetime64(origin.date()), whole, roll='forward')
    out[ok] = (dates - np.datetime64(origin.date())).astype(int) + (b[ok] - whole)
    return out


def build_sim_model(V, caps):
    """Extrai do histórico tudo o que o simulador precisa (rotas, durações, esperas, funções, capacidade, custos)."""
    P, T, A, R = V['P'], V['T'], V['A'], V['R']
    has_cap = caps_ok({'RECURSOS', 'TEMPOS'}, caps) and T['resource_type'].notna().mean() > 0.5
    has_cost = caps_ok({'CUSTO'}, caps) and not A.empty
    has_eff = caps_ok({'ESFORCO'}, caps) and not A.empty
    T = T.dropna(subset=['start_date', 'end_date']).copy()
    origin = T['start_date'].min().normalize()
    T['bs'] = _bpos(T['start_date'], origin); T['be'] = _bpos(T['end_date'], origin)
    T['bdur'] = (T['be'] - T['bs']).clip(lower=0)
    T['bprev'] = T.groupby('project_id')['be'].shift(1)
    T['bwait'] = (T['bs'] - T['bprev']).clip(lower=0)
    ctype = P.set_index('project_id')['path_name'] if 'path_name' in P.columns else pd.Series(dtype=str)
    T['ctype'] = T['project_id'].map(ctype).fillna('Geral')

    # Rotas: probabilidade da atividade seguinte, por tipo de processo (Markov de 1ª ordem)
    seq = T[['project_id', 'ctype', 'task_name']].copy()
    seq['prev'] = seq.groupby('project_id')['task_name'].shift(1).fillna(_START)
    last = seq.groupby('project_id').tail(1)[['project_id', 'ctype', 'task_name']].rename(columns={'task_name': 'prev'}); last['task_name'] = _END
    trans = pd.concat([seq[['ctype', 'prev', 'task_name']], last[['ctype', 'prev', 'task_name']]])

    def _route_table(df):
        out = {}
        for prev, g in df.groupby('prev'):
            vc = g['task_name'].value_counts()
            out[prev] = (list(vc.index), (vc / vc.sum()).values)
        return out
    routing = {'*': _route_table(trans)}
    for ct, g in trans.groupby('ctype'):
        if g['prev'].eq(_START).sum() >= 20: routing[ct] = _route_table(g)

    # Pares (espera antes, duração) observados juntos: preserva a relação entre ambos (ex.: fins de semana, agendamentos)
    dur = {('*', a): g['bdur'].values for a, g in T.groupby('task_name')}
    for (ct, a), g in T.groupby(['ctype', 'task_name']):
        if len(g) >= 30: dur[(ct, a)] = g['bdur'].values
    wt = T.dropna(subset=['prev_task'])
    pairs = {('*', p, a): g[['bwait', 'bdur']].values for (p, a), g in wt.groupby(['prev_task', 'task_name'])}
    for (ct, p, a), g in wt.groupby(['ctype', 'prev_task', 'task_name']):
        if len(g) >= 30: pairs[(ct, p, a)] = g[['bwait', 'bdur']].values
    exo_act = {a: g['bwait'].values for a, g in wt.groupby('task_name')}
    # 'Processos-molde': cada processo simulado segue os tempos de um processo real do mesmo tipo (preserva a relação entre tarefas do mesmo processo)
    donors = {}
    for (pid, ct), g in T.groupby(['project_id', 'ctype'], sort=False):
        d = {}
        for pv, a, w, du in zip(g['prev_task'].fillna(_START), g['task_name'], g['bwait'].fillna(0), g['bdur']):
            d.setdefault((pv, a), []).append((float(w), float(du)))
        donors.setdefault(ct, []).append((pid, d))
    n_obs = {(p, a): len(g) for (p, a), g in wt.groupby(['prev_task', 'task_name'])}
    obs_wait = {(p, a): float(g['bwait'].mean()) for (p, a), g in wt.groupby(['prev_task', 'task_name'])}

    # Funções e capacidade (nº de tarefas em simultâneo por função)
    roles, slots, people, conc_role, cap_src, util_target = {}, {}, {}, {}, None, {}
    if has_cap:
        for a, g in T.dropna(subset=['resource_type']).groupby('task_name'):
            vc = g['resource_type'].value_counts(); roles[a] = (list(vc.index), (vc / vc.sum()).values)
        pc = _person_concurrency(T)
        people = pc.groupby('resource_type').size().to_dict()
        horizon = max(T['be'].max() - T['bs'].min(), 1)
        busy = T.dropna(subset=['resource_type']).groupby('resource_type')['bdur'].sum()
        util = _role_utilization_recent(A, R, months=999) if has_eff else {}
        util_target = {r: min(u, 95) for r, u in util.items() if u and 5 < u < 150}
        for r, n in people.items():
            u = util.get(r)
            if u and 5 < u < 150:   # capacidade ancorada na ocupação real em horas
                slots[r] = int(max(n, np.ceil(busy.get(r, 0) / (min(u, 98) / 100 * horizon)))); cap_src = 'horas'
            else:                   # sem horas: nº de tarefas que cada pessoa costuma ter em simultâneo
                slots[r] = int(pc[pc['resource_type'] == r]['conc'].sum()); cap_src = cap_src or 'simultaneidade'
            conc_role[r] = slots[r] / n

    # Esforço e custo por tarefa
    hours, rate = {}, {}
    if has_cost:
        h = A.groupby(['task_id', 'task_name'])['hours_worked'].sum().reset_index()
        hours = {a: g['hours_worked'].values for a, g in h.groupby('task_name')}
        rate = A.groupby('resource_type').apply(lambda g: g['cost'].sum() / max(g['hours_worked'].sum(), 1e-9)).to_dict()
        rate['*'] = A['cost'].sum() / max(A['hours_worked'].sum(), 1e-9)

    # Chegadas históricas (para validar: os mesmos processos, nas mesmas datas)
    first = T.groupby('project_id')['bs'].min()
    arr = pd.DataFrame({'case': first.index, 't0': first.values})
    arr['ctype'] = arr['case'].map(ctype).fillna('Geral')
    if caps_ok({'PLANO'}, caps) and 'planned_end_date' in P.columns:
        pl = pd.to_datetime(P.set_index('project_id')['planned_end_date'], errors='coerce')
        arr['plan_cal'] = ((arr['case'].map(pl) - origin).dt.total_seconds() / 86400).values
    else:
        arr['plan_cal'] = np.nan
    arr['plan_off'] = arr['plan_cal'] - _bday_to_cal(arr['t0'].values, origin)
    # Para a política de priorização: duração esperada de cada atividade e tempo que costuma faltar até ao fim do processo
    cend = T.groupby('project_id')['be'].transform('max')
    rem = (cend - T['bs']).groupby(T['task_name']).mean().to_dict()
    dmean = T.groupby('task_name')['bdur'].mean().to_dict()
    ctypes_list = sorted(T['ctype'].unique())
    return dict(rem=rem, dmean=dmean, ctypes=ctypes_list,routing=routing, dur=dur, pairs=pairs, donors=donors, exo_act=exo_act, obs_wait=obs_wait, n_obs=n_obs, exo_factor={k: 1.0 for k in obs_wait}, default_factor=1.0,
                roles=roles, slots=slots, people=people, conc=conc_role, cap_src=cap_src, util_target=util_target, hours=hours, rate=rate, has_cap=has_cap, has_cost=has_cost,
                arrivals=arr.sort_values('t0').reset_index(drop=True), origin=origin, acts=sorted(T['task_name'].unique()))


POLICY_FEATS = ['Urgência (pouca folga até ao prazo)', 'Tempo já em fila', 'Duração esperada da tarefa', 'Processo já com retrabalho', 'Progresso do processo']


def policy_features(M):
    return POLICY_FEATS + [f"Tipo: {c}" for c in M['ctypes']]


def simulate_process(M, arrivals=None, seed=0, slots=None, exo_mult=None, max_steps=60, dur_mult=None, rework_mult=1.0, policy=None, log=None):
    """Simulação por eventos discretos (relógio em dias úteis). Devolve (casos, tarefas, ocupação por função).
    slots: capacidade por função (nº de tarefas em simultâneo); exo_mult: multiplicador da espera estrutural (por passagem ou global)."""
    rng = np.random.default_rng(seed)
    arrivals = M['arrivals'] if arrivals is None else arrivals
    slots = dict(M['slots'] if slots is None else slots)
    free = dict(slots); queues = {r: _deque() for r in slots}; busy = {r: 0.0 for r in slots}
    heap, cnt = [], _itertools.count()
    push = lambda t, kind, data: _heapq.heappush(heap, (t, next(cnt), kind, data))
    ctypes = arrivals['ctype'].values; t0s = arrivals['t0'].values
    steps = np.zeros(len(arrivals), dtype=int); case_end = np.full(len(arrivals), np.nan)
    rec = []
    cases_ids = arrivals['case'].values
    donor = []
    all_d = [x for v in M.get('donors', {}).values() for x in v]
    for c in range(len(arrivals)):
        pool = M.get('donors', {}).get(ctypes[c]) or all_d
        if not pool: donor.append({}); continue
        for _ in range(3):
            pid, d = pool[rng.integers(len(pool))]
            if pid != cases_ids[c] or len(pool) == 1: break
        donor.append({k: list(v) for k, v in d.items()})

    def from_donor(c, prev, a):
        lst = donor[c].get((prev, a))
        return lst.pop(0) if lst else None

    _crngs = {}

    def crng(c):   # um gerador por processo: decisões diferentes noutros processos não mudam a sorte deste (comparações justas)
        g = _crngs.get(c)
        if g is None: g = _crngs[c] = np.random.default_rng((int(seed), int(c)))
        return g

    _cum = {}

    def choose(lst_probs, c=None):
        lst, pr = lst_probs
        k = id(pr); cum = _cum.get(k)
        if cum is None: cum = _cum[k] = np.cumsum(pr)
        i = int(np.searchsorted(cum, (rng if c is None else crng(c)).random() * cum[-1], side='right'))
        return lst[min(i, len(lst) - 1)]

    visited = [set() for _ in range(len(arrivals))]
    rwc = np.zeros(len(arrivals), dtype=bool)
    if policy is not None:
        po = arrivals['plan_off'].values if 'plan_off' in arrivals else np.full(len(arrivals), np.nan)
        plan_b = t0s + po * 5 / 7
        tix = {ct: i for i, ct in enumerate(M['ctypes'])}
        cti = np.array([tix.get(x, -1) for x in ctypes])
        W = np.asarray(policy['w'], dtype=float); flex = policy.get('flex', np.inf)

    def feats(item, t):
        c, a, prev, t_req, e, t_ready, d = item
        sl = plan_b[c] - t - M['rem'].get(a, 0)
        f = [0.0 if np.isnan(sl) else float(np.clip(-sl, -30, 30)) / 10, (t - t_ready) / 5, M['dmean'].get(a, 0) / 5,
             1.0 if rwc[c] else 0.0, steps[c] / 10]
        oh = [0.0] * len(M['ctypes'])
        if cti[c] >= 0: oh[cti[c]] = 1.0
        return np.array(f + oh)

    def pick(q, t):
        if policy is None or len(q) == 1: return q.popleft()
        F = np.array([feats(it, t) for it in q]); sc = F @ W
        i = int(np.argmax(sc))
        if log is not None:
            oldest = int(np.argmin([it[5] for it in q]))
            log.append((i != oldest, F[i], np.delete(F, i, axis=0).mean(axis=0)))
        it = q[i]; del q[i]
        return it

    def route(c, prev):
        tab = M['routing'].get(ctypes[c], M['routing']['*'])
        opts = tab.get(prev) or M['routing']['*'].get(prev)
        if opts is None or steps[c] >= max_steps: return _END
        a = choose(opts, c)
        if a in visited[c]: rwc[c] = True
        if a in visited[c] and rework_mult < 1.0 and crng(c).random() > rework_mult:
            # retrabalho evitado: segue para a atividade seguinte mais provável que ainda não foi feita
            lst, pr = opts
            alt = [(x, q) for x, q in zip(lst, pr) if x not in visited[c]]
            if alt:
                xs, qs = zip(*alt); qs = np.array(qs) / sum(qs); a = xs[crng(c).choice(len(xs), p=qs)]
            else:
                fwd = M['routing']['*'].get(a)
                a = next((x for x in (fwd[0] if fwd else []) if x not in visited[c]), _END)
        if a != _END: visited[c].add(a)
        return a

    def sample_dur(c, a):
        v = M['dur'].get((ctypes[c], a), M['dur'].get(('*', a)))
        return float(v[crng(c).integers(len(v))]) if v is not None and len(v) else 0.0

    def mult(prev, a):
        if isinstance(exo_mult, dict): return exo_mult.get((prev, a), exo_mult.get(a, exo_mult.get('*', 1.0)))
        return 1.0 if exo_mult is None else exo_mult

    def request(c, a, t, prev):
        steps[c] += 1
        dn = from_donor(c, _START if prev is None else prev, a)
        if prev is None:
            e, d = 0.0, (dn[1] if dn else sample_dur(c, a))
        else:
            pr = M['pairs'].get((ctypes[c], prev, a), M['pairs'].get(('*', prev, a)))
            if dn:
                w, d = dn
            elif pr is not None and len(pr):
                w, d = pr[crng(c).integers(len(pr))]
            else:
                v = M['exo_act'].get(a); w = float(v[crng(c).integers(len(v))]) if v is not None and len(v) else 0.0; d = sample_dur(c, a)
            e = float(w) * M['exo_factor'].get((prev, a), M['default_factor']) * mult(prev, a)
        if dur_mult: d = float(d) * dur_mult.get(a, dur_mult.get('*', 1.0))
        push(t + e, 'ready', (c, a, prev, t, e, float(d)))

    def start(c, a, role, t, prev, t_req, e, t_ready, d):
        push(t + d, 'end', (c, a, role, prev, t_req, e, t_ready, t, d))

    for c in range(len(arrivals)):
        push(float(t0s[c]), 'arr', c)
    while heap:
        t, _, kind, data = _heapq.heappop(heap)
        if kind == 'arr':
            a = route(data, _START)
            if a == _END: case_end[data] = t
            else: request(data, a, t, None)
        elif kind == 'ready':
            c, a, prev, t_req, e, d = data
            role = choose(M['roles'][a], c) if M['has_cap'] and a in M['roles'] else None
            if role is None or role not in free:
                start(c, a, role, t, prev, t_req, e, t, d)
            elif free[role] > 0:
                free[role] -= 1; start(c, a, role, t, prev, t_req, e, t, d)
            else:
                if policy is not None and len(queues[role]) >= flex:
                    # função saturada: outra função que já fez esta atividade no histórico ajuda (se tiver capacidade livre)
                    lst, pr = M['roles'][a]
                    alt = [r for r, q in sorted(zip(lst, pr), key=lambda x: -x[1]) if r != role and q >= 0.05 and free.get(r, 0) > 0]
                    if alt:
                        r2 = alt[0]; free[r2] -= 1; start(c, a, r2, t, prev, t_req, e, t, d * 1.1)
                        if log is not None: log.append(('flex', role, r2))
                        continue
                queues[role].append((c, a, prev, t_req, e, t, d))
        else:
            c, a, role, prev, t_req, e, t_ready, t_st, d = data
            rec.append((c, a, role, prev, t_req, t_ready, t_st, t, e, t_st - t_ready))
            if role in free:
                busy[role] += d
                if queues[role]:
                    c2, a2, p2, tr2, e2, ty2, d2 = pick(queues[role], t)
                    start(c2, a2, role, t, p2, tr2, e2, ty2, d2)
                else:
                    free[role] += 1
            nxt = route(c, a)
            if nxt == _END: case_end[c] = t
            else: request(c, nxt, t, a)

    tasks = pd.DataFrame(rec, columns=['c', 'task_name', 'role', 'prev', 't_req', 't_ready', 't_start', 't_end', 'exo_wait', 'queue_wait'])
    o = M['origin']
    for col in ['t_req', 't_start', 't_end']:
        tasks[col.replace('t_', 'cal_')] = _bday_to_cal(tasks[col].values, o)
    tasks['wait'] = (tasks['cal_start'] - tasks['cal_req']).clip(lower=0); tasks['dur'] = (tasks['cal_end'] - tasks['cal_start']).clip(lower=0)
    tasks.loc[tasks['prev'].isna(), 'wait'] = 0.0
    if M['has_cost'] and len(tasks):
        h = np.array([float(M['hours'][a][rng.integers(len(M['hours'][a]))]) if a in M['hours'] and len(M['hours'][a]) else 0.0 for a in tasks['task_name']])
        tasks['cost'] = h * tasks['role'].map(M['rate']).fillna(M['rate'].get('*', 0)).values
    g = tasks.groupby('c')
    cases = pd.DataFrame({'c': np.arange(len(arrivals)), 'case': arrivals['case'].values, 'ctype': ctypes, 't0': t0s,
                          'plan': arrivals['plan_cal'].values if 'plan_cal' in arrivals else np.nan})
    if 'plan_off' in arrivals:   # prazo relativo ao início (cenários com chegadas reamostradas)
        cases['plan'] = _bday_to_cal(t0s, o) + arrivals['plan_off'].values
    cases['first'] = cases['c'].map(g['cal_start'].min()); cases['end'] = _bday_to_cal(case_end, o)
    cases['lead'] = cases['end'] - cases['first']
    cases['wait'] = cases['c'].map(g['wait'].sum()).fillna(0)
    cases['n_tasks'] = cases['c'].map(g.size()).fillna(0)
    cases['rework'] = cases['c'].map(tasks.duplicated(['c', 'task_name']).groupby(tasks['c']).any()).fillna(False)
    cases['late'] = np.where(pd.notna(cases['plan']), cases['end'] > cases['plan'] + 0.5, np.nan)
    if 'cost' in tasks: cases['cost'] = cases['c'].map(g['cost'].sum())
    horizon = max(np.nanmax(case_end) - float(np.min(t0s)), 1) if len(arrivals) else 1
    util = {r: busy[r] / (slots[r] * horizon) * 100 for r in slots if slots[r] > 0}
    return cases, tasks, util


def _real_reference(V, caps, M):
    """Indicadores reais calculados da mesma forma que no simulador (para comparar maçãs com maçãs)."""
    T = V['T'].dropna(subset=['start_date', 'end_date'])
    g = T.groupby('project_id')
    c = pd.DataFrame({'first': g['start_date'].min(), 'end': g['end_date'].max(), 'wait': g['wait_days'].sum(), 'n_tasks': g.size(),
                      'rework': g['repeat'].any()})
    c['lead'] = (c['end'] - c['first']).dt.total_seconds() / 86400
    if caps_ok({'PLANO'}, caps) and 'planned_end_date' in V['P'].columns:
        pl = pd.to_datetime(V['P'].set_index('project_id')['planned_end_date'], errors='coerce')
        c['late'] = (c['end'] > c.index.map(pl) + pd.Timedelta(hours=12)).astype(float).where(c.index.map(pl).notna())
    if M['has_cost']:
        c['cost'] = V['A'].groupby('project_id')['cost'].sum()
    c['end_month'] = c['end'].dt.to_period('M').dt.to_timestamp()
    acts = T.groupby('task_name').agg(wait=('wait_days', 'mean'), dur=('dur_days', 'mean'))
    return c, acts


def _summ(cases):
    s = {'Duração média (dias)': cases['lead'].mean(), 'Duração mediana (dias)': cases['lead'].median(), 'Duração P90 (dias)': cases['lead'].quantile(0.9),
         'Espera média por processo (dias)': cases['wait'].mean(), 'Tarefas por processo': cases['n_tasks'].mean(), '% com retrabalho': cases['rework'].mean() * 100}
    if 'late' in cases and cases['late'].notna().any(): s['% atrasados'] = pd.to_numeric(cases['late'], errors='coerce').mean() * 100
    if 'cost' in cases and cases['cost'].notna().any(): s['Custo médio por processo (€)'] = cases['cost'].mean()
    return s


def calibrate_and_validate(V, caps, reps=5, progress=None):
    """1) Constrói o modelo; 2) separa a espera em 'falta de capacidade' (fila) e 'estrutural'; 3) valida contra o histórico."""
    M = build_sim_model(V, caps)
    step = lambda p, txt: progress(p, txt) if progress else None
    obs = M['obs_wait']
    # 1. Só filas (sem espera estrutural): a capacidade explica quanta espera?
    if M['has_cap']:
        M['exo_factor'] = {k: 0.0 for k in obs}; M['default_factor'] = 0.0
        # 0. Ancorar a capacidade na ocupação real (horas trabalhadas ÷ horas disponíveis)
        for it in range(2):
            if not M['util_target']: break
            step(0.05 + it * 0.03, "A ajustar a capacidade de cada função à ocupação real...")
            u = pd.DataFrame([simulate_process(M, seed=50 + s_)[2] for s_ in range(2)]).mean()
            for r, tgt in M['util_target'].items():
                if r in M['slots'] and u.get(r, 0) > 0:
                    M['slots'][r] = int(max(M['people'].get(r, 1), round(M['slots'][r] * u[r] / tgt)))
        for it in range(6):
            step(0.1 + it * 0.05, "A medir a espera causada por falta de capacidade...")
            q = pd.concat([simulate_process(M, seed=s)[1] for s in range(2)])
            qw = q.dropna(subset=['prev']).groupby(['prev', 'task_name'])['queue_wait'].mean().to_dict()
            tot_q = sum(qw.get(k, 0) * M['n_obs'][k] for k in obs); tot_o = sum(obs[k] * M['n_obs'][k] for k in obs)
            if tot_q <= 0.9 * max(tot_o, 1e-9): break
            M['slots'] = {r: int(np.ceil(v * 1.25)) for r, v in M['slots'].items()}   # capacidade estimada demasiado baixa
        M['exo_factor'] = {k: float(np.clip((obs[k] - qw.get(k, 0)) / obs[k], 0, 1)) if obs[k] > 0 else 0.0 for k in obs}
        M['default_factor'] = 1.0
        M['queue_wait_base'] = qw
    # 2. Ajuste fino da espera estrutural (2 iterações)
    for it in range(2):
        step(0.45 + it * 0.15, "A calibrar as esperas entre atividades...")
        s_t = pd.concat([simulate_process(M, seed=10 + s)[1] for s in range(2)])
        sw = s_t.dropna(subset=['prev']).assign(bw=lambda x: x['t_start'] - x['t_req']).groupby(['prev', 'task_name'])['bw'].mean().to_dict()
        for k in obs:
            if sw.get(k, 0) > 1e-6 and obs[k] > 0:
                M['exo_factor'][k] = float(np.clip(M['exo_factor'][k] * obs[k] / sw[k], 0, 3))
    # 3. Validação: os mesmos processos, nas mesmas datas, várias réplicas
    runs = []
    for s in range(reps):
        step(0.75 + 0.2 * s / reps, f"A validar contra o histórico (réplica {s + 1} de {reps})...")
        runs.append(simulate_process(M, seed=100 + s))
    real_c, real_a = _real_reference(V, caps, M)
    rs = _summ(real_c)
    sims = pd.DataFrame([_summ(r[0]) for r in runs])
    rows = []
    for k, v in rs.items():
        if k not in sims: continue
        m, lo, hi = sims[k].mean(), sims[k].min(), sims[k].max()
        err = (m - v) if k.startswith('%') else ((m - v) / v * 100 if v else np.nan)
        rows.append({'Indicador': k, 'Real': v, 'Simulado': m, 'Intervalo simulado': f"{lo:,.1f} – {hi:,.1f}".replace(',', ' '),
                     'Diferença': (f"{err:+.1f} pp" if k.startswith('%') else f"{err:+.1f}%"), '_err': abs(err)})
    val = pd.DataFrame(rows)
    key = val[val['Indicador'].isin(['Duração média (dias)', 'Duração P90 (dias)', 'Espera média por processo (dias)', 'Custo médio por processo (€)'])]
    score = key['_err'].mean() if len(key) else np.nan
    # Detalhe por atividade e por mês
    st_all = pd.concat([r[1] for r in runs])
    sim_a = st_all.groupby('task_name').agg(wait=('wait', 'mean'), dur=('dur', 'mean'), queue=('queue_wait', 'mean'))
    sim_c = pd.concat([r[0].assign(rep=i) for i, r in enumerate(runs)])
    sim_c['end_month'] = (M['origin'] + pd.to_timedelta(sim_c['end'], unit='D')).dt.to_period('M').dt.to_timestamp()
    util = pd.DataFrame([r[2] for r in runs]).mean().to_dict() if M['has_cap'] else {}
    # Decomposição da espera por passagem: fila (capacidade) vs estrutural
    dec = st_all.dropna(subset=['prev']).groupby(['prev', 'task_name']).agg(fila=('queue_wait', 'mean'), estrutural=('exo_wait', 'mean'), n=('c', 'size')).reset_index()
    dec['n'] = dec['n'] / reps
    step(1.0, "Concluído.")
    return dict(model=M, validation=val, score=score, real_cases=real_c, real_acts=real_a, sim_cases=sim_c, sim_acts=sim_a, util=util, wait_split=dec, reps=reps)


# ---------- Gráficos da validação ----------
def fig_sim_monthly(SV):
    r = SV['real_cases'].groupby('end_month')['lead'].mean()
    s = SV['sim_cases'].groupby(['rep', 'end_month'])['lead'].mean().unstack(0)
    cnt = SV['real_cases'].groupby('end_month').size(); full = cnt[cnt >= 0.3 * cnt.median()].index
    r = r[r.index.isin(full)]; s = s[s.index.isin(full)]
    if len(r) < 2: return None
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.max(axis=1), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip')
    fig.add_scatter(x=s.index, y=s.min(axis=1), mode='lines', fill='tonexty', fillcolor='rgba(37,99,235,0.12)', line=dict(width=0), name='Intervalo simulado', hoverinfo='skip')
    fig.add_scatter(x=s.index, y=s.mean(axis=1), name='Simulado', mode='lines', line=dict(color=UI['blue'], width=2.5, dash='dot'))
    fig.add_scatter(x=r.index, y=r.values, name='Real', mode='lines+markers', line=dict(color=UI['ink'], width=2.5))
    fig.update_layout(yaxis_title='Duração média (dias)', xaxis_title='Mês de conclusão')
    return _style(fig, 320)


def fig_sim_acts(SV, col='wait'):
    r, s = SV['real_acts'][col], SV['sim_acts'][col]
    d = pd.DataFrame({'Real': r, 'Simulado': s}).dropna().sort_values('Real')
    fig = go.Figure()
    fig.add_bar(y=d.index, x=d['Real'], name='Real', orientation='h', marker_color=UI['ink'])
    fig.add_bar(y=d.index, x=d['Simulado'], name='Simulado', orientation='h', marker_color=UI['blue'])
    fig.update_layout(barmode='group', xaxis_title='Dias (média)' if col == 'wait' else 'Dias de execução (média)', yaxis_title='')
    return _style(fig, max(300, 40 * len(d)))


def fig_wait_split(SV):
    d = SV['wait_split'].copy()
    d['tot'] = (d['fila'] + d['estrutural']) * d['n']
    d = d[d['n'] >= 0.01 * len(SV['real_cases'])].sort_values('tot').tail(10)
    if d.empty or d['fila'].sum() == 0: return None
    d['lbl'] = d['prev'] + ' → ' + d['task_name']
    fig = go.Figure()
    fig.add_bar(y=d['lbl'], x=d['fila'], name='Falta de capacidade (fila)', orientation='h', marker_color=UI['red'],
                hovertemplate='%{y}<br>Fila: %{x:.2f} dias<extra></extra>')
    fig.add_bar(y=d['lbl'], x=d['estrutural'], name='Estrutural (não depende de pessoas)', orientation='h', marker_color=UI['amber'],
                hovertemplate='%{y}<br>Estrutural: %{x:.2f} dias<extra></extra>')
    fig.update_layout(barmode='stack', xaxis_title='Dias úteis (média)', yaxis_title='')
    return _style(fig, max(320, 36 * len(d)))


def fig_sim_util(SV):
    u = pd.Series(SV['util']).sort_values()
    if u.empty: return None
    cols = [UI['red'] if v >= 85 else (UI['amber'] if v >= 70 else UI['teal']) for v in u.values]
    fig = go.Figure(go.Bar(x=u.values, y=u.index, orientation='h', marker_color=cols, text=[f"{v:.0f}%" for v in u.values], textposition='inside',
                           insidetextanchor='end', textfont=dict(color='white'), hovertemplate='%{y}: %{x:.0f}%<extra></extra>'))
    fig.add_vline(x=85, line=dict(color=UI['red'], dash='dash', width=1))
    fig.update_layout(xaxis_title='Ocupação média da capacidade (%)', yaxis_title='', xaxis_range=[0, max(100, u.max() * 1.1)])
    return _style(fig, max(260, 36 * len(u)), legend=False)


def render_sim_validation(caps):
    V = st.session_state.view
    ui_section_intro("O simulador reproduz o seu processo?",
                     "Antes de testar cenários, o modelo é calibrado com o histórico e posto à prova: simula os mesmos processos, nas mesmas datas, e compara com o que aconteceu.")
    if not caps_ok({'TEMPOS'}, caps):
        ui_note("O simulador precisa de datas de início e de fim por tarefa."); return
    c1, c2 = st.columns([5, 1.4], vertical_alignment='center')
    with c2:
        rerun = st.button("Recalibrar", icon=":material/refresh:", use_container_width=True, key='sim_recal',
                          help="Volta a construir e a ajustar o simulador a partir do histórico. Só é preciso depois de carregar dados novos; os resultados podem variar ligeiramente por causa da aleatoriedade.")
    if st.session_state.get('sim_validation') is None or rerun:
        bar = st.progress(0.0, text="A construir o modelo do processo...")
        try:
            st.session_state.sim_validation = calibrate_and_validate(V, caps, reps=5, progress=lambda p, t: bar.progress(min(p, 1.0), text=t))
        except Exception as e:
            bar.empty(); st.error(f"Não foi possível calibrar o simulador: {e}"); return
        bar.empty()
    SV = st.session_state.sim_validation; M = SV['model']
    with c1:
        st.caption(f"{len(M['arrivals']):,} processos simulados × {SV['reps']} réplicas · {len(M['acts'])} atividades".replace(',', ' ')
                   + (f" · {sum(M['people'].values())} pessoas em {len(M['people'])} funções" if M['has_cap'] else " · sem dados de pessoas: capacidade ilimitada"))
    sc = SV['score']
    lab, icon = ('Boa', 'bi-patch-check') if sc < 10 else (('Aceitável', 'bi-exclamation-circle') if sc < 20 else ('Fraca', 'bi-x-octagon'))
    v = SV['validation'].set_index('Indicador')
    k = [dict(icon=icon, value=lab, label=f"Fidelidade do modelo (erro médio {sc:.1f}%)", good=sc < 10 if sc < 10 or sc >= 20 else None,
              help='Média do erro absoluto em duração média, P90, espera e custo.')]
    for ind, ic in [('Duração média (dias)', 'bi-stopwatch'), ('Espera média por processo (dias)', 'bi-pause-circle'), ('% atrasados', 'bi-alarm')]:
        if ind in v.index:
            k.append(dict(icon=ic, value=f"{v.loc[ind, 'Simulado']:.1f}" + ('%' if ind.startswith('%') else ' d'),
                          label=f"{ind.split(' (')[0]} simulada · real {v.loc[ind, 'Real']:.1f}" + ('%' if ind.startswith('%') else ' d'),
                          help='Valor obtido pelo simulador (média de 5 réplicas) comparado com o valor real do histórico.'))
    ui_kpis(k)
    with st.container(border=True):
        ui_card_header('Real vs. simulado', 'Indicadores principais com os mesmos processos e datas de chegada', 'bi-clipboard-data')
        show = SV['validation'].drop(columns='_err').copy()
        st.dataframe(show.style.format({'Real': '{:,.1f}', 'Simulado': '{:,.1f}'}), hide_index=True, use_container_width=True)
    ui_chart(fig_sim_monthly(SV), 'Duração ao longo do tempo: real vs. simulado', 'Se as linhas acompanham os altos e baixos, o modelo capta o efeito do volume e da capacidade', 'bi-graph-up', key='sim_m')
    c1, c2 = st.columns(2)
    with c1: ui_chart(fig_sim_acts(SV, 'wait'), 'Espera antes de cada atividade', 'Real vs. simulado', 'bi-hourglass-split', key='sim_aw')
    with c2: ui_chart(fig_sim_acts(SV, 'dur'), 'Duração de cada atividade', 'Real vs. simulado', 'bi-stopwatch', key='sim_ad')
    if M['has_cap']:
        c1, c2 = st.columns(2)
        with c1: ui_chart(fig_wait_split(SV), 'De onde vem a espera?', 'Fila por falta de pessoas vs. espera estrutural (aprovações, terceiros, agendamentos)', 'bi-diagram-2', key='sim_ws')
        with c2: ui_chart(fig_sim_util(SV), 'Ocupação da capacidade por função', 'Acima de 85% as filas crescem depressa com qualquer aumento de volume', 'bi-speedometer2', key='sim_ut')
        ws = SV['wait_split']; tot_f = (ws['fila'] * ws['n']).sum(); tot_e = (ws['estrutural'] * ws['n']).sum()
        if tot_f + tot_e > 0:
            ui_note(f"Leitura: {tot_f / (tot_f + tot_e):.0%} da espera deve-se a falta de capacidade (reduz-se com mais pessoas ou melhor alocação); "
                    f"{tot_e / (tot_f + tot_e):.0%} é estrutural (reduz-se mudando regras, aprovações ou prazos de terceiros, não com mais pessoas).")
    with st.expander("Pressupostos do modelo"):
        st.markdown("- **Chegadas:** os processos reais, nas datas reais (na validação). Nos cenários, o volume pode ser alterado.\n"
                    "- **Caminho de cada processo:** probabilidade da atividade seguinte observada no histórico, por tipo de processo (inclui retrabalho).\n"
                    "- **Duração das tarefas:** sorteada da distribuição real de cada atividade.\n"
                    "- **Capacidade:** cada função tem tantas pessoas como no histórico; cada pessoa trabalha em paralelo no nº de tarefas que costuma ter em simultâneo (percentil 90).\n"
                    "- **Espera:** parte surge naturalmente da fila por falta de capacidade; o restante é espera estrutural calibrada para reproduzir o histórico.")
        if M['has_cap']:
            st.dataframe(pd.DataFrame({'Pessoas': M['people'], 'Tarefas em simultâneo por pessoa (média)': pd.Series(M['conc']).round(1),
                                       'Capacidade total (tarefas em simultâneo)': M['slots']}), use_container_width=True)


# ---------- Fase 2: cenários "e se" ----------
def _ensure_sim(V, caps):
    if st.session_state.get('sim_validation') is None:
        bar = st.progress(0.0, text="A calibrar o simulador com o histórico (só na primeira vez)...")
        st.session_state.sim_validation = calibrate_and_validate(V, caps, reps=5, progress=lambda p, t: bar.progress(min(p, 1.0), text=t))
        bar.empty()
    return st.session_state.sim_validation


def scenario_arrivals(M, volume_pct=0, seed=0):
    base = M['arrivals']
    if not volume_pct: return base
    rng = np.random.default_rng(1000 + seed)
    n0 = len(base); n = max(1, int(round(n0 * (1 + volume_pct / 100))))
    idx = np.sort(rng.choice(n0, n, replace=n > n0))
    a = base.iloc[idx].copy().reset_index(drop=True)
    dup = pd.Series(idx).duplicated().values
    a.loc[dup, 't0'] = (a.loc[dup, 't0'] + rng.uniform(-5, 5, dup.sum())).clip(lower=0)
    a['case'] = [f"sim_{i}" for i in range(len(a))]
    return a.sort_values('t0').reset_index(drop=True)


def run_scenario(M, params, reps=5):
    """Corre o cenário com as mesmas sementes da base (comparação justa). Devolve lista de (casos, tarefas, ocupação)."""
    slots = dict(M['slots'])
    for r, d in (params.get('people') or {}).items():
        if r in slots and d:
            n = M['people'].get(r, 1); slots[r] = max(1, int(round(slots[r] * max(n + d, 1) / n)))
    exo = {k: 1 - v / 100 for k, v in (params.get('exo') or {}).items()} or None
    dur = {a: 1 - v / 100 for a, v in (params.get('dur') or {}).items()} or None
    runs = []
    for s_ in range(reps):
        arr = scenario_arrivals(M, params.get('volume', 0), seed=s_)
        runs.append(simulate_process(M, arrivals=arr, seed=500 + s_, slots=slots, exo_mult=exo, dur_mult=dur,
                                     rework_mult=1 - params.get('rework', 0) / 100))
    return runs


def summarize_runs(runs):
    rows = []
    for c, t, u in runs:
        c = c.dropna(subset=['lead'])
        r = {'lead': c['lead'].mean(), 'p90': c['lead'].quantile(0.9), 'wait': c['wait'].mean(), 'n': len(c),
             'rework': c['rework'].mean() * 100, 'umax': max(u.values()) if u else np.nan}
        if c['late'].notna().any(): r['late'] = pd.to_numeric(c['late'], errors='coerce').mean() * 100
        if 'cost' in c: r['cost'] = c['cost'].mean(); r['cost_tot'] = c['cost'].sum()
        rows.append(r)
    out = pd.DataFrame(rows).mean().to_dict()
    out['util'] = pd.DataFrame([u for _, _, u in runs]).mean().to_dict() if runs and runs[0][2] else {}
    allt = pd.concat([t for _, t, _ in runs])
    out['acts'] = allt.groupby('task_name').agg(wait=('wait', 'mean'), fila=('queue_wait', 'mean'))
    out['leads'] = pd.concat([c['lead'] for c, _, _ in runs]).dropna().values
    return out


def _scen_preset(kind):
    SV = st.session_state.sim_validation; M = SV['model']
    for k in list(st.session_state.keys()):
        if k.startswith('sc_p_'): st.session_state[k] = 0
    st.session_state.update({'sc_vol': 0, 'sc_rw': 0, 'sc_exo_sel': [], 'sc_exo_pct': 50, 'sc_dur_sel': [], 'sc_dur_pct': 20})
    if kind == 'vol':
        st.session_state.sc_vol = 20; st.session_state.sc_name = 'Volume +20%'
    elif kind == 'people' and SV['util']:
        r = max(SV['util'], key=SV['util'].get); st.session_state[f'sc_p_{r}'] = 1; st.session_state.sc_name = f'+1 {r}'
    elif kind == 'exo':
        ws = SV['wait_split'].assign(tot=lambda d: d['estrutural'] * d['n']).sort_values('tot', ascending=False)
        if len(ws):
            k = f"{ws.iloc[0]['prev']} → {ws.iloc[0]['task_name']}"; st.session_state.sc_exo_sel = [k]; st.session_state.sc_exo_pct = 50
            st.session_state.sc_name = f'Metade da espera {k}'
    elif kind == 'rework':
        st.session_state.sc_rw = 50; st.session_state.sc_name = 'Retrabalho −50%'
    elif kind == 'clear':
        st.session_state.sc_name = ''


def _delta_txt(new, old, unit='', pp=False, dec=1):
    if new is None or old is None or pd.isna(new) or pd.isna(old): return None
    d = new - old
    if pp: return f"{d:+.{dec}f} pp"
    return f"{d:+,.{dec}f}{unit}".replace(',', ' ') + (f" ({d / old:+.0%})" if old else "")


def fig_scen_util(B, S):
    roles = sorted(set(B['util']) | set(S['util']))
    if not roles: return None
    fig = go.Figure()
    fig.add_bar(y=roles, x=[B['util'].get(r, 0) for r in roles], name='Base', orientation='h', marker_color='#cbd5e1')
    fig.add_bar(y=roles, x=[S['util'].get(r, 0) for r in roles], name='Cenário', orientation='h',
                marker_color=[UI['red'] if S['util'].get(r, 0) >= 85 else UI['blue'] for r in roles])
    fig.add_vline(x=85, line=dict(color=UI['red'], dash='dash', width=1))
    fig.update_layout(barmode='group', xaxis_title='Ocupação da capacidade (%)', yaxis_title='')
    return _style(fig, max(280, 44 * len(roles)))


def fig_scen_acts(B, S):
    d = pd.DataFrame({'Base': B['acts']['wait'], 'Cenário': S['acts']['wait']}).dropna().sort_values('Base')
    fig = go.Figure()
    fig.add_bar(y=d.index, x=d['Base'], name='Base', orientation='h', marker_color='#cbd5e1')
    fig.add_bar(y=d.index, x=d['Cenário'], name='Cenário', orientation='h', marker_color=UI['blue'])
    fig.update_layout(barmode='group', xaxis_title='Espera média antes da atividade (dias)', yaxis_title='')
    return _style(fig, max(300, 40 * len(d)))


def fig_scen_dist(B, S):
    fig = go.Figure()
    fig.add_histogram(x=B['leads'], name='Base', marker_color='#94a3b8', opacity=0.6, nbinsx=40, histnorm='percent')
    fig.add_histogram(x=S['leads'], name='Cenário', marker_color=UI['blue'], opacity=0.6, nbinsx=40, histnorm='percent')
    fig.update_layout(barmode='overlay', xaxis_title='Duração do processo (dias)', yaxis_title='% dos processos')
    return _style(fig, 300)


def render_sim_scenarios(caps):
    V = st.session_state.view
    if not caps_ok({'TEMPOS'}, caps):
        ui_note("O simulador precisa de datas de início e de fim por tarefa."); return
    SV = _ensure_sim(V, caps); M = SV['model']
    for k_, v_ in {'sc_vol': 0, 'sc_rw': 0, 'sc_exo_sel': [], 'sc_exo_pct': 50, 'sc_dur_sel': [], 'sc_dur_pct': 20, 'sc_name': ''}.items():
        st.session_state.setdefault(k_, v_)
    for r_ in M['people']: st.session_state.setdefault(f'sc_p_{r_}', 0)
    ui_section_intro('E se...?', 'Altere volume, equipa, esperas, retrabalho ou duração das atividades e veja o efeito antes de decidir. '
                     'Cada cenário é comparado com a base simulada, nos mesmos processos e com as mesmas sementes aleatórias.')
    if SV['score'] >= 20:
        ui_note(f"Atenção: a fidelidade do modelo é fraca (erro médio {SV['score']:.0f}%). Use os resultados como indicação de sentido, não como valores exatos.")
    n_sc = len(st.session_state.get('sim_scen_list') or [])
    if n_sc:
        cc1, cc2 = st.columns([4, 1.4], vertical_alignment='center')
        cc1.caption(f"{n_sc} cenário(s) testado(s) nesta sessão.")
        if cc2.button("Recomeçar do zero", icon=":material/delete_sweep:", use_container_width=True, key='sc_reset_all',
                      help="Apaga todos os cenários testados e repõe o formulário na situação atual."):
            _scen_preset('clear'); st.session_state.sim_scen_list = []; st.session_state.sim_last = None
            for k_ in [k for k in st.session_state if k.startswith('dir_txt_')]: st.session_state.pop(k_)
            st.rerun()
    st.markdown("<div class='pm-side-label' style='margin:4px 0 6px'>Cenários rápidos</div>", unsafe_allow_html=True)
    b = st.columns(5)
    b[0].button("Volume +20%", on_click=_scen_preset, args=('vol',), use_container_width=True, icon=":material/trending_up:", help="Preenche o formulário com mais 20% de processos a entrar. Depois carregue em Simular cenário.")
    b[1].button("+1 pessoa", on_click=_scen_preset, args=('people',), use_container_width=True, icon=":material/person_add:", disabled=not M['has_cap'], help="Junta uma pessoa à função mais ocupada.")
    b[2].button("Metade da espera", on_click=_scen_preset, args=('exo',), use_container_width=True, icon=":material/hourglass_bottom:", help="Reduz para metade a maior espera estrutural (a passagem onde os processos ficam mais tempo parados).")
    b[3].button("Retrabalho −50%", on_click=_scen_preset, args=('rework',), use_container_width=True, icon=":material/replay:", help="Metade dos retornos a atividades já feitas deixa de acontecer.")
    b[4].button("Limpar", on_click=_scen_preset, args=('clear',), use_container_width=True, icon=":material/restart_alt:", help="Repõe todos os parâmetros na situação atual.")

    ws = SV['wait_split'].assign(tot=lambda d: d['estrutural'] * d['n']).sort_values('tot', ascending=False)
    exo_opts = [f"{r.prev} → {r.task_name}" for r in ws.head(10).itertuples()]
    exo_opts += [k for k in (st.session_state.get('sc_exo_sel') or []) if k not in exo_opts]
    autorun = st.session_state.pop('sc_autorun', False)
    if autorun: ui_note(f"Cenário vindo do Diagnóstico: «{st.session_state.get('sc_name')}». Pode ajustar os parâmetros e simular de novo.")
    with st.form('scen_form', border=True):
        ui_card_header('Definir cenário', None, 'bi-sliders')
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Volume de processos (%)", -50, 100, step=5, key='sc_vol', help='Variação do nº de processos que entram, no mesmo período.')
            st.slider("Redução do retrabalho (%)", 0, 100, step=10, key='sc_rw', help='Percentagem dos retornos a atividades já feitas que deixam de acontecer.')
            st.multiselect("Reduzir esperas estruturais nestas passagens", exo_opts, key='sc_exo_sel', placeholder='Escolher passagens',
                           help='Esperas que não dependem de pessoas: agendamentos, aprovações, terceiros.')
            st.slider("Redução dessas esperas (%)", 0, 100, step=10, key='sc_exo_pct', help='Ex.: 50% = os processos passam a esperar metade do tempo nessas passagens.')
            st.multiselect("Acelerar estas atividades (automação, simplificação)", M['acts'], key='sc_dur_sel', placeholder='Escolher atividades', help='Atividades cujo tempo de execução diminui (ex.: com automação ou menos verificações).')
            st.slider("Redução da duração dessas atividades (%)", 0, 90, step=10, key='sc_dur_pct', help='Ex.: 20% = a atividade passa a demorar 80% do tempo atual.')
        with c2:
            if M['has_cap']:
                st.markdown("**Pessoas por função** (variação face ao atual)")
                for r in sorted(M['people']):
                    st.number_input(f"{r} · hoje {M['people'][r]}", min_value=-(M['people'][r] - 1), max_value=50, step=1, key=f'sc_p_{r}',
                                    help='Quantas pessoas a mais (positivo) ou a menos (negativo) nesta função.')
            else:
                st.caption("Sem dados de pessoas por tarefa: não é possível simular alterações de equipa.")
        st.text_input("Nome do cenário", key='sc_name', placeholder='ex.: +1 avaliador e metade da espera na avaliação')
        go_ = st.form_submit_button("Simular cenário", type='primary', icon=":material/play_arrow:", help="Corre o cenário 5 vezes e compara com a base (situação atual). Demora alguns segundos.")

    if 'sim_base_summary' not in st.session_state or st.session_state.get('sim_base_sig') != id(SV):
        with st.spinner("A simular a base..."):
            st.session_state.sim_base_summary = summarize_runs(run_scenario(M, {}))
            st.session_state.sim_base_sig = id(SV)
    B = st.session_state.sim_base_summary

    if go_ or autorun:
        exo_sel = st.session_state.get('sc_exo_sel') or []
        params = {'volume': st.session_state.sc_vol, 'rework': st.session_state.sc_rw,
                  'people': {r: st.session_state.get(f'sc_p_{r}', 0) for r in M['people']},
                  'exo': {tuple(k.split(' → ', 1)): st.session_state.sc_exo_pct for k in exo_sel},
                  'dur': {a: st.session_state.sc_dur_pct for a in (st.session_state.get('sc_dur_sel') or [])}}
        if not any([params['volume'], params['rework'], any(params['people'].values()), params['exo'] and st.session_state.sc_exo_pct, params['dur'] and st.session_state.sc_dur_pct]):
            st.info("Altere pelo menos um parâmetro (ou use um cenário rápido).");
        else:
            with st.spinner("A simular o cenário (5 réplicas)..."):
                S = summarize_runs(run_scenario(M, params))
            name = st.session_state.get('sc_name') or f"Cenário {len(st.session_state.get('sim_scen_list', [])) + 1}"
            st.session_state.sim_last = (name, params, S)
            st.session_state.setdefault('sim_scen_list', []).append((name, params, S))
            st.rerun()

    if st.session_state.get('sim_last'):
        name, params, S = st.session_state.sim_last
        st.markdown(f"<div class='pm-h5'>Resultado: {html.escape(name)}</div>", unsafe_allow_html=True)
        k = [dict(icon='bi-stopwatch', value=_fmt_days(S['lead']), label='Duração média', delta=_delta_txt(S['lead'], B['lead'], ' d'), good=S['lead'] <= B['lead']),
             dict(icon='bi-graph-up-arrow', value=_fmt_days(S['p90']), label='Duração P90 (10% mais lentos)', delta=_delta_txt(S['p90'], B['p90'], ' d'), good=S['p90'] <= B['p90']),
             dict(icon='bi-pause-circle', value=_fmt_days(S['wait']), label='Espera por processo', delta=_delta_txt(S['wait'], B['wait'], ' d'), good=S['wait'] <= B['wait'])]
        if 'late' in S and 'late' in B:
            k.append(dict(icon='bi-alarm', value=f"{S['late']:.1f}%", label='Processos atrasados', delta=_delta_txt(S['late'], B['late'], pp=True), good=S['late'] <= B['late']))
        if 'cost' in S and 'cost' in B:
            k.append(dict(icon='bi-currency-euro', value=_fmt_eur(S['cost']), label='Custo de mão-de-obra por processo', delta=_delta_txt(S['cost'], B['cost'], ' €', dec=0), good=S['cost'] <= B['cost']))
        if S['util']:
            r = max(S['util'], key=S['util'].get)
            k.append(dict(icon='bi-speedometer2', value=f"{S['util'][r]:.0f}%", label=f"Ocupação máx.: {r}", delta=_delta_txt(S['util'][r], B['util'].get(r), pp=True, dec=0),
                          good=S['util'][r] <= B['util'].get(r, 0)))
        for it in k:   # variações abaixo do ruído da simulação ficam neutras (cinzento)
            dl = it.get('delta') or ''
            try:
                num = float(dl.split()[0].replace('+', '').replace('−', '-'))
            except (ValueError, IndexError):
                continue
            rel = re.search(r'\(([+-]?\d+)%\)', dl)
            if (('pp' in dl and abs(num) < 0.5) or (rel and abs(int(rel.group(1))) < 2) or abs(num) < 0.05):
                it['good'] = None
        ui_kpis(k)
        # leitura económica
        P = V['P']; yrs = max((P['end_date'].max() - P['start_date'].min()).days / 365, 0.1)
        per_year = S['n'] / yrs
        notes = [f"≈ {per_year:,.0f} processos/ano no cenário.".replace(',', ' ')]
        if B['lead'] - S['lead'] > 0.05: notes.append(f"Poupa ≈ {(B['lead'] - S['lead']) * per_year:,.0f} dias de processo por ano.".replace(',', ' '))
        if B['lead'] - S['lead'] < -0.05: notes.append(f"Acrescenta ≈ {(S['lead'] - B['lead']) * per_year:,.0f} dias de processo por ano.".replace(',', ' '))
        add = {r: d for r, d in (params.get('people') or {}).items() if d}
        if add and M['has_cost']:
            capd = V['R'].groupby('resource_type')['daily_capacity'].mean() if 'daily_capacity' in V['R'].columns else pd.Series(dtype=float)
            yc = sum(d * M['rate'].get(r, M['rate'].get('*', 0)) * capd.get(r, 8) * 220 for r, d in add.items())
            notes.append(f"Custo anual estimado da variação de pessoas: {_fmt_eur(yc)} (custo/hora × horas/dia × 220 dias).")
        if S['util'] and max(S['util'].values()) >= 85:
            notes.append("Há funções acima de 85% de ocupação: neste cenário as filas tornam-se sensíveis a qualquer pico de volume.")
        ui_note(" ".join(notes))
        c1, c2 = st.columns(2)
        with c1: ui_chart(fig_scen_dist(B, S), 'Distribuição da duração', 'Base vs. cenário', 'bi-bar-chart', key='sc_dist')
        with c2:
            if S['util']: ui_chart(fig_scen_util(B, S), 'Ocupação por função', 'Base vs. cenário (linha = 85%)', 'bi-speedometer2', key='sc_util')
        ui_chart(fig_scen_acts(B, S), 'Espera antes de cada atividade', 'Base vs. cenário', 'bi-hourglass-split', key='sc_acts')

    lst = st.session_state.get('sim_scen_list') or []
    if lst:
        with st.container(border=True):
            ui_card_header('Comparação de cenários', 'Todos os cenários simulados nesta sessão, face à base', 'bi-table')
            rows = [{'Cenário': 'Base (atual)', 'Duração média': B['lead'], 'P90': B['p90'], 'Espera': B['wait'], **({'% atrasados': B['late']} if 'late' in B else {}),
                     **({'Custo/processo (€)': B['cost']} if 'cost' in B else {}), 'Ocupação máx. (%)': B['umax']}]
            for n_, _, S_ in lst:
                rows.append({'Cenário': n_, 'Duração média': S_['lead'], 'P90': S_['p90'], 'Espera': S_['wait'], **({'% atrasados': S_.get('late')} if 'late' in B else {}),
                             **({'Custo/processo (€)': S_.get('cost')} if 'cost' in B else {}), 'Ocupação máx. (%)': S_['umax']})
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({c: '{:,.1f}' for c in df.columns if c != 'Cenário'}), hide_index=True, use_container_width=True)
            if st.button("Limpar cenários", key='sc_clear_list'):
                st.session_state.sim_scen_list = []; st.session_state.sim_last = None; st.rerun()
        key_m = 'late' if 'late' in B else 'lead'
        best = min(lst, key=lambda x: x[2].get(key_m, np.inf))
        lines = []
        for n_, pr_, S_ in lst:
            dl = S_['lead'] - B['lead']; dlt = (S_.get('late', 0) - B.get('late', 0)) if 'late' in B else None
            eff = "melhora" if dl < -0.3 or (dlt is not None and dlt < -0.5) else ("piora" if dl > 0.3 or (dlt is not None and dlt > 0.5) else "não muda significativamente")
            lines.append(f"**{n_}**: {eff} — duração {dl:+.1f} dias" + (f", atrasados {dlt:+.1f} pp" if dlt is not None else "") + (f", ocupação máx. {S_['umax']:.0f}%" if S_.get('umax') == S_.get('umax') else ""))
        lines.insert(0, f"Cenário com melhor resultado em {'atrasos' if key_m == 'late' else 'duração'}: **{best[0]}**.")
        render_director_summary('scen', sim_facts(), lines)


# ---------- Fase 3: otimização da política de decisão (RL por pesquisa direta de política) ----------
RL_OBJECTIVES = {'atrasos': 'Menos processos atrasados', 'duracao': 'Menor duração média', 'p90': 'Menos processos muito lentos (P90)'}


def _objective(cases, kind):
    c = cases.dropna(subset=['lead'])
    if c.empty: return 1e9
    if kind == 'atrasos' and c['late'].notna().any():
        late = pd.to_numeric(c['late'], errors='coerce')
        over = (c['end'] - c['plan']).clip(lower=0)
        return late.mean() * 100 + over.mean()
    if kind == 'p90': return c['lead'].quantile(0.9)
    return c['lead'].mean()


def _eval_policy(M, arrs, slots, policy, kind, seeds):
    return float(np.mean([_objective(simulate_process(M, arrivals=a, seed=s, slots=slots, policy=policy)[0], kind) for a, s in zip(arrs, seeds)]))


def train_policy(M, volume_pct=40, people=None, kind='atrasos', allow_flex=True, iters=8, pop=14, progress=None):
    """Cross-Entropy Method: o agente experimenta políticas no simulador, fica com as melhores e refina. Interpretável e rápido."""
    nf = len(policy_features(M))
    slots = dict(M['slots'])
    for r, d in (people or {}).items():
        if r in slots and d:
            n = M['people'].get(r, 1); slots[r] = max(1, int(round(slots[r] * max(n + d, 1) / n)))
    # treino numa janela de ~9 meses (mais rápido); avaliação final no período completo
    full = [scenario_arrivals(M, volume_pct, seed=s) for s in range(2)]
    arrs = []
    for a in full:
        lo = a['t0'].quantile(0.25); hi = lo + 190
        arrs.append(a[(a['t0'] >= lo) & (a['t0'] < hi)].reset_index(drop=True))
    seeds = [900, 901]
    mu = np.zeros(nf + 1); mu[0] = 0.5; mu[1] = 0.5; mu[-1] = 2.0   # parte de uma mistura de prazo e ordem de chegada
    sd = np.ones(nf + 1); sd[-1] = 1.5
    if not allow_flex: sd[-1] = 0; mu[-1] = 99
    to_pol = lambda x: {'w': x[:nf], 'flex': (np.inf if not allow_flex else float(np.exp(np.clip(x[-1], 0, 4))))}
    best, best_x, hist = np.inf, mu.copy(), []
    fifo_score = _eval_policy(M, arrs, slots, to_pol(np.r_[np.eye(nf)[1], 99]), kind, seeds)
    rng = np.random.default_rng(7)
    for it in range(iters):
        X = mu + sd * rng.standard_normal((pop, nf + 1)); X[0] = mu
        scores = []
        for j, x in enumerate(X):
            scores.append(_eval_policy(M, arrs, slots, to_pol(x), kind, seeds))
            if progress: progress((it * pop + j + 1) / (iters * pop), f"Treino: iteração {it + 1} de {iters} · melhor até agora {min(best, min(scores)):.2f} (prática atual {fifo_score:.2f})")
        scores = np.array(scores); el = X[np.argsort(scores)[:max(3, pop // 4)]]
        if scores.min() < best: best, best_x = float(scores.min()), X[np.argmin(scores)].copy()
        mu = 0.7 * el.mean(axis=0) + 0.3 * mu; sd = 0.7 * el.std(axis=0) + 0.3 * sd * 0.6 + 0.02
        hist.append((it + 1, float(scores.min()), float(np.median(scores))))
    # Validação noutro sorteio e no período completo: só fica a política aprendida se bater as alternativas simples (evita sobre-ajuste)
    if progress: progress(1.0, "A validar a política aprendida noutro sorteio...")
    cands = {'rl': to_pol(best_x), 'rl_mean': to_pol(mu), 'edd': {'w': np.eye(nf)[0] + 0.05 * np.eye(nf)[1], 'flex': np.inf},
             'fifo': {'w': np.eye(nf)[1], 'flex': np.inf}}
    va = [scenario_arrivals(M, volume_pct, seed=s_) for s_ in (11, 12)]
    vs = {k: _eval_policy(M, va, slots, pol, kind, [1311, 1312]) for k, pol in cands.items()}
    win = min(vs, key=vs.get)
    return dict(policy=cands[win], chosen=win, val=vs, slots=slots, kind=kind, volume=volume_pct, people=people or {}, hist=hist, fifo_train=fifo_score, best_train=best)


def evaluate_policies(M, R, reps=5):
    """Compara, no período completo e com as mesmas sementes: prática atual (FIFO), regra simples do prazo (EDD) e política aprendida."""
    nf = len(policy_features(M))
    fifo = {'w': np.eye(nf)[1], 'flex': np.inf}
    edd = {'w': np.eye(nf)[0] + 0.05 * np.eye(nf)[1], 'flex': np.inf}
    out = {}
    logs = []
    for name, pol in [('Prática atual (por ordem de chegada)', fifo), ('Regra simples: prazo mais próximo primeiro', edd), ('Política aprendida (RL)', R['policy'])]:
        runs = []
        for s_ in range(reps):
            lg = [] if name.startswith('Política') else None
            runs.append(simulate_process(M, arrivals=scenario_arrivals(M, R['volume'], seed=s_), seed=700 + s_, slots=R['slots'], policy=pol, log=lg))
            if lg is not None: logs += lg
        S = summarize_runs(runs)
        S['obj'] = float(np.mean([_objective(r[0], R['kind']) for r in runs]))
        S['by_type'] = pd.concat([r[0] for r in runs]).groupby('ctype')['lead'].mean().to_dict()
        out[name] = S
    return out, logs


def policy_rules(M, R, logs, ev):
    """Traduz a política aprendida em regras legíveis, com base nos pesos e nas decisões tomadas na simulação."""
    names = policy_features(M); w = np.asarray(R['policy']['w'])
    dec = [l for l in logs if l[0] is not True and l[0] is not False and l[0] == 'flex']
    pri = [l for l in logs if l[0] in (True, False)]
    rules = []
    if pri:
        jumped = np.mean([l[0] for l in pri])
        ch = np.array([l[1] for l in pri if l[0]]); ot = np.array([l[2] for l in pri if l[0]])
        rules.append(("Mudar a ordem da fila em algumas decisões",
                      f"Em {jumped:.0%} das vezes em que havia fila, o agente não escolheu o trabalho que chegou primeiro.",
                      None))
        if len(ch):
            diff = ch.mean(axis=0) - ot.mean(axis=0)
            order = np.argsort(-np.abs(diff * np.sign(w)))
            for i in order[:4]:
                if abs(diff[i]) < 0.05: continue
                n = names[i]
                if i == 0:
                    if diff[i] > 0:
                        rules.append(("Dar prioridade a processos com o prazo mais apertado",
                                      f"Quando passou à frente, o processo escolhido tinha em média {abs(diff[i]) * 10:.1f} dias úteis menos de folga do que os que ficaram à espera.", n))
                    else:
                        rules.append(("Dar prioridade aos processos que ainda vão a tempo de cumprir o prazo",
                                      f"Em sobrecarga, o agente atende primeiro quem ainda pode cumprir o prazo (em média {abs(diff[i]) * 10:.1f} dias úteis mais de folga) "
                                      "em vez de insistir nos que já estão atrasados. Salva mais processos, mas os já atrasados demoram mais: é uma decisão de gestão.", n))
                elif i == 1:
                    rules.append(("Não deixar trabalho envelhecer na fila" if diff[i] > 0 else "Não dar prioridade automática ao que está há mais tempo na fila",
                                  f"O escolhido estava em fila {abs(diff[i]) * 5:.1f} dias úteis {'mais' if diff[i] > 0 else 'menos'} do que a média dos restantes.", n))
                elif i == 2:
                    rules.append(("Despachar primeiro as tarefas curtas" if diff[i] < 0 else "Começar primeiro as tarefas longas",
                                  f"A tarefa escolhida durava em média {abs(diff[i]) * 5:.1f} dias úteis {'menos' if diff[i] < 0 else 'mais'}.", n))
                elif i == 3:
                    rules.append(("Dar prioridade a processos que já tiveram retrabalho" if diff[i] > 0 else "Processos com retrabalho não passam à frente",
                                  "Recupera o tempo perdido nos retornos." if diff[i] > 0 else "Evita que os retornos atrasem os restantes.", n))
                elif i == 4:
                    rules.append(("Fechar primeiro os processos quase concluídos" if diff[i] > 0 else "Fazer avançar os processos no início",
                                  "Termina o que está mais perto do fim antes de pegar em novos." if diff[i] > 0 else "Evita que processos novos fiquem parados no início.", n))
                else:
                    rules.append((f"{'Dar prioridade a' if diff[i] > 0 else 'Baixar a prioridade de'} processos do tipo {n.split(': ', 1)[1]}",
                                  f"Nas decisões em que o agente alterou a ordem, este tipo foi {'mais' if diff[i] > 0 else 'menos'} escolhido do que a sua presença na fila ({abs(diff[i]) * 100:.0f} pontos percentuais).", n))
    if dec and len(dec) / 5 >= 1:
        vc = pd.Series([f"{a} → {b}" for _, a, b in dec]).value_counts()
        rules.append(("Usar outras funções quando uma está saturada",
                      f"{len(dec) / 5:,.0f} tarefas por período foram feitas por outra função que já as fazia no histórico (assumindo +10% de duração). Mais comum: {vc.index[0]}.".replace(',', ' '), None))
    elif np.isfinite(R['policy']['flex']):
        rules.append(("Ajuda entre funções não foi necessária", "Neste contexto, o agente não precisou de passar tarefas para outra função.", None))
    return rules


def fig_rl_compare(ev, key, label):
    names = list(ev); vals = [ev[n].get(key, np.nan) for n in names]
    cols = ['#94a3b8', '#60a5fa', UI['blue']]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=cols[:len(names)], text=[f"{v:.1f}" for v in vals], textposition='inside',
                           insidetextanchor='end', textfont=dict(color='white')))
    fig.update_layout(xaxis_title=label, yaxis_title='', yaxis_autorange='reversed')
    return _style(fig, 220, legend=False)


def fig_rl_types(ev):
    names = list(ev); types = sorted(set().union(*[ev[n]['by_type'] for n in names]))
    if len(types) < 2: return None
    fig = go.Figure()
    for n, col in zip(names, ['#94a3b8', '#60a5fa', UI['blue']]):
        fig.add_bar(x=types, y=[ev[n]['by_type'].get(t, np.nan) for t in types], name=n.split(' (')[0], marker_color=col)
    fig.update_layout(barmode='group', yaxis_title='Duração média (dias)')
    return _style(fig, 300)


def fig_rl_training(R):
    h = pd.DataFrame(R['hist'], columns=['it', 'best', 'med'])
    fig = go.Figure()
    fig.add_scatter(x=h['it'], y=h['med'], name='Mediana das políticas testadas', mode='lines+markers', line=dict(color='#94a3b8'))
    fig.add_scatter(x=h['it'], y=h['best'], name='Melhor política', mode='lines+markers', line=dict(color=UI['blue'], width=3))
    fig.add_hline(y=R['fifo_train'], line=dict(color=UI['red'], dash='dash'), annotation_text='Prática atual', annotation_position='top left')
    fig.update_layout(xaxis_title='Iteração', yaxis_title='Objetivo (menor = melhor)')
    return _style(fig, 260)


RL_SITUATIONS = [('Como hoje', 0), ('Mais trabalho (+20%)', 20), ('Pico de trabalho (+40%)', 40), ('Sobrecarga (+60%)', 60), ('Outro valor', None)]


def _load_preview(M, vol):
    """Como fica a carga numa dada situação (1 simulação rápida): ocupação da função mais carregada e fila por processo."""
    cache = st.session_state.setdefault('rl_preview', {})
    if vol not in cache:
        c, t, u = simulate_process(M, arrivals=scenario_arrivals(M, vol, seed=0), seed=1)
        r = max(u, key=u.get) if u else None
        cache[vol] = dict(role=r, util=u.get(r, 0) if r else 0, queue=t['queue_wait'].sum() / max(len(c), 1),
                          wait=c['wait'].mean(), n=len(c))
    return cache[vol]


def render_sim_rl(caps):
    V = st.session_state.view
    if not caps_ok({'TEMPOS'}, caps):
        ui_note("O simulador precisa de datas de início e de fim por tarefa."); return
    SV = _ensure_sim(V, caps); M = SV['model']
    ui_section_intro('Quando há trabalho em espera, o que se deve fazer primeiro?',
                     'Hoje, quando várias tarefas esperam pela mesma equipa, assume-se que são atendidas por ordem de chegada. '
                     'Aqui, um agente de inteligência artificial (reinforcement learning) testa no simulador outras formas de decidir: '
                     'dar prioridade ao prazo mais próximo, às tarefas mais curtas, a certos tipos de processo, pedir ajuda a outra equipa... '
                     'e diz qual funciona melhor, em regras que a equipa pode aplicar.')
    if not M['has_cap']:
        ui_note("Sem dados de quem executa as tarefas não há filas de trabalho para priorizar: a otimização precisa de pessoas/funções por tarefa."); return
    with st.container(border=True):
        ui_card_header('1. O que quer melhorar?', None, 'bi-bullseye',
                       help="O agente procura a forma de decidir que mais melhora este indicador.")
        kind = st.radio('Objetivo', list(RL_OBJECTIVES), format_func=lambda k: RL_OBJECTIVES[k], key='rl_kind', horizontal=True,
                        index=0 if caps_ok({'PLANO'}, caps) else 1, label_visibility='collapsed')
        if kind == 'atrasos' and not caps_ok({'PLANO'}, caps):
            st.caption("Sem datas planeadas, 'menos atrasos' usa a duração média.")
    with st.container(border=True):
        ui_card_header('2. Em que situação de trabalho?', 'A ordem de atendimento só importa quando há filas, ou seja, quando há mais trabalho do que as equipas conseguem fazer de imediato.', 'bi-speedometer2',
                       help="Escolha o volume de processos a entrar, face ao que aconteceu no histórico. Ex.: +40% = entram 40% mais processos no mesmo período, com as mesmas pessoas. "
                            "Serve para preparar a equipa para picos ou crescimento.")
        labels = [l for l, _ in RL_SITUATIONS]
        sit = st.radio('Situação', labels, index=2, key='rl_sit', horizontal=True, label_visibility='collapsed')
        vol = dict(RL_SITUATIONS)[sit]
        if vol is None:
            vol = st.slider('Variação do volume de processos face ao histórico (%)', -20, 100, value=30, step=10, key='rl_vol_custom')
        pv = _load_preview(M, vol); pv0 = _load_preview(M, 0)
        cols = st.columns(3)
        cols[0].metric("Processos a entrar", f"{pv['n']:,}".replace(',', ' '), f"{vol:+d}% face ao histórico" if vol else "igual ao histórico", delta_color='off',
                       help="Nº de processos simulados no período histórico, com a variação escolhida.")
        cols[1].metric(f"Ocupação de {pv['role']}", f"{pv['util']:.0f}%", f"{pv['util'] - pv0['util']:+.0f} pp vs. hoje" if vol else None, delta_color='inverse',
                       help="A equipa mais carregada nesta situação. Acima de 85% formam-se filas.")
        cols[2].metric("Espera em fila por processo", f"{pv['queue']:.1f} dias", f"{pv['queue'] - pv0['queue']:+.1f} vs. hoje" if vol else None, delta_color='inverse',
                       help="Dias que cada processo passa, em média, à espera de alguém livre. É este tempo que uma melhor priorização pode reduzir.")
        if pv['queue'] < 0.3:
            ui_note("Nesta situação quase não há filas: as equipas têm folga e a ordem de atendimento praticamente não muda nada. Escolha uma situação com mais trabalho para ver diferenças.")
        elif pv['util'] >= 92:
            ui_note("Nesta situação as equipas estão em sobrecarga: nenhuma regra de prioridade compensa a falta de capacidade (veja os Cenários para testar reforços). O agente ainda pode reduzir os danos.")
    with st.container(border=True):
        ui_card_header('3. Opções', None, 'bi-gear')
        c1, c2 = st.columns([2, 1])
        flex = c1.checkbox('Permitir que outra equipa ajude quando uma está sobrecarregada', value=True, key='rl_flex',
                           help="Só em atividades que essa outra equipa já fez no histórico (ex.: Analistas de Operações a fazer parte da Análise Documental). Assume-se que demora mais 10%.")
        mode = c2.selectbox('Treino', ['Rápido (~1 min)', 'Completo (~3 min)'], key='rl_mode', label_visibility='collapsed',
                            help="O treino completo testa mais alternativas e costuma encontrar resultados um pouco melhores.")
        go_ = st.button('Procurar a melhor forma de decidir', type='primary', icon=":material/model_training:", key='rl_go',
                        help="O agente experimenta centenas de regras de prioridade no simulador, confirma a melhor noutra amostra e compara-a com a prática atual.")
    if go_:
        bar = st.progress(0.0, text="A preparar o treino...")
        it, pop = (6, 10) if mode.startswith('Rápido') else (10, 16)
        try:
            R = train_policy(M, volume_pct=vol, kind=kind, allow_flex=flex, iters=it, pop=pop, progress=lambda p, t: bar.progress(min(p, 1.0), text=t))
            R['situation'] = sit
            bar.progress(1.0, text="A comparar no período completo: prática atual, regra simples e política aprendida...")
            ev, logs = evaluate_policies(M, R)
            st.session_state.rl_result = (R, ev, policy_rules(M, R, logs, ev))
        except Exception as e:
            st.error(f"Não foi possível treinar: {e}")
        bar.empty()
    if not st.session_state.get('rl_result'): return
    R, ev, rules = st.session_state.rl_result
    names = list(ev); base, edd, rl = ev[names[0]], ev[names[1]], ev[names[2]]
    gain = (base['obj'] - rl['obj']) / base['obj'] * 100 if base['obj'] else 0
    if abs(gain) < 0.5: gain = 0.0
    sit_txt = R.get('situation') or (f"volume {R['volume']:+d}%" if R['volume'] else 'como hoje')
    pol_txt = {'fifo': 'Manter a ordem de chegada', 'edd': 'Atender primeiro os processos com o prazo mais próximo'}.get(R.get('chosen'), 'Aplicar a política aprendida pelo agente (regras abaixo)')
    if R['kind'] == 'atrasos' and 'late' in rl:
        eff = f"os processos atrasados passam de {base['late']:.1f}% para {rl['late']:.1f}%"
    elif R['kind'] == 'p90':
        eff = f"os 10% de processos mais lentos passam de {base['p90']:.1f} para {rl['p90']:.1f} dias"
    else:
        eff = f"a duração média passa de {base['lead']:.1f} para {rl['lead']:.1f} dias"
    if abs(gain) <= 2:
        vt, vd, colr = ("Nesta situação, a ordem de atendimento não faz diferença",
                        "As equipas têm capacidade suficiente para quase não haver filas, por isso decidir o que fazer primeiro não muda os resultados. "
                        "Para melhorar, atue nas esperas estruturais ou no retrabalho (ver Cenários), ou teste uma situação com mais trabalho.", '#64748b')
    else:
        vt, vd, colr = (f"{pol_txt}", f"Na situação «{sit_txt}», {eff} (melhoria de {gain:.0f}% no objetivo face à prática atual).", '#16a34a')
        if rl['p90'] > base['p90'] * 1.05:
            vd += (f" Atenção: os processos mais lentos ficam ainda mais lentos (P90 de {base['p90']:.0f} para {rl['p90']:.0f} dias). "
                   "A política sacrifica alguns casos já perdidos para salvar mais processos dentro do prazo.")
            colr = '#d97706'
    st.markdown(f"""<div class='pm-verdict' style='border-left-color:{colr}'><div class='pm-verdict-top'><span class='pm-pill' style='background:{colr}1a;color:{colr}'>Resposta</span>
        <span class='pm-src'>Objetivo: {RL_OBJECTIVES[R['kind']].lower()} · situação: {html.escape(sit_txt)}</span></div>
        <div class='pm-verdict-title'>{html.escape(vt)}</div><div class='pm-verdict-text'>{html.escape(vd)}</div></div>""", unsafe_allow_html=True)
    k = [dict(icon='bi-trophy', value=(f"{gain:+.0f}%" if abs(gain) > 2 else "≈ 0%"), label='Melhoria do objetivo face à prática atual', good=gain > 2 if abs(gain) > 2 else None,
              help='Quanto o objetivo escolhido melhora com a política recomendada, face a atender por ordem de chegada. Avaliado no período completo, 5 réplicas.'),
         dict(icon='bi-stopwatch', value=_fmt_days(rl['lead']), label='Duração média com a política', delta=_delta_txt(rl['lead'], base['lead'], ' d'), good=(rl['lead'] < base['lead'] * 0.99) or None),
         dict(icon='bi-graph-up-arrow', value=_fmt_days(rl['p90']), label='P90 com a política', delta=_delta_txt(rl['p90'], base['p90'], ' d'), good=(rl['p90'] < base['p90'] * 0.99) or None)]
    if 'late' in rl:
        k.append(dict(icon='bi-alarm', value=f"{rl['late']:.1f}%", label='Atrasados com a política', delta=_delta_txt(rl['late'], base['late'], pp=True), good=rl['late'] < base['late'] - 0.3 or None))
    ui_kpis(k)
    if abs(gain) > 2 and R.get('chosen') == 'edd':
        ui_note("O agente testou regras mais complexas, mas a que melhor resultou foi a regra simples do prazo mais próximo. Boa notícia: aplica-se já, sem tecnologia.")
    elif abs(gain) > 2 and R.get('chosen') not in ('edd', 'fifo') and (edd['obj'] - rl['obj']) / max(edd['obj'], 1e-9) * 100 < 2:
        ui_note("A regra simples 'prazo mais próximo primeiro' consegue quase o mesmo que o agente: é uma regra fácil de aplicar já, sem tecnologia.")
    if abs(gain) <= 2: rules = []
    with st.container(border=True) if rules else st.empty():
        if rules: ui_card_header('Regras de decisão recomendadas', 'O que o agente faz de diferente, em linguagem de negócio', 'bi-signpost-2')
        if False:
            st.caption("Sem diferenças relevantes face à prática atual.")
        for i, (t, d, f) in enumerate(rules):
            st.markdown(f"<div class='pm-prio' style='margin:6px 0'><div class='pm-prio-num'>{i + 1}</div><div style='flex:1'><div class='pm-prio-title'>{html.escape(t)}</div>"
                        f"<div class='pm-prio-problem'>{html.escape(d)}</div></div></div>", unsafe_allow_html=True)
    lab = {'atrasos': '% atrasados + dias de atraso', 'duracao': 'Duração média (dias)', 'p90': 'Duração P90 (dias)'}[R['kind']]
    c1, c2 = st.columns(2)
    with c1: ui_chart(fig_rl_compare(ev, 'obj', lab), 'Prática atual vs. regra simples vs. agente', 'Período completo, 5 réplicas, mesmas sementes', 'bi-bar-chart', key='rl_cmp')
    with c2: ui_chart(fig_rl_types(ev), 'Efeito por tipo de processo', 'Verifique que nenhum tipo fica prejudicado', 'bi-diagram-3', key='rl_types')
    with st.expander("Como o agente aprendeu"):
        ui_chart(fig_rl_training(R), 'Evolução do treino', 'Em cada iteração o agente testa várias políticas no simulador e refina as melhores', 'bi-activity', key='rl_tr')
        w = pd.Series(R['policy']['w'], index=policy_features(M))
        st.dataframe(pd.DataFrame({'Critério': w.index, 'Peso na prioridade': w.round(2).values}).sort_values('Peso na prioridade', key=abs, ascending=False),
                     hide_index=True, use_container_width=True)
        st.caption("Método: pesquisa direta de política (Cross-Entropy Method), uma família de reinforcement learning sem modelo de gradiente. A prioridade de cada trabalho em fila "
                   "é uma soma ponderada dos critérios acima; o agente aprende os pesos e o limiar a partir do qual pede ajuda a outra função. "
                   "Escolhido por ser rápido e explicável: cada decisão pode ser justificada.")
    render_director_summary('rl', sim_facts(), [f"{vt}. {vd}"] +
                            [f"{t}." for t, _, _ in rules[:3]])


def rl_page():
    ui_page_header("Simulação e Otimização", "Teste decisões antes de as tomar: um simulador do seu processo, calibrado com o histórico")
    with st.expander("Como funciona esta página (ler primeiro)", expanded=not st.session_state.get('sim_help_seen', False)):
        st.session_state.sim_help_seen = True
        st.markdown("""
**O que é o simulador?** É uma réplica virtual do seu processo, construída a partir do histórico: que atividades existem, por que ordem acontecem,
quanto tempo demoram, quanto tempo os processos esperam entre elas e quantas pessoas há em cada função. Em vez de experimentar mudanças na realidade
(contratar, mudar regras, reorganizar), experimenta-as primeiro aqui e vê o efeito provável.

**Os três passos**
1. **Modelo e validação:** a app constrói o simulador e põe-no à prova, repetindo os processos reais. Se os resultados simulados ficam perto dos reais, pode confiar nos passos seguintes.
2. **Cenários "e se":** altera uma coisa (mais volume, mais pessoas, menos espera, menos retrabalho...) e compara com a situação atual.
3. **Otimização (RL):** um agente de inteligência artificial experimenta muitas formas de decidir *o que atender primeiro* e recomenda a melhor, em regras simples.

**Termos usados**
- **Base:** a situação atual, simulada. Os cenários comparam-se sempre com ela.
- **Réplica:** cada simulação tem alguma aleatoriedade (como a realidade). Por isso cada cenário corre 5 vezes e mostramos a média.
- **Calibrar / Recalibrar:** ajustar o simulador ao histórico. Acontece sozinho na primeira vez; só precisa de recalibrar se carregar dados novos.
- **Fila (falta de capacidade):** espera porque as pessoas da função seguinte estão ocupadas. Desce com mais pessoas ou melhor organização.
- **Espera estrutural:** espera que não depende de pessoas livres: agendamentos, aprovações, documentos de terceiros. Só desce mudando regras ou prazos.
- **Ocupação:** percentagem do tempo em que a capacidade de uma função está em uso. Acima de 85% qualquer pico de trabalho gera filas.
- **P90:** a duração que só 10% dos processos ultrapassam (os casos mais lentos).
""")
    if not st.session_state.analysis_run:
        st.warning("É necessário executar a análise inicial primeiro. Vá à página de 'Dados e Configurações' para carregar os dados."); return
    caps = st.session_state.get('caps')
    if st.session_state.get('view') is None:
        _dfp = st.session_state.data_frames_processed
        _alloc = st.session_state.get('dfs_norm', st.session_state.dfs)['resource_allocations']
        st.session_state.view = prepare_view(_dfp['projects'], _dfp['tasks'], _dfp['resources'], _alloc)
    tabs = {'modelo': '1 · Modelo e validação', 'cenarios': '2 · Cenários "e se"', 'rl': '3 · Otimização (RL)'}
    ch = st.segmented_control("Etapa", list(tabs), format_func=lambda x: tabs[x], default=st.session_state.get('sim_tab', 'modelo'),
                              key='sim_tab_ctl', label_visibility='collapsed')
    st.session_state.sim_tab = ch or st.session_state.get('sim_tab', 'modelo')
    t = st.session_state.sim_tab
    if t == 'modelo':
        render_sim_validation(caps)
    elif t == 'cenarios':
        render_sim_scenarios(caps)
    elif t == 'rl':
        render_sim_rl(caps)


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
            st.markdown("<div class='pm-brand'><span class='pm-brand-badge'><i class='bi bi-diagram-3-fill'></i></span>Process Intelligence</div>", unsafe_allow_html=True)
            cur = st.session_state.current_page
            for label, icon, page in [("Análise de Processos", ":material/monitoring:", "Dashboard"),
                                      ("Simulação", ":material/smart_toy:", "RL"),
                                      ("Dados e Configurações", ":material/database:", "Settings")]:
                if st.button(label, icon=icon, use_container_width=True, type="primary" if cur == page else "secondary", key=f"navp_{page}"):
                    st.session_state.current_page = page
                    st.rerun()
            st.markdown("<hr style='margin:0.8rem 0; border-color:#f1f5f9'>", unsafe_allow_html=True)
            _u = st.session_state.get("user_name", "Admin")
            st.markdown(f"<div class='pm-user'><span class='pm-avatar'>{_u[:2].upper()}</span><div><div class='pm-user-name'>{_u}</div><div class='pm-user-role'>Analista</div></div></div>", unsafe_allow_html=True)
            if st.button("Terminar sessão", icon=":material/logout:", use_container_width=True, key="navp_logout"):
                st.session_state.authenticated = False
                for key in list(st.session_state.keys()):
                    if key not in ['authenticated']: del st.session_state[key]
                st.rerun()
                
        if st.session_state.current_page == "Dashboard":
            dashboard_page()
            _scroll_top_if_moved()
        elif st.session_state.current_page == "RL":
            rl_page()
        elif st.session_state.current_page == "Settings":
            settings_page()
        if st.session_state.current_page != "Dashboard":
            _scroll_top_if_moved()

if __name__ == "__main__":
    main()
