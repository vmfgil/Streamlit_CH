import difflib
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
    .stApp { background-color: var(--background-color); color: var(--text-color); }
    h1, h2, h3, h4, h5 { color: var(--text-color); font-weight: 700; letter-spacing: -0.01em; }
    .block-container { padding-top: 2.2rem; max-width: 1500px; }
    header[data-testid="stHeader"] { background: transparent; }

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
    d = raw.get('dependencies')
    if d is None or d.empty:
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
if 'rl_params_expanded' not in st.session_state: st.session_state.rl_params_expanded = True# Alteração Sugerida
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

# --- FUNÇÃO DE ANÁLISE (REINFORCEMENT LEARNING) ---
#@st.cache_data # Removido para permitir interatividade e barra de progresso
def run_rl_analysis(dfs, project_id_to_simulate, num_episodes, reward_config, progress_bar, status_text, agent_params=None):
    if agent_params is None:
        agent_params = {}    
    dfs = {key: df.copy() for key, df in dfs.items()}
    
    # --- PASSO 1: CONVERTER TODAS AS DATAS NOS DADOS ORIGINAIS PRIMEIRO ---
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
    
    # --- AMBIENTE E AGENTE (CLASSES) --- 
    class PortfolioManagementEnv:
        def __init__(self, dfs, reward_config):
            self.rewards = reward_config

            # 1. Carregar e pré-processar TODOS os dados de uma vez
            self.df_projects = dfs['projects'].sort_values('start_date').reset_index(drop=True)
            self.df_resources = dfs['resources']
            self.df_dependencies = dfs['dependencies']
            self.df_tasks = dfs['tasks']
            self.df_allocations = dfs['resource_allocations']
            
            # 2. Juntar alocações, tarefas e recursos para saber "quem fez o quê"
            # Assegura que os IDs são strings para o merge
            self.df_allocations['task_id'] = self.df_allocations['task_id'].astype(str)
            self.df_allocations['resource_id'] = self.df_allocations['resource_id'].astype(str)
            self.df_tasks['task_id'] = self.df_tasks['task_id'].astype(str)
            self.df_resources['resource_id'] = self.df_resources['resource_id'].astype(str)
            
            df_merged_logic = self.df_allocations.merge(self.df_tasks[['task_id', 'task_type']], on='task_id')
            df_merged_logic = df_merged_logic.merge(self.df_resources[['resource_id', 'resource_type']], on='resource_id')
            
            # 3. Agrupar para encontrar as combinações únicas
            df_map = df_merged_logic[['task_type', 'resource_type']].drop_duplicates().dropna()
            
            # 4. Construir o TASK_TYPE_RESOURCE_MAP dinamicamente
            self.TASK_TYPE_RESOURCE_MAP = {}
            for task_type, group in df_map.groupby('task_type'):
                self.TASK_TYPE_RESOURCE_MAP[task_type] = group['resource_type'].tolist()
            
            print("--- MAPA DE RECURSOS/TAREFAS GERADO DINAMICAMENTE ---")
            print(self.TASK_TYPE_RESOURCE_MAP)
                        
            # 5. Criar dicionários para acesso rápido
            self.tasks_by_project = {pid: df_group.to_dict('records') for pid, df_group in dfs['tasks'].groupby('project_id')}
            self.dependencies_by_project = {pid: {str(row['task_id_successor']): str(row['task_id_predecessor']) for _, row in df_group.iterrows()} for pid, df_group in self.df_dependencies.groupby('project_id')}
            
            self.resource_types = sorted(self.df_resources['resource_type'].unique().tolist())
            self.resources_by_type = {rt: self.df_resources[self.df_resources['resource_type'] == rt] for rt in self.resource_types}
            self.resource_capacity_map = pd.Series(self.df_resources.daily_capacity.values, index=self.df_resources.resource_id).to_dict()
            
        def reset(self):
            # 6. Reset GERAL da simulação
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
            # 7. O estado descreve o PORTFÓLIO, não um projeto
            num_active = len(self.active_projects)
            pending_tasks_count = 0
            high_prio_tasks_count = 0
        
            max_pending_days = 0 
            
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
                    
            # Estado para o RL
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
                allowed_resources = self.TASK_TYPE_RESOURCE_MAP.get(task_type, [])
                return check_resource_type in allowed_resources
            
            return True # Retorna True se não estiver a verificar um recurso específico (elegibilidade geral)

        def get_possible_actions_for_state(self):
            # 8. Procura tarefas elegíveis em TODOS os projetos ativos
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

            # O loop  processa a lista de ações detalhada
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

                    # --- Atualizar Elegibilidade de Sucessoras ---
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

                    
            for proj_id, proj_state in self.active_projects.items():
                if proj_id not in self.detailed_logs:
                    self.detailed_logs[proj_id] = []
                
                # Obter o último estado de progresso ou começar do zero
                last_log = self.detailed_logs[proj_id][-1] if self.detailed_logs[proj_id] else {'cumulative_cost': 0.0, 'cumulative_hours': 0.0}
                
                self.detailed_logs[proj_id].append({
                    'date': self.current_date,
                    'day': np.busday_count(proj_state['start_date'].date(), self.current_date.date()), 
                    'cumulative_cost': proj_state['current_cost'], 
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
            # Calcular Penalização DINÂMICA por Tarefas Pendentes/Atrasadas ---
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
    
    # Calcular o total de horas reais trabalhadas por tarefa a partir das alocações
    real_hours_per_task_series = df_resource_allocations.groupby('task_id')['hours_worked'].sum()
    real_hours_per_task = real_hours_per_task_series.to_dict()
                
    # --- Lógica de Simulação de Portfólio ---
    
    # 1. Instanciar o novo ambiente de portfólio.
    # Amostra completa de projetos para simular o portfólio.
    env = PortfolioManagementEnv(dfs={'projects': df_projects, 'tasks': df_tasks, 'resources': df_resources, 'dependencies': df_dependencies, 'resource_allocations': df_resource_allocations}, reward_config=reward_config)

    # 2. Simplificar as ações para o agente.
    # Mantémm a Q-Table com um tamanho gerenciável.
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
            
            # ---  LÓGICA DE ALOCAÇÃO DIÁRIA ---
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
        # As métricas de duração e custo referem-se ao desempenho do portfólio inteiro.
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

    # 5. Geração de Gráficos de Treino
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    rewards, durations, costs, epsilon_history = agent.episode_rewards, agent.episode_durations, agent.episode_costs, agent.epsilon_history
    rolling_avg_window = 50
    axes[0, 0].plot(rewards, alpha=0.3, label='Recompensa do Episódio'); axes[0, 0].plot(pd.Series(rewards).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)'); axes[0, 0].set_title('Recompensa por Episódio'); axes[0, 0].set_xlabel('Episódio'); axes[0, 0].set_ylabel('Recompensa Total'); axes[0, 0].legend(); axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].plot(durations, alpha=0.3, label='Duração do Portfólio'); axes[0, 1].plot(pd.Series(durations).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)'); axes[0, 1].set_title('Duração Total do Portfólio'); axes[0, 1].set_xlabel('Episódio'); axes[0, 1].set_ylabel('Soma das Durações (dias)'); axes[0, 1].legend(); axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].plot(epsilon_history, color='green'); axes[1, 0].set_title('Decaimento do Epsilon (Exploração)'); axes[1, 0].set_xlabel('Episódio'); axes[1, 0].set_ylabel('Valor de Epsilon'); axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].plot(costs, alpha=0.3, label='Custo do Portfólio'); axes[1, 1].plot(pd.Series(costs).rolling(rolling_avg_window, min_periods=1).mean(), lw=2.5, color='orange', label=f'Média Móvel ({rolling_avg_window} ep)'); axes[1, 1].set_title('Custo Total do Portfólio'); axes[1, 1].set_xlabel('Episódio'); axes[1, 1].set_ylabel('Custo Total (€)'); axes[1, 1].legend(); axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    plots['training_metrics'] = convert_fig_to_bytes(fig)

    # 6. AVALIAÇÃO FINAL
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
            # Remove a tarefa escolhida da lista de trabalho para este dia.
            work_actions.remove(best_task_action)
            simplified_options = list(set([(a[0], a[1]) for a in work_actions]))
            
            # Se a tarefa ficar sem esforço restante, removemo-la das opções para o resto do dia
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
    st.info("Carregue os ficheiros que tiver. Só o ficheiro de **tarefas** é obrigatório; com os 5 ficheiros a análise fica completa. "
            "A app deteta automaticamente que análises os seus dados permitem e esconde as restantes.")
    # Dicionário com os textos para as tooltips
    tooltips = {
        'projects': "OPCIONAL. 1 linha por processo. Colunas: project_id; recomendadas: project_name, start_date, end_date, planned_end_date (ativa a análise de atrasos), budget_impact (ativa a análise de orçamento), project_status, path_name. Se não existir, início/fim de cada processo são derivados das tarefas.",
        'tasks': "OBRIGATÓRIO. 1 linha por tarefa. Colunas mínimas: project_id, task_name e uma data (start_date ou end_date). Com start_date E end_date ativam-se os tempos de execução e espera. Opcionais: task_id, task_type, priority, estimated_effort. Se não tiver resource_allocations, pode indicar aqui quem fez a tarefa com uma coluna resource_name (ou resource_id) e opcionalmente resource_type (função); várias pessoas separadas por ';'.",
        'resources': "OPCIONAL. 1 linha por recurso: resource_id, resource_name, resource_type (função), cost_per_hour (ativa custos), daily_capacity, skill_level (ativa competências).",
        'resource_allocations': "OPCIONAL. Horas efetivamente trabalhadas: task_id, resource_id, allocation_date, hours_worked. Ativa as análises de esforço, eficiência e (com cost_per_hour) custos.",
        'dependencies': "OPCIONAL. project_id, task_id_predecessor, task_id_successor. Se não existir, a sequência é derivada da ordem das tarefas."
    }
    file_names = ['projects', 'tasks', 'resources', 'resource_allocations', 'dependencies']
    labels = {'projects': 'projects.csv', 'tasks': 'tasks.csv ★', 'resources': 'resources.csv',
              'resource_allocations': 'resource_allocations.csv', 'dependencies': 'dependencies.csv'}

    upload_cols = st.columns(5)
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

    raw = {k: v for k, v in st.session_state.dfs.items() if v is not None}
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
                    ["Vários ficheiros (até 5 tabelas)", "Um único ficheiro (event log)"],
                    horizontal=True, key="modo_upload",
                    help="Vários ficheiros: projects, tasks, resources, resource_allocations, dependencies (só tasks é obrigatório). "
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
    if st.button("🚀 Iniciar Análise Inicial (PM & EDA)", use_container_width=True):
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
                    (plots_eda.get('plot_26'), "Gráfico de Dependências: Processo Exemplo"),
                    (plots_eda.get('plot_27'), "Relação entre Complexidade e Atraso"),
                    (plots_eda.get('plot_28'), "Relação entre Dependências e Desvio de Custo"),
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


def ui_card_header(title, subtitle=None, icon=None):
    ic = f"<span class='pm-card-icon'><i class='bi {icon}'></i></span>" if icon else ""
    sub = f"<div class='pm-card-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"<div class='pm-card-head'>{ic}<div><div class='pm-card-title'>{title}</div>{sub}</div></div>", unsafe_allow_html=True)


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
            lines.append(f'"__start" -> "{a}" [color="#94a3b8" penwidth={1 + 4 * n / total_cases:.2f} label="{n / total_cases:.0%}"];')
    for a, n in ends.items():
        if a in keep and n >= min_share * total_cases:
            lines.append(f'"{a}" -> "__end" [color="#94a3b8" penwidth={1 + 4 * n / total_cases:.2f}];')
    for _, r in e.iterrows():
        pw = 1 + 5 * r['n'] / total_cases
        if metric == 'wait':
            lab = f"{r['wait']:.1f} d" if r['wait'] >= 0.05 else "0 d"; col = wcol(r['wait'])
        else:
            lab = f"{r['n'] / total_cases:.0%}"; col = '#475569'
        lines.append(f'"{r["prev_task"]}" -> "{r["task_name"]}" [label="{lab}" color="{col}" penwidth={pw:.2f}];')
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
        pe = gate_outputs(st.session_state.plots_eda, caps)
        for k_, t_ in [('plot_26', 'Dependências de um processo exemplo'), ('plot_24', 'Distribuição da complexidade dos processos')]:
            if pe.get(k_) is not None:
                st.markdown(f"**{t_}**"); st.image(pe[k_].getvalue(), use_container_width=True)


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
    c1, c2 = st.columns(2)
    with c1: ui_chart(fig_role_handoffs(T), 'Passagens de trabalho entre funções', 'Nº de passagens; no hover, a espera média em cada passagem', 'bi-arrow-left-right', key='rs_rh')
    with c2:
        if has_h: ui_chart(fig_efficiency(T, A), 'Eficiência comparada dentro de cada função', 'Cada ponto é uma pessoa (≥5 tarefas)', 'bi-person-check', key='rs_ef')
        else: ui_chart(fig_teamsize(P), 'Duração vs. nº de pessoas envolvidas', None, 'bi-people', key='rs_ts0')
    if has_h:
        ui_chart(fig_resource_wait(T), 'Espera gerada a seguir a cada pessoa', 'Tempo médio até a tarefa seguinte começar (pessoas com ≥10 tarefas)', 'bi-sign-stop', key='rs_rw')
        ui_chart(fig_resource_activity(A, has_h), 'Especialização: horas por pessoa e atividade', 'Top 25 pessoas por horas', 'bi-grid', key='rs_ra')
    with st.expander("Análise detalhada: tamanho das equipas, competências e composição"):
        if has_h: ui_chart(fig_teamsize(P), 'Duração vs. nº de pessoas envolvidas', None, 'bi-people', key='rs_ts')
        pe = gate_outputs(st.session_state.plots_eda, caps); po = gate_outputs(st.session_state.plots_post_mining, caps)
        for src, k_, t_ in [(po, 'skill_vs_performance_adv', 'Competência vs. horas por tarefa'), (pe, 'plot_23', 'Atraso por nível de competência'),
                            (pe, 'plot_14', 'Atraso médio por recurso'), (pe, 'plot_07', 'Nº médio de recursos por processo, por trimestre')]:
            if src.get(k_) is not None:
                st.markdown(f"**{t_}**"); st.image(src[k_].getvalue(), use_container_width=True)


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
        pe = gate_outputs(st.session_state.plots_eda, caps)
        for k_, t_ in [('plot_29', 'Matriz de correlação'), ('plot_18', 'Custo real vs. atraso'), ('plot_19', 'Custo/hora médio vs. atraso'),
                       ('plot_20', 'Nº de recursos vs. custo'), ('plot_22', 'Atraso por faixa de orçamento'), ('plot_27', 'Complexidade vs. atraso'),
                       ('plot_28', 'Dependências vs. desvio de custo'), ('plot_17', 'Custo por faixa de orçamento e função')]:
            if pe.get(k_) is not None:
                st.markdown(f"**{t_}**"); st.image(pe[k_].getvalue(), use_container_width=True)


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


def dashboard_page():

    ### START PDF/AI STATE INIT (if not global) ###
    if 'show_ai_modal' not in st.session_state: st.session_state.show_ai_modal = False
    if 'gemini_analysis' not in st.session_state: st.session_state.gemini_analysis = None
    if 'pdf_bytes_ready' not in st.session_state: st.session_state.pdf_bytes_ready = None
    ### END PDF/AI STATE INIT ###

    ### START PDF/AI BUTTONS & MODAL ###
    if st.session_state.analysis_run:
        col_buffer, col_pdf, col_ai = st.columns([6, 1.25, 1.25], vertical_alignment='center')
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
        
        # --- Botão Toggle + Área de Contexto ---
        with col_ai:
            # Botão para MOSTRAR/ESCONDER a área de preparação
            button_label = "🤖 Esconder AI Prep" if st.session_state.get("show_ai_prep_area", False) else "🤖 AI Prep"
            if st.button(button_label, help="Preparar dados para análise externa com Gemini", use_container_width=True):
                # Inverte o estado de visibilidade da área de preparação
                st.session_state.show_ai_prep_area = not st.session_state.get("show_ai_prep_area", False)
                st.rerun() # Rerun para mostrar/esconder a área abaixo

    # --- Área Condicional para Preparar Contexto ---
    if st.session_state.get("show_ai_prep_area", False):
        st.markdown("---") # Separador visual
        with st.container(border=True): # Container para agrupar visualmente
            st.subheader("Preparar Análise com Gemini (Externo)")
            st.info("Siga estes passos:\n1. Copie o prompt abaixo.\n2. Descarregue o PDF com os resultados.\n3. Clique no link 'Abrir Gemini' (abrirá numa nova aba).\n4. No Gemini, cole o prompt e faça upload do PDF descarregado.")

            pdf_generated = False
            pdf_bytes_context = None
            prompt_text = ""
            error_generating = None

            # Gerar o PDF e o Prompt
            
            @st.cache_data(show_spinner="Preparando dados para IA...")
            def prepare_ai_data():
                _pdf_bytes = None
                _prompt = "# Erro ao gerar prompt #"
                _pdf_ok = False
                _error = None
                try:
                    # Gerar PDF
                    _plots_pre, _tables_pre, _plots_post, _plots_eda, _tables_eda = get_visible_outputs()
                    _pdf_bytes = generate_pdf_report(_plots_pre, _tables_pre, _plots_post, _plots_eda, _tables_eda)
                    if _pdf_bytes and len(_pdf_bytes) > 100:
                        _pdf_ok = True
                    else: _error = "Falha ao gerar conteúdo do PDF."


                    # Gerar Prompt
                    _app_code = "# Erro ao ler código #"
                    try:
                         _script_path = Path(inspect.getfile(inspect.currentframe())).resolve()
                         _app_code = _script_path.read_text(encoding='utf-8')
                    except Exception as _e1:
                         print(f"Erro AI Prep - Ler código Path: {_e1}")
                         try: _app_code = inspect.getsource(sys.modules[__name__])
                         except Exception as _e2: print(f"Erro AI Prep - Ler código Inspect: {_e2}")

                    _prompt_instruction = "Coloco em anexo o Relatório em PDF com os resultados das análises obtidas (gráficos/tabelas) sobre o meu processo, e preciso que considerando todas a análises constantes do PDF, me faças um resumo alargado dos principais aspetos, conclusões, problemas ou insights sobre o meu processo, e isto deve incluir mas não se resumir a recursos, atrasos, custos, gargalos, tempos de espera, tempos de execução, rework, entre outros aspetos."
                    _prompt = f"{_prompt_instruction}" # REMOVIDA a parte que adicionava o _app_code

                except Exception as e:
                    print(f"Erro geral em prepare_ai_data: {e}")
                    _error = f"Erro ao preparar dados: {e}"

                return _prompt, _pdf_bytes, _pdf_ok, _error

            # Chama a função cacheada para obter os dados
            prompt_text, pdf_bytes_context, pdf_generated, error_generating = prepare_ai_data()

            if error_generating:
                st.error(error_generating)

            # Mostrar Prompt e Botões
            st.text_area("1. Prompt para Gemini:", prompt_text, height=250, key="ai_prompt_text_area")

            if pdf_generated and pdf_bytes_context:
                st.download_button(
                    label="2. Descarregar PDF com Resultados",
                    data=pdf_bytes_context,
                    file_name="relatorio_process_mining_analise.pdf",
                    mime="application/pdf",
                    key="pdf_download_context_ai"
                )
            else:
                st.warning("Geração do PDF falhou ou PDF está vazio. O Gemini não poderá analisar os gráficos.")

            
            st.markdown('[3. Abrir Gemini (Nova Aba)](https://gemini.google.com/)', unsafe_allow_html=True)
            # ----------------------------------------

        st.markdown("---") 
        ### END AI CONTEXT AREA ###
    
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
    if sec == "diagnostico": render_diagnostics_page()
    elif sec == "visao_geral": section_overview(V, caps)
    elif sec == "fluxo": section_flow(V, caps)
    elif sec == "tempos": section_times(V, caps)
    elif sec == "recursos": section_resources(V, caps)
    elif sec == "custos": section_costs(V, caps)


# --- (REINFORCEMENT LEARNING) ---
def rl_page():
    ui_page_header("Simulação (Reinforcement Learning)", "Treine um agente para testar políticas de alocação de recursos")

    if not st.session_state.analysis_run:
        st.warning("É necessário executar a análise inicial primeiro. Vá à página de 'Configurações' para carregar os dados.")
        return

    caps = st.session_state.get('caps')
    if not caps_ok({'TEMPOS', 'RECURSOS'}, caps):
        st.warning("A simulação precisa de datas de início e fim por tarefa **e** de saber quem executou cada tarefa. "
                   "Os dados carregados não têm essa informação (ver 'Dados detetados' em Configurações).")
        return
    if not caps_ok({'ESFORCO'}, caps):
        st.info("ℹ️ Sem horas trabalhadas: o esforço de cada tarefa é estimado em 8h por dia útil de duração. Os resultados são indicativos.")

    #  CRIAR A AMOSTRA DE RL APENAS UMA VEZ ---
    # Se a amostra de IDs ainda não foi criada, cria-a e guarda-a no estado da sessão.
    if 'rl_sample_ids' not in st.session_state:
        proj_ids = st.session_state.get('dfs_norm', st.session_state.dfs)['projects']['project_id'].astype(str)
        n = min(100, len(proj_ids))
        st.session_state['rl_sample_ids'] = proj_ids.sample(n=n, random_state=42).tolist()

    # -----------------------------------------------------------------

    st.info("Esta secção permite treinar um agente de IA para otimizar a gestão de processos. O treino e a análise correm sobre uma amostra de 100 processos para garantir a performance.")

    with st.expander("⚙️ Parâmetros da Simulação", expanded=st.session_state.rl_params_expanded):
        st.markdown("<p><strong>Parâmetros Gerais</strong></p>", unsafe_allow_html=True)
        
        # As opções vêm da amostra que acabou de ser criada.
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
        
        # Criar reward_config DENTRO do botão para usar valores atuais
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
                st.session_state.get('dfs_norm', st.session_state.dfs),
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
            
            
    
        st.session_state.plots_rl = plots_rl
        st.session_state.tables_rl = tables_rl
        st.session_state.logs_rl = logs_rl
        st.session_state.rl_analysis_run = True
        st.rerun()
    if st.session_state.rl_analysis_run:

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
                st.metric(label="Duração (dias úteis)", value=f"{sim_duration:.0f}", delta=f"{sim_duration - real_duration:.0f} vs Real")
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
            st.markdown("<div class='pm-brand'><span class='pm-brand-badge'><i class='bi bi-diagram-3-fill'></i></span>Process Intelligence</div>", unsafe_allow_html=True)
            cur = st.session_state.current_page
            for label, icon, page in [("Análise de Processos", ":material/monitoring:", "Dashboard"),
                                      ("Simulação (RL)", ":material/smart_toy:", "RL"),
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
        elif st.session_state.current_page == "RL":
            rl_page()
        elif st.session_state.current_page == "Settings":
            settings_page()

if __name__ == "__main__":
    main()
