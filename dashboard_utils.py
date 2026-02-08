import os
import ast
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output

BASE_DIR = os.getcwd()

_LAST_BASE_LEVERAGE = None

DEFAULT_VIX_FILE = 'VIX.csv'
DEFAULT_I500_FILE = 'AUM5 ETF Stock Price History.csv'

DEFAULT_START = datetime(2006, 1, 1).date()
DEFAULT_END = datetime.today().date()


def _resolve_path(path):
    if path is None:
        raise ValueError('Path is required.')
    path = path.strip()
    if not path:
        raise ValueError('Path is required.')
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _coerce_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(',', '')
    return pd.to_numeric(series, errors='coerce')


def _load_asset_returns(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        date_col = None
        for c in df.columns:
            if 'date' in c.lower():
                date_col = c
                break
        if date_col is None:
            raise ValueError(f'No date column found in {path}.')
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    if df['Date'].isna().all():
        raise ValueError(f'Could not parse any dates in {path}.')

    if 'Change %' in df.columns:
        df['Daily_Return'] = df['Change %'].astype(str).str.rstrip('%').str.replace(',', '').astype(float) / 100
    elif 'Change' in df.columns and df['Change'].dtype == object:
        df['Daily_Return'] = df['Change'].astype(str).str.rstrip('%').str.replace(',', '').astype(float) / 100
    else:
        price_col = None
        for candidate in ['Price', 'Close', 'Adj Close', 'AdjClose']:
            if candidate in df.columns:
                price_col = candidate
                break
        if price_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c.lower() not in ['vol', 'volume']]
            price_col = numeric_cols[0] if numeric_cols else None
        if price_col is None:
            raise ValueError(f'No price or change column found in {path}.')
        df[price_col] = _coerce_numeric(df[price_col])
        df = df.sort_values('Date')
        df['Daily_Return'] = df[price_col].pct_change()

    df = df[['Date', 'Daily_Return']].dropna().copy()
    return df


def _load_vix(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    date_col = None
    for c in df.columns:
        if 'date' in c.strip().lower():
            date_col = c
            break
    if date_col is None:
        first_col = df.columns[0]
        parsed_try = pd.to_datetime(df[first_col], errors='coerce')
        if parsed_try.notna().any():
            date_col = first_col
    if date_col is None:
        raise ValueError('No date-like column found in VIX file.')

    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    if df['Date'].isna().all():
        raise ValueError('Could not parse any dates in VIX file.')

    vix_col = None
    for candidate in ['VIX', 'Close', 'Price', 'Adj Close', 'AdjClose']:
        if candidate in df.columns:
            vix_col = candidate
            break
    if vix_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != date_col]
        vix_col = numeric_cols[0] if numeric_cols else None
    if vix_col is None:
        raise ValueError('No numeric column found in VIX file.')

    df[vix_col] = _coerce_numeric(df[vix_col])
    df = df[['Date', vix_col]].rename(columns={vix_col: 'VIX'})
    df = df.sort_values('Date').dropna().reset_index(drop=True)
    return df


def build_merged_returns(files, vix_path, start=None, end=None):
    asset_returns = {}
    for name, path in files.items():
        resolved = _resolve_path(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f'File not found: {resolved}')
        df = _load_asset_returns(resolved)
        df = df.rename(columns={'Daily_Return': name})
        asset_returns[name] = df

    merged = None
    for name, df in asset_returns.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='Date', how='inner')
    merged = merged.sort_values('Date').reset_index(drop=True)
    merged = merged.dropna().reset_index(drop=True)

    vix_resolved = _resolve_path(vix_path)
    if not os.path.exists(vix_resolved):
        raise FileNotFoundError(f'File not found: {vix_resolved}')
    df_vix = _load_vix(vix_resolved)
    merged = merged.merge(df_vix, on='Date', how='left')

    if start is not None:
        merged = merged[merged['Date'] >= pd.to_datetime(start)]
    if end is not None:
        merged = merged[merged['Date'] <= pd.to_datetime(end)]
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged


def compute_vol_and_zscores(df, ret_col='I500', vix_col='VIX', ewm_span=252, z_window=252, vix_ewm_span=10):
    vol_real = df[ret_col].ewm(span=ewm_span, adjust=False).std() * np.sqrt(252)
    log_vol_real = pd.Series(np.where(vol_real > 0, np.log(vol_real), np.nan), index=vol_real.index)
    real_mean = log_vol_real.rolling(z_window).mean()
    real_std = log_vol_real.rolling(z_window).std()
    z_real = (log_vol_real - real_mean) / real_std

    vix_sm = df[vix_col].ewm(span=vix_ewm_span, adjust=False).mean()
    log_vix_sm = pd.Series(np.where(vix_sm > 0, np.log(vix_sm), np.nan), index=vix_sm.index)
    vix_mean = log_vix_sm.rolling(z_window).mean()
    vix_std = log_vix_sm.rolling(z_window).std()
    z_impl = (log_vix_sm - vix_mean) / vix_std

    out = df.copy()
    out['Vol_Realized'] = vol_real
    out['VIX_Smoothed'] = vix_sm
    out['Z_Realized'] = z_real
    out['Z_Implied'] = z_impl
    return out


def compute_drawdown_signal(df, ret_col='I500', window_days=63):
    price = (1 + df[ret_col]).cumprod()
    rolling_high = price.rolling(window_days, min_periods=1).max()
    drawdown = (price / rolling_high) - 1.0
    out = df.copy()
    out['Price_Index'] = price
    out['Rolling_High'] = rolling_high
    out['Drawdown_3M'] = drawdown
    return out


ALLOWED_NODE_TYPES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
    ast.Eq, ast.NotEq, ast.And, ast.Or, ast.BoolOp, ast.IfExp, ast.Call
)

ALLOWED_FUNCS = {
    'min': np.minimum,
    'max': np.maximum,
    'abs': np.abs,
    'where': np.where
}


def safe_eval_formula(expr, context):
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            raise ValueError(f'Unsupported expression element: {type(node).__name__}')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError('Only min, max, abs, where functions are allowed.')
        if isinstance(node, ast.Name):
            if node.id not in context and node.id not in ALLOWED_FUNCS:
                raise ValueError(f'Unknown variable: {node.id}')
    safe_globals = {'__builtins__': {}}
    safe_locals = {**ALLOWED_FUNCS, **context}
    return eval(compile(tree, '<formula>', 'eval'), safe_globals, safe_locals)


def apply_monthly_rebalance(df, leverage_series):
    tmp = df[['Date']].copy()
    tmp['Leverage'] = leverage_series
    tmp['Month'] = tmp['Date'].dt.to_period('M')
    return tmp.groupby('Month')['Leverage'].transform('first')


def compute_regime_leverage(z_series, on_threshold=-0.5, off_threshold=-0, on_value=2.0, off_value=1.0):
    z_series = pd.Series(z_series)
    regime = np.full(z_series.shape[0], np.nan)
    state = off_value
    for idx, z_val in enumerate(z_series):
        if np.isnan(z_val):
            regime[idx] = state
            continue
        if z_val < on_threshold:
            state = on_value
        elif z_val > off_threshold:
            state = off_value
        regime[idx] = state
    return regime


def compute_metrics(value_series):
    series = value_series.dropna()
    if series.shape[0] < 2:
        return {
            'CAGR': np.nan,
            'Annual_Vol': np.nan,
            'Sharpe': np.nan,
            'Max_DD': np.nan
        }
    days = series.shape[0]
    daily_ret = series.pct_change().dropna()
    if daily_ret.empty:
        return {
            'CAGR': np.nan,
            'Annual_Vol': np.nan,
            'Sharpe': np.nan,
            'Max_DD': np.nan
        }
    ann_vol = daily_ret.std() * np.sqrt(252)
    ann_ret = (series.iloc[-1] / series.iloc[0]) ** (252 / days) - 1
    sharpe = ann_ret / ann_vol if ann_vol and not np.isnan(ann_vol) else 0.0
    peak = series.cummax()
    dd = (series - peak) / peak
    max_dd = dd.min()
    return {
        'CAGR': round(ann_ret, 4),
        'Annual_Vol': round(ann_vol, 4),
        'Sharpe': round(sharpe, 4),
        'Max_DD': round(max_dd, 4)
    }


def build_value_series(returns_series, start_value=100):
    return (1 + returns_series).cumprod() * start_value


def build_value_series_with_tax(returns_series, leverage_series, tax_rate, start_value=100):
    values = np.zeros(len(returns_series))
    value = start_value
    entry_value = None
    prev_lev = None

    for idx, (ret, lev) in enumerate(zip(returns_series, leverage_series)):
        if np.isnan(ret):
            values[idx] = value
            prev_lev = lev
            continue

        value = value * (1 + ret)

        if prev_lev is None:
            prev_lev = lev

        if prev_lev <= 1.0 and lev > 1.0:
            entry_value = value
        elif prev_lev > 1.0 and lev <= 1.0 and entry_value is not None:
            gain = value - entry_value
            if gain > 0:
                value -= gain * tax_rate
            entry_value = None

        values[idx] = value
        prev_lev = lev

    return pd.Series(values, index=returns_series.index)


def _parse_formulas(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed = []
    for idx, line in enumerate(lines, start=1):
        if '=' in line:
            name, expr = line.split('=', 1)
            name = name.strip() or f'Custom {idx}'
            expr = expr.strip()
        else:
            name = f'Custom {idx}'
            expr = line
        if not expr:
            continue

        shift_all = False
        if ';' in expr:
            parts = [p.strip() for p in expr.split(';') if p.strip()]
            expr = parts[0]
            for flag in parts[1:]:
                if flag.replace(' ', '').lower() == 'shift_all=true':
                    shift_all = True
                else:
                    raise ValueError(f'Unknown formula flag: {flag}')

        parsed.append((name, expr, {'shift_all': shift_all}))
    if not parsed:
        raise ValueError('At least one formula is required.')
    return parsed


def _build_dashboard(vix_path, i500_path, start, end, ewm_span, z_window, vix_ewm_span, min_leverage, max_leverage, rebal_freq, formulas_text, rf_annual, ter_annual, leakage_annual, initial_amount, monthly_contribution):
    i500_resolved = _resolve_path(i500_path)
    if not os.path.exists(i500_resolved):
        raise FileNotFoundError(f'File not found: {i500_resolved}')

    df_i500 = _load_asset_returns(i500_resolved).rename(columns={'Daily_Return': 'I500'})
    df = df_i500.copy()

    vix_resolved = _resolve_path(vix_path)
    if not os.path.exists(vix_resolved):
        raise FileNotFoundError(f'File not found: {vix_resolved}')
    df_vix = _load_vix(vix_resolved)
    df = df.merge(df_vix, on='Date', how='left')

    if start is not None:
        df = df[df['Date'] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df['Date'] <= pd.to_datetime(end)]
    df = df.sort_values('Date').reset_index(drop=True)
    df['VIX'] = df['VIX'].ffill()

    df = compute_vol_and_zscores(
        df,
        ret_col='I500',
        vix_col='VIX',
        ewm_span=ewm_span,
        z_window=z_window,
        vix_ewm_span=vix_ewm_span
    )

    df = compute_drawdown_signal(df, ret_col='I500', window_days=63)

    regime_lev = compute_regime_leverage(df['Z_Realized'].values)
    implied_regime_lev = compute_regime_leverage(df['Z_Implied'].values)

    context = {
        'z_real': df['Z_Realized'].values,
        'z_impl': df['Z_Implied'].values,
        'z_imp': df['Z_Implied'].values,
        'z_real_raw': df['Z_Realized'].values,
        'z_impl_raw': df['Z_Implied'].values,
        'vix': df['VIX'].values,
        'vol_real': df['Vol_Realized'].values,
        'regime_lev': regime_lev,
        'implied_regime_lev': implied_regime_lev,
        'max_leverage': max_leverage,
        'min_leverage': min_leverage,
        'dd_3m': df['Drawdown_3M'].values
    }

    formula_items = _parse_formulas(formulas_text)
    custom_strategies = {}
    custom_flags = {}
    for name, expr, flags in formula_items:
        leverage = safe_eval_formula(expr, context)
        leverage = np.clip(leverage, min_leverage, max_leverage)
        leverage = pd.Series(leverage, index=df.index)
        if rebal_freq == 'monthly':
            leverage = apply_monthly_rebalance(df, leverage)
        custom_strategies[name] = leverage
        custom_flags[name] = flags

    cost_rate = rf_annual + ter_annual + leakage_annual

    def _synthetic_returns(leverage_series):
        lev = pd.Series(leverage_series, index=df.index).astype(float)
        lev = lev.clip(lower=min_leverage, upper=max_leverage)
        cost = (lev - 1.0) * (cost_rate / 252)
        return (lev * df['I500']) - cost

    def _simulate_with_contributions(leverage_series, initial_leverage, shift_all=False):
        lev = pd.Series(leverage_series, index=df.index).astype(float)
        lev = lev.clip(lower=min_leverage, upper=max_leverage)
        if rebal_freq == 'monthly':
            lev = apply_monthly_rebalance(df, lev)

        tranches = [{'value': float(initial_amount), 'lev': float(initial_leverage)}]
        values = []
        eff_lev = []
        contrib = np.zeros(len(df))
        current_month = None

        for idx, (date, base_ret, lev_new) in enumerate(zip(df['Date'], df['I500'], lev)):
            month = date.to_period('M')
            if current_month is None:
                current_month = month
                if shift_all and not np.isnan(lev_new):
                    for tranche in tranches:
                        tranche['lev'] = float(lev_new)
            elif month != current_month:
                current_month = month
                if shift_all and not np.isnan(lev_new):
                    for tranche in tranches:
                        tranche['lev'] = float(lev_new)
                if monthly_contribution and monthly_contribution != 0:
                    tranches.append({'value': float(monthly_contribution), 'lev': float(lev_new)})
                    contrib[idx] = float(monthly_contribution)

            total_value = 0.0
            total_exposure = 0.0
            for tranche in tranches:
                t_lev = tranche['lev']
                cost = (t_lev - 1.0) * (cost_rate / 252)
                tranche['value'] *= (1.0 + (t_lev * base_ret) - cost)
                total_value += tranche['value']
                total_exposure += tranche['value'] * t_lev

            values.append(total_value)
            eff_lev.append((total_exposure / total_value) if total_value else np.nan)

        return (
            pd.Series(values, index=df.index),
            pd.Series(eff_lev, index=df.index),
            lev,
            pd.Series(contrib, index=df.index)
        )

    def _xirr(dates, cashflows, guess=0.08, max_iter=100, tol=1e-6):
        dates = pd.to_datetime(dates)
        days = (dates - dates.iloc[0]).dt.days.values.astype(float)
        cashflows = np.array(cashflows, dtype=float)
        if np.allclose(cashflows, 0):
            return np.nan

        rate = guess
        for _ in range(max_iter):
            denom = (1 + rate) ** (days / 365.0)
            f = np.sum(cashflows / denom)
            df_rate = np.sum(-(days / 365.0) * cashflows / denom / (1 + rate))
            if df_rate == 0:
                break
            new_rate = rate - f / df_rate
            if abs(new_rate - rate) < tol:
                return new_rate
            rate = new_rate
        return rate

    def _compute_metrics_with_flows(values, contrib_series):
        series = values.dropna()
        if series.shape[0] < 2:
            return {'CAGR': np.nan, 'Annual_Vol': np.nan, 'Sharpe': np.nan, 'Max_DD': np.nan}

        # Net-of-flows daily returns
        daily_ret = []
        for i in range(1, len(series)):
            v_prev = series.iloc[i - 1]
            v_now = series.iloc[i]
            flow = contrib_series.iloc[i]
            if v_prev and not np.isnan(v_prev):
                daily_ret.append((v_now - v_prev - flow) / v_prev)
        daily_ret = pd.Series(daily_ret).dropna()
        ann_vol = daily_ret.std() * np.sqrt(252) if not daily_ret.empty else np.nan

        cashflows = np.zeros(len(series))
        cashflows[0] = -float(initial_amount)
        cashflows += -contrib_series.values[:len(series)]
        cashflows[-1] += series.iloc[-1]
        ann_ret = _xirr(df['Date'].iloc[:len(series)], cashflows)

        sharpe = ann_ret / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
        peak = series.cummax()
        dd = (series - peak) / peak
        max_dd = dd.min()
        return {
            'CAGR': round(ann_ret, 4) if ann_ret == ann_ret else np.nan,
            'Annual_Vol': round(ann_vol, 4) if ann_vol == ann_vol else np.nan,
            'Sharpe': round(sharpe, 4) if sharpe == sharpe else np.nan,
            'Max_DD': round(max_dd, 4)
        }

    start_idx = 0
    for name, leverage in custom_strategies.items():
        returns = _synthetic_returns(leverage)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        first_idx = returns.first_valid_index()
        if first_idx is None:
            raise ValueError(f'No valid leverage values for {name}.')
        start_idx = max(start_idx, int(first_idx))
    if start_idx > 0:
        df = df.loc[start_idx:].reset_index(drop=True)
        for name in list(custom_strategies.keys()):
            custom_strategies[name] = custom_strategies[name].loc[start_idx:].reset_index(drop=True)

    base_name = None
    for name in custom_strategies:
        if name.strip().lower() == 'base':
            base_name = name
            break
    if base_name is None:
        base_name = next(iter(custom_strategies))
    base_current = float(pd.Series(custom_strategies[base_name]).dropna().iloc[-1])
    global _LAST_BASE_LEVERAGE
    _LAST_BASE_LEVERAGE = base_current

    df = df.reset_index(drop=True)

    value_unlev, eff_unlev, lev_unlev, contrib_unlev = _simulate_with_contributions(1.0, 1.0)
    value_15x, eff_15x, lev_15x, contrib_15x = _simulate_with_contributions(1.5, 1.5)
    value_2x, eff_2x, lev_2x, contrib_2x = _simulate_with_contributions(2.0, 2.0)
    value_25x, eff_25x, lev_25x, contrib_25x = _simulate_with_contributions(2.5, 2.5)
    value_3x, eff_3x, lev_3x, contrib_3x = _simulate_with_contributions(3.0, 3.0)

    df['Value_Unlevered'] = value_unlev
    df['Value_1_5x'] = value_15x
    df['Value_2x'] = value_2x
    df['Value_2_5x'] = value_25x
    df['Value_3x'] = value_3x

    custom_values = {}
    custom_eff_lev = {}
    custom_lev_current = {}
    custom_contrib = {}
    for name, leverage in custom_strategies.items():
        flags = custom_flags.get(name, {})
        values, eff_lev, lev_used, contrib = _simulate_with_contributions(
            leverage,
            1.0,
            shift_all=bool(flags.get('shift_all'))
        )
        custom_values[name] = values
        custom_eff_lev[name] = eff_lev
        custom_contrib[name] = contrib
        custom_lev_current[name] = float(pd.Series(lev_used).dropna().iloc[-1])

    metrics_payload = {
        'Unlevered': _compute_metrics_with_flows(df['Value_Unlevered'], contrib_unlev),
        '1.5x': _compute_metrics_with_flows(df['Value_1_5x'], contrib_15x),
        '2x': _compute_metrics_with_flows(df['Value_2x'], contrib_2x),
        '2.5x': _compute_metrics_with_flows(df['Value_2_5x'], contrib_25x),
        '3x': _compute_metrics_with_flows(df['Value_3x'], contrib_3x)
    }
    for name, series in custom_values.items():
        metrics_payload[name] = _compute_metrics_with_flows(series, custom_contrib[name])
    metrics_df = pd.DataFrame(metrics_payload)

    time_weighted = {
        'Unlevered': round(float(np.nanmean(eff_unlev)), 4),
        '1.5x': round(float(np.nanmean(eff_15x)), 4),
        '2x': round(float(np.nanmean(eff_2x)), 4),
        '2.5x': round(float(np.nanmean(eff_25x)), 4),
        '3x': round(float(np.nanmean(eff_3x)), 4)
    }
    for name, eff_lev in custom_eff_lev.items():
        time_weighted[name] = round(float(np.nanmean(eff_lev)), 4)
    metrics_df.loc['Time_Weighted_Lev'] = time_weighted

    current_lev = {
        'Unlevered': 1.0,
        '1.5x': 1.5,
        '2x': 2.0,
        '2.5x': 2.5,
        '3x': 3.0
    }
    for name, lev_val in custom_lev_current.items():
        current_lev[name] = round(float(lev_val), 4)
    metrics_df.loc['Current_Lev'] = current_lev

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08
    )

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value_Unlevered'], name='Unlevered'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value_1_5x'], name='1.5x Fixed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value_2x'], name='2x Fixed', visible='legendonly'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value_2_5x'], name='2.5x Fixed', visible='legendonly'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value_3x'], name='3x Fixed', visible='legendonly'), row=1, col=1)

    for idx, (name, series) in enumerate(custom_values.items()):
        width = 3 if idx == 0 else 2
        fig.add_trace(go.Scatter(x=df['Date'], y=series, name=name, line=dict(width=width)), row=1, col=1)

    for name, eff_lev in custom_eff_lev.items():
        fig.add_trace(go.Scatter(x=df['Date'], y=eff_lev, name=f'{name} Leverage'), row=2, col=1)

    fig.add_hline(y=1.0, line_dash='dash', line_color='gray', row=2, col=1)

    x_min = df['Date'].min()
    x_max = df['Date'].max()

    fig.update_layout(
        height=800,
        width=1200,
        title=dict(text='Backtests: Fixed vs Custom Leverage (1x ETF)', y=0.98),
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='left', x=0),
        margin=dict(t=140)
    )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(title_text='Value (start=100)', row=1, col=1)
    fig.update_yaxes(title_text='Leverage', row=2, col=1)

    vol_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    vol_fig.add_trace(go.Scatter(x=df['Date'], y=df['Vol_Realized'] * 100, name='Realized Vol (%)'), row=1, col=1)
    vol_fig.add_trace(go.Scatter(x=df['Date'], y=df['VIX_Smoothed'], name='VIX (smoothed)'), row=1, col=1)

    vol_fig.add_trace(go.Scatter(x=df['Date'], y=df['Z_Realized'], name='Z Realized'), row=2, col=1)
    vol_fig.add_trace(go.Scatter(x=df['Date'], y=df['Z_Implied'], name='Z Implied'), row=2, col=1)
    vol_fig.add_hline(y=0.0, line_dash='dash', line_color='gray', row=2, col=1)

    vol_fig.update_layout(
        width=1200,
        height=700,
        title='Realized vs Implied Volatility (VIX) and Z-Scores',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    vol_fig.update_xaxes(range=[x_min, x_max])
    vol_fig.update_yaxes(title_text='Vol (%) / VIX', row=1, col=1)
    vol_fig.update_yaxes(title_text='Z-score', row=2, col=1)

    last_row = df.dropna().iloc[-1]
    info_html = f'''
    <div style="font-family: 'Segoe UI', Tahoma, Arial; line-height: 1.5;">
        <div style="font-size: 16px; font-weight: 600; margin-bottom: 6px;">Current Snapshot</div>
        <div style="background: #f6f7fb; border: 1px solid #e4e7f0; border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 14px; color: #2f2f2f;">
                <div style="margin-bottom: 8px;">Realized Vol (EWMA): <b>{last_row['Vol_Realized']:.3f}</b></div>
                <div style="margin-bottom: 8px;">VIX (smoothed): <b>{last_row['VIX_Smoothed']:.2f}</b></div>
                <div style="margin-bottom: 8px;">Z Realized: <b>{last_row['Z_Realized']:.2f}</b></div>
                <div style="margin-bottom: 8px;">Z Implied: <b>{last_row['Z_Implied']:.2f}</b></div>
            </div>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #dde2ee;">
                <div style="font-size: 13px; color: #60667a; text-transform: uppercase; letter-spacing: 0.06em;">Base leverage</div>
                <div style="font-size: 28px; font-weight: 700; color: #1f2a44;">{base_current:.2f}</div>
            </div>
        </div>
    </div>
    '''

    return df, fig, vol_fig, metrics_df, info_html


def create_dashboard(alloc_widget=None):
    vix_file = widgets.Text(value=DEFAULT_VIX_FILE, description='VIX CSV:')

    start_date = widgets.DatePicker(value=DEFAULT_START, description='Start')
    end_date = widgets.DatePicker(value=DEFAULT_END, description='End')

    ewm_span = widgets.IntSlider(value=252, min=50, max=400, step=1, description='EWMA')
    z_window = widgets.IntSlider(value=252, min=50, max=400, step=1, description='Z-Window')
    vix_ewm_span = widgets.IntSlider(value=1, min=1, max=60, step=1, description='VIX-EWMA')

    max_leverage = widgets.FloatSlider(value=3.0, min=1.0, max=5.0, step=0.1, description='MaxLev')
    min_leverage = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, description='MinLev')
    rebal_freq = widgets.Dropdown(options=['monthly', 'none'], value='monthly', description='Rebal')
    rf_annual = widgets.FloatText(value=0.045, description='RF/yr')
    ter_annual = widgets.FloatText(value=0.002, description='TER/yr')
    leakage_annual = widgets.FloatText(value=0.001, description='Leakage/yr')

    i500_file = widgets.Text(value=DEFAULT_I500_FILE, description='1x ETF CSV:')
    initial_amount = widgets.FloatText(value=10000.0, description='Start Amt')
    monthly_contribution = widgets.FloatText(value=1000.0, description='Monthly Add')

    formulas = widgets.Textarea(
        value='Real=where(z_real>-0.5,1,2)\nImp=where(z_imp>-0.5,1,2)\nDD30=where(dd_3m<=-0.30,2,1);shift_all=True',
        description='Formulas',
        layout=widgets.Layout(width='100%', height='90px')
    )

    update_btn = widgets.Button(description='Update dashboard', button_style='primary')

    out_backtest = widgets.Output()
    out_vols = widgets.Output()
    out_info = widgets.Output()

    controls = widgets.VBox([
        widgets.HBox([vix_file, i500_file]),
        widgets.HBox([start_date, end_date]),
        widgets.HBox([ewm_span, z_window, vix_ewm_span]),
        widgets.HBox([min_leverage, max_leverage, rebal_freq]),
        widgets.HBox([rf_annual, ter_annual, leakage_annual]),
        widgets.HBox([initial_amount, monthly_contribution]),
        formulas,
        update_btn
    ])

    def _render_dashboard():
        with out_backtest:
            clear_output()
        with out_vols:
            clear_output()
        with out_info:
            clear_output()

        vix_path = vix_file.value.strip()
        start = start_date.value or DEFAULT_START
        end = end_date.value or DEFAULT_END

        df, fig, vol_fig, metrics_df, info_html = _build_dashboard(
            vix_path,
            i500_file.value.strip(),
            start,
            end,
            ewm_span.value,
            z_window.value,
            vix_ewm_span.value,
            min_leverage.value,
            max_leverage.value,
            rebal_freq.value,
            formulas.value,
            rf_annual.value,
            ter_annual.value,
            leakage_annual.value,
            initial_amount.value,
            monthly_contribution.value
        )

        metrics_html = metrics_df.round(2).to_html(border=0)
        metrics_html = metrics_html.replace(
            '<table border="0" class="dataframe">',
            '<table border="0" class="dataframe" style="font-family: \'Segoe UI\', Tahoma, Arial; font-size: 13px; border-collapse: collapse; background: #f6f7fb; border: 1px solid #e4e7f0; border-radius: 10px;">'
        ).replace(
            '<th>',
            '<th style="padding: 6px 10px; text-align: left; border-bottom: 1px solid #e4e7f0; color: #4a4f5a;">'
        ).replace(
            '<td>',
            '<td style="padding: 6px 10px; border-bottom: 1px solid #f0f2f7;">'
        )

        with out_backtest:
            display(fig)

        with out_vols:
            display(vol_fig)

        info_panel = widgets.HTML(info_html)
        info_panel.layout = widgets.Layout(width='33%')
        metrics_panel = widgets.HTML(metrics_html)
        metrics_panel.layout = widgets.Layout(width='33%')

        row_items = [info_panel, metrics_panel]
        if alloc_widget is not None:
            alloc_widget.layout = widgets.Layout(width='33%')
            row_items.insert(0, alloc_widget)

        with out_info:
            display(widgets.HBox(
                row_items,
                layout=widgets.Layout(gap='16px', align_items='stretch')
            ))

        return df, fig, vol_fig, metrics_df

    def _on_update_clicked(_):
        _render_dashboard()

    update_btn.on_click(_on_update_clicked)

    container = widgets.VBox([controls, out_info, out_backtest, out_vols])
    _render_dashboard()
    return container


def create_allocation_calculator():
    amount = widgets.FloatText(value=100000, description='')
    msci_weight = widgets.FloatText(value=0.7, description='')
    gold_weight = widgets.FloatText(value=0.3, description='')
    default_lev = _LAST_BASE_LEVERAGE if _LAST_BASE_LEVERAGE is not None else 1.0
    min_lev = 0.5
    max_lev = 5.0
    default_lev = min(max(default_lev, min_lev), max_lev)
    msci_leverage = widgets.FloatText(value=default_lev, description='')
    alloc_out = widgets.HTML(
        value="<div style=\"font-family: 'Segoe UI', Tahoma, Arial; font-size: 13px; color: #2f2f2f;\"></div>"
    )
    base_lev_badge = widgets.HTML(
        value=f"<div style=\"font-family: 'Segoe UI', Tahoma, Arial; font-size: 12px; color: #4a4f5a;\">Base leverage (current): <b>{default_lev:.2f}</b></div>"
    )
    alloc_card = widgets.HTML(
        value=(
            "<div style=\"font-family: 'Segoe UI', Tahoma, Arial; font-size: 16px; font-weight: 600; margin-bottom: 6px;\">"
            "Allocation Calculator</div>"
            "<div style=\"height: 4px; width: 40px; background: #4a6cf7; border-radius: 4px; margin-bottom: 10px;\"></div>"
        )
    )

    def update_alloc(_=None):
        w_m = msci_weight.value
        w_g = gold_weight.value
        if w_m + w_g == 0:
            alloc_out.value = (
                "<div style=\"font-family: 'Segoe UI', Tahoma, Arial; font-size: 13px; color: #b00020;\">"
                "Weights sum to zero."
                "</div>"
            )
            return
        total_w = w_m + w_g
        w_m = w_m / total_w
        w_g = w_g / total_w
        cap_m = amount.value * w_m
        cap_g = amount.value * w_g

        target_lev = msci_leverage.value
        notional_m = cap_m * target_lev
        total_notional = notional_m + cap_g

        mix_3x = (target_lev - 1.0) / 2.0
        mix_3x = max(0.0, min(1.0, mix_3x))
        msci_3x_cap = cap_m * mix_3x
        msci_1x_cap = cap_m * (1.0 - mix_3x)
        msci_notional = msci_1x_cap + 3.0 * msci_3x_cap

        alloc_out.value = (
            "<div style=\"font-family: 'Segoe UI', Tahoma, Arial; font-size: 13px; color: #2f2f2f; line-height: 1.6;\">"
            f"<div>Gold: <b>{cap_g:,.2f}</b></div>"
            f"<div>MSCI 1x: <b>{msci_1x_cap:,.2f}</b></div>"
            f"<div>MSCI 3x: <b>{msci_3x_cap:,.2f}</b></div>"
            "</div>"
        )

    for w in [amount, msci_weight, gold_weight, msci_leverage]:
        w.observe(update_alloc, names='value')

    def _label(text):
        return widgets.HTML(
            value=(
                "<div style=\"font-family: 'Segoe UI', Tahoma, Arial; font-size: 12px; color: #4a4f5a;\">"
                f"{text}" 
                "</div>"
            )
        )

    input_width = '90px'
    amount.layout = widgets.Layout(width=input_width)
    msci_weight.layout = widgets.Layout(width=input_width)
    gold_weight.layout = widgets.Layout(width=input_width)
    msci_leverage.layout = widgets.Layout(width=input_width)

    amount_box = widgets.VBox([_label('Amount'), amount], layout=widgets.Layout(gap='4px'))
    msci_box = widgets.VBox([_label('MSCI %'), msci_weight], layout=widgets.Layout(gap='4px'))
    gold_box = widgets.VBox([_label('Gold %'), gold_weight], layout=widgets.Layout(gap='4px'))
    lev_box = widgets.VBox([_label('MSCI Lev'), msci_leverage], layout=widgets.Layout(gap='4px'))

    container = widgets.VBox([
        alloc_card,
        widgets.HBox([amount_box, msci_box, gold_box, lev_box], layout=widgets.Layout(gap='12px')),
        base_lev_badge,
        alloc_out
    ], layout=widgets.Layout(gap='8px', border='1px solid #e4e7f0', padding='12px', border_radius='10px', background='#f6f7fb'))
    update_alloc()
    return container
