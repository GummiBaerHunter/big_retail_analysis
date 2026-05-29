"""
Skript zur Einzelhandelsdatenanalyse
===================================
Analysiert fake_retail_data.csv mit:
  - Deskriptiven Statistiken (numpy)
  - Umsatztrend-Analyse (monatliche Aggregation + numpy polyfit Regression)
  - Kategorie- und Regionauswertungen
  - Rabattwirkung-Analyse (t-Test via numpy)
  - Kundengoldsegmentierung (numpy Histogramm / Binning)
  - Ausreißererkennung (Z-Score & IQR Methoden)
  - Korrelationsanalyse (numpy corrcoef)
  - Matplotlib-Visualisierungen (gespeichert in ./charts/)
"""

import csv
import math
import os
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = 'fake_retail_data.csv'
CHART_DIR = 'charts'


def ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


def parse_bool(value):
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def load_data():
    rows = []
    with open(DATA_FILE, newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                sales = float(row['Sales']) if row['Sales'].strip() != '' else np.nan
            except ValueError:
                sales = np.nan
            try:
                age = float(row['Customer_Age']) if row['Customer_Age'].strip() != '' else np.nan
            except ValueError:
                age = np.nan

            parsed_row = {
                'date': datetime.strptime(row['Date'], '%Y-%m-%d'),
                'month': datetime.strptime(row['Date'], '%Y-%m-%d').strftime('%Y-%m'),
                'product_id': row['Product_ID'].strip(),
                'category': row['Product_Category'].strip(),
                'region': row['Region'].strip(),
                'sales': sales,
                'age': age,
                'gender': row['Customer_Gender'].strip(),
                'discount': parse_bool(row['Discount_Applied']),
            }
            rows.append(parsed_row)
    return rows


def describe_numeric(values):
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {}
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        'min': float(np.min(arr)),
        '25%': float(np.percentile(arr, 25)),
        '50%': float(np.percentile(arr, 50)),
        '75%': float(np.percentile(arr, 75)),
        'max': float(np.max(arr)),
    }


def aggregate_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if not np.isnan(row['sales']):
            groups[row[key]].append(row['sales'])
    return {group: groups[group] for group in sorted(groups)}


def summary_by_group(values_by_group):
    return {
        group: {
            'count': len(values),
            'sum': float(np.sum(values)),
            'mean': float(np.mean(values)) if values else 0.0,
        }
        for group, values in values_by_group.items()
    }


def monthly_sales_trend(rows):
    monthly = defaultdict(list)
    for row in rows:
        if not np.isnan(row['sales']):
            monthly[row['month']].append(row['sales'])
    months = sorted(monthly)
    totals = [float(np.sum(monthly[m])) for m in months]
    return months, totals


def fit_trend_line(x_values, y_values):
    if len(x_values) < 2:
        return None, None
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def apply_age_bins(rows):
    bin_edges = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    counts = {label: 0 for label in labels}
    for row in rows:
        age = row['age']
        if np.isnan(age):
            continue
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= age < bin_edges[i + 1]:
                counts[labels[i]] += 1
                break
    return counts


def t_test(group_a, group_b):
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    n_a = a.size
    n_b = b.size
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return np.nan
    return (mean_a - mean_b) / se


def detect_outliers(values):
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return []
    z_scores = (arr - np.mean(arr)) / np.std(arr, ddof=1)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [float(v) for v, z in zip(arr, z_scores) if abs(z) > 2 or v < lower or v > upper]


def correlation_matrix(rows):
    records = []
    for row in rows:
        if not np.isnan(row['sales']) and not np.isnan(row['age']):
            records.append((row['sales'], row['age'], float(row['discount'])))
    if not records:
        return None
    arr = np.array(records, dtype=float)
    return np.corrcoef(arr, rowvar=False)


def save_chart(fig, filename):
    ensure_chart_dir()
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path


def plot_monthly_sales(months, totals):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(months))
    ax.bar(x, totals, color='#4c72b0')
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_title('Monatliche Umsatzwerte')
    ax.set_ylabel('Umsatz')
    ax.set_xlabel('Monat')
    save_chart(fig, 'monthly_sales_totals.png')

    if len(months) >= 2:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, totals, marker='o')
        slope, intercept = fit_trend_line(x, totals)
        if slope is not None:
            ax.plot(x, slope * x + intercept, linestyle='--', color='orange', label='Trendlinie')
            ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_title('Monatlicher Umsatz-Trend')
        ax.set_ylabel('Umsatz')
        ax.set_xlabel('Monat')
        save_chart(fig, 'monthly_sales_trend.png')


def plot_category_region_breakdown(rows):
    category_sales = summary_by_group(aggregate_by(rows, 'category'))
    region_sales = summary_by_group(aggregate_by(rows, 'region'))

    if category_sales:
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = list(category_sales.keys())
        values = [category_sales[k]['sum'] for k in categories]
        ax.bar(categories, values, color='#55a868')
        ax.set_title('Umsatz nach Produktkategorie')
        ax.set_ylabel('Gesamtumsatz')
        save_chart(fig, 'sales_by_category.png')

    if region_sales:
        fig, ax = plt.subplots(figsize=(8, 5))
        regions = list(region_sales.keys())
        values = [region_sales[k]['sum'] for k in regions]
        ax.bar(regions, values, color='#c44e52')
        ax.set_title('Umsatz nach Region')
        ax.set_ylabel('Gesamtumsatz')
        save_chart(fig, 'sales_by_region.png')


def plot_age_distribution(rows):
    ages = [row['age'] for row in rows if not np.isnan(row['age'])]
    if not ages:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ages, bins=[18, 25, 35, 45, 55, 65, 75, 85], color='#8172b3', edgecolor='black')
    ax.set_title('Altersverteilung der Kunden')
    ax.set_xlabel('Alter')
    ax.set_ylabel('Anzahl der Transaktionen')
    save_chart(fig, 'customer_age_distribution.png')


def plot_discount_comparison(rows):
    with_discount = [row['sales'] for row in rows if row['discount'] and not np.isnan(row['sales'])]
    without_discount = [row['sales'] for row in rows if not row['discount'] and not np.isnan(row['sales'])]
    if not with_discount or not without_discount:
        return

    totals = [float(np.sum(without_discount)), float(np.sum(with_discount))]
    labels = ['Kein Rabatt', 'Rabatt']
    explode = (0.05, 0.05)
    colors = ['#4c72b0', '#55a868']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(totals, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, colors=colors,
           wedgeprops=dict(edgecolor='white'))
    ax.set_title('Umsatzanteil: Rabatt vs Kein Rabatt')
    ax.axis('equal')
    save_chart(fig, 'discount_sales_comparison.png')


def print_summary(rows):
    sales_stats = describe_numeric([row['sales'] for row in rows])
    age_stats = describe_numeric([row['age'] for row in rows])
    months, totals = monthly_sales_trend(rows)
    discount_groups = {
        'discount': [row['sales'] for row in rows if row['discount'] and not np.isnan(row['sales'])],
        'no_discount': [row['sales'] for row in rows if not row['discount'] and not np.isnan(row['sales'])],
    }
    discount_z = t_test(discount_groups['discount'], discount_groups['no_discount'])
    outliers = detect_outliers([row['sales'] for row in rows])
    corr = correlation_matrix(rows)

    print('\n=== Analyse der Einzelhandelsdaten ===\n')
    print('Gesamtanzahl Datensätze:', len(rows))
    print('Deskriptive Statistik für Umsatz:')
    for key, value in sales_stats.items():
        print(f'  {key:>4}: {value:.2f}' if isinstance(value, float) else f'  {key:>4}: {value}')
    print('\nDeskriptive Statistik für Kundenalter:')
    for key, value in age_stats.items():
        print(f'  {key:>4}: {value:.2f}' if isinstance(value, float) else f'  {key:>4}: {value}')
    print('\nMonate mit Umsätzen:', ', '.join(months))
    print('Monatliche Umsatzzahlen:', ', '.join(f'{total:.2f}' for total in totals))
    print('\nAnalyse der Rabattwirkung:')
    print(f'  Transaktionen mit Rabatt: {len(discount_groups["discount"])}')
    print(f'  Transaktionen ohne Rabatt: {len(discount_groups["no_discount"])}')
    print(f'  Ungefähre t-Score-Differenz: {discount_z:.3f}')
    age_bins = apply_age_bins(rows)
    print('\nAlterssegmente der Kunden:')
    for segment, count in age_bins.items():
        print(f'  {segment}: {count}')
    print('\nAusreißererkennung:')
    print(f'  Erfasste Umsatz-Ausreißer: {len(outliers)}')
    if outliers:
        print('  Beispiel-Ausreißer:', ', '.join(f'{value:.2f}' for value in sorted(set(outliers))[:5]))
    if corr is not None:
        print('\nNumerische Korrelationsmatrix (Umsatz, Alter, Rabatt):')
        print(np.array2string(corr, precision=3, suppress_small=True))
    else:
        print('\nNumerische Korrelationsmatrix: Nicht genug Daten')
    print('\nDiagramme gespeichert in ./charts/')


def main():
    ensure_chart_dir()
    rows = load_data()
    print_summary(rows)
    months, totals = monthly_sales_trend(rows)
    if months and totals:
        plot_monthly_sales(months, totals)
    plot_category_region_breakdown(rows)
    plot_age_distribution(rows)
    plot_discount_comparison(rows)


if __name__ == '__main__':
    main()
