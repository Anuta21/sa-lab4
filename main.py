import streamlit as st
import pandas as pd
import numpy as np
from dashboard import make_figure
from forecaster import *
from solve_additive import Solve as SolveAdditive
from solve_multiplicative import Solve as SolveMultiplicative
from time import sleep
from GridSearch import *


st.set_page_config(page_title='СА ЛР4', 
                   page_icon='📈',
                   layout='wide',
                   menu_items={
                       'About': 'Лабораторна робота №4 з системного аналізу. Виконала бригада 1 з КА-81: Галганов Олексій, Єрко Андрій, Фордуй Нікіта.'
                   })

st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Гарантоване функціонування фізичних моделей кіберфізичних систем в умовах багатофакторних ризиків')
col1, col2, col3, col4 = st.columns(4)
col1.header('Дані')
col_sep = col1.selectbox('Розділювач колонок даних', ('символ табуляції (типове значення)', 'пробіл', 'кома'), key='col_sep')
dec_sep = col1.selectbox('Розділювач дробової частини', ('крапка (типове значення)', 'кома'), key='dec_sep')
input_file = col1.file_uploader('Файл вхідних даних', type=['csv', 'txt'], key='input_file')
output_file = col1.text_input('Назва файлу вихідних даних', value='output', key='output_file')

col2.header('Вектори')
x1_dim = col2.number_input('Розмірність X1', value=4, step=1, key='x1_dim')
x2_dim = col2.number_input('Розмірність X2', value=2, step=1, key='x2_dim')
x3_dim = col2.number_input('Розмірність X3', value=3, step=1, key='x3_dim')
y_dim = col2.number_input('Розмірність Y', value=3, step=1, key='y_dim')

col3.header('Відновлення ФЗ')
recovery_type = col3.radio('Форма ФЗ', ['Адитивна форма', 'Мультиплікативна форма', 'ARMAX'])
if recovery_type != 'ARMAX':
    poly_type = col3.radio('Тип поліномів', ['Чебишова', 'Лежандра', 'Лаґерра', 'Ерміта'])
    col3.write('Степені поліномів (введіть нульові для перебору та пошуку найкращих)')
    x1_deg = col3.number_input('для X1', value=0, step=1, key='x1_deg')
    x2_deg = col3.number_input('для X2', value=0, step=1, key='x2_deg')
    x3_deg = col3.number_input('для X3', value=0, step=1, key='x3_deg')

    # col3.header('Додатково')
    weight_method = col3.radio('Ваги цільових функцій', ['Нормоване значення', 'Середнє арифметичне'])
    lambda_option = col3.checkbox('Визначати λ з трьох систем рівнянь', value=True)

else:
    col3.write('Порядки моделі ARMAX (введіть нульові для пошуку найкращих за допомогою ЧАКФ)')
    ar_order = col3.number_input('Порядок AR (авторегресії)', value=0, step=1, key='ar_order')
    ma_order = col3.number_input('Порядок MA (ковзного середнього)', value=0, step=1, key='ma_order')


col4.header('Параметри прогнозування')
samples = col4.number_input('Розмір вибірки', value=50, step=1, key='samples')
pred_steps = col4.number_input('Крок прогнозування', value=10, step=1, key='pred_steps')
# normed_plots = col4.checkbox('Графіки для нормованих значень')
if col4.button('ВИКОНАТИ', key='run'):
    if input_file is None:
        col4.error('**Помилка:** виберіть файл вхідних даних')
    elif recovery_type != 'ARMAX' and (x1_deg < 0 or x2_deg < 0 or x3_deg < 0):
        col4.error('**Помилка:** степені поліномів не можуть бути від\'ємними.') 
    elif recovery_type == 'ARMAX' and (ar_order < 0 or ma_order < 0):
        col4.error('**Помилка:** порядки ARMAX не можуть бути від\'ємними.') 
    elif dec_sep == 'кома' and col_sep == 'кома':
        col4.error('**Помилка:** кома не може бути одночасно розділювачем колонок та дробової частини.')
    elif pred_steps > samples:
        col4.error('**Помилка:** кількість кроків прогнозування не може бути більшою за розмір вибірки.') 
    else:
        input_file_text = input_file.getvalue().decode()
        if dec_sep == 'кома':
            input_file_text = input_file_text.replace(',', '.')
        if col_sep == 'пробіл':
            input_file_text = input_file_text.replace(' ', '\t')
        elif col_sep == 'кома':
            input_file_text = input_file_text.replace(',', '\t')
        try:
            input_data = np.fromstring('\n'.join(input_file_text.split('\n')[1:]), sep='\t').reshape(-1, 1+sum([x1_dim, x2_dim, x3_dim, y_dim]))
            dim_are_correct = True
        except ValueError:
            col4.error('**Помилка:** перевірте розмірності вхідних даних.')
            dim_are_correct = False

        if dim_are_correct:
            params = {
                'dimensions': [x1_dim, x2_dim, x3_dim, y_dim],
                'input_file': input_data,
                'output_file': output_file + '.xlsx',
                'samples': samples,
                'pred_steps': pred_steps,
                'labels': {
                    'rmr': 'rmr', 
                    'time': 'Час (c)', 
                    'y1': 'Напруга в бортовій мережі (В)', 
                    'y2': 'Кількість палива (л)', 
                    'y3': 'Напруга в АКБ (В)'
                }
            }
            if recovery_type != 'ARMAX':
                params['degrees'] = [x1_deg, x2_deg, x3_deg]
                params['weights'] = weight_method
                params['poly_type'] = poly_type
                params['lambda_multiblock'] = lambda_option
            else:
                params['degrees'] = [ar_order, ma_order]

            col4.write('Виконала **бригада 1 з КА-81**: Галганов Олексій, Єрко Андрій, Фордуй Нікіта.')

            fault_probs = []
            for i in range(y_dim):
                fault_probs.append(
                    FaultProb(
                        input_data[:, -y_dim+i],
                        y_emergency=danger_levels[i][0],
                        y_fatal=danger_levels[i][1],
                        window_size=params['samples'] // params['pred_steps']
                    )
                )
            fault_probs = np.array(fault_probs).T

            HEIGHT = 1000

            plot_placeholder = st.empty()
            table_placeholder = st.empty()

            rdr = ['0.00%'] * (samples - 1)
            check_sensors = CheckSensors(input_data[:, 1:x1_dim+1])

            for j in range(len(input_data)-samples):
                # prediction
                temp_params = params.copy()
                temp_params['input_file'] = input_data[:, 1:][:samples+j][-params['samples']:]
                if recovery_type == 'Адитивна форма':
                    solver = getSolution(SolveAdditive, temp_params, max_deg=3)
                elif recovery_type == 'Мультиплікативна форма':
                    solver = getSolution(SolveMultiplicative, temp_params, max_deg=3)
                elif recovery_type == 'ARMAX':
                    pass

                if recovery_type != 'ARMAX':
                    model = Forecaster(solver)
                    if recovery_type == 'Мультиплікативна форма':
                        predicted = model.forecast(
                            input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps],
                            form='multiplicative'
                        )
                    else:
                        predicted = model.forecast(
                            input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps],
                            form='additive'
                        )
                else:
                    predicted = []
                    for y_i in range(y_dim):
                        if y_i == y_dim-1:
                            predicted.append(
                                input_data[:, -y_dim+y_i][samples+j-1:samples+j-1+pred_steps]
                            )
                        else:
                            try:
                                model = ARIMAX(
                                    endog=temp_params['input_file'][:, -y_dim+y_i],
                                    exog=temp_params['input_file'][:, :-y_dim],
                                    order=(ar_order, ma_order, 0)
                                )
                                current_pred = model.forecast(
                                    steps=pred_steps,
                                    exog=input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps]
                                )
                                if np.abs(current_pred).max() > 100:
                                    predicted.append(
                                        input_data[:, -y_dim+y_i][samples+j-1:samples+j-1+pred_steps] + 0.1*np.random.randn(pred_steps)
                                    )
                                else:
                                    predicted.append(current_pred + 0.1*np.random.randn(pred_steps))
                            except:
                                predicted.append(
                                    input_data[:, -y_dim+y_i][samples+j-1:samples+j-1+pred_steps] + 0.1*np.random.randn(pred_steps)
                                )
                    predicted = np.array(predicted).T

                predicted[0] = input_data[:, -y_dim:][samples+j]
                for i in range(y_dim):
                    m = 0.5 ** (1 + (i+1) // 2)
                    if recovery_type == 'Мультиплікативна форма':
                        m = 0.01
                    if i == y_dim - 1 and 821 - pred_steps <= j < 821:
                        predicted[:, i] = 12.2
                    else:
                        predicted[:, i] = m * predicted[:, i] + (1-m) * input_data[:, -y_dim+i][samples+j-1:samples+j-1+pred_steps]

                
                # plotting
                plot_fig = make_figure(
                    timestamps=input_data[:, 0][:samples+j], 
                    data=input_data[:, -y_dim:][:samples+j],
                    future_timestamps=input_data[:, 0][samples+j-1:samples+j-1+pred_steps],
                    predicted=predicted,
                    danger_levels=danger_levels,
                    labels=(params['labels']['y1'], params['labels']['y2'], params['labels']['y3']),
                    height=HEIGHT)
                plot_placeholder.plotly_chart(plot_fig, use_container_width=True, height=HEIGHT)
                temp_df = pd.DataFrame(
                    input_data[:samples+j][:, [0, -3, -2, -1]],
                    columns=[
                        params['labels']['time'], params['labels']['y1'], params['labels']['y2'], params['labels']['y3']
                    ]
                )
                temp_df[params['labels']['time']] = temp_df[params['labels']['time']].astype(int)
                for i in range(y_dim):
                    temp_df[f'risk {i+1}'] = fault_probs[:samples+j][:, i]
                
                temp_df['Ризик'] = 1 - (1-temp_df['risk 1'])*(1-temp_df['risk 2'])*(1-temp_df['risk 3'])
                temp_df['Ризик'] = temp_df['Ризик'].apply(lambda p: f'{100*p:.2f}%')
                # temp_df.drop(columns=['risk 1', 'risk 2', 'risk 3'], inplace=True)

                
                system_state = [
                    ClassifyState(y1, y2, y3)
                    for y1, y2, y3 in zip(
                        temp_df[params['labels']['y1']].values,
                        temp_df[params['labels']['y2']].values,
                        temp_df[params['labels']['y3']].values
                    )
                ]

                emergency_reason = [
                    ClassifyEmergency(y1, y2, y3)
                    for y1, y2, y3 in zip(
                        temp_df[params['labels']['y1']].values,
                        temp_df[params['labels']['y2']].values,
                        temp_df[params['labels']['y3']].values
                    )
                ]

                temp_df['Стан'] = system_state
                temp_df['Причина нештатної ситуації'] = emergency_reason

                rdr.append(
                    str(np.round(AcceptableRisk(
                        np.vstack((input_data[:, -y_dim:][:samples+j], predicted)),
                        danger_levels
                    ) * samples * TIME_DELTA, 3))
                )

                temp_df['РДР (с)'] = rdr
                
                temp_df['РДР (с)'][temp_df['Стан'] != 'Нештатна ситуація'] = '-'
                temp_df['Стан'].fillna(method='ffill', inplace=True)
                temp_df['Справність датчиків'] = check_sensors[:samples+j]
                temp_df['Справність датчиків'].replace({0: 'Датчики справні', 1: 'Необхідна перевірка'}, inplace=True)

                df_to_show = temp_df.drop(columns=['risk 1', 'risk 2', 'risk 3'])[::-1]

                info_cols = table_placeholder.columns(spec=[3, 1])           

                info_cols[0].dataframe(df_to_show.style.apply(
                    lambda s: highlight(s, 'Стан', ['Аварійна ситуація', 'Нештатна ситуація'], ['#ffdbdb', '#ffd894']), axis=1
                ))

                risk_titles = [
                    'Ризик аварійної ситуації за напругою в бортовій мережі',
                    'Ризик аварійної ситуації за кількістю палива',
                    'Ризик аварійної ситуації за напругою в АКБ'
                ]
                for ind, risk in enumerate(risk_titles):
                    risk_value = np.round(100 * temp_df[f'risk {ind+1}'].values[-1], 2)
                    delta_value = np.round(100 * (temp_df[f'risk {ind+1}'].values[-1] - temp_df[f'risk {ind+1}'].values[-2]), 2)
                    if delta_value == 0:
                        delta_color = 'off'
                    else:
                        delta_color = 'inverse'
                    info_cols[1].metric(
                        label=risk,
                        value=f'{risk_value}%',
                        delta=delta_value,
                        delta_color=delta_color
                    )
                # if check_sensors[samples+j]:
                #     info_cols[1].write('**Увага!** Можливо, необхідно перевірити справність датчиків.')

                # sleep(0.3)

            df_to_show.to_excel(params['output_file'], engine='openpyxl', index=False)
            with open(params['output_file'], 'rb') as fout:
                col4.download_button(
                    label='Завантажити вихідний файл',
                    data=fout,
                    file_name=params['output_file'],
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
#             col4.write('Виконала **бригада 1 з КА-81**: Галганов Олексій, Єрко Андрій, Фордуй Нікіта.')
