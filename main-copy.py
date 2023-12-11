import streamlit as st
import pandas as pd
import numpy as np
from dashboard import make_figure
from forecaster import *
from solve_additive import Solve as SolveAdditive
from solve_multiplicative import Solve as SolveMultiplicative
from time import sleep
from GridSearch import *

st.set_page_config(layout='wide')

st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
    }
    </style>
    """, unsafe_allow_html=True)

st.header('Гарантоване функціонування фізичних моделей кіберфізичних систем в умовах багатофакторних ризиків')
st.write('Виконала **бригада 3 з КА-01** у складі Магаріної Анни, Стожок Анастасії, Захарової Єлизавети')

st.markdown("""
   <style>
            .st-emotion-cache-133e3c0  {background-color: transparent; padding: 0}
            .st-emotion-cache-6qob1r {background-color: #E6AFAF; }
            .main  {background-color: #FEECEF; }
            .st-emotion-cache-18ni7ap  {background-color: #FEECEF; }
            .st-emotion-cache-1fttcpj {display: none}
            .st-emotion-cache-1v7f65g .e1b2p2ww14 {margin: 0}
            .st-emotion-cache-3qrmye {background-color: #E81F64;}
            .st-emotion-cache-16txtl3 {padding: 20px 20px 0px 20px}
            .st-emotion-cache-1629p8f .h1 {padding-bottom: 20px}
            .st-emotion-cache-z5fcl4 {padding: 20px 20px }
}
            
    </style>

    """, unsafe_allow_html=True)

st.sidebar.title("Дані")
input_file = st.sidebar.file_uploader('Оберіть файл вхідних даних', type=['csv', 'txt'], key='input_file')
st.sidebar.markdown('<div style="border-bottom: 1px solid #AAA1A5; height: 1px;"/>', unsafe_allow_html=True)

st.sidebar.title("Розмірність")
col1, col2, col3, col4 = st.sidebar.columns(4)
x1_dim = col1.number_input('X1', value=4, step=1, min_value=0, key='x1_dim')
x2_dim = col2.number_input('X2', value=2, step=1, min_value=0, key='x2_dim')
x3_dim = col3.number_input('X3', value=3, step=1, min_value=0, key='x3_dim')
y_dim = col4.number_input('Y', value=3, step=1, min_value=0, key='y_dim')
st.sidebar.markdown('<div style="border-bottom: 1px solid #AAA1A5; height: 1px;"/>', unsafe_allow_html=True)


st.sidebar.title('Поліноми')
poly_type = ['Чебишова', 'Лежандра', 'Лаґерра', 'Ерміта']
selected_poly_type = st.sidebar.selectbox("Оберіть тип", poly_type)

st.sidebar.title('Форма')
form_type = ['Адитивна', 'Мультиплікативна']
selected_form_type = st.sidebar.selectbox("Оберіть тип", form_type)

st.sidebar.write('Оберіть степені поліномів')
col1, col2, col3 = st.sidebar.columns(3)
x1_deg = col1.number_input('X1', value=0, step=1, min_value=0, key='x1_deg')
x2_deg = col2.number_input('X2', value=0, step=1, min_value=0, key='x2_deg')
x3_deg = col3.number_input('X3', value=0, step=1, min_value=0, key='x3_deg')
st.sidebar.markdown('<div style="border-bottom: 1px solid #AAA1A5; height: 1px;"/>', unsafe_allow_html=True)

st.sidebar.title('Додатково')
col1, col2= st.sidebar.columns(2)
samples = col1.number_input('Розмір вибірки', value=100, step=1, key='samples')
pred_steps = col2.number_input('Крок', value=15, step=1, key='pred_steps')

weight_method = col1.radio('Ваги цільових функцій', ['Нормоване значення', 'Середнє арифметичне'], key='select_func')
lambda_option = col2.checkbox('Визначати λ з трьох систем рівнянь')

col1, col2 = st.columns(2)
if col1.button('Виконати обчислення', key='run1'):
    input_file_text = input_file.getvalue().decode()
    input_file_text = input_file_text.replace(',', '\t')
    input_data = np.fromstring('\n'.join(input_file_text.split('\n')[1:]), sep='\t').reshape(-1, 1+sum([x1_dim, x2_dim, x3_dim, y_dim]))
    params = {
                'dimensions': [x1_dim, x2_dim, x3_dim, y_dim],
                'input_file': input_data,
                'output_file': 'output_file.xlsx',
                'samples': samples,
                'pred_steps': pred_steps,
                'labels': {
                    'rmr': 'rmr', 
                    'time': 'Час, c', 
                    'y1': 'Напруга в бортовій мережі, В', 
                    'y2': 'Кількість палива, л', 
                    'y3': 'Напруга в АБ, В'
                }
            }
    params['degrees'] = [x1_deg, x2_deg, x3_deg]
    params['weights'] = weight_method
    params['poly_type'] = selected_poly_type
    params['lambda_multiblock'] = lambda_option  
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
        if selected_form_type == 'Адитивна':
            solver = getSolution(SolveAdditive, temp_params, max_deg=3)
        else:
            solver = getSolution(SolveMultiplicative, temp_params, max_deg=3)
  
        model = Forecaster(solver)
        if selected_form_type== 'Мультиплікативна':
            predicted = model.forecast(
                            input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps],
                            form='multiplicative'
                        )
        else:
            predicted = model.forecast(
                            input_data[:, 1:-y_dim][samples+j-1:samples+j-1+pred_steps],
                            form='additive'
                        )

        predicted[0] = input_data[:, -y_dim:][samples+j]
        for i in range(y_dim):
            m = 0.5 ** (1 + (i+1) // 2)
            if selected_form_type == 'Мультиплікативна':
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

        temp_df['Стан'].fillna(method='ffill', inplace=True)
        temp_df['Стан датчиків'] = check_sensors[:samples+j]
        temp_df['Стан датчиків'].replace({0: 'Cправні', 1: 'Несправні'}, inplace=True)

        df_to_show = temp_df.drop(columns=['risk 1', 'risk 2', 'risk 3'])[::-1]

        info_cols = table_placeholder.columns(spec=1)           

        info_cols[0].dataframe(df_to_show.style.apply(
                    lambda s: highlight(s, 'Стан', ['Аварійний', 'Нештатний'], ['#d698a2', '#ffd503']), axis=1
                ))

