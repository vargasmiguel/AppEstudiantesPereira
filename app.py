import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import pydeck as pdk

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

#####
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Agregar filtro")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar datos por", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valores para {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Valores para {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valores para {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Palabra de busqueda {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


####



@st.cache_data
def data():
    filtro=['I.E MARIA DOLOROSA','I.E JORGE ELIECER GAITAN', 'I.E MATECAÑA', 'I.E LENINGRADO', 'I.E MANOS UNIDAS', 'I.E SOFIA HERNANDEZ',
        'I.E VILLA SANTANA', 'I.E AUGUSTO ZULUAGA', 'I.E ESCUELA DE LA PALABRA', 'I.E INEM FELIPE PEREZ', 'I.E CIUDAD BOQUIA']
        #'I.E LA CARBONERA',
        # 'C.E MARIA CRISTINA GOMEZ','I.E EL RETIRO','I.E GABRIEL TRUJILLO','I.E GONZALO MEJIA ECHEVERRY',
        # 'I.E LA BELLA','I.E LA PALMILLA','I.E MUNDO NUEVO']
    df=pd.read_csv("estudiantesPereira.csv")
    dfA=df[df["INSTITUCION"].isin(filtro)]
    return dfA


dfA = data()


with st.sidebar:
    choose = option_menu("Mis Opciones", ["Datos", "MAPA","SEDE", "ESTRATO", "DISCAPACIDAD", "CAPACIDADES", "ETNIA", "NIVEL", "METODOLOGIA","JORNADA","EDAD"],
                         icons=['house', 'geo-alt', 'building', 'coin', 'universal-access', 'eyeglasses', 'person','book', "vector-pen", "sun-fill", "calendar"],
                         menu_icon="app-indicator", default_index=0)
    
if choose == "Datos":
    st.header("DATOS CRUDOS")
    st.dataframe(filter_dataframe(dfA))

if choose in ["SEDE", "ESTRATO", "DISCAPACIDAD", "CAPACIDADES","ETNIA", "NIVEL", "METODOLOGIA","JORNADA"]:
    st.header("DATOS POR "+choose)
    fig = px.bar(dfA[["INSTITUCION",choose]].value_counts().reset_index(), x="INSTITUCION", y="count", color=choose)
    st.plotly_chart(fig)


if choose == "EDAD":
    st.header("DATOS POR "+choose)
    dfA['EDAD']=pd.cut(dfA["EDAD"], bins=[0,6,10,16,18,100], labels=["0-6","6-10","10-16","16-18",">18"])
    fig = px.bar(dfA[["INSTITUCION","EDAD"]].value_counts().reset_index(), x="INSTITUCION", y="count", color='EDAD')
    st.plotly_chart(fig)



if choose == "MAPA":
    st.header("MAPA")
    coord={'I.E MARIA DOLOROSA':[4.81466, -75.70622],
       'I.E JORGE ELIECER GAITAN': [4.81167, -75.67667],
       'I.E MATECAÑA': [4.81628, -75.73208], 
       'I.E LENINGRADO': [4.79873, -75.73778],
       'I.E MANOS UNIDAS': [4.79444, -75.66637], 
       'I.E SOFIA HERNANDEZ': [4.80075, -75.73994],
       'I.E VILLA SANTANA': [4.8013, -75.66729], 
       'I.E AUGUSTO ZULUAGA': [4.81916, -75.70615], 
       'I.E ESCUELA DE LA PALABRA':[4.82073, -75.70153],
       'I.E INEM FELIPE PEREZ':[4.811635897782964, -75.71674771709239],
       'I.E CIUDAD BOQUIA':[4.8216234049201425, -75.73258535204414]}
       #'I.E LA CARBONERA':[],
       #'C.E MARIA CRISTINA GOMEZ':[],
       #'I.E EL RETIRO':[],
       #'I.E GABRIEL TRUJILLO':[],
       #'I.E GONZALO MEJIA ECHEVERRY':[],
       #'I.E LA BELLA':[],
       #'I.E LA PALMILLA':[],
       #'I.E MUNDO NUEVO':[]
    
    dfC=dfA[["INSTITUCION"]].value_counts().reset_index()
    
    
    dfC["lat"]=dfC["INSTITUCION"].map(lambda x: coord[x][0])
    dfC["lon"]=dfC["INSTITUCION"].map(lambda x: coord[x][1])
    print(dfC)


    #st.dataframe(dfA[["INSTITUCION","lon","lat"]])

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=4.80,
            longitude=-75.70,
            zoom=12,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
            'ColumnLayer',
            data=dfC,
            get_position='[lon, lat]',
            get_elevation="count",
            radius=80,
            elevation_scale=1,
            #elevation_range=[0, 10],
            get_fill_color="[255-(count/1000)*255, 255-(count/1000)*255, (count/1000)*255, 95]",
            pickable=True,
            extruded=True,
            ),
        ],
    tooltip={"text": "{INSTITUCION} \n Estudiantes: {count}"}))
