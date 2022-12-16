# -*- coding: utf-8 -*-
"""
Version du 15/12/2022 pour streamlit sharing
Les modèles ne sont plus en direct mais remplacés
par les fichiers de résultats issus du programme original
( cf PYcycle_2022_original)
"""



import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pydeck as pdk
from datetime import date


import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind , pearsonr

#Configuration de la page
st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

#Fichiers de données à utiliser
velo_J=pd.read_csv("velo_J.csv",sep=",",index_col=0)
velo_J=velo_J.reset_index(drop=True)
velo_S=pd.read_csv("velo_S.csv",sep=",",index_col=0)
velo_S=velo_S.reset_index(drop=True)
infos_compteurs=pd.read_csv("infos_compteurs.csv",sep=",",index_col=0)
infos_compteurs=infos_compteurs.reset_index(drop=True)
equip_arronds=pd.read_csv("equipements_arronds.csv",sep=",",index_col=0)
df_calendrier=pd.read_csv("data_calendrier.csv",sep=";")
df_calendrier["Date"]=pd.to_datetime(df_calendrier["Date"])
df_calendrier["Date"]=df_calendrier["Date"].apply(lambda x:x.date())

df_meteo=pd.read_csv("data_meteo.csv",sep=";")
df_meteo["Date"]=pd.to_datetime(df_meteo["Date"])
df_meteo["Date"]=df_meteo["Date"].apply(lambda x:x.date())
soleil1=df_meteo["Ensoleillement"].apply(lambda x: x.split("h")[0]).astype('int')
soleil2=(df_meteo["Ensoleillement"].apply(lambda x: x.split("h")[1]).astype('int')/60)
df_meteo["Ensoleillement"]=soleil1+soleil2

velo_extrait=pd.read_csv("velo_extrait.csv",sep=",",index_col=0)

#cas particulier reconstitution de velo_H
velo_H1=pd.read_csv("velo_H1.csv",sep=",",index_col=0)
velo_H2=pd.read_csv("velo_H2.csv",sep=",",index_col=0)
velo_H3=pd.read_csv("velo_H3.csv",sep=",",index_col=0)
velo_H4=pd.read_csv("velo_H4.csv",sep=",",index_col=0)

velo_H=pd.concat([velo_H1,velo_H2,velo_H3,velo_H4])
velo_H["Date"]=pd.to_datetime(velo_H["Date"])
velo_H["Date"]=velo_H["Date"].apply(lambda x:x.date())

velo_H=velo_H.merge(right=df_calendrier,on="Date",how="left")
velo_H=velo_H.merge(right=df_meteo,on="Date",how="left")
velo_H=velo_H.merge(right=infos_compteurs,on="Id_compteur",how="inner")
velo_H=velo_H.merge(right=equip_arronds,left_on="Arrondissement",right_on="DEPCOM",how="inner")
velo_H=velo_H.drop("DEPCOM",axis=1)
velo_H=velo_H[velo_H["Arrondissement"]!=0]
velo_H=velo_H.sort_values(by="Date",ascending=True)
velo_H=velo_H.drop(["Id_compteur","Date","Nom_métro","Superficie","Population","Nom_compteur","Comptage_moyen","CodeClasse","Classe_comptage"],axis=1)

#Modif du fichier pour affichage
equip_arronds=equip_arronds.rename({"DEPCOM":"Arrondissement"},axis=1)
equip_arronds=equip_arronds.set_index('Arrondissement')
#Tableaux extraits des fichiers pour la présentation
infos_synthese=infos_compteurs[["Nom_compteur","Arrondissement","Comptage_moyen"]]
infos_synthese=infos_synthese.rename({"Nom_compteur":"Nom du compteur"},axis=1)

infos_compteurs=infos_compteurs.sort_values(by="Comptage_moyen", ascending=True)
infos_compteurs=infos_compteurs.replace(to_replace=["3 avenue de la Porte D'Orléans (gare routière) S-N 3 avenue de la Porte D'Orléans S-N"],
                                     value=("3 avenue de la Porte D'Orléans S-N"))


#gestion des codes couleurs pour la carte
infos_compteurs["codeR"]=infos_compteurs["CodeClasse"]
infos_compteurs["codeG"]=infos_compteurs["CodeClasse"]
infos_compteurs["codeB"]=infos_compteurs["CodeClasse"]
for i in infos_compteurs.index:
    if infos_compteurs["CodeClasse"][i]==0:
        infos_compteurs["codeR"][i]=0.11765
        infos_compteurs["codeG"][i]=0.66470
        infos_compteurs["codeB"][i]=1
    elif infos_compteurs["CodeClasse"][i]==1:
        infos_compteurs["codeR"][i]=0
        infos_compteurs["codeG"][i]=0.39217
        infos_compteurs["codeB"][i]=0    
    elif infos_compteurs["CodeClasse"][i]==2:
        infos_compteurs["codeR"][i]=0
        infos_compteurs["codeG"][i]=1
        infos_compteurs["codeB"][i]=1
    else: 
        infos_compteurs["codeR"][i]=1
        infos_compteurs["codeG"][i]=0
        infos_compteurs["codeB"][i]=0
        
        
#Péparation des modélisations


#lecture des résuktats des modèles
scores=pd.read_csv("scores.csv",sep=",")

df_S_rfr_variables=pd.read_csv("df_S_rfr_variables.csv",sep=",",index_col=0)
df_S_xgbr_variables=pd.read_csv("df_S_xgbr_variables.csv",sep=",",index_col=0)
df_J_rfr_variables=pd.read_csv("df_J_rfr_variables.csv",sep=",",index_col=0)
df_J_xgbr_variables=pd.read_csv("df_J_xgbr_variables.csv",sep=",",index_col=0)
df_H_rfr_variables=pd.read_csv("df_H_rfr_variables.csv",sep=",",index_col=0)
df_H_xgbr_variables=pd.read_csv("df_H_xgbr_variables.csv",sep=",",index_col=0)


df_S_rfr_test=pd.read_csv("df_S_rfr_test.csv",sep=",")
df_S_xgbr_test=pd.read_csv("df_S_xgbr_test.csv",sep=",")
df_J_rfr_test=pd.read_csv("df_J_rfr_test.csv",sep=",")
df_J_xgbr_test=pd.read_csv("df_J_xgbr_test.csv",sep=",")
df_H_rfr_test=pd.read_csv("df_H_rfr_test.csv",sep=",")
df_H_xgbr_test=pd.read_csv("df_H_xgbr_test.csv",sep=",")

df_S_rfr=pd.read_csv("df_S_rfr.csv",sep=",")
df_S_xgbr=pd.read_csv("df_S_xgbr.csv",sep=",")
df_J_rfr=pd.read_csv("df_J_rfr.csv",sep=",")
df_J_xgbr=pd.read_csv("df_J_xgbr.csv",sep=",")
df_H_rfr_moy=pd.read_csv("df_H_rfr_moy.csv",sep=",")
df_H_xgbr_moy=pd.read_csv("df_H_xgbr_moy.csv",sep=",")

#début de la présentation streamlit

st.sidebar.title("Projet PYcycle 2022")
pages=["Présentation","Carte des compteurs","Dataviz-Variables spatiales","Dataviz-Variables temporelles","Modélisation","Conclusions"]
page=st.sidebar.radio("Aller vers", pages)

#période suivie
periode=velo_J[["An_comptage","Nummois_comptage","Numjourmois_comptage"]]
periode["date"]=periode["An_comptage"]
for i in periode.index:
   periode["date"][i]=date(periode["An_comptage"][i],periode["Nummois_comptage"][i],periode["Numjourmois_comptage"][i])
periode=periode.drop_duplicates(subset=["date"],keep="first").reset_index(drop=True)
datemin=periode["date"].min()
datemax=periode["date"].max()

if page==pages[0]:
    new_title = '<p style="font-family:sans-serif; color:cornflowerblue; font-size: 15px;"><b>Projet fil rouge mené dans le cadre de la formation Data Analyst - avril 2022 en continu de Datascientest  -  Auteurs :C.Dreneau-E.Tchoué-D.Thémines</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown("""
            https://formation.datascientest.com/    
                """, unsafe_allow_html=True)
   
    st.write("\n")
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 17px;"><b>Présentation du projet</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    texte0 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Contexte</b></p>'
    st.markdown(texte0, unsafe_allow_html=True)
    
    col1,col2=st.columns([6,4])
    with col1:
        st.write("\n")
        st.markdown("""
  La Ville de Paris déploie depuis plusieurs années des compteurs à vélo permanents pour évaluer le développement de la pratique cycliste.
  Nous nous proposons d’effectuer une analyse des données récoltées par ces compteurs afin de visualiser les horaires et les zones d’affluences.    
                
                """)
    with col2:           
        st.image("affiche velib.jpg",caption=None,use_column_width="auto", output_format="auto")
            
    
    texte01 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Données de base</b></p>'
    st.markdown(texte01, unsafe_allow_html=True)
    
    st.markdown("""
    Les données de base sont constituées du fichier "comptage-velo-donnees-compteurs.csv" téléchargé à l'adresse https://opendata.paris.fr/pages/home/
    
    La présente étude est basée sur un téléchargement du 15/06/2022.
    
    Ci-dessous un extrait de ce fichier :
                """)
    
    st.dataframe(velo_extrait)
    
    st.write("\nLe nombre total de compteurs est  :",infos_compteurs.shape[0])
    st.write ("La période de comptage couverte par l' étude s'étend du " ,datemin ," au ",datemax ,"  soit ",(datemax-datemin).days," jours.")
    st.write("Pour chaque compteur et chaque jour , on dispose d' 1 valeur de comptage par heure de la fréquentation cycliste.")
    st.write("\n")
    texte02 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Données ajoutées</b></p>'
    st.markdown(texte02, unsafe_allow_html=True)

    st.write("Les données ajoutées sont :")
    st.write(
          
          "- la météo journalière relevée à la station de Paris-Montsouris https://www.meteo60.fr/stations-releves/station-jour "
    )
    st.write(  
          "- des données calendaires:jours fériés ,vacances scolaires , périodes de restrictions Covid "
    ) 
    st.write(     
          "- des distances par rapport aux stations de métro après détermination de la station la plus proche en se basant sur les informations de la région Ile de France https://www.iledefrance-mobilites.fr/open-data ."
    )
    st.write(   
          "- des distances par rapport au centre de Paris dont les coordonnées définies par l'IGN sont : latitude 48.8566667 / longitude 2.34222222 , https://www.ign.fr/reperes/centre-geographique-des-departements-metropolitains ."
    )
    st.write(      
          "- des informations sur les équipements/infrastructures des arrondissements de Paris après retraitement de la base permanente des équipements bpe1621_pres_equip_DEPCOM.csv  de l' INSEE https://www.insee.fr/fr/statistiques/3606476?sommaire=3568656 ."
    )    
       
       
elif page==pages[1]:
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 17px;"><b>Carte des compteurs</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    st.markdown("""
        Le positionnement des compteurs à vélo est présenté avec un code couleur fonction
        du comptage horaire moyen sur toute la période d'étude ( en rouge les plus fréquentés > 175 passages par heure , en bleu 
        les moins fréquentés < 50 passages par heure )
                """)
    
    infos_compteurs=infos_compteurs.rename({"lng":"lon"},axis=1)
    st.pydeck_chart(pdk.Deck(
    map_style='road',
    map_provider='mapbox',
    initial_view_state=pdk.ViewState(
        latitude=48.8566667,
        longitude=2.34222222,
        zoom=11,
        max_zoom=50,
        pitch=1, #inclinaison de la carte - perspective
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=infos_compteurs,
            pickable=True,
            get_position='[lon, lat]',
            get_fill_color='[codeR*255,codeG*255,codeB*255]',
            get_radius='[175]',
        ),
    ],
    tooltip =  {
            "html":"<b>Nom du compteur :</b> {Nom_compteur}<br/><b>Comptage moyen:</b> {Comptage_moyen}"
                   "<br/><b>Classe de comptage :</b> {Classe_comptage}",  
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
                }
            },
    ))
    #fin afficahge carte
    #Affichage des infos compeurs
    st.write("\n\n")
    st.markdown("""
                
        Les graphes ci-dessous donnent une idée de l'hétérogénéité de la fréquentation cycliste mesurée par les compteurs.
                        
                """)
    
    col1,col2=st.columns([6, 2])
    with col1:
        st.write("\n")
        st.dataframe(infos_synthese)
        
    with col2:    
        counts, bins = np.histogram(infos_synthese.Comptage_moyen, bins=range(0, 350, 25))
        bins = 0.5 * (bins[:-1] + bins[1:])

        fig = px.bar(x=bins, y=counts,
                 labels={'x':'Comptage moyen', 'y':'Nombre de compteurs'},
                 title = "Distribution des compteurs selon le comptage moyen sur la période étudiée",
                 width=600, height=400,
                 color_discrete_sequence=["lightskyblue"],
                 #hover_data=list(infos_synthese.columns)
                 )
        st.plotly_chart(fig)
   # cmap=["dodgerblue","darkgreen","cyan","red"])
    st.write("\n\n")
    
    with col1:    
       
        fig = px.bar(data_frame= infos_compteurs.tail(5),x="Comptage_moyen",y="Nom_compteur",
            labels={'Comptage_moyen':'Comptage moyen', 'Nom_compteur':'Compteurs'},
            title = "Les 5 compteurs les plus fréquentés",
            width=500, height=300,
            color_discrete_sequence=["sandybrown"],
                 )
        st.plotly_chart(fig)
        
    with col2:
        st.write("\n")
        fig = px.bar(data_frame= infos_compteurs.head(5),x="Comptage_moyen",y="Nom_compteur",
            labels={'Comptage_moyen':'Comptage moyen', 'Nom_compteur':'Compteurs'},
            title = "Les 5 compteurs les moins fréquentés",
            width=500, height=300,
            color_discrete_sequence=["yellowgreen"],
                 )
        st.plotly_chart(fig)    
        
elif page==pages[2]:
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 17px;"><b>Variables spatiales</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown("""
               
On s'interesse ici aux variables supplémentaires introduites pour expliciter la dépendance du comptage à la position des compteurs :
d'une part deux variables de distance , distances au centre de Paris , distances aux stations de métro les plus proches 
et d'autre part des variables catégorielles , à l'échelle des arrondissements , qui indiquent la présence ou non de certains équipements.

"""  )
    st.write("\n")
    texte1 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Distance des compteurs au centre de Paris</b></p>'
    st.markdown(texte1, unsafe_allow_html=True)
    st.write("\n")
    
#DEBUT DE MODIF    
    fig=px.scatter(
        infos_compteurs ,x="Distance_centre_Paris",y="Comptage_moyen",
        trendline="ols",
        title ="Relation entre distance du compteur au centre ville  et comptage moyen",
        width=800,height=400,
        labels=dict(Distance_centre_Paris="Distance au centre de Paris en km", Comptage_moyen="Comptage horaire moyen"),
            )
    st.plotly_chart(fig)
 
    #test statistique
    resul = pearsonr(infos_compteurs["Distance_centre_Paris"],infos_compteurs["Comptage_moyen"])
    st.write("*Résultat du test de pearson :*","r = ",resul[0].round(4)," ","p_value = ",resul[1].round(6))
     
    st.markdown("""
Plus la distance au centre de Paris est faible , plus la fréquentation cycliste est élevée.      
            """)
    st.write("\n")
    texte2 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Distance des compteurs à la station de métro la plus proche</b></p>'
    st.markdown(texte2, unsafe_allow_html=True)
    st.write("\n")
      
    fig=px.scatter(
        infos_compteurs ,x="Distance_métro",y="Comptage_moyen",
        trendline="ols",
        title ="Relation entre distance du compteur au métro le plus proche  et comptage moyen",
        width=800,height=400,
        labels=dict(Distance_métro="Distance à la station de métro en km", Comptage_moyen="Comptage horaire moyen"),
            )
    st.plotly_chart(fig)
 #FIN DE MODIF    
    
    #test statistique
    resul = pearsonr(infos_compteurs["Distance_métro"],infos_compteurs["Comptage_moyen"])
    st.write(" ","*Résultat du test de pearson :*","r = ",resul[0].round(4)," ","p_value = ",resul[1].round(6))
    
    st.markdown("""
Plus on est proche d'une station de métro, plus la fréquentation cycliste est élevée.      
            """)
    st.write("\n")
    texte2 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Equipements/Infrastructures</b></p>'
    st.markdown(texte2, unsafe_allow_html=True)
    st.write("\n")
    
    #reconstitution simple de velo equip différente du notebook de référence
    velo_equip=velo_H[["Comptage_horaire","Arrondissement"]].groupby(by=["Arrondissement"],as_index=False).agg({"Comptage_horaire":"mean"})
    velo_equip=velo_equip.merge(right=equip_arronds,on="Arrondissement",how="inner")
    velo_equip=velo_equip.rename({"Comptage_horaire":"Comptage_moyen"}, axis=1)
    
    coeffcorr=[]
    pval=[]
    for var in velo_equip.columns[2:]:
        coeffcorr.append(pearsonr(velo_equip["Comptage_moyen"],velo_equip[var])[0])
        pval.append(pearsonr(velo_equip["Comptage_moyen"],velo_equip[var])[1])
    corr_comptage_equip=pd.DataFrame({"Variables":velo_equip.columns[2:],"correlation":coeffcorr,"p-value":pval}) 
    corr_comptage_equip["val_abs"]=corr_comptage_equip["correlation"].abs()
    corr_comptage_equip=corr_comptage_equip.sort_values(by="val_abs",ascending=False).reset_index(drop=True) 
    corr_comptage_equip=corr_comptage_equip.drop("val_abs",axis=1).head(20)
    
    st.write("Les 20 premiers équipements classés par valeur absolue de correlation sont: ")
    st.dataframe(corr_comptage_equip)   
    st.markdown("""
         On note que les premières corrélations sont toutes négatives : l'absence de l'équipement 
         correspond à une plus grande fréquentation cycliste dans l'arrondissement.
                """)
    st.write("Nous présentons deux graphes à titre d'illustration.")
    
  #DEBUT DE MODIF  
     
    df = df=velo_equip[["Comptage_moyen","TERRAINS DE GRANDS JEUX","STATION-SERVICE"]]
    df=df.replace(to_replace=[0,1],value=["absent","présent"])
    
    # TERRAINS DE GRANDS JEUX - t student
    st.write("***Premier exemple : Terrains de grands jeux***")
   
    fig = px.box(
        df, x="TERRAINS DE GRANDS JEUX", y="Comptage_moyen",
        title="Effet de TERRAINS DE GRANDS JEUX sur le comptage horaire moyen",
        width=800,height=400,
        )
    st.plotly_chart(fig)
    
    absent=velo_equip["Comptage_moyen"][velo_equip["TERRAINS DE GRANDS JEUX"]==0]
    present=velo_equip["Comptage_moyen"][velo_equip["TERRAINS DE GRANDS JEUX"]==1]
    resul = ttest_ind(absent,present)
    st.write("*Résultat du test de Student :*","t = ",resul[0].round(4)," ","p_value = ",resul[1].round(6))
    
    # STATION-SERVICE - t student
    st.write("***Deuxième exemple : Station-Service***")
    
    fig = px.box(
        df, x="STATION-SERVICE", y="Comptage_moyen",
        title="Effet de STATION-SERVICE sur le comptage horaire moyen",
        width=800,height=400,
        )
    st.plotly_chart(fig)
    
    absent=velo_equip["Comptage_moyen"][velo_equip["STATION-SERVICE"]==0]
    present=velo_equip["Comptage_moyen"][velo_equip["STATION-SERVICE"]==1]
    resul = ttest_ind(absent,present)
    st.write("*Résultat du test de Student :*","t = ",resul[0].round(4)," ","p_value = ",resul[1].round(6))

#FIN DE MODIF

elif page==pages[3]:
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 17px;"><b>Variables temporelles</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write("\n")
    st.markdown("""
               
On analyse ici les relations entre le comptage horaire et différentes variables temporelles ; jour , heure ,semaine et mois de comptage , périodes de vacances , météorologie quotidienne .

"""  )
    st.write("\n")
    texte1 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Correlation de Comptage horaire avec les données de date , de calendrier et de meteo</b></p>'
    st.markdown(texte1, unsafe_allow_html=True)
    st.write("\n")
    
    velo_heure=velo_H[["Comptage_horaire","Heure_comptage","Numjoursem_comptage","Numjourmois_comptage","Semaine_comptage","Nummois_comptage","An_comptage", "Vacances","Ferie",
            "Restrictions_covid","Temperature_min","Temperature_max","Temperature_moy","Precipitations","Ensoleillement"]]
    
    coeffcorr=[]
    pval=[]
    for var in velo_heure.columns[1:]:
        coeffcorr.append(pearsonr(velo_heure["Comptage_horaire"],velo_heure[var])[0])
        pval.append(pearsonr(velo_heure["Comptage_horaire"],velo_heure[var])[1])
    corr_velo_heure=pd.DataFrame({"Variables":velo_heure.columns[1:],"correlation":coeffcorr,"p-value":pval}) 
    corr_velo_heure["val_abs"]=corr_velo_heure["correlation"].abs()
    corr_velo_heure=corr_velo_heure.sort_values(by="val_abs",ascending=False).reset_index(drop=True) 
    corr_velo_heure=corr_velo_heure.drop("val_abs",axis=1)
    
    col1,col2,col3=st.columns([8,2,6])
    with col1:
        st.dataframe(corr_velo_heure)
        
    with col3:
        st.markdown("""
    Les 5 variables les plus corrélées au comptage horaire sont , par ordre décroissant de valeur absolue:
    l'heure de comptage ,le temps de  l'ensoleillement , le numéro du jour dans la semaine,le fait d'être en période de vacances ou non 
    et la température maximale dans la journée ( les 4 dernières sont définies à la maille de la journée et non de l'heure).
    A noter que les 2 variables ttemps d'ensoleillement et température max sont corrélées entre elles (0.57)        
                
        """)
    
    st.markdown("""
    Nous présentons les relations entre le comptage horaire et les 4 variables les 
    plus corrélées.            
                """)
    st.write("\n")
    #1ere variable
    texte3 =  '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Relation avec la variable heure de comptage</b></p>'
    st.markdown(texte3, unsafe_allow_html=True)
    
    velo_heurejour =velo_H.groupby(["Heure_comptage","Nomjour_comptage","Numjoursem_comptage"],as_index=False).agg({"Comptage_horaire":"mean"})
    
    
    #construction du df à représenter
    df=velo_heurejour[velo_heurejour["Nomjour_comptage"]=="Monday"][["Heure_comptage","Comptage_horaire"]].rename({"Comptage_horaire":"Lundi"},axis=1)
    df=df.reset_index(drop=True)
    liste_jour=["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    #boucle de création des autres colonnes
    for i in range(1,7,1):
        dfprov=velo_heurejour[velo_heurejour["Numjoursem_comptage"]==i].drop(["Nomjour_comptage","Numjoursem_comptage"],axis=1)
        dfprov=dfprov.rename({"Comptage_horaire": liste_jour[i]},axis=1)
        dfprov=dfprov.reset_index(drop=True)
        df=df.merge(right=dfprov,on="Heure_comptage",how="left")
    fig = px.line(        
        df, #Data Frame
        x = "Heure_comptage", #Columns from the data frame
        y = liste_jour,
        color_discrete_map={
            "Lundi": "yellowgreen",
            "Mardi": "peru",
            "Mercredi":"goldenrod",
            "Jeudi":"orange",
            "Vendredi":"darkkhaki",
            "Samedi":"steelblue",
            "Dimanche":"lightskyblue",   
            },
        title = "Comptage horaire en fonction de l'heure de comptage - Moyenne des compteurs",
        width=800, height=400,
        labels=dict(value="Comptage horaire",variable="jour de la semaine")  
    )
    st.plotly_chart(fig)
    #Test Anova Heure de comptage
    test_heure=ols('Comptage_horaire ~ Heure_comptage',data=velo_heure).fit()
    resul=sm.stats.anova_lm(test_heure)
    st.write("*Résultat du test de l'anova comptage horaire vs heure de comptage :*","F = ",resul["F"][0].round(4)," ","p_value = ",resul["PR(>F)"][0].round(6))
    
    st.markdown("""
    On observe , en semaine du lundi au vendredi , deux horaires de pointes à 7 heures et 17 heures qui correspondent aux horaires de travail.
    Les journées de samedi et dimanche sont différentes avec un maximum de comptage à 15 heures.           
                """)
    
    #2ième variable
    texte4 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Relation avec la variable ensoleillement</b></p>'
    st.markdown(texte4, unsafe_allow_html=True)
    st.write("\n")
    velo_soleil=velo_H.groupby(["Ensoleillement"],as_index=False).agg({"Comptage_horaire":"mean"})
    velo_soleil=velo_soleil.rename({"Comptage_horaire":"Comptage_moyen"},axis=1)
#DEBUT DE MODIF    
    fig=px.scatter(
        velo_soleil ,x="Ensoleillement",y="Comptage_moyen",
        trendline="ols",
        title ="Relation entre temps d'ensoleillement et comptage moyen",
        width=800,height=400,
        labels=dict(Ensoleillement="Temps d'ensoleillement en heures", Comptage_moyen="Comptage horaire moyen"),
        )
    st.plotly_chart(fig)
#FIN DE MODIF    
    #test statistique
    resul = pearsonr(velo_soleil["Ensoleillement"],velo_soleil["Comptage_moyen"])
    st.write("*Résultat du test de pearson :*","r = ",resul[0].round(4)," ","p_value = ",resul[1].round(6))
    
    st.markdown("""
L'effet du temps d'ensoleilement est bien marqué.La correlation positive est très significative.
Plus le temps d'ensoleillement d'une journée est élevé , plus la fréquentation est élevée.       
            """)
    
    texte5 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Relation avec la variable jour de comptage</b></p>'
    st.markdown(texte5, unsafe_allow_html=True)
    st.write("\n")
    velo_jour =velo_H.groupby(["Nomjour_comptage","Numjoursem_comptage"],as_index=False).agg({"Comptage_horaire":"mean"})
    velo_jour=velo_jour.rename({"Comptage_horaire":"Comptage_moyen"},axis=1)
    velo_jour=velo_jour.sort_values(by="Numjoursem_comptage",ascending=True)
    velo_jour["Nomjour_comptage"]=velo_jour["Nomjour_comptage"].replace(to_replace=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                    ,value=["Lundi","Mardi", "Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
    fig = px.bar(x=velo_jour["Nomjour_comptage"], y=velo_jour["Comptage_moyen"],
             labels={'x':'Jour de la semaine', 'y':'Comptage moyen'},
             title = "Comptage moyen en fonction du jour de la semaine",
             width=600, height=400,
             color_discrete_sequence=["lightskyblue"],
             
             )
    st.plotly_chart(fig)
    
    #Test Anova Jour de comptage
    test_jour=ols('Comptage_moyen ~ Numjoursem_comptage',data=velo_jour).fit()
    resul=sm.stats.anova_lm(test_jour)
    st.write("*Résultat du test de l'anova comptage horaire vs jour de comptage :*","F = ",resul["F"][0].round(4)," ","p_value = ",resul["PR(>F)"][0].round(6))

    st.markdown("""
La variation du niveau moyen de comptage en fonction du jour de la semaine est statistiquement établi.
La fréquentation commence à baisser dès le vendredi pour atteindre son niveau le plus bas le dimanche.           
            """)
    #4ième graphique
    texte6 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Relation avec la variable vacances</b></p>'
    st.markdown(texte6, unsafe_allow_html=True)
    st.write("\n")
    
    velo_vacances=velo_H.groupby(["Nomjour_comptage","Numjoursem_comptage","Vacances"],as_index=False).agg({"Comptage_horaire":"mean"})
#DEBUT DE MODIF
    df = df=velo_vacances[["Comptage_horaire","Vacances"]]
    df=df.replace(to_replace=[0,1],value=["hors","pendant"])
    
   
    fig = px.box(
        df, x="Vacances", y="Comptage_horaire",
        title="Effet de vacances sur le comptage moyen par jour",
        width=800,height=400,
        )
    st.plotly_chart(fig)

    absent=velo_vacances["Comptage_horaire"][velo_vacances["Vacances"]==0]
    present=velo_vacances["Comptage_horaire"][velo_vacances["Vacances"]==1]
    resul = ttest_ind(absent,present)
    st.write("*Résultat du test de Student :*","t = ",resul[0].round(4)," ","p_value = ",resul[1].round(6))
#FIN MODIF
    st.markdown("""
    La fréquention cycliste mesurée par les compteurs est significativement plus basse en période de vacances scolaires
    que pendant le reste de l'année.           
                """)
    
elif page==pages[4]:

    titre = '<p style="font-family:sans-serif; color:Green; font-size: 17px;"><b>Modélisation</b></p>'
    st.markdown(titre, unsafe_allow_html=True)
    
    new_title = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Conditions des modélisations</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown("**Variable cible : le comptage horaire,nombre de vélos enregistrés en 1 heure par un compteur.**")
    st.write("**Variables explicatives : 89 variables**")
   
    st.markdown("**La modélisation est faite à 3 échelles de temps : semaine , jour et heure.**")
    st.markdown("**Deux modèles de régression sont utlisés : RandomForestRegressor et XGBoostRegressor.**")
    st.markdown("""
        - **La période d'entraînement s'étend du 01/05/2021 au 23/03/2022 (80% )**
        - **La période de test s'étend du 24/03/2022 au 14/06/2022 (20%)**
        """)
             
    new_title = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Résultats des modélisations</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
 
    col1,col2=st.columns([2, 4])
    with col1:
        st.write("\n")
        option1=st.selectbox(
        "Semaine/jour/heure",    
        ("comptage par semaine","comptage par jour","comptage par heure"),
        )
        option2=st.selectbox(
        "RandomForest/XGBoost",
        ("RandomForestRegressor","XGBoostRegressor"),
        )
        if option1=="comptage par heure":
            periode="H"
            if option2=="XGBoostRegressor":
                scortrain=scores["xgbr_H"][0]
                scortest=scores["xgbr_H"][1]
                rmsetrain=scores["xgbr_H"][2]
                rmsetest=scores["xgbr_H"][3]
                df_variables_importantes=df_H_xgbr_variables
                y_test=df_H_xgbr_test["reel"]
                predtest=df_H_xgbr_test["pred"]
                df_H_moy=df_H_xgbr_moy
            
            elif option2=="RandomForestRegressor":
                scortrain=scores["rfr_H"][0]
                scortest=scores["rfr_H"][1]
                rmsetrain=scores["rfr_H"][2]
                rmsetest=scores["rfr_H"][3]    
                df_variables_importantes=df_H_rfr_variables
                y_test=df_H_rfr_test["reel"]
                predtest=df_H_rfr_test["pred"]
                df_H_moy=df_H_rfr_moy
        elif option1=="comptage moyen par jour":
            periode="J"
            if option2=="XGBoostRegressor":
                scortrain=scores["xgbr_J"][0]
                scortest=scores["xgbr_J"][1]
                rmsetrain=scores["xgbr_J"][2]
                rmsetest=scores["xgbr_J"][3]
                df_variables_importantes=df_J_xgbr_variables
                y_test=df_J_xgbr_test["reel"]
                predtest=df_J_xgbr_test["pred"]
                df_J=df_J_xgbr
            
            
            elif option2=="RandomForestRegressor":
                scortrain=scores["rfr_J"][0]
                scortest=scores["rfr_J"][1]
                rmsetrain=scores["rfr_J"][2]
                rmsetest=scores["rfr_J"][3]    
                df_variables_importantes=df_J_rfr_variables
                y_test=df_J_rfr_test["reel"]
                predtest=df_J_rfr_test["pred"]
                df_J=df_J_rfr
                
        elif option1=="comptage moyen par semaine":
            periode="S"
            if option2=="XGBoostRegressor":
                scortrain=scores["xgbr_S"][0]
                scortest=scores["xgbr_S"][1]
                rmsetrain=scores["xgbr_S"][2]
                rmsetest=scores["xgbr_S"][3]
                df_variables_importantes=df_S_xgbr_variables
                y_test=df_S_xgbr_test["reel"]
                predtest=df_S_xgbr_test["pred"]
                df_S=df_S_xgbr
            
            elif option2=="RandomForestRegressor":
               scortrain=scores["rfr_S"][0]
               scortest=scores["rfr_S"][1]
               rmsetrain=scores["rfr_S"][2]
               rmsetest=scores["rfr_S"][3]    
               df_variables_importantes=df_S_rfr_variables
               y_test=df_S_rfr_test["reel"]
               predtest=df_S_rfr_test["pred"]
               df_S=df_S_rfr
      #Résultas métriques et graphes
        st.write("Métriques du modèle:")
        st.write("scoretrain =",scortrain.round(2))
        st.write("scoretest =",scortest.round(2))
        st.write("rmse train=",rmsetrain.round(2))
        st.write("rmse test=",rmsetest.round(2))
    
    with col2:
    #tableau des importances
     st.write("Les 10 variables les plus importantes sont :")
     st.dataframe(df_variables_importantes)
     
    fig=px.scatter(
        x=y_test,y=predtest,
        trendline="ols",
        title ="Relation entre prédiction et réalité - Echantillon test",
        width=800,height=400,
        labels=dict(x="Valeurs réelles", y="Valeurs prédites"),
        )
    st.plotly_chart(fig)

    if periode=="H":
        
        #on recalcule la date sur chaque tranche --> 3 tranches plus rapide que sur l'ensemble
        #graphique pour le comptage à 7h
        
        df_H_7=df_H_moy[df_H_moy["Heure_comptage"]==7].reset_index(drop=True)
        df_H_12=df_H_moy[df_H_moy["Heure_comptage"]==12].reset_index(drop=True)
        df_H_17=df_H_moy[df_H_moy["Heure_comptage"]==17].reset_index(drop=True)
    
        fig = px.line(        
            df_H_7, #Data Frame
            x = "Date", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage à 7 Heures - Réalité et modèle- Moyenne des compteurs",
            width=800, height=400,
            labels=dict(value="Comptage horaire")
            
        )
        st.plotly_chart(fig)
    
        #graphique pour le comptage à 12h
       
        
        fig = px.line(        
            df_H_12, #Data Frame
            x = "Date", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage à 12 heures- Réalité et modèle-Moyenne des compteurs",
            width=800, height=400,
            labels=dict(value="Comptage horaire")
            
        )
        st.plotly_chart(fig)
        
        #graphique pour le ccomptage à 17h
        
        fig = px.line(        
            df_H_17, #Data Frame
            x = "Date", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage à 17 heures - Réalité et modèle-Moyenne des compteurs",
            width=800, height=400,
            labels=dict(value="Comptage horaire")
            
        )
        st.plotly_chart(fig)
    
        #graphique pour le comptage à 12h
       
     #Graphes supplémentaire pour jour et semaine
    latmax= 48.87756
    lngmax= 2.35535
    latmoy= 48.869831  
    lngmoy= 2.307076
    
    if periode=="J":
                
        #graphique pour la moyenne de compteurs
        df_J_moy=df_J.groupby(by=["Date"],as_index=False).agg({"y_train":"mean","y_test":"mean","pred_train":"mean","pred_test":"mean"})
       
        fig = px.line(        
            df_J_moy, #Data Frame
            x = "Date", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage moyen par jour - Réalité et modèle- Moyenne des compteurs",
            width=800, height=400,
            labels=dict(value="Comptage horaire")
            
        )
        st.plotly_chart(fig)
    
        #graphique pour le compteur le plus fréquenté 89 Bld de Magenta NO-SE
        df_J_max=df_J[(df_J["lat"]==latmax)&(df_J["lng"]==lngmax)]
        
        fig = px.line(        
            df_J_max, #Data Frame
            x = "Date", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage moyen par jour - Réalité et modèle-Compteur 89 Bld de Magenta NO-SE",
            width=800, height=400,
            labels=dict(value="Comptage horaire")
            
        )
        st.plotly_chart(fig)
        
        #graphique pour le compteur 33 avenue des Champs Elysées NO-SE
        df_J_33=df_J[(df_J["lat"]==latmoy)&(df_J["lng"]==lngmoy)]
        
        fig = px.line(        
            df_J_33, #Data Frame
            x = "Date", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage moyen par jour - Réalité et modèle-Compteur 33 avenue des Champs Elysées NO-SE",
            width=800, height=400,
            labels=dict(value="Comptage horaire")
            
        )
        st.plotly_chart(fig)
        
    if periode=="S":
        
        #graphique pour la moyenne de compteurs
        df_S_moy=df_S.groupby(by=["An_semaine","Semaine"],as_index=False).agg({"y_train":"mean","y_test":"mean","pred_train":"mean","pred_test":"mean"})
        
        fig = px.line(        
            df_S_moy, #Data Frame
            x = "Semaine", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage moyen par semaine - Réalité et modèle- Moyenne des compteurs",
            width=800, height=400,
            labels=dict( value="Comptage moyen")
            
        )
        st.plotly_chart(fig)
        
        #graphique pour le compteur le plus fréquenté 89 Bld de Magenta NO-SE
        df_S_max=df_S[(df_S["lat"]==latmax)&(df_S["lng"]==lngmax)]
        
        fig = px.line(        
            df_S_max, #Data Frame
            x = "Semaine", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage moyen par semaine - Réalité et modèle-Compteur 89 Bld de Magenta NO-SE",
            width=800, height=400,
            labels=dict(value="Comptage moyen")
            
        )
        st.plotly_chart(fig)
        
        #graphique pour le compteur 33 avenue des Champs Elysées NO-SE
        df_S_33=df_S[(df_S["lat"]==latmoy)&(df_S["lng"]==lngmoy)]
                
        fig = px.line(        
            df_S_33, #Data Frame
            x = "Semaine", #Columns from the data frame
            y = ["y_train","pred_train","y_test","pred_test"],
            color_discrete_map={
                "y_train": "steelblue",
                "pred_train": "orange",
                "y_test": "green",
                "pred_test": "red",
                },
            title = "Comptage moyen par semaine - Réalité et modèle-Compteur 33 avenue des Champs Elysées NO-SE",
            width=800, height=400,
            labels=dict( value="Comptage moyen"),
            
        )
        st.plotly_chart(fig)
       
            
    
elif page==pages[5]:
    titre = '<p style="font-family:sans-serif; color:Green; font-size: 17px;"><b>Conclusions</b></p>'
    st.markdown(titre, unsafe_allow_html=True)
    texte1 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Synthèse des principaux résultats</b></p>'
    st.markdown(texte1, unsafe_allow_html=True)
    st.markdown("""
  Les analyses du fichier de base montrent que  le comptage horaire  présente :
    - une dépendance à l'heure de comptage : deux horaires de pointes en semaine
    - une dépendance au jour de la semaine : opposition semaine/week-end
    - une forte dépendance au compteur : dix compteurs au-delà de 175 passages par heure en moyenne , 41 compteurs en dessous de 50 par heure , sur un total de 94 compteurs.

  Des données supplémentaires ont été introduites pour expliquer des effets saisonniers , données météo et  données calendaires ( vacances , jour ,fériés..).On constate :
    - une dépendance au temps d'ensoleillement : plus le temps d'ensoleillement est important, plus le comptage horaire est élevé
    - un effet de la variable vacances : le comptage est significativement plus fort hors période de vacances scolaires.

  D'autres données ont été introduites pour tenter d'expliciter l'effet compteur , d'une part des données de distance - distance au centre de Paris , distance à la station métro/RER la plus proche , d'autre part des données à l'échelle des arrondissements - densité de population , équipements des arrondissements.
  On constate  pour la variable cible :
    - une dépendance à la distance au centre de Paris : plus le compteur est proche , plus le comptage est élevé
    - une dépendance à la distance à la station de métro la plus proche : plus celle-ci est faible, plus le comptage est élevé
    - des corrélations significatives avec la présence ou l'absence de certains équipements : les 20 premières corrélations , sur 70 variables , sont toutes négatives , c'est à dire que l’absence de l'équipement considéré est associée à une fréquentation cycliste plus élevée. L’interprétation reste délicate.         
                """)
    
    texte1 = '<p style="font-family:sans-serif; color:DarkBlue; font-size: 15px;"><b>Commentaires sur les modélisations</b></p>'
    st.markdown(texte1, unsafe_allow_html=True)
    st.markdown("""
               
Les deux types de modèles présentent un fort surapprentissage.
Le modèle XGBoostRegressor obtient de meilleurs scores que le RandomForestRegressor sur la partie test.
Les deux modèles échouent à prédire le comportement des compteurs les plus fréquentés au-delà du 01/05/2022.
Les modèles arrivent à mieux prédire la tendance sur toute la période de test pour les compteurs situés dans la moyenne.

Une partie des problèmes provient des données d'entrée avec lesquelles on a essayé de caractériser 
les compteurs.Elles sont à la maille arrondissement et cela ne peut pas permettre de
différencier plusieurs compteurs du même arrondissement.

En termes de métier, il paraît important d'augmenter le nombre de compteurs, 
D’une part en assurant une meilleure répartition dans les arrondissements,
D’autre part en densifiant dans la zone centrale pour mieux comprendre les écarts entre les compteurs dans la moyenne (vers 80 passages par heure) et les compteurs à plus de 175 passages par heure.

"""  )

    st.markdown("_Deux compteurs situés dans le 4ième arrondissement_")
    col1,col2=st.columns(2)
    with col1:
        st.image("Totem 64 Rue de Rivoli.jpg",caption="Totem 64 Rue de Rivoli - 75104",use_column_width="auto", output_format="auto")

    with col2:
        st.image("18 quai de l'Hotel de ville.jpg",caption="18 quai de l'Hotel de ville - 75104",use_column_width="auto" ,output_format="auto")
    
    st.markdown("_Quatre compteurs situés dans le 10ième arrondissement_")    
    st.image("Quatre capteurs dans le 10ième.jpg",caption=None,use_column_width="auto" ,output_format="auto")

#fin du code

