# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np
import requests
import json
import operator
from tensorflow import keras
from flask_cors import CORS
from keras.models import load_model



app = Flask(__name__, static_url_path='/static')
CORS(app)

############################################################## 전역변수들 ###########################################################################
token = 'Ekrxj25DkdlvmfpdlaXhdtlsrkqt'

dx_dic = {0 : "간농양" , 1 : "간암", 2 : "간경화", 3 : "감염성 장염", 4 : "거짓막성 결장염", 5 : "골반염", 6 : "과민성대장증후군", 7 : "궤양성 결장염",
            8 :  "급성 간염", 9 : "급성 게실염", 10 : "급성 복막염", 11 : "급성 신우신염 또는 요로감염", 12 : "급성 위장관염", 13 : "급성 장염", 14 :  "급성 충수돌기염",
            15 : "기계적 장폐색", 16 : "기능성 위장관 질환", 17 : "난소낭종파열", 18 : "난소의 염전",  19 : "난소의 혹", 20 : "담관결석", 21 : "담관염",       
            22 :  "담낭염", 23 :  "담도산통", 24 : "담도암", 25 : "대상포진", 26 : "대동맥박리 또는 파열", 27 : "대장암", 28 : "마비성 장폐색", 29 : "방광염",
            30 : "방광암", 31 :  "변비",  32 : "비브리오 패혈증" ,  33 : "비장의 농양 또는 경색증", 34 : "상세불명의 수신증", 35 : "소화성 궤양",  36 :  "신장의 경색증",           
            37 : "요로결석", 38 : "월경통", 39 : "위경련", 40 : "위암", 41 : "위염 또는 위식도역류염", 42 :  "자궁내막증", 43 : "자궁근종", 44 : "허혈성 장염", 
            45 : "장간막 림프절염", 46 : "췌장암", 47 : "췌장염", 48 : "크론병", 49 : "헤노흐-쇤라인 자반"}

reverse_dx_dic = dict(map(reversed, dx_dic.items()))


ktas_dic = {0 : 2, 1 : 3, 2 : 4, 3 : 5}


KTAS_dic = {2 : "KTAS 2점으로 즉시 병원에 가서 검사 받아보셔야 합니다.", 
            3 : "KTAS 3점으로 24시간 이내에 병원에 방문하여 진료를 받아보는 것이 필요합니다.",
            4 : "KTAS 4점으로 가까운 시일내에 병의원에서 진료가 필요합니다.",
            5 : "KTAS 5점으로 근처 의원에서 증상조절 먼저 해보세요."}

ktas_dx_dic= {1 : [], 2 : ['급성 복막염',  '난소의 염전',  '대동맥박리 또는 파열', '비브리오 패혈증' ,  '신장의 경색증', '허혈성 장염', '췌장염'],
3 : ['간농양', '감염성 장염',  '거짓막성 결장염', '골반염', '궤양성 결장염',  '급성 간염', '급성 게실염', '급성 신우신염 또는 요로감염', '급성 충수돌기염', '기계적 장폐색', '난소낭종파열', '담관결석', '담관염', '담낭염', '담도산통', '담도암', '대장암', '마비성 장폐색', '비장의 농양 또는 경색증',  '상세불명의 수신증', '소화성 궤양', '요로결석', '자궁내막증', '장간막 림프절염', '크론병', '대상포진'],
4 : ['간암', '간경화', '췌장암', '난소의 혹', '자궁근종', '헤노흐-쇤라인 자반', '방광암', '위암'],
5: ['급성 위장관염',  '급성 장염',  '기능성 위장관 질환',  '방광염',  '변비', '월경통', '위경련',  '위염 또는 위식도역류염'] }




final_Sx_list_Korean =["나이", "성별", "맥박", "38'C 이상의 발열", #0, 1, 2, 3
                       "37.5'C-38'C 사이의 발열", "열감이나 오한", "고혈압", "당뇨병", "고지혈증",  # 4, 5, 6, 7, 8
                       "심방세동", "위염/위식도역류염", "위/십이지장 궤양",  # 9, 10, 11
                       "담석, 담낭염 이나 담도염", "만성 바이러스성 간염 또는 독성/자가면역성 간염", # 12, 13 
                       "만성 췌장염", "게실염", #14, 15
                       "과민성장증후군", "자궁근종 또는 선종", "자궁내막증", #16, 17, 18
                       "난소의 혹", "골반염", #19, 20
                       "크론", "궤양성대장염", "요로결석","신우신염", # 21, 22, 23, 24
                       "쓸개절제술", "충수돌기절제술", "자궁절제술", "위절제술", #25, 26, 27, 28
                       "그 외 복부수술력", #29
                       "매일 소주 1병 이상 음주", # 30
                       "복부전체/", "상복부/", "하복부/",  # 31, 32, 33
                       "명치/", "배꼽주위/", "치골상부/", # 34, 35, 36
                       "우상복부/", "우중복부/", "우하복부/", # 37, 38, 39 
                       "상복부/", "좌중복부/", "좌하복부/", #40, 41, 42   
                       "오른쪽 옆구리/", "왼쪽 옆구리/", # 43, 44
                       "등으로 방사통", "사타구니로 방사통", "오른쪽 어깨로 방사통", # 45, 46, 47
                       "우하복부로 통증의 이동", # 48
                       "갑자기 시작된 급격한 복통", "며칠 전부터 시작되어 점차 심해지는 복통", # 49, 50
                       "며칠 전부터 시작되어 악화/완화를 반복하는 복통", "만성적으로 반복되는 복통", #51, 52
                       "생리주기와 연관", # 53
                       "1일 이내", "1-3일", "3-7일", "1주-2주", "2-4주", "1달 이상", # 54, 55, 56, 57, 58, 59
                       "타는 듯한", "쓰리고 신물이 올라오는", "찌르는 듯한", "쥐어트는 듯한", # 60, 61, 62, 63
                       "둔중하고 묵직한", "찢어지는 듯한", "욱신거리는", "복부가 불쾌한", "복부가 부푸는", # 64, 65, 66, 67, 68
                       "과식", "익히지 않은 음식이나 오래된 음식 복용", "기름진 음식(고기, 튀김 등) 복용", # 69, 70, 71
                       "맵고 짜고 자극적인 음식 복용", "심리적인 스트레스", "식사", "음주", # 72, 73, 74, 75
                       "일주일 이상 항생제 복용", #76                    
                       "식은땀", "소화불량", "식욕부진", "오심", "구토", # 77, 78, 79, 80, 81
                       "무른변", "물설사", "콧물 같은 점액변", "변비", "황달", #82, 83, 84, 85, 86
                       "빈뇨", "절박뇨", "배뇨통", "혈뇨", # 87, 88, 89, 90
                       "냉이 많이 나오고 냄새남", # 91
                       "복부나 다리에 멍/점상출혈", # 92
                       "통증 부위 피부의 물집", # 93
                       "급격한 체중 감소", "심한 피로감", # 94, 95
                       "배변습관 변화", "방귀는", "토혈", "혈변", "흑색변", # 96, 97, 98, 99, 100
                       "복막자극증상", "압통", "척추늑골각압통"]  #101, 102, 103


sx_list_female = []
sx_list_male = []

################################################### 뒤로가기 방지 - 결과에서도 뒤로가기가 안되는데????, 프런트에서 하는 방법??? ##########################################################################

## @app.after_request
## def add_header(resp):
##    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
##    resp.headers["Pragma"] ="no-cache"
##    resp.headers["Expires"] = "0"
##    return resp


################################################################ 머신러닝 시작  ######################################################################

@app.route('/doctor25_abdominal_pain')
def abdominal_pain_start():
    sx_list_female.clear()   ## 이전에 수행했던 리스트 초기화
    sx_list_male.clear()
    return render_template('/abdominal_pain/doctor25mlstart.html')  ## 남성 여성 선택하는 페이지로 가기


##################################################################  Female  #########################################################################

@app.route('/doctor25ml_abdominal_pain_female', methods=['POST'])   ## 여성 선택하면 넘어오는 페이지
def doctor25ml_abdominal_pain_female():
    return render_template('/abdominal_pain/female_vs.html')  ## 여성 v/s 페이지로 넘어가기




@app.route('/female_vs', methods=['POST'])   ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_VS():
    age = request.form.get('user_age')
    HR = request.form.get('hr')
    High_fever = request.form.get('>=38')
    Mod_fever = request.form.get('37.5-38')
    Febrile_sense = request.form.get('febrile sense')

    sx_list_female.insert(0, age)
    sx_list_female.insert(1, 2)
    sx_list_female.insert(2, HR)
    sx_list_female.insert(3, High_fever)
    sx_list_female.insert(4, Mod_fever)
    sx_list_female.insert(5, Febrile_sense)

    return render_template('/abdominal_pain/female_phx_1.html')   ## 다음 페이지로 넘겨주고

@app.route('/female_phx_1', methods=['POST'])   ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_past_hx_1():
    HTN = request.form.get('HTN')
    DM = request.form.get('DM')
    Dyslipidemia = request.form.get('dyslipidemia')
    A_fib = request.form.get('a.fib')
    

    sx_list_female.insert(6, HTN)
    sx_list_female.insert(7, DM)
    sx_list_female.insert(8, Dyslipidemia)
    sx_list_female.insert(9, A_fib)
    
    return render_template('/abdominal_pain/female_phx_2.html')

@app.route('/female_phx_2', methods=['POST']) 
def abdominal_pain_past_hx_2():
    Gastritis= request.form.get('gastritis')
    Peptic_ulcer = request.form.get('peptic ulcer')
    Biliary = request.form.get('biliary')
    Hepatitis = request.form.get('hepatitis')
    Pancreatitis = request.form.get('pancreatitis')
    Diverticulitis = request.form.get('diverticulitis')
    IBS = request.form.get('IBS')

    sx_list_female.insert(10, Gastritis)
    sx_list_female.insert(11, Peptic_ulcer)
    sx_list_female.insert(12, Biliary)
    sx_list_female.insert(13, Hepatitis)
    sx_list_female.insert(14, Pancreatitis)
    sx_list_female.insert(15, Diverticulitis)
    sx_list_female.insert(16, IBS)
    
    return render_template('/abdominal_pain/female_phx_3.html')



@app.route('/female_phx_3', methods=['POST']) 
def abdominal_pain_past_hx_3():
    Myoma = request.form.get('myoma')
    Endometriosis = request.form.get('endometriosis')
    Ovarian_mass = request.form.get('ovarian mass')
    PID = request.form.get('pid')
    Crohn = request.form.get('crohn')
    UC = request.form.get('UC')
    Ureter_stone = request.form.get('ureter stone')
    APN = request.form.get('APN')

    sx_list_female.insert(17, Myoma)
    sx_list_female.insert(18, Endometriosis)
    sx_list_female.insert(19, Ovarian_mass)
    sx_list_female.insert(20, PID)
    sx_list_female.insert(21, Crohn)
    sx_list_female.insert(22, UC)
    sx_list_female.insert(23, Ureter_stone)
    sx_list_female.insert(24, APN)


    return render_template('/abdominal_pain/female_surgical_hx.html')   ## 다음 페이지로 넘겨주고
    

@app.route('/female_surgical_hx', methods=['POST'])    ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_surgical_hx():
    Cholecystectomy = request.form.get('cholecystectomy')
    Appendectomy = request.form.get('appendectomy')
    Hysterectomy = request.form.get('hysterectomy')
    Gastrectomy = request.form.get('gastrectomy')
    Others = request.form.get('others')
    Alcohol = request.form.get('alcohol')

    sx_list_female.insert(25, Cholecystectomy)
    sx_list_female.insert(26, Appendectomy)
    sx_list_female.insert(27, Hysterectomy)
    sx_list_female.insert(28, Gastrectomy)
    sx_list_female.insert(29, Others)
    sx_list_female.insert(30, Alcohol)

    return render_template('/abdominal_pain/female_location.html')    ## 다음 페이지로 넘겨주고


@app.route('/female_location', methods=['POST'])   ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_location():
    Whole_abdomen = request.form.get('whole_abdomen')
    Upper_abdomen = request.form.get('upper_abdomen')
    Lower_abdomen = request.form.get('lower_abdomen')
    Epigastric = request.form.get('epigastric')
    Periumblical = request.form.get('periumblical')
    Suprapubic = request.form.get('suprapubic')
    RUQ = request.form.get('RUQ')
    Rt_mid = request.form.get('Rt_mid')
    RLQ = request.form.get('RLQ')
    LUQ = request.form.get('LUQ')
    Lt_mid = request.form.get('Lt_mid')
    LLQ = request.form.get('LLQ')
    Rt_flank = request.form.get('Rt_flank')
    Lt_flank = request.form.get('Lt_flank')

    sx_list_female.insert(31, Whole_abdomen)
    sx_list_female.insert(32, Upper_abdomen)
    sx_list_female.insert(33, Lower_abdomen)
    sx_list_female.insert(34, Epigastric)
    sx_list_female.insert(35, Periumblical)
    sx_list_female.insert(36, Suprapubic)
    sx_list_female.insert(37, RUQ)
    sx_list_female.insert(38, Rt_mid)
    sx_list_female.insert(39, RLQ)
    sx_list_female.insert(40, LUQ)
    sx_list_female.insert(41, Lt_mid)
    sx_list_female.insert(42, LLQ)
    sx_list_female.insert(43, Rt_flank)
    sx_list_female.insert(44, Lt_flank)

    return render_template('/abdominal_pain/female_radiation.html')  ## 다음 페이지로 넘겨주고

@app.route('/female_radiation', methods=['POST'])    ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_radiation():
    Back_radiation = request.form.get('back radiation')
    Inguinal_radiation = request.form.get('inguinal radiation')
    Shoulder_radiation= request.form.get('shoulder radiation')
    RLQ_migration = request.form.get('RLQ migration')
    
    sx_list_female.insert(45, Back_radiation)
    sx_list_female.insert(46, Inguinal_radiation)
    sx_list_female.insert(47, Shoulder_radiation)
    sx_list_female.insert(48, RLQ_migration)     
    
    return render_template('/abdominal_pain/female_onset.html')  ## 다음 페이지로 넘겨주고

@app.route('/female_onset', methods=['POST'])     ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_onset():
    Sudden_onset = request.form.get('sudden onset')
    Gradual_increasing = request.form.get('gradual increasing')
    Gradual_intermittent = request.form.get('gradual intermittent')
    Chronic = request.form.get('chronic')
    Mens = request.form.get('mens')
    
    
    sx_list_female.insert(49, Sudden_onset)
    sx_list_female.insert(50, Gradual_increasing)
    sx_list_female.insert(51, Gradual_intermittent)
    sx_list_female.insert(52, Chronic) 
    sx_list_female.insert(53, Mens) 
    
    return render_template('/abdominal_pain/female_duration.html')  ## 다음 페이지로 넘겨주고
    

@app.route('/female_duration', methods=['POST'])  ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_duration():
    A_day = request.form.get('a day')
    Three_day = request.form.get('3 day')
    A_week = request.form.get('1 week')
    Two_week = request.form.get('2 week')
    Four_week = request.form.get('4 week')
    Over_month = request.form.get('over month')
    
    
    sx_list_female.insert(54, A_day)
    sx_list_female.insert(55, Three_day)
    sx_list_female.insert(56, A_week)
    sx_list_female.insert(57, Two_week) 
    sx_list_female.insert(58, Four_week) 
    sx_list_female.insert(59, Over_month) 
    
    return render_template('/abdominal_pain/female_character.html')   ## 다음 페이지로 넘겨주고

@app.route('/female_character', methods=['POST'])  ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_character():
    Burning = request.form.get('burning')
    Soarness = request.form.get('soarness')
    Stabbing = request.form.get('stabbing')
    Squeezing = request.form.get('squeezing')
    Dull = request.form.get('dull')
    Tearing = request.form.get('tearing')
    Throbbing = request.form.get('throbbing')
    Discomfort = request.form.get('discomfort')
    Distension = request.form.get('distension')    
    
    
    
    sx_list_female.insert(60, Burning)
    sx_list_female.insert(61, Soarness)
    sx_list_female.insert(62, Stabbing)
    sx_list_female.insert(63, Squeezing) 
    sx_list_female.insert(64, Dull) 
    sx_list_female.insert(65, Tearing) 
    sx_list_female.insert(66, Throbbing) 
    sx_list_female.insert(67, Discomfort) 
    sx_list_female.insert(68, Distension) 
    
    return render_template('/abdominal_pain/female_agg_factor.html')  ## 다음 페이지로 넘겨주고

@app.route('/female_agg_factor', methods=['POST'])   ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_agg():
    Over_eat = request.form.get('over eat')
    Poisoned = request.form.get('poisoned')
    Oily = request.form.get('oily')
    Irritant = request.form.get('irritant')
    Stress = request.form.get('stress')
    Meal = request.form.get('meal')
    Drinking = request.form.get('drinking')
    Antibiotics = request.form.get('antibiotics')   
    
    
    
    sx_list_female.insert(69, Over_eat)
    sx_list_female.insert(70, Poisoned)
    sx_list_female.insert(71, Oily)
    sx_list_female.insert(72, Irritant) 
    sx_list_female.insert(73, Stress) 
    sx_list_female.insert(74, Meal) 
    sx_list_female.insert(75, Drinking) 
    sx_list_female.insert(76, Antibiotics) 
    
    
    return render_template('/abdominal_pain/female_asso_sx.html') ## 다음 페이지로 넘겨주고


@app.route('/female_asso_sx', methods=['POST'])  ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_asso_sx():
    Sweating = request.form.get('sweating')
    Dyspepsia = request.form.get('dyspepsia')
    Anorexia = request.form.get('anorexia')
    Nausea = request.form.get('nausea')
    Vomiting = request.form.get('vomiting')
    Loose_stool = request.form.get('loose stool')
    Watery_diarrhea = request.form.get('watery diarrhea')
    Mucuous = request.form.get('mucuous')  
    Constipation = request.form.get('constipation')  
    Jaundice = request.form.get('jaundice')    
    
    
    
    sx_list_female.insert(77, Sweating)
    sx_list_female.insert(78, Dyspepsia)
    sx_list_female.insert(79,  Anorexia)
    sx_list_female.insert(80, Nausea) 
    sx_list_female.insert(81, Vomiting) 
    sx_list_female.insert(82, Loose_stool) 
    sx_list_female.insert(83, Watery_diarrhea) 
    sx_list_female.insert(84, Mucuous) 
    sx_list_female.insert(85, Constipation) 
    sx_list_female.insert(86, Jaundice) 
    
    
    return render_template('/abdominal_pain/female_urinary_sx.html') ## 다음 페이지로 넘겨주고


@app.route('/female_urinary_sx', methods=['POST']) ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_urinary_sx():
    Frequency = request.form.get('frequency')
    Urgency = request.form.get('urgency')
    Dysuria = request.form.get('dysuria')
    Hematuria = request.form.get('hematuria')
    Vaginal_discharge = request.form.get('vaginal discharge')
    Petechiae = request.form.get('petechiae')
    Vesicle = request.form.get('vesicle')
    Weight_loss = request.form.get('weight loss')  
    Fatigue = request.form.get('fatigue')    
    
        
    sx_list_female.insert(87, Frequency)
    sx_list_female.insert(88, Urgency)
    sx_list_female.insert(89, Dysuria)
    sx_list_female.insert(90, Hematuria) 
    sx_list_female.insert(91, Vaginal_discharge) 
    sx_list_female.insert(92, Petechiae) 
    sx_list_female.insert(93, Vesicle) 
    sx_list_female.insert(94, Weight_loss) 
    sx_list_female.insert(95, Fatigue) 
       
    
    return render_template('/abdominal_pain/female_additional_sx.html')  ## 다음 페이지로 넘겨주고


@app.route('/female_additional_sx', methods=['POST'])  ## 폼에서 넘어오는 값 받아서 sx_list_female 에 리스트로 저장
def abdominal_pain_additional_sx():
    Bowel_habbit = request.form.get('bowel habbit')
    Gas = request.form.get('gas')
    Hematemesis = request.form.get('hematemesis')
    Hematochezia = request.form.get('hematochezia')
    Melena = request.form.get('melena')
    Peritoneal = request.form.get('peritoneal')
    Tenderness = request.form.get('tenderness')
    CVAT = request.form.get('cvat')  
      
        
    sx_list_female.insert(96, Bowel_habbit)
    sx_list_female.insert(97, Gas)
    sx_list_female.insert(98, Hematemesis)
    sx_list_female.insert(99, Hematochezia) 
    sx_list_female.insert(100, Melena) 
    sx_list_female.insert(101, Peritoneal) 
    sx_list_female.insert(102, Tenderness) 
    sx_list_female.insert(103, CVAT) 
   
    

    sx = pd.DataFrame(sx_list_female)           ## sx_list_female 를 판다스로 넘겨주기
    Sx_list = sx.fillna(2)                      ## NaN 값 2로 채우기
    final_Sx_list_female = list(np.array(Sx_list[0].tolist()))  ## 넘파이 어레이를 파이썬 리스트 형식으로 바꾸기
    final_Sx_list_female = [int (i) for i in final_Sx_list_female]  ## 소수 타입을 정수로 변환
    final_Sx_list_female_nump_array = [final_Sx_list_female]  ## 다시 넘파이 어레이 형태로
    print(final_Sx_list_female_nump_array)  ## 어레이 잘 만들어졌나 확인
        
    loaded_abdominal_pain= keras.models.load_model("doctor25_abdominal_pain")  ## 복통 모델 로드하기
      

    a = loaded_abdominal_pain.predict(final_Sx_list_female_nump_array)  ## 예측한 값 a 변수에 저장, 이차원 어레이로 출력됨
    b = np.array(a).flatten().tolist()  ## 1차원 리스트로 변환해서 b 변수에 저장
    b_list = [] ## 빈 리스트 선언

    
    for i in range (0,50) :
        b_list.append(format(b[i],'.3f'))   ## b[i]일 때 확률값을 3f 까지 출력해서 b_list 에 삽입

    b_floatList = list(map(float, b_list))  ## b_lis에 있는 값을 float 으로 변환해서 b_floatList 변수에 저장

    dx_per_list = []  ## 빈 리스트 선언 

    for i in range (0,50):
        dx_per_list.append([dx_dic[i],b_floatList[i]])  ## 빈 리스트에 i 값에 따른 진단명 및 확률값 삽입

    dx_per_list_sort = sorted(dx_per_list, key=operator.itemgetter(1), reverse=True)   ## 잘 나왔는지 확인 진당명 : 급성 충수돌기염, 확률 : 38% 형태로 출력됨.
    print("진단명 :", dx_per_list_sort[0][0], ",",  "확률 :", format(dx_per_list_sort[0][1]*100, '.1f'), "%")
    print("진단명 :", dx_per_list_sort[1][0], ",",  "확률 :", format(dx_per_list_sort[1][1]*100, '.1f'), "%")
    print("진단명 :", dx_per_list_sort[2][0], ",",  "확률 :", format(dx_per_list_sort[2][1]*100, '.1f'), "%")
   
    
    description_list = []   ## 나중에 결과값에 보여줄 dsecription list 
    phx_list1 = []  ## 조건 만족하면 '과거력으로' 라는 문구 보여줄 list  
    phx_list2 = []  ## 조건 만족하면 과거력 나열해주는 list 
    abdomen_surgical_hx_list1 = [] ## 조건 만족하면 '수숳력으로' 라는 문구 보여줄 list
    abdomen_surgical_hx_list2 = [] ## 조건 만족하면 수술력 나열해주는 list 
    hr_list = []  ## hr 삽입해 놓을 list
    shx_list = []  ## 음주력 삽입해 높을 list 
    ktas_0= [ ]  ## ktas 순위별로 따로 저장하기 
    ktas_1= [ ]  ## ktas 순위별로 따로 저장하기
    ktas_2= [ ]  ## ktas 순위별로 따로 저장하기


    if final_Sx_list_female[6] == 1 or final_Sx_list_female[7] == 1 or final_Sx_list_female[8] == 1 or final_Sx_list_female[9] == 1 or final_Sx_list_female[10] == 1 or final_Sx_list_female[11] == 1 or  final_Sx_list_female[12] == 1 or final_Sx_list_female[13] == 1 or final_Sx_list_female[14] == 1 or final_Sx_list_female[15] == 1 or final_Sx_list_female[16] == 1 or final_Sx_list_female[17] == 1 or  final_Sx_list_female[18] == 1 or final_Sx_list_female[19] == 1 or final_Sx_list_female[20] == 1 or final_Sx_list_female[21] == 1 or final_Sx_list_female[22] == 1 or final_Sx_list_female[23] == 1 or final_Sx_list_female[24] == 1:
        phx_list1.append("기저질환")
    ### 과거력 input 값 중 하나라도 만족하면 기저질환이라는 문구 삽입

    for i in range (6, 25): 
        if final_Sx_list_female[i] == 1 :
            phx_list2.append(final_Sx_list_Korean[i]+"/")
    ### 과거력 input 값 들어와 있으면 korean list 에서 해당 매칭 값 리스트에 저장
        

    if final_Sx_list_female[25] == 1 or final_Sx_list_female[26] == 1 or final_Sx_list_female[27] == 1 or final_Sx_list_female[28] == 1 or final_Sx_list_female[29] == 1 :
        abdomen_surgical_hx_list1.append("복부 수술력")
    ### 수술력 input 값 중 하나라도 만족하면 복부수술력이라는 문구 삽입

    for i in range (25, 30):  
        if final_Sx_list_female[i] == 1 :
            abdomen_surgical_hx_list2.append(final_Sx_list_Korean[i]+"/")
     ### 수술력 input 값 들어와 있으면 korean list 에서 해당 매칭 값 리스트에 저장
        
   
    hr_list.append("맥박수 : " )
    hr_list.append(final_Sx_list_female[2])  
    hr_list.append( "회")          
    ## 맥박수 list 에 '맥박수:' 입력받은 맥박수 '회' 저장하기  
    

    if final_Sx_list_female[30] == 1 :
        shx_list.append("매일 소주 1병 이상 음주를 합니다.")
    ## 음주력 input 값 들어와 있으면 위 문구 리스트에 삽입
   
    description_list.append(" " )
    description_list.append(final_Sx_list_female[0])
    description_list.append("세")
    ## 나이부터는 dexcription list 에 저장 

    if final_Sx_list_female[1] == 1:
        description_list.append("남성인 당신은,")
    else :
        description_list.append("여성인 당신은,")


    for i in range (49, 53):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i]+ "이")

    for i in range (54, 60):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i]+ " 지속되고 있습니다.")    
    
    if final_Sx_list_female[53] == 1 :
        description_list.append("또한 통증은" + final_Sx_list_Korean[i] + "이 있습니다.")
    
    description_list.append("통증의 위치는 ")

    for i in range (31, 45):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
      
    description_list.append("이고")
    
    for i in range (45, 49):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i] )

    if final_Sx_list_female[45] == 1 or final_Sx_list_female[46] == 1  or final_Sx_list_female[47] == 1 or final_Sx_list_female[48] == 1:
        description_list.append("이 동반 되며, ")

    for i in range (60, 69):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
        
    description_list.append("양상입니다.")        
      
    
    for i in range (69, 77):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
        
    if final_Sx_list_female[69] == 1 or final_Sx_list_female[70] == 1 or final_Sx_list_female[71] == 1 or final_Sx_list_female[72] == 1 or final_Sx_list_female[73] == 1 or final_Sx_list_female[74] == 1 or final_Sx_list_female[75] == 1 or final_Sx_list_female[76] == 1 : 
        description_list.append(" 후 증상이 악화 됩니다.")    
        

    if final_Sx_list_female[77] == 1 or final_Sx_list_female[78] == 1 or final_Sx_list_female[79] == 1 or final_Sx_list_female[80] == 1 or final_Sx_list_female[81] == 1 or final_Sx_list_female[82] == 1 or final_Sx_list_female[83] == 1 or final_Sx_list_female[84] == 1 or final_Sx_list_female[85] == 1 or final_Sx_list_female[86] == 1 or final_Sx_list_female[87] == 1 or final_Sx_list_female[88] == 1 or final_Sx_list_female[89] == 1 or final_Sx_list_female[90] == 1 or final_Sx_list_female[91] == 1 or final_Sx_list_female[92] == 1 or final_Sx_list_female[93] == 1 :
        description_list.append("동반 증상으로 ")
    
    for i in range (77, 94):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
        
    if final_Sx_list_female[77] == 1 or final_Sx_list_female[78] == 1 or final_Sx_list_female[79] == 1 or final_Sx_list_female[80] == 1 or final_Sx_list_female[81] == 1 or final_Sx_list_female[82] == 1 or final_Sx_list_female[83] == 1 or final_Sx_list_female[84] == 1 or final_Sx_list_female[85] == 1 or final_Sx_list_female[86] == 1 or final_Sx_list_female[87] == 1 or final_Sx_list_female[88] == 1 or final_Sx_list_female[89] == 1 or final_Sx_list_female[90] == 1 or final_Sx_list_female[91] == 1 or final_Sx_list_female[92] == 1 or final_Sx_list_female[93] == 1:
        description_list.append(" 가(이) 있습니다.")       
    
    for i in range (94, 97):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])  
            
    if final_Sx_list_female[94] == 1 or final_Sx_list_female[95] == 1 or final_Sx_list_female[96] == 1:
        description_list.append("가(이) 있으며,") 
    

    for i in range (98, 101):
        if final_Sx_list_female[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])

    if final_Sx_list_female[98] == 1 or final_Sx_list_female[99] == 1  or final_Sx_list_female[100] == 1:
        description_list.append("이 있고")    
    
       
    if final_Sx_list_female[97] == 1 :
        description_list.append(final_Sx_list_Korean[97] + " 잘 나오지 않습니다.")
    else:
        description_list.append(final_Sx_list_Korean[97] + " 잘 나오는 편입니다.")


    if final_Sx_list_female[101] == 1 :
        description_list.append(final_Sx_list_Korean[101]) 
    if final_Sx_list_female[102] == 1 :
        description_list.append(final_Sx_list_Korean[102]) 
    if final_Sx_list_female[103] == 1 :
        description_list.append(final_Sx_list_Korean[103]) 

    if final_Sx_list_female[101] == 1 or final_Sx_list_female[102] == 1 or final_Sx_list_female[103] == 1:
        description_list.append("이 있습니다.")


    description_list.append("위 사항으로 미루어 보아 " )
    description_list.append(format(dx_per_list_sort[0][1]*100, '.1f'))
    description_list.append( "% 의 확률로") 
    description_list.append(dx_per_list_sort[0][0] + "이 의심됩니다." )

   
    if reverse_dx_dic[dx_per_list_sort[0][0]] == 10 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 44 or final_Sx_list_female[98] == 1 or final_Sx_list_female[99] == 1 or final_Sx_list_female[100] == 1 :
           description_list.append(KTAS_dic[ktas_dic[0]])
    else :
        for i in range(1,6):
            if dx_per_list_sort[0][0] in ktas_dx_dic[i]:
                description_list.append(KTAS_dic[i])

    if reverse_dx_dic[dx_per_list_sort[0][0]] == 10 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 44 or final_Sx_list_female[98] == 1 or  final_Sx_list_female[99] == 1 or final_Sx_list_female[100] == 1 :
        ktas_0.append(2) 
    else :
        for i in range(1,6):
            if dx_per_list_sort[0][0] in ktas_dx_dic[i]:
                ktas_0.append(i)   
    
    if reverse_dx_dic[dx_per_list_sort[1][0]] == 10 or reverse_dx_dic[dx_per_list_sort[1][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[1][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[1][0]]  == 44 :
        ktas_1.append(2) 
    else :
        for i in range(1,6):
            if dx_per_list_sort[1][0] in ktas_dx_dic[i]:
                ktas_1.append(i)   
    
    if reverse_dx_dic[dx_per_list_sort[2][0]] == 10 or reverse_dx_dic[dx_per_list_sort[2][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[2][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[2][0]]  == 44 :
        ktas_2.append(2)
    else :
        for i in range(1,6):
            if dx_per_list_sort[1][0] in ktas_dx_dic[i]:
                ktas_2.append(i)

    female_list_slicing = final_Sx_list_female[3:]

    data= {"token" : token, 
           "resultDescription" : [{"과거력 제목": phx_list1, "과거력" : phx_list2, "수술력 제목": abdomen_surgical_hx_list1, "수술력" : abdomen_surgical_hx_list2, "사회력" : shx_list, "증상 서술문" : description_list}], 
           "mainSymptomStatementId" : dx_per_list_sort[0][0],
           "reason" : "복통" ,
           "urgency " : ktas_0,
           "sessionConclusions" : [{"description" : dx_per_list_sort[0][0] , "percentage" : format(dx_per_list_sort[0][1]*100, '.1f') , "specialties" : ""}, {"description" : dx_per_list_sort[1][0] , "percentage" : format(dx_per_list_sort[0][1]*100, '.1f') , "specialties" : ""} ]
          }        
    
    json_data = json.dumps(data, ensure_ascii=False)
       
    if all(element == 2 for element in  female_list_slicing):
        return  render_template('/abdominal_pain/result_error.html')     
    else:
        return render_template('/abdominal_pain/result.html', a = json_data ) 
 


######################################################################## Male  #####################################################################

@app.route('/doctor25ml_abdominal_pain_male', methods=['POST'])
def doctor25ml_abdominal_pain_male():
    return render_template('/abdominal_pain/male_vs.html')



@app.route('/male_vs', methods=['POST'])
def abdominal_pain_male_VS():
    age = request.form.get('user_age')
    HR = request.form.get('hr')
    High_fever = request.form.get('>=38')
    Mod_fever = request.form.get('37.5-38')
    Febrile_sense = request.form.get('febrile sense')

    sx_list_male.insert(0, age)
    sx_list_male.insert(1, 1)
    sx_list_male.insert(2, HR)
    sx_list_male.insert(3, High_fever)
    sx_list_male.insert(4, Mod_fever)
    sx_list_male.insert(5, Febrile_sense)

    return render_template('/abdominal_pain/male_phx_1.html')


@app.route('/male_phx_1', methods=['POST'])
def abdominal_pain_male_past_hx_1():
    HTN = request.form.get('HTN')
    DM = request.form.get('DM')
    Dyslipidemia = request.form.get('dyslipidemia')
    A_fib = request.form.get('a.fib')
    Gastritis= request.form.get('gastritis')
    Peptic_ulcer = request.form.get('peptic ulcer')
    
    

    sx_list_male.insert(6, HTN)
    sx_list_male.insert(7, DM)
    sx_list_male.insert(8, Dyslipidemia)
    sx_list_male.insert(9, A_fib)
    sx_list_male.insert(10, Gastritis)
    sx_list_male.insert(11, Peptic_ulcer)
    

    return render_template('/abdominal_pain/male_phx_2.html')

@app.route('/male_phx_2', methods=['POST'])
def abdominal_pain_male_past_hx_2():
    Biliary = request.form.get('biliary')
    Hepatitis = request.form.get('hepatitis')
    Pancreatitis = request.form.get('pancreatitis')
    Diverticulitis = request.form.get('diverticulitis')
    IBS = request.form.get('IBS')
    Crohn = request.form.get('crohn')
    UC = request.form.get('UC')
    Ureter_stone = request.form.get('ureter stone')
    APN = request.form.get('APN')

    sx_list_male.insert(12, Biliary)    
    sx_list_male.insert(13, Hepatitis)
    sx_list_male.insert(14, Pancreatitis)
    sx_list_male.insert(15, Diverticulitis)
    sx_list_male.insert(16, IBS)
    sx_list_male.insert(17, 2)
    sx_list_male.insert(18, 2)
    sx_list_male.insert(19, 2)
    sx_list_male.insert(20, 2)
    sx_list_male.insert(21, Crohn)
    sx_list_male.insert(22, UC)
    sx_list_male.insert(23, Ureter_stone)
    sx_list_male.insert(24, APN)
    
    return render_template('/abdominal_pain/male_surgical_hx.html')

@app.route('/male_surgical_hx', methods=['POST'])
def abdominal_pain_male_surgical_hx():
    Cholecystectomy = request.form.get('cholecystectomy')
    Appendectomy = request.form.get('appendectomy')
    Gastrectomy = request.form.get('gastrectomy')
    Others = request.form.get('others')
    Alcohol = request.form.get('alcohol')

    sx_list_male.insert(25, Cholecystectomy)
    sx_list_male.insert(26, Appendectomy)
    sx_list_male.insert(27, 2)
    sx_list_male.insert(28, Gastrectomy)
    sx_list_male.insert(29, Others)
    sx_list_male.insert(30, Alcohol)

    return render_template('/abdominal_pain/male_location.html')


@app.route('/male_location', methods=['POST'])
def abdominal_pain_male_location():
    Whole_abdomen = request.form.get('whole_abdomen')
    Upper_abdomen = request.form.get('upper_abdomen')
    Lower_abdomen = request.form.get('lower_abdomen')
    Epigastric = request.form.get('epigastric')
    Periumblical = request.form.get('periumblical')
    Suprapubic = request.form.get('suprapubic')
    RUQ = request.form.get('RUQ')
    Rt_mid = request.form.get('Rt_mid')
    RLQ = request.form.get('RLQ')
    LUQ = request.form.get('LUQ')
    Lt_mid = request.form.get('Lt_mid')
    LLQ = request.form.get('LLQ')
    Rt_flank = request.form.get('Rt_flank')
    Lt_flank = request.form.get('Lt_flank')

    sx_list_male.insert(31, Whole_abdomen)
    sx_list_male.insert(32, Upper_abdomen)
    sx_list_male.insert(33, Lower_abdomen)
    sx_list_male.insert(34, Epigastric)
    sx_list_male.insert(35, Periumblical)
    sx_list_male.insert(36, Suprapubic)
    sx_list_male.insert(37, RUQ)
    sx_list_male.insert(38, Rt_mid)
    sx_list_male.insert(39, RLQ)
    sx_list_male.insert(40, LUQ)
    sx_list_male.insert(41, Lt_mid)
    sx_list_male.insert(42, LLQ)
    sx_list_male.insert(43, Rt_flank)
    sx_list_male.insert(44, Lt_flank)

    return render_template('/abdominal_pain/male_radiation.html')
   

@app.route('/male_radiation', methods=['POST'])
def abdominal_pain_male_radiation():
    Back_radiation = request.form.get('back radiation')
    Inguinal_radiation = request.form.get('inguinal radiation')
    Shoulder_radiation= request.form.get('shoulder radiation')
    RLQ_migration = request.form.get('RLQ migration')
    
    sx_list_male.insert(45, Back_radiation)
    sx_list_male.insert(46, Inguinal_radiation)
    sx_list_male.insert(47, Shoulder_radiation)
    sx_list_male.insert(48, RLQ_migration)     
    
    return render_template('/abdominal_pain/male_onset.html')

@app.route('/male_onset', methods=['POST'])
def abdominal_pain_male_onset():
    Sudden_onset = request.form.get('sudden onset')
    Gradual_increasing = request.form.get('gradual increasing')
    Gradual_intermittent = request.form.get('gradual intermittent')
    Chronic = request.form.get('chronic')
      
    
    sx_list_male.insert(49, Sudden_onset)
    sx_list_male.insert(50, Gradual_increasing)
    sx_list_male.insert(51, Gradual_intermittent)
    sx_list_male.insert(52, Chronic) 
    sx_list_male.insert(53, 2) 
    
    return render_template('/abdominal_pain/male_duration.html')


@app.route('/male_duration', methods=['POST'])
def abdominal_pain_male_duration():
    A_day = request.form.get('a day')
    Three_day = request.form.get('3 day')
    A_week = request.form.get('1 week')
    Two_week = request.form.get('2 week')
    Four_week = request.form.get('4 week')
    Over_month = request.form.get('over month')
    
    
    sx_list_male.insert(54, A_day)
    sx_list_male.insert(55, Three_day)
    sx_list_male.insert(56, A_week)
    sx_list_male.insert(57, Two_week) 
    sx_list_male.insert(58, Four_week) 
    sx_list_male.insert(59, Over_month) 
    
    return render_template('/abdominal_pain/male_character.html')

@app.route('/male_character', methods=['POST'])
def abdominal_pain_male_character():
    Burning = request.form.get('burning')
    Soarness = request.form.get('soarness')
    Stabbing = request.form.get('stabbing')
    Squeezing = request.form.get('squeezing')
    Dull = request.form.get('dull')
    Tearing = request.form.get('tearing')
    Throbbing = request.form.get('throbbing')
    Discomfort = request.form.get('discomfort')
    Distension = request.form.get('distension')    
    
    
    
    sx_list_male.insert(60, Burning)
    sx_list_male.insert(61, Soarness)
    sx_list_male.insert(62, Stabbing)
    sx_list_male.insert(63, Squeezing) 
    sx_list_male.insert(64, Dull) 
    sx_list_male.insert(65, Tearing) 
    sx_list_male.insert(66, Throbbing) 
    sx_list_male.insert(67, Discomfort) 
    sx_list_male.insert(68, Distension) 
    
    return render_template('/abdominal_pain/male_agg_factor.html')


@app.route('/male_agg_factor', methods=['POST'])
def abdominal_pain_male_agg():
    Over_eat = request.form.get('over eat')
    Poisoned = request.form.get('poisoned')
    Oily = request.form.get('oily')
    Irritant = request.form.get('irritant')
    Stress = request.form.get('stress')
    Meal = request.form.get('meal')
    Drinking = request.form.get('drinking')
    Antibiotics = request.form.get('antibiotics')   
    
    
    
    sx_list_male.insert(69, Over_eat)
    sx_list_male.insert(70, Poisoned)
    sx_list_male.insert(71, Oily)
    sx_list_male.insert(72, Irritant) 
    sx_list_male.insert(73, Stress) 
    sx_list_male.insert(74, Meal) 
    sx_list_male.insert(75, Drinking) 
    sx_list_male.insert(76, Antibiotics) 
    
    
    return render_template('/abdominal_pain/male_asso_sx.html')


@app.route('/male_asso_sx', methods=['POST'])
def abdominal_pain_male_asso_sx():
    Sweating = request.form.get('sweating')
    Dyspepsia = request.form.get('dyspepsia')
    Anorexia = request.form.get('anorexia')
    Nausea = request.form.get('nausea')
    Vomiting = request.form.get('vomiting')
    Loose_stool = request.form.get('loose stool')
    Watery_diarrhea = request.form.get('watery diarrhea')
    Mucuous = request.form.get('mucuous')  
    Constipation = request.form.get('constipation')  
    Jaundice = request.form.get('jaundice')     
    
    
    sx_list_male.insert(77, Sweating)
    sx_list_male.insert(78, Dyspepsia)
    sx_list_male.insert(79,  Anorexia)
    sx_list_male.insert(80, Nausea) 
    sx_list_male.insert(81, Vomiting) 
    sx_list_male.insert(82, Loose_stool) 
    sx_list_male.insert(83, Watery_diarrhea) 
    sx_list_male.insert(84, Mucuous) 
    sx_list_male.insert(85, Constipation) 
    sx_list_male.insert(86, Jaundice) 
    
    
    return render_template('/abdominal_pain/male_urinary_sx.html')


@app.route('/male_urinary_sx', methods=['POST'])
def abdominal_pain_male_urinary_sx():
    Frequency = request.form.get('frequency')
    Urgency = request.form.get('urgency')
    Dysuria = request.form.get('dysuria')
    Hematuria = request.form.get('hematuria')
    Petechiae = request.form.get('petechiae')
    Vesicle = request.form.get('vesicle')
    Weight_loss = request.form.get('weight loss')  
    Fatigue = request.form.get('fatigue')    
    
        
    sx_list_male.insert(87, Frequency)
    sx_list_male.insert(88, Urgency)
    sx_list_male.insert(89, Dysuria)
    sx_list_male.insert(90, Hematuria) 
    sx_list_male.insert(91, 2) 
    sx_list_male.insert(92, Petechiae) 
    sx_list_male.insert(93, Vesicle) 
    sx_list_male.insert(94, Weight_loss) 
    sx_list_male.insert(95, Fatigue) 
       
    
    return render_template('/abdominal_pain/male_additional_sx.html')

@app.route('/male_additional_sx', methods=['POST'])
def abdominal_pain_male_additional_sx():
    Bowel_habbit = request.form.get('bowel habbit')
    Gas = request.form.get('gas')
    Hematemesis = request.form.get('hematemesis')
    Hematochezia = request.form.get('hematochezia')
    Melena = request.form.get('melena')
    Peritoneal = request.form.get('peritoneal')
    Tenderness = request.form.get('tenderness')
    CVAT = request.form.get('cvat')  
      
        
    sx_list_male.insert(96, Bowel_habbit)
    sx_list_male.insert(97, Gas)
    sx_list_male.insert(98, Hematemesis)
    sx_list_male.insert(99, Hematochezia) 
    sx_list_male.insert(100, Melena) 
    sx_list_male.insert(101, Peritoneal) 
    sx_list_male.insert(102, Tenderness) 
    sx_list_male.insert(103, CVAT) 


    sx = pd.DataFrame(sx_list_male)
    Sx_list = sx.fillna(2)
    final_Sx_list_male = list(np.array(Sx_list[0].tolist()))
    final_Sx_list_male = [int (i) for i in final_Sx_list_male] 
    final_Sx_list_male_nump_array = [final_Sx_list_male]
    print(final_Sx_list_male_nump_array)
        
    loaded_abdominal_pain= keras.models.load_model("doctor25_abdominal_pain")
        

    a = loaded_abdominal_pain.predict(final_Sx_list_male_nump_array)
    b = np.array(a).flatten().tolist()
    b_list = []

    
    for i in range (0,50) :
        b_list.append(format(b[i],'.3f'))

    b_floatList = list(map(float, b_list))

    dx_per_list = []

    for i in range (0,50):
        dx_per_list.append([dx_dic[i],b_floatList[i]])

    dx_per_list_sort = sorted(dx_per_list, key=operator.itemgetter(1), reverse=True)
    print("진단명 :", dx_per_list_sort[0][0], ",",  "확률 :", format(dx_per_list_sort[0][1]*100, '.1f'), "%")
    print("진단명 :", dx_per_list_sort[1][0], ",",  "확률 :", format(dx_per_list_sort[1][1]*100, '.1f'), "%")
    print("진단명 :", dx_per_list_sort[2][0], ",",  "확률 :", format(dx_per_list_sort[2][1]*100, '.1f'), "%")
   
    
    description_list = []
    phx_list1 = []
    phx_list2 = []
    abdomen_surgical_hx_list1 = []
    abdomen_surgical_hx_list2 = []
    hr_list = []
    shx_list = []
    ktas_0= [ ]
    ktas_1= [ ]
    ktas_2= [ ]     


    if final_Sx_list_male[6] == 1 or final_Sx_list_male[7] == 1 or final_Sx_list_male[8] == 1 or final_Sx_list_male[9] == 1 or final_Sx_list_male[10] == 1 or final_Sx_list_male[11] == 1 or  final_Sx_list_male[12] == 1 or final_Sx_list_male[13] == 1 or final_Sx_list_male[14] == 1 or final_Sx_list_male[15] == 1 or final_Sx_list_male[16] == 1 or final_Sx_list_male[17] == 1 or  final_Sx_list_male[18] == 1 or final_Sx_list_male[19] == 1 or final_Sx_list_male[20] == 1 or final_Sx_list_male[21] == 1 or final_Sx_list_male[22] == 1 or final_Sx_list_male[23] == 1 or final_Sx_list_male[24] == 1:
        phx_list1.append("기저질환")
    

    for i in range (6, 25): 
        if final_Sx_list_male[i] == 1 :
            phx_list2.append(final_Sx_list_Korean[i]+"/")
        

    if final_Sx_list_male[25] == 1 or final_Sx_list_male[26] == 1 or final_Sx_list_male[27] == 1 or final_Sx_list_male[28] == 1 or final_Sx_list_male[29] == 1 :
        abdomen_surgical_hx_list1.append("복부 수술력")
    

    for i in range (25, 30):  
        if final_Sx_list_male[i] == 1 :
            abdomen_surgical_hx_list2.append(final_Sx_list_Korean[i]+"/")
        
   
    hr_list.append("맥박수 : " )
    hr_list.append(final_Sx_list_male[2])  
    hr_list.append( "회")          
    
    

    if final_Sx_list_male[30] == 1 :
        shx_list.append("매일 소주 1병 이상 음주를 합니다.")
   
    description_list.append(" " )
    description_list.append(final_Sx_list_male[0])
    description_list.append("세")


    if final_Sx_list_male[1] == 1:
        description_list.append("남성인 당신은,")
    else :
        description_list.append("여성인 당신은,")


    for i in range (49, 53):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i]+ "이")

    for i in range (54, 60):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i]+ " 지속되고 있습니다.")    
    
    if final_Sx_list_male[53] == 1 :
        description_list.append("또한 통증은" + final_Sx_list_Korean[i] + "이 있습니다.")
    
    description_list.append("통증의 위치는 ")

    for i in range (31, 45):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
      
    description_list.append("이고")
    
    for i in range (45, 49):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i] )

    if final_Sx_list_male[45] == 1 or final_Sx_list_male[46] == 1  or final_Sx_list_male[47] == 1 or final_Sx_list_male[48] == 1:
        description_list.append("이 동반 되며, ")

    for i in range (60, 69):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
        
    description_list.append("양상입니다.")        
      
    
    for i in range (69, 77):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
        
    if final_Sx_list_male[69] == 1 or final_Sx_list_male[70] == 1 or final_Sx_list_male[71] == 1 or final_Sx_list_male[72] == 1 or final_Sx_list_male[73] == 1 or final_Sx_list_male[74] == 1 or final_Sx_list_male[75] == 1 or final_Sx_list_male[76] == 1 : 
        description_list.append(" 후 증상이 악화 됩니다.")    
        

    if final_Sx_list_male[77] == 1 or final_Sx_list_male[78] == 1 or final_Sx_list_male[79] == 1 or final_Sx_list_male[80] == 1 or final_Sx_list_male[81] == 1 or final_Sx_list_male[82] == 1 or final_Sx_list_male[83] == 1 or final_Sx_list_male[84] == 1 or final_Sx_list_male[85] == 1 or final_Sx_list_male[86] == 1 or final_Sx_list_male[87] == 1 or final_Sx_list_male[88] == 1 or final_Sx_list_male[89] == 1 or final_Sx_list_male[90] == 1 or final_Sx_list_male[91] == 1 or final_Sx_list_male[92] == 1 or final_Sx_list_male[93] == 1 :
        description_list.append("동반 증상으로 ")
    
    for i in range (77, 94):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])
        
    if final_Sx_list_male[77] == 1 or final_Sx_list_male[78] == 1 or final_Sx_list_male[79] == 1 or final_Sx_list_male[80] == 1 or final_Sx_list_male[81] == 1 or final_Sx_list_male[82] == 1 or final_Sx_list_male[83] == 1 or final_Sx_list_male[84] == 1 or final_Sx_list_male[85] == 1 or final_Sx_list_male[86] == 1 or final_Sx_list_male[87] == 1 or final_Sx_list_male[88] == 1 or final_Sx_list_male[89] == 1 or final_Sx_list_male[90] == 1 or final_Sx_list_male[91] == 1 or final_Sx_list_male[92] == 1 or final_Sx_list_male[93] == 1:
        description_list.append(" 가(이) 있습니다.")       
    
    for i in range (94, 97):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])  
            
    if final_Sx_list_male[94] == 1 or final_Sx_list_male[95] == 1 or final_Sx_list_male[96] == 1:
        description_list.append("가(이) 있으며,") 
    

    for i in range (98, 101):
        if final_Sx_list_male[i] == 1 :
            description_list.append(final_Sx_list_Korean[i])

    if final_Sx_list_male[98] == 1 or final_Sx_list_male[99] == 1  or final_Sx_list_male[100] == 1:
        description_list.append("이 있고")    
    
       
    if final_Sx_list_male[97] == 1 :
        description_list.append(final_Sx_list_Korean[97] + " 잘 나오지 않습니다.")
    else:
        description_list.append(final_Sx_list_Korean[97] + " 잘 나오는 편입니다.")


    if final_Sx_list_male[101] == 1 :
        description_list.append(final_Sx_list_Korean[101]) 
    if final_Sx_list_male[102] == 1 :
        description_list.append(final_Sx_list_Korean[102]) 
    if final_Sx_list_male[103] == 1 :
        description_list.append(final_Sx_list_Korean[103]) 

    if final_Sx_list_male[101] == 1 or final_Sx_list_male[102] == 1 or final_Sx_list_male[103] == 1:
        description_list.append("이 있습니다.")


    if reverse_dx_dic[dx_per_list_sort[0][0]] == 10 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 44 or final_Sx_list_male[98] == 1 or  final_Sx_list_male[99] == 1 or final_Sx_list_male[100] == 1 :
           description_list.append(KTAS_dic[ktas_dic[0]])
    else :
        for i in range(1,6):
            if dx_per_list_sort[0][0] in ktas_dx_dic[i]:
                description_list.append(KTAS_dic[i])

    if reverse_dx_dic[dx_per_list_sort[0][0]] == 10 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[0][0]]  == 44 or final_Sx_list_male[98] == 1 or  final_Sx_list_male[99] == 1 or final_Sx_list_male[100] == 1 :
        ktas_0.append(2) 
    else :
        for i in range(1,6):
            if dx_per_list_sort[0][0] in ktas_dx_dic[i]:
                ktas_0.append(i)   
    
    if reverse_dx_dic[dx_per_list_sort[1][0]] == 10 or reverse_dx_dic[dx_per_list_sort[1][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[1][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[1][0]]  == 44 :
        ktas_1.append(2) 
    else :
        for i in range(1,6):
            if dx_per_list_sort[1][0] in ktas_dx_dic[i]:
                ktas_1.append(i)   
    
    if reverse_dx_dic[dx_per_list_sort[2][0]] == 10 or reverse_dx_dic[dx_per_list_sort[2][0]]  == 26 or reverse_dx_dic[dx_per_list_sort[2][0]]  == 32 or reverse_dx_dic[dx_per_list_sort[2][0]]  == 44 :
        ktas_2.append(2)
    else :
        for i in range(1,6):
            if dx_per_list_sort[1][0] in ktas_dx_dic[i]:
                ktas_2.append(i)
    

    male_list_slicing = final_Sx_list_male[3:]

    data= {"token" : token, 
           "resultDescription" : [{"과거력 제목": phx_list1, "과거력" : phx_list2, "수술력 제목": abdomen_surgical_hx_list1, "수술력" : abdomen_surgical_hx_list2, "사회력" : shx_list, "증상 서술문" : description_list}], 
           "mainSymptomStatementId" :  dx_per_list_sort[0][0],
           "reason" : "복통" ,
           "urgency " : ktas_0,
           "sessionConclusions" : [{"description" : dx_per_list_sort[0][0] , "percentage" : format(dx_per_list_sort[0][1]*100, '.1f') , "specialties" : ""}, {"description" : dx_per_list_sort[1][0] , "percentage" : format(dx_per_list_sort[0][1]*100, '.1f') , "specialties" : ""} ]
          }        
    
    json_data = json.dumps(data, ensure_ascii=False)
       
    if all(element == 2 for element in  male_list_slicing):
        return render_template('/abdominal_pain/result_error.html')     
    else:
        return render_template('/abdominal_pain/result.html', a = json_data )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')