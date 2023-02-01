from flask import Flask, render_template, request, flash, g
from werkzeug.utils import secure_filename
import sqlite3 as sql
import base64
import requests
from time import sleep
import urllib3
import json
import os, pyscreenshot, random, string
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import easyocr

app = Flask(__name__)


################### 6번째 게임 : STT ###################
@app.route('/stt')
def stt():
    return render_template('6th_test.html')

DATABASE_URI = 'sttdb.db'
# ---- DB에서 데이터를 불러오기 ----
conn = sql.connect(DATABASE_URI, isolation_level=None)
cur = conn.cursor()
# cur.execute(
#     'CREATE TABLE IF NOT EXISTS STT (id TEXT, p TEXT, url TEXT)')
# id = 1
# p = "안녕하세요. 오늘도 멋진 하루 되세요"
# url = 'C:/Users/admin/Desktop/SoftLand (1)/game/6_stt/sound/정답1.wav'

# conn = sql.connect(DATABASE_URI, isolation_level=None)
# cur = conn.cursor()
# cur.execute("""INSERT INTO STT(id, p, url) 
#                     VALUES(?, ?, ?)""", (id, p, url))

cur.execute("SELECT * FROM STT")
db_text = str(cur.fetchmany(size=1))

# 경로와 정답Text만 추출하기 위한 처리
db_List = db_text.split("'")

global sound_target
sound_url = db_List[5]    # 경로
sound_target = db_List[3] # 정답Text
dic = {'1' : sound_target} # 정답 Text


@app.route('/sound')
def sound():
    
    return render_template('6th_test.html', target=dic['1'])

@app.route('/STT', methods=['POST', 'GET'])
def STT():
    
    String_sound = ''  # 녹음파일 Text
    String_target = '' # 정답 Text
    
    sleep(5)
    count = 1
    
    #---------------------------------------------------------------------------
    #      STT Open API
    #---------------------------------------------------------------------------
    if request.method == 'POST':
        openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
        accessKey = "f0f9fd15-daef-4655-b516-d7a9711c696a" 
        audioFilePath = "C:/Users/admin/Downloads/정답1.wav" # 다운로드한 음성파일을 여기에 넣어서 Text로 바꾸기
            
        languageCode = "korean"
        
        file = open(audioFilePath, "rb")
        audioContents = base64.b64encode(file.read()).decode("utf8")
        file.close()
        
        requestJson = {    
            "argument": {
                "language_code": languageCode,
                "audio": audioContents
            }
        }
        
        http = urllib3.PoolManager()
        response = http.request(
        "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8","Authorization": accessKey},
            body=json.dumps(requestJson)
        )
        
        print("[responseCode] " + str(response.status))
        print("[responBody]")
        print("===== 결과 확인 ====")

        # 출력결과는 쓸때없는 내용이 들어가기 때문에 필요한 부분만 가져오기
        string = str(response.data,"utf-8")
        List = string.split('"')
        List = List[-2]
        List = List[:-1]
        print(List)
        # 녹음한 음성을 처리한 결과를 List변수에 담는다.
        
        
        # dic = {'1' : "안녕하세요. 오늘도 멋진 하루 되세요"}
        
        
        # NLP 유사도검사를 위해 정답Text와 녹음하고 Text로 바꾼 결과를 변수에 담에서 NLP모델에 넘긴다.
        # 녹음파일 Text
        String_sound = List
        
        # 정답Text
        String_target = sound_target
        
        print(List)
        
        #---------------------------------------------------------------------------
        #       유사도 검사 NLP Open API
        #---------------------------------------------------------------------------
        
        openApiURL = "http://aiopen.etri.re.kr:8000/ParaphraseQA"
        accessKey = "f0f9fd15-daef-4655-b516-d7a9711c696a"
        sentence1 = String_sound
        sentence2 = String_target
        
        requestJson = {
        "argument": {
            "sentence1": sentence1,
            "sentence2": sentence2
            }
        }
        
        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8","Authorization" :  accessKey},
            body=json.dumps(requestJson)
        )
        
        print("[responseCode] " + str(response.status))
        print("[responBody]")
        print(str(response.data,"utf-8"))

        NLP_String = str(response.data,"utf-8")
        NLP_List = NLP_String.split('"')
        print(NLP_List)
        
        NLP_reuslt = NLP_List[-2]
        # NLP_reuslt = NLP_target[:-1]
        print(NLP_reuslt)
        
        #--------------------------------------------------------------------------
        #     검증 결과 추출 및 전송
        #--------------------------------------------------------------------------
        
        String = ''
        if NLP_reuslt == 'paraphrase' :
            String += '유사합니다'
        else:
            String += '유사하지 않습니다'
            
        os.remove(audioFilePath)
        #                                             정답문장          TTS        체크 결과
        return render_template('6th_test.html', target = sentence2, sound = sentence1, ck=String)




