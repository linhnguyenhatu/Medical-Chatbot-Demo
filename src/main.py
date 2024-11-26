from flask import Flask, render_template, request,session
import openai
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'

openai.api_key = '...'
#dashboard
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')



#chatbot
dialog = []
@app.route('/chatbot', methods=['GET','POST'])

def chat():
   

    question = request.form.get('question')
    if request.method == 'POST':
        if request.form.get('button_action') == 'activated':
            question = request.form.get('question')

            completion = openai.chat.completions.create(
            model = 'gpt-3.5-turbo',
            messages = [
            {'role': 'user', 'content': question}
            ],
            temperature = 0  
            )
            answer =  completion['choices'][0]['message']['content']
            dialog.append((question,answer))

    
    
    
    return render_template('chatbot.html',question = question, dialog=dialog)

#vision

@app.route('/vision', methods=['GET', 'POST'])

def vision():
    
    if request.form.get("button_start") == "started":

        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        while True:
            success, img = cap.read()
            if not success:
                break

            hands, img = detector.findHands(img)

            if hands:
                for hand in hands:
                    lmList = hand["lmList"]
                    bbox = hand["bbox"]
                    centerPoint = hand['center']
                    handType = hand["type"]

                    fingers = detector.fingersUp(hand)
                    finger_count = fingers.count(1)
                    display_text = f"{handType}: {finger_count} fingers"
                    cv2.putText(img, display_text, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
            
                


            cv2.imshow("Image", img)

            if cv2.waitKey(1) == ord('q'):
                break
    

        cap.release()
        cv2.destroyAllWindows()


    result = None    
    if request.form.get("button_morse_start") == "start":

        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        message = []
        last_signal_time = 0
        morse_code = { 'A':'.-', 'B':'-...',
                            'C':'-.-.', 'D':'-..', 'E':'.',
                            'F':'..-.', 'G':'--.', 'H':'....',
                            'I':'..', 'J':'.---', 'K':'-.-',
                            'L':'.-..', 'M':'--', 'N':'-.',
                            'O':'---', 'P':'.--.', 'Q':'--.-',
                            'R':'.-.', 'S':'...', 'T':'-',
                            'U':'..-', 'V':'...-', 'W':'.--',
                            'X':'-..-', 'Y':'-.--', 'Z':'--..',
                            '1':'.----', '2':'..---', '3':'...--',
                            '4':'....-', '5':'.....', '6':'-....',
                            '7':'--...', '8':'---..', '9':'----.',
                            '0':'-----', ', ':'--..--', '.':'.-.-.-',
                            '?':'..--..', '/':'-..-.', '-':'-....-',
                            '(':'-.--.', ')':'-.--.-', '' : ''}
        def interpret(message_str):
            message_str += ' '
            result = ''
            morse_word = ''
            i = 0
            index = 0
            if message_string == "":
                return ("Không có mã Morse nhập vào")
            else:
                while message_str[index] == " ":
                    message_str.replace(" ","")
                    index = index + 1
                for code in message_str:
                        if (code != ' '):
                            i = 0
                            morse_word += code
                        else:
                            i += 1
                            if i == 2 :
                                result += ' '
                            else:
                                if  morse_word in morse_code.values():
                                    result += list(morse_code.keys())[list(morse_code.values()).index(morse_word)]
                                    morse_word = ''
                                else:
                                    return "Sai mã Morse"
                                    break
                return result
        while True:
            success, img = cap.read()
            if not success:
                break

            hands, img = detector.findHands(img)

            if hands:
                for hand in hands:
                    lmList = hand["lmList"]
                    bbox = hand["bbox"]
                    centerPoint = hand['center']
                    handType = hand["type"]

                    fingers = detector.fingersUp(hand)
                    finger_count = fingers.count(1)
                    display_text = f"{handType}: {finger_count} fingers"
                    cv2.putText(img, display_text, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    
                    if time.time() - last_signal_time >= 1.0:
                            if finger_count == 5 :
                                message.append("-")
                                last_signal_time = time.time()
                                print(message)
                            elif finger_count == 0:
                                message.append(".")
                                last_signal_time = time.time()
                                print(message)
                            elif finger_count == 2:
                                message.append(" ")
                                last_signal_time = time.time()
                                print(message)
            
                


            cv2.imshow("Image", img)

            if cv2.waitKey(1) == ord('q'):
                break
    
        message_string = "".join(message)
        result = interpret(message_string)

        cap.release()
        cv2.destroyAllWindows()
        print(result)
    return render_template("vision.html",result = result)
if __name__ == '__main__':
    app.run(debug=True)
