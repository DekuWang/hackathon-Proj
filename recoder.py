import pyaudio,wave
import numpy as np


def listen(output_name):
    temp = 20
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = output_name

    mindb=500    #最小声音，大于则开始录音，否则结束
    delayTime=8  #小声1.3秒后自动终止
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("开始!计时")

    frames = []
    start_recording = False            # 开始录音节点
    continue_recording = True                #判断是否继续录音
    low_volume = False            #判断声音小了

    tempnum = 0                #tempnum、tempnum2、tempnum3为时间
    tempnum2 = 0

    while continue_recording:
        data = stream.read(CHUNK,exception_on_overflow = False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.short)
        temp = np.max(audio_data)
        if temp > mindb and start_recording==False:
            start_recording =True
            print("开始录音")
            tempnum2=tempnum

        if start_recording:

            if(temp < mindb and low_volume==False):
                low_volume = True
                tempnum2 = tempnum
                print("声音小，且之前是是大的或刚开始，记录当前点")
            if(temp > mindb):
                low_volume =False
                tempnum2 = tempnum
                #刷新

            if(tempnum > tempnum2 + delayTime*15 and low_volume==True):
                print("间隔%.2lfs后开始检测是否还是小声"%delayTime)
                if(low_volume and temp < mindb):
                    start_recording = False
                    #还是小声，则continue_recording=True
                    print("小声！")
                else:
                    low_volume = False
                    print("大声！")

        print(str(temp)  +  "      " +  str(tempnum))
        tempnum = tempnum + 1
        if tempnum > 150:                #超时直接退出
            continue_recording = False
    print("录音结束")

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# listen()

if __name__ == "__main__":
    listen("whoareyou_cache/0.wav")
