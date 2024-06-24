import pyaudio
import wave
import argparse
import speech_recognition as sr
import requests
import json
import logging
import os
from datetime import datetime
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 设置 Tavily API 密钥
os.environ["TAVILY_API_KEY"] = ""

# 创建 Tavily 搜索 API 检索器
retriever = TavilySearchAPIRetriever(k=5)

# 创建 ChatOllama 对象
llm = ChatOllama(model="qwen2:7b")

# 获取当前日期
current_date = datetime.now().strftime("%Y-%m-%d")

# 定义文本生成模板
main_template = ChatPromptTemplate.from_template(
    """根据问题，抓住核心，汇总 {topic} 的内容。

日期: {date}
上下文: {context}

来源:
{sources}"""
)

def format_sources(results):
    sources = []
    for i, result in enumerate(results):
        link = result.metadata.get('link')
        title = result.metadata.get('title', 'No Title')
        if link:
            sources.append(f"{i+1}. {link} ({title})")
    return "\n".join(sources)

def retrieve_and_format_sources(topic):
    search_results = retriever.invoke(topic)
    context = "\n".join(result.page_content for result in search_results)
    sources = format_sources(search_results)
    return context, sources

# 录音设置
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 录音通道数
RATE = 44100  # 采样率
CHUNK = 1024  # 每个数据块的帧数
RECORD_SECONDS = 10  # 录音时长（秒）
WAVE_OUTPUT_FILENAME = "output.wav"  # 输出文件名

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=RECORD_SECONDS, help="录音时长（秒）")
parser.add_argument("--output", type=str, default=WAVE_OUTPUT_FILENAME, help="输出文件名")
args = parser.parse_args()

def record_audio(duration, output_filename):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        logging.info(f"开始录音，时长为 {duration} 秒...")
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        logging.info("录音结束。")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        input_directory = "input"
        os.makedirs(input_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(input_directory, f"input_{timestamp}.wav")
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        logging.info(f"音频文件已保存到 {output_filename}")
        return output_filename
    except Exception as e:
        logging.error(f"录音时发生错误：{e}")
        return None

def recognize_speech_from_wav(wav_filename, max_attempts=3):
    recognizer = sr.Recognizer()
    attempt = 1
    while attempt <= max_attempts:
        try:
            with sr.AudioFile(wav_filename) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="zh-CN")
                return text
        except sr.UnknownValueError:
            logging.error("无法识别音频")
        except sr.RequestError as e:
            logging.error(f"无法请求语音识别服务: {e}")
        except Exception as e:
            logging.error(f"识别时发生错误：{e}")
        attempt += 1
    logging.error(f"已达到最大尝试次数 ({max_attempts})，无法识别音频")
    return "无法识别音频"

def connect_to_model():
    logging.info("连接到大模型...")
    return True

def get_response_from_model(prompt):
    try:
        full_prompt = f"用户：{prompt}\n助手："
        logging.info(f"向模型发送的输入：{full_prompt}")
        response = requests.post("http://localhost:11434/api/generate", 
                                 json={"model": "qwen2:7b", "prompt": full_prompt}, 
                                 headers={"Content-Type": "application/json; charset=utf-8"})
        if response.status_code != 200:
            logging.error(f"调用模型时出错：{response.status_code}, {response.text}")
            return "抱歉，我无法生成回答。"
        response.encoding = 'utf-8'
        responses = [json.loads(line) for line in response.text.splitlines() if line]
        response_texts = [item.get("response", "") for item in responses if "response" in item]
        if not response_texts:
            logging.error("模型未能生成回答。")
            return "抱歉，我无法生成回答。"
        response_text = "".join(response_texts)
        logging.info(f"从模型接收的回答：{response_text}")
        return response_text
    except requests.exceptions.RequestException as e:
        logging.error(f"调用模型时出错：{e}")
        return "抱歉，我无法生成回答。"
    except json.JSONDecodeError as e:
        logging.error(f"解析 JSON 数据时出错：{e}")
        return "抱歉，我无法生成回答。"

def synthesize_speech(text):
    url_generate = "http://127.0.0.1:5005/generate_audio"
    out_directory = "out"
    os.makedirs(out_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(out_directory, f"generated_audio_{timestamp}.wav")
    payload = {"text": text, "temperature": 0.7, "audio_seed_input": -1}
    headers = {"Content-Type": "application/json"}
    try:
        response_generate = requests.post(url_generate, json=payload, headers=headers)
        if response_generate.status_code == 200:
            data = response_generate.json()
            output_path = data.get('output_path')
            if output_path and os.path.exists(output_path):
                with open(output_filename, 'wb') as f_out, open(output_path, 'rb') as f_in:
                    f_out.write(f_in.read())
                logging.info(f"生成的音频文件已保存到 {output_filename}")
                return output_filename
            else:
                logging.error(f"生成的音频文件路径不存在: {output_path}")
                return None
        else:
            logging.error(f"API 请求失败，状态码：{response_generate.status_code}，响应：{response_generate.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"请求 API 时发生错误：{e}")
        return None

def play_audio_and_delete(file_path):
    try:
        wf = wave.open(file_path, 'rb')
        audio = pyaudio.PyAudio()
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        wf.close()
        logging.info(f"播放完毕，删除文件：{file_path}")
        os.remove(file_path)
    except Exception as e:
        logging.error(f"播放或删除音频文件时发生错误：{e}")

def check_for_commands(text):
    prompt = f"这是用户的输入：“{text}”。请判断这是否是一个命令，并返回命令类型（例如：“连接大模型”或“停止”），如果不是，请回答“无”。"
    response = get_response_from_model(prompt)
    if "启动" in response:
        return "连接大模型"
    elif "停止" in response:
        return "停止"
    else:
        return "无"

def evaluate_model_response(response):
    # 简单评估：检查回答中是否包含“抱歉”字符
    return "抱歉" in response

if __name__ == "__main__":
    connected_to_model = False

    while True:
        # 录音
        audio_filename = record_audio(args.duration, args.output)
        if not audio_filename:
            continue
        
        # 语音识别
        recognized_text = recognize_speech_from_wav(audio_filename)
        logging.info("识别结果: %s", recognized_text)

        if recognized_text == "无法识别音频":
            continue  # 继续下一次循环进行重新录音和识别

        if not connected_to_model:
            # 检查命令
            command = check_for_commands(recognized_text)
            if command == "连接大模型":
                connected_to_model = connect_to_model()
                if connected_to_model:
                    logging.info("大模型已成功连接。")
                else:
                    logging.error("无法连接到大模型，继续检测其他命令。")
                continue

        if connected_to_model:
            if "停止" in recognized_text:
                logging.info("检测到'停止'命令，终止程序。")
                break
            
            # 获取大模型回复
            model_response = get_response_from_model(recognized_text)
            logging.info("大模型回复: %s", model_response)

            # 保存识别结果到 input 文件夹
            try:
                output_directory = os.path.dirname(audio_filename)
                recognized_text_path = os.path.join(output_directory, "recognized_text.txt")
                with open(recognized_text_path, "w", encoding="utf-8") as file:
                    file.write("识别结果:\n")
                    file.write(recognized_text)
                    file.write("\n\n大模型回复:\n")
                    file.write(model_response)
                logging.info(f"识别结果和大模型回复已保存到 {recognized_text_path} 文件中")
            except Exception as e:
                logging.error(f"保存文件时发生错误：{e}")

            # 生成回复的语音文件并保存在 out 文件夹下
            response_audio_path = synthesize_speech(model_response)
            if response_audio_path:
                logging.info(f"生成的回复语音文件路径：{response_audio_path}")

                # 播放生成的语音文件并删除
                play_audio_and_delete(response_audio_path)

            # 判断大模型回复是否足够
            if  evaluate_model_response(model_response):
                # 如果大模型回复不足，进行 Tavily 搜索
                logging.info("大模型回复不足，通过 Tavily 搜索补充信息。")
                topic = recognized_text  # 将识别的文本作为搜索的主题
                context, sources = retrieve_and_format_sources(topic)
                full_prompt = main_template.format(topic=topic, date=current_date, context=context, sources=sources)
                logging.info(f"Tavily 搜索结果和模型回复的结合：\n{full_prompt}")

                # 生成补充信息的语音文件并播放
                supplement_audio_path = synthesize_speech(full_prompt)
                if supplement_audio_path:
                    logging.info(f"生成的补充信息语音文件路径：{supplement_audio_path}")
                    play_audio_and_delete(supplement_audio_path)

        else:
            logging.error("程序无法连接到大模型，无法继续。")
            break
