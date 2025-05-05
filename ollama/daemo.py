# import ollama
from ollama import Client


client = Client(
    host='http://localhost:11434',
    headers={'x-some-header': 'some-value'}
)
condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
        "请用中文回答。"
        "回到条数不能少于5条"
    )
response = client.chat(model='deepseek-r1:1.5b', messages=[
    {
        'role': 'user',
        'content': '你是谁?',
    },
])
print(response['message']['content'])

