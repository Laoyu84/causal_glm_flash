import os
import sys
import json
from zai import ZhipuAiClient

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt,wait_fixed


from util import constant
from util.logger import logger

load_dotenv()

def launch_client():
    API_KEY = os.getenv('API_KEY')
    if API_KEY is None:
        API_KEY = os.getenv('GLM_API_KEY')
    client = ZhipuAiClient(api_key= API_KEY)
    return client

def format_messages(sys_msg, user_msg, historicals=None):
    messages = []
    if sys_msg != "":
        messages.append({"role": "system", "content": sys_msg})
    
    if historicals:
        for s in historicals:
            messages.append({"role": s["role"], "content": s["content"]})
    
    if user_msg != "":
        messages.append( {"role": "user", "content": user_msg})   
    #print(f"messages are sent to llms: \n\n {messages}")
    return messages

@retry(stop=stop_after_attempt(constant.MAX_LLM_API_RETRY), wait=wait_fixed(1))
def completion(
        msgs,
        model = constant.GLM45,
        stream = False,
        temperature = 0.05,
        web_search_enabled = False,
        thinking_type = "enabled"  # "enabled", "disabled"
    ):

    client = launch_client()
    full_response = ""
    logger.debug(msgs)
    response = client.chat.completions.create(
        model=model,  # 填写需要调用的模型名称
        messages=msgs,
        stream=stream,
        thinking={
            "type": thinking_type,    # 启用深度思考模式
        },
        max_tokens = 8192,
        temperature=temperature,
        tools=[{"type": "web_search", "web_search": {"enable": web_search_enabled,"search_result": web_search_enabled}}]
    )
    if stream:
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                print(chunk.choices[0].delta.reasoning_content, end='', flush=True)

            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
                full_response += chunk.choices[0].delta.content
    else:
        full_response += response.choices[0].message.content
    
    return full_response

def completion_with_function_calling(
        msgs,
        tools,
        model = constant.GLM45,
        func = None
):  
    
    logger.debug(f"准备使用{msgs}调用function")
    client = launch_client()
    response = client.chat.completions.create(
        model=model,  
        messages=msgs,
        tools= tools,
        tool_choice= {"type": "function", "function": {"name": func}} if func is not None else "auto",
    )
    logger.info(f"准备调用function：{response.choices[0].message}")
    
    if response.choices[0].message.tool_calls :
        function = response.choices[0].message.tool_calls[0].function
        func_args = json.loads(function.arguments)
        func = None
        for module_name, module in sys.modules.items():
            if module and hasattr(module, function.name):
                func = getattr(module,  function.name)
                break
        if func:
            result = func(func_args) 
            #logger.debug(f"调用{function.name}查到数据 : {result}")
            return result
    else: 
        logger.error("未能进行function calling...") 
        raise Exception("调用function calling失败....")
    return None

def get_embeddings(s):
    client = launch_client()
    response = client.embeddings.create(
        model=constant.EMBEDDING, #填写需要调用的模型名称
        input=s,
    )
    return response.data[0].embedding

